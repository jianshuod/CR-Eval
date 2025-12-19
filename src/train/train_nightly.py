# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import time
import torch
import deepspeed
import contextlib
import transformers
from pathlib import Path
from datasets import Split
import src.utils.conversation as conversation_lib
from src.utils import print_rank_0, warning_rank_0
from src.utils.file import get_all_files_in_directory
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from src.train.trainer import CRiticTrainer, CRiticDatasetCommunicationCallback
from src.train.utils import find_all_linear_names, safe_save_model_for_hf_trainer
from src.train.data.crdataset import (
    CPTCRiticDataset,
    DataManager,
    DataCollatorForCRiticSFTDataset,
    BlendedDataset,
)
from src.train.args import (
    inherit_args,
    ModelArguments,
    DataArguments,
    TrainingArguments,
    save_args_to_json,
)


def replace_with_scaled_rope(model, scale=1.0):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    for name, module in model.named_modules():
        if isinstance(module, LlamaRotaryEmbedding):
            module.register_buffer(
                "inv_freq", module.inv_freq * scale, persistent=False
            )

    return model


def override_tokenizer_config(model_dir):
    import os
    import json

    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")

    if os.path.exists(tokenizer_config_path):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                with open(tokenizer_config_path, "r") as f:
                    configs = json.load(f)  # Correct way to load JSON from file

                # Modify the desired key
                configs["tokenizer_class"] = "LlamaTokenizerFast"

                with open(tokenizer_config_path, "w") as f:
                    f.write(
                        json.dumps(configs, indent=4)
                    )  # Writing back the changes with proper formatting

                warning_rank_0(
                    f"[Attention] Have overridden the {tokenizer_config_path} with "
                    f"`tokenizer_class` as {configs['tokenizer_class']}. "
                    "Make sure that you are aware of its effect.",
                )
            else:
                time.sleep(5)


def delete_unexpected_tokens(tokenizer):
    from TokenizerChanger import TokenizerChanger

    changer = TokenizerChanger(tokenizer)
    changer.delete_tokens(["<unk>"], include_substrings=False)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    inherit_args(data_args, training_args, "dataloader_num_workers")
    inherit_args(data_args, training_args, "num_train_epochs")

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_func,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
    )

    model.config.use_cache = False

    if model_args.emulate_long_context:
        scaling_factor = (
            training_args.model_max_length / data_args.max_chunk_size_in_pretraining
        )
        replace_with_scaled_rope(model, scaling_factor)

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        print_rank_0("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

        if training_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)
                    if training_args.fp16:
                        module = module.to(torch.float16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        if training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
                        if training_args.fp16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.float16)
        model.print_trainable_parameters()
    elif training_args.enable_prefix_tuning:
        from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

        peft_config = PromptTuningConfig(
            task_type="CAUSAL_LM",
            prompt_tuning_init=PromptTuningInit.RANDOM,
            num_virtual_tokens=training_args.num_virtual_tokens,
            tokenizer_name_or_path=model_args.model_name_or_path,
        )
        # task_inst = """<|start_header_id|>system<|end_header_id|>\n\nYou are a cellular network protocol expert. Given a revision of the 3GPP protocol statements, you summarize the revision, reason about the revision, and envision the negative outcomes without the revision. The revision is given in the form of diff, with [+] being added, [-] being removed, and [ ] being untouched.  Then, you prepare a change request, which has the three fields:\n- **SUMMARY OF CHANGE**: Provide a summary of the necessary changes to the specifications.\n- **REASON FOR CHANGE**: Explain why the identified flaws need to be addressed.\n- **CONSEQUENCES IF NOT REVISED**: Describe the potential negative impacts if the proposed changes are not made.\n\nYou should avoid missing important statements and try your best to return detailed responses rich in reasoning.<|eot_id|>"""
        # peft_config = PromptTuningConfig(
        #     task_type="CAUSAL_LM",
        #     prompt_tuning_init=PromptTuningInit.TEXT,
        #     prompt_tuning_init_text=task_inst,
        #     tokenizer_name_or_path=model_args.model_name_or_path,
        #     num_virtual_tokens=training_args.num_virtual_tokens,
        # )

        print_rank_0("Adding PrefixTuning adapters...")
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    # override_tokenizer_config(model_args.model_name_or_path)

    # Different platforms (or codebase version) has different add_special_tokens behavior when loading PreTrainedTokenizerFast
    # This will instantiate a LLaMATokenizerFast
    # LLaMATokenizerFast will automatically add an `<unk>` token.
    # See `https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/156` for reference
    # To avoid the problem, set the unk_token to None
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_bos=True,
        verbose=False,
        legacy=False,
    )

    """
        There are a series of workarounds (not sure about correctness).
        Finally, I adopt the following solution and combine the smart_resize_tokenizer of llava:  
        https://huggingface.co/docs/transformers/main/en/model_doc/llama3
    """
    if tokenizer.pad_token is None:
        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict=dict(pad_token="<pad>"),
        #     tokenizer=tokenizer,
        #     model=model,
        # )
        # It is noticed that LLaMA-3.1 herd has a great number of reserved special tokens.
        if model_args.base_model == "llama-3.1":
            tokenizer.pad_token = "<|reserved_special_token_247|>"
        elif model_args.base_model == "codellama":
            tokenizer.pad_token = "▁<EOT><EOT><EOT><EOT><EOT><EOT>"
        elif model_args.base_model == "codellama-2":
            tokenizer.pad_token = "给"
        # So, we just use on reserved token (or any token that will never be used in the target task) to escape the warning and performance regression
        # You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding dimension will be 128257. This might induce some performance reduction as *Tensor Cores* will not be available. For more details about this, or help on choosing the correct value for resizing, refer to this guide: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
        warning_rank_0(f"Adding pad token as '{tokenizer.pad_token}'")

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "llama3"
        ]
    print_rank_0(
        f"Using conversation format: {conversation_lib.default_conversation.version}"
    )

    if not training_args.is_pretraining:
        data_manager = DataManager(tokenizer, data_args)
        train_dataset = data_manager.get_dataset(Split.TRAIN)
        val_dataset = data_manager.get_dataset(Split.VALIDATION)
        test_dataset = data_manager.get_dataset(Split.TEST)
        max_seq_length4training_collator = training_args.model_max_length
    elif data_args.two_stage_combined:
        # read all files in data_dir as pretraining data
        data_manager = DataManager(tokenizer, data_args)
        # In this case, we use the train_file arg to specify the SFT data;
        # use the cache_file to specify the CPT data
        train_dataset_1 = CPTCRiticDataset(
            [], tokenizer, data_args, cache_path=data_args.cache_path
        )
        train_dataset_2 = data_manager.get_dataset(Split.TRAIN)
        train_dataset = BlendedDataset(
            [train_dataset_1, train_dataset_2],
            [len(train_dataset_1), len(train_dataset_2)],
            size=None,
            data_args=data_args,
        )
        val_dataset = data_manager.get_dataset(Split.VALIDATION)
        test_dataset = data_manager.get_dataset(Split.TEST)
        max_seq_length4training_collator = training_args.model_max_length
    else:
        # read all files in data_dir as pretraining data
        data_manager = DataManager(tokenizer, data_args, training_args.is_pretraining)
        data_path_list = get_all_files_in_directory(data_args.train_file)
        train_dataset = CPTCRiticDataset(data_path_list, tokenizer, data_args)
        val_dataset = data_manager.get_dataset(Split.VALIDATION)
        test_dataset = data_manager.get_dataset(Split.TEST)
        max_seq_length4training_collator = data_args.max_chunk_size_in_pretraining

    if data_args.training_sample_max_length >= 0:
        max_seq_length4training_collator = data_args.training_sample_max_length
    warning_rank_0(
        (
            f"The maximal training_sample token number is {max_seq_length4training_collator}.\n"
            f"Any training examples with longer length will be cut off to be within the setting."
        )
    )

    warning_rank_0(
        f"Tokenizer vocab size: {len(tokenizer)}\n"
        f"Model embedding size: {model.get_input_embeddings()}\n"
        "They should be aligned to pass the indexing assertion."
    )

    context = (
        deepspeed.zero.GatheredParameters(
            model.get_input_embeddings().weight, modifier_rank=0
        )
        if is_deepspeed_zero3_enabled()
        else contextlib.nullcontext()
    )
    with context:
        assert (
            len(tokenizer) == model.get_input_embeddings().weight.shape[0]
        ), "Failed to pass the assertion"

    # early_stop = EarlyStoppingAfterOneEpochCallback(20, 0)

    callbacks = [CRiticDatasetCommunicationCallback()]

    trainer = CRiticTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=DataCollatorForCRiticSFTDataset(
            tokenizer=tokenizer, max_seq_length=max_seq_length4training_collator
        ),
        eval_data_collator=DataCollatorForCRiticSFTDataset(
            tokenizer=tokenizer, max_seq_length=tokenizer.model_max_length
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    if training_args.resume_from:
        trainer.train(resume_from_checkpoint=training_args.resume_from)
    elif list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if test_dataset is not None:
        trainer.predict(test_dataset)

    if model_args.emulate_long_context:  # Restore the rope setting before saving
        scaling_factor = (
            data_args.max_chunk_size_in_pretraining / training_args.model_max_length
        )
        replace_with_scaled_rope(model, scaling_factor)

    trainer.save_state()
    save_args_to_json(model_args, data_args, training_args, training_args.output_dir)

    model.config.use_cache = True

    if training_args.lora_enable:
        merged_model = model.merge_and_unload()
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            merged_model.config.save_pretrained(training_args.output_dir)
            merged_model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
    elif training_args.enable_prefix_tuning:
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            trainer.model.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()

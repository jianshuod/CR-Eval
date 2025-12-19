import time
import torch
import transformers
from peft import PeftModel
import src.utils.conversation as conversation_lib
from src.utils import print_rank_0, warning_rank_0
from src.train.data.crdataset import smart_tokenizer_and_embedding_resize
from src.train.args import (
    inherit_args,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)


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


def merge_lora(attn_implementation="flash_attention_2"):

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

    output_dir = training_args.output_dir

    peft_model_id = training_args.use_lora

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
        attn_implementation=attn_implementation,
        torch_dtype=compute_dtype,
        **bnb_model_from_pretrained_args,
    )

    # model.config.use_cache = False

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

    model = PeftModel.from_pretrained(model, peft_model_id)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(output_dir)

    # override_tokenizer_config(model_args.model_name_or_path)

    # Different platforms (or codebase version) has different add_special_tokens behavior when loading PreTrainedTokenizerFast
    # This will instantiate a LLaMATokenizerFast
    # LLaMATokenizerFast will automatically add an `<unk>` token.
    # See `https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/156` for reference
    # To avoid the problem, set the unk_token to None
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    """
        There are a series of workarounds (not sure about correctness).
        Finally, I adopt the following solution and combine the smart_resize_tokenizer of llava:  
        https://huggingface.co/docs/transformers/main/en/model_doc/llama3
    """
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            print_rank_0(f"Adding pad token as '[PAD]'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    else:
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

    tokenizer.save_pretrained(output_dir)

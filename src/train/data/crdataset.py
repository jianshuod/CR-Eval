import os
import time
import json
import copy
import ftfy
import torch
import random
import hashlib
import tokenizers
import numpy as np
import transformers
from tqdm import tqdm
import multiprocessing
import torch.distributed
from pathlib import Path
from datasets import Split
import lm_dataformat as lmd
from packaging import version
from functools import partial
from threading import Semaphore
from collections import Counter
from dataclasses import dataclass
from collections import OrderedDict
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize
from src.train.args import DataArguments
from src.utils import generate_16_char_hash
from accelerate.utils import wait_for_everyone
from src.utils import print_rank_0, warning_rank_0
from src.make_data.framing import apply_task_framing
from src.make_data.play_with_cr import ChangeRequest
from src.utils import conversation as conversation_lib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union, Sequence
from src.utils.constants import IGNORE_INDEX, MAX_RESPONSE_TOKENS
from src.configs import (
    CACHE_DATASET_TOKENIZED_DIR,
    CACHE_DATASET_MESSAGES_DIR,
    MAGIC_NUM,
)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)

PERSONALIZATION_ANCHOR = (
    "You are a specialist in cellular network protocol and security analysis.\n\n"
)


@dataclass
class DataCollatorForCRiticSFTDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    max_seq_length: int
    pad_to_multiple_of: int = 8  # [24/11/03] New argument to specify padding multiple

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # The following truncation adheres to ==>
        #   length = min(self.tokenizer.model_max_length, input_ids.size(1))
        input_ids = input_ids[:, : self.max_seq_length]
        labels = labels[:, : self.max_seq_length]

        # If pad_to_multiple_of is set, pad up to the next multiple
        if self.pad_to_multiple_of:
            pad_length = (
                self.pad_to_multiple_of - input_ids.size(1) % self.pad_to_multiple_of
            ) % self.pad_to_multiple_of
            if pad_length > 0:
                # Pad input_ids and labels to pad_length along the last dimension
                input_ids = torch.nn.functional.pad(
                    input_ids, (0, pad_length), value=self.tokenizer.pad_token_id
                )
                labels = torch.nn.functional.pad(
                    labels, (0, pad_length), value=IGNORE_INDEX
                )

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch


def load_crs_from_jsonl(data_path):

    with open(data_path, "r") as f:
        cr_jsons = [json.loads(line) for line in f]

    crs = list()
    for cr_json in cr_jsons:
        try:
            cr = ChangeRequest(**cr_json)
            crs.append(cr)
        except TypeError:
            continue

    return crs


def load_crs_from_dir(data_dir):
    crs = list()
    for file in os.listdir(data_dir):
        if file.endswith(".jsonl"):
            crs.extend(load_crs_from_jsonl(os.path.join(data_dir, file)))
        elif file.endswith(".json"):
            with open(os.path.join(data_dir, file), "r") as f:
                cr_json = json.load(f)
                cr = ChangeRequest(**cr_json)
                crs.append(cr)

    return crs


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    To understand it, refer to https://nlp.stanford.edu/~johnhew/vocab-expansion.html
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,  # overried by the config
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # Bypass the system prompt
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    speakers = speakers[1:]
    target[:cur_idx] = IGNORE_INDEX

    ##? Mask out user prompts (+2?)
    ##* https://github.com/DAMO-NLP-SG/Video-LLaMA/issues/44
    ##* That corresponds to the `###`
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "user":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["role"]
        if from_str.lower() == "user":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "assistant":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["content"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["content"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["content"]
    conversation += BEGIN_SIGNAL
    return conversation


def _control_input_size(source, tokenizer):
    max_length = tokenizer.model_max_length - 10

    token_nums = _tokenize_fn([s["content"] for s in source], tokenizer)[
        "input_ids_lens"
    ]

    if sum(token_nums) < max_length:
        return source

    # Truncate source[2]["content"] if it exceeds MAX_RESPONSE_TOKENS
    if token_nums[2] > MAX_RESPONSE_TOKENS:
        source[2]["content"] = tokenizer.decode(
            tokenizer.encode(source[2]["content"])[:MAX_RESPONSE_TOKENS],
            skip_special_tokens=True,
        )
        token_nums[2] = MAX_RESPONSE_TOKENS  # Update the token count for source[2]

    # Calculate current token sum again after possible truncation of source[2]
    current_token_sum = sum(token_nums)

    # Directly truncate source[1]["content"] to fit within the max_length
    if current_token_sum >= max_length:
        allowed_tokens_for_source_1 = max_length - (current_token_sum - token_nums[1])
        source[1]["content"] = tokenizer.decode(
            tokenizer.encode(source[1]["content"])[:allowed_tokens_for_source_1],
            skip_special_tokens=True,
        )

    return source


def alter_introduction_formatting(content):
    """
    Randomly replace occurrences of ">>>" in the given content with a single consistent candidate from replacements_for_chevrons.

    Args:
        content (str): The input string containing occurrences of ">>>".

    Returns:
        str: The modified string with consistent replacements.
    """
    # List of replacement candidates
    replacements_for_chevrons = [
        ">>>",
        ">>>",
        ">>>",
        ">>>",
        "-->",
        ">>",
        ">>|",
        "***",
        "###",
        "===",
    ]

    # Select one random candidate for consistency
    selected_replacement = random.choice(replacements_for_chevrons)

    # Replace all occurrences of ">>>" with the selected candidate
    import re

    modified_content = re.sub(r">>>", selected_replacement, content)

    return modified_content


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    adaptive_masking=False,
    only_inst=False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"user": conv.roles[0], "assistant": conv.roles[1], "system": None}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["role"]] != conv.roles[0]:
            if source[0]["role"] == "system":
                conv.set_system(source[0]["content"])
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["content"])
        conversations.append(conv.get_prompt())

    if only_inst:
        return conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            # Adaptive masking: mask tokens in parts[1] that occur in parts[0]
            if adaptive_masking:
                instruction_tokens = target[
                    cur_len : cur_len + instruction_len
                ].tolist()
                response_tokens_start = cur_len + instruction_len

                for j in range(
                    len(target) - response_tokens_start - 1
                ):  # <EOT> should be predicted
                    if target[response_tokens_start + j] in instruction_tokens:
                        target[response_tokens_start + j] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank_0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        conversations=conversations,
    )


def preprocess_llama2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    adaptive_masking=False,
    only_inst=False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"user": conv.roles[0], "assistant": conv.roles[1], "system": None}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["role"]] != conv.roles[0]:
            if source[0]["role"] == "system" and source[0]["content"] != "":
                conv.set_system(source[0]["content"])
                source = source[1:]
            elif source[0]["role"] == "system" and source[0]["content"] == "":
                conv.use_default_system()
                source = source[1:]
            else:  # Missing system prompt
                conv.use_default_system()
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["content"])
        conversations.append(conv.get_prompt())

    if only_inst:
        return conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA2

    cur_lens = []
    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            # Adaptive masking: mask tokens in parts[1] that occur in parts[0]
            if adaptive_masking:
                instruction_tokens = target[
                    cur_len : cur_len + instruction_len
                ].tolist()
                response_tokens_start = cur_len + instruction_len

                for j in range(
                    len(target) - response_tokens_start - 1
                ):  # <EOT> should be predicted
                    if target[response_tokens_start + j] in instruction_tokens:
                        target[response_tokens_start + j] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank_0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        cur_lens.append(cur_len)

    return dict(
        input_ids=input_ids,
        labels=targets,
        conversations=conversations,
        token_num=cur_lens,
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    adaptive_masking=False,
    only_inst=False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"user": conv.roles[0], "assistant": conv.roles[1], "system": None}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["role"]] != conv.roles[0]:
            if source[0]["role"] == "system" and source[0]["content"] != "":
                conv.set_system(source[0]["content"])
                source = source[1:]
            elif source[0]["role"] == "system" and source[0]["content"] == "":
                conv.use_default_system()
                source = source[1:]
            else:  # Missing system prompt
                conv.use_default_system()
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["content"])
        conversations.append(conv.get_prompt())

    if only_inst:
        return conversations

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    cur_lens = []
    # Mask targets
    sep = conv.sep + conv.roles[1]  # to split the instruction and the response
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(
            3, len(rounds), 2
        ):  # process the rest of the conversation
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = (
                len(tokenizer(rou).input_ids) + 1
            )  # why +1? Answer: the tokenizer corresponds to the (conv.sep)|<EOT_ID>| after the assistant message
            if i != 0:
                round_len -= 1  # As we explicitly force the tokenizer to add the bos, the following round should ignore the non-existing <bos>
            instruction_len = len(tokenizer(parts[0]).input_ids)
            if i != 0:
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            # Adaptive masking: mask tokens in parts[1] that occur in parts[0]
            if adaptive_masking:
                instruction_tokens = target[
                    cur_len : cur_len + instruction_len
                ].tolist()
                response_tokens_start = cur_len + instruction_len

                for j in range(
                    len(target) - response_tokens_start - 1
                ):  # <EOT> should be predicted
                    if target[response_tokens_start + j] in instruction_tokens:
                        target[response_tokens_start + j] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print_rank_0(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
        cur_lens.append(cur_len)

    return dict(
        input_ids=input_ids,
        labels=targets,
        conversations=conversations,
        token_num=cur_lens,
    )


def preprocess_plain_like_chat(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    adaptive_masking=False,
    only_inst=False,
) -> Dict:
    new_sources = []
    for source in sources:
        # split the content
        new_source = [
            {
                "role": "system",
                "content": "You are a specialist in cellular network protocol and security. Given an incomplete content, you are asked to complete the content.",
            }
        ]
        conversation = ""
        for idx, sentence in enumerate(source):
            if idx != 0:
                conversation += conversation_lib.default_conversation.sep  # '\n'
            conversation += sentence["content"]

        # Randomly split the conversation into two parts
        conversation_len = len(conversation)
        if conversation_len > 1:
            split_idx = random.randint(
                1, conversation_len - 1
            )  # Ensure a split that isn't at the start or end

            # First part: before the split index
            first_part = conversation[:split_idx].strip()
            # Second part: after the split index
            second_part = conversation[split_idx:].strip()

            # Add both parts to the new source in roles
            if first_part:
                new_source.append({"role": "user", "content": first_part})
            if second_part:
                new_source.append({"role": "assistant", "content": second_part})

        # Add the new source to the new_sources list
        new_sources.append(new_source)

    return preprocess(new_sources, tokenizer)


def preprocess_plain(
    sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, only_inst=False
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        conversation = ""
        for idx, sentence in enumerate(source):
            if sentence["role"] == "system":
                continue  # No system message.
            if idx > 1:
                conversation += "\n"
            conversation += sentence["content"]
        conversations.append(conversation)
    if only_inst:
        return conversations

    cur_lens = []
    input_ids = _tokenize_fn(conversations, tokenizer)["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        # Learn all tokens except the first one
        target[:1] = IGNORE_INDEX
        cur_lens.append(total_len)

    return dict(
        input_ids=input_ids,
        labels=targets,
        conversations=conversations,
        token_num=cur_lens,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    adaptive_masking=False,
    only_inst=False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, adaptive_masking, only_inst)
    if (
        conversation_lib.default_conversation.version == "llama2"
        or conversation_lib.default_conversation.version == "codellama"
    ):
        return preprocess_llama2(sources, tokenizer, adaptive_masking, only_inst)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer, only_inst)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(sources, tokenizer, adaptive_masking, only_inst)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    if only_inst:
        return conversations

    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn(
            [header] + [s["content"] for s in source], tokenizer
        )["input_ids_lens"]
        speakers = [sentence["role"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets, conversations=conversations)


def establish_message(instance, task_name, tokenizer):
    from src.analysis.task_worker import Tool
    from src.eval import ResponseEval

    tool = Tool(task_name, model="gpt-4o", using_shot=False)
    input_framing = lambda x: partial(tool.chat_engine.chat, only_prepare=True)(
        apply_task_framing(x, task_name)
    )
    eval_engine = ResponseEval(task_name, [], None)
    output_framing = partial(eval_engine.form_desirable_response, chat=True)

    # Use oracle retrieval
    instance["input_text"] = instance["change_list"][0]
    inputs = input_framing(instance)
    response = output_framing(instance)
    # Control the input size to avoid the missing of the response
    source = _control_input_size(inputs + response, tokenizer)

    return source


def establish_messages(task_name, instances, tokenizer, num_workers=None):
    # Determine the number of num_proc to use
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 4)

    # Create a partial function with fixed task_name and tokenizer parameters
    worker_func = partial(establish_message, task_name=task_name, tokenizer=tokenizer)

    sources = []
    # Use ThreadPoolExecutor for non-blocking parallel execution
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(worker_func, instance): instance for instance in instances
        }
        for future in tqdm(
            as_completed(futures),
            total=len(instances),
            desc=f"Establishing messages for {task_name}",
        ):
            sources.append(future.result())

    return sources


def filter_task_instances_via_task(instances, task_names, accepted_tasks):
    assert len(instances) == len(
        task_names
    ), "The number of instances and task names should be the same."

    # a set of task names that are ruled out
    deprecated_tasks = set()

    filtered_instances = []
    filtered_task_names = []
    for instance, task_name in zip(instances, task_names):
        if task_name in accepted_tasks:
            filtered_instances.append(instance)
            filtered_task_names.append(task_name)
        else:
            deprecated_tasks.add(task_name)

    warning_msg = (
        f"Accepted tasks: {accepted_tasks}\n"
        f"Only {len(filtered_instances)} instances are kept.\n"
        f"Deprecated tasks: {deprecated_tasks}"
    )

    warning_rank_0(warning_msg)

    return filtered_instances, filtered_task_names


def remove_task_instances_via_task(instances, task_names, removed_tasks):
    assert len(instances) == len(
        task_names
    ), "The number of instances and task names should be the same."

    # a set of task names that are ruled out
    sustained_tasks = set()

    sustained_instances = []
    sustained_task_names = []
    for instance, task_name in zip(instances, task_names):
        if task_name not in removed_tasks:
            sustained_instances.append(instance)
            sustained_task_names.append(task_name)
        else:
            sustained_tasks.add(task_name)

    warning_msg = (
        f"Removed tasks: {removed_tasks}\n"
        f"Only {len(sustained_instances)} instances are kept.\n"
        f"Sustained tasks: {sustained_tasks}"
    )

    warning_rank_0(warning_msg)

    return sustained_instances, sustained_task_names


def filter_task_instances_lt_response_token_num(
    instances, task_names, the_least_response_token_num, tokenizer
):
    # Assert that instances and task names match in length
    assert len(instances) == len(
        task_names
    ), "The number of instances and task names should be the same."

    # List comprehension to filter based on response token count
    filtered_data = [
        (instance, task_name)
        for instance, task_name in tqdm(zip(instances, task_names))
        if tokenizer(
            instance[2 if task_name != "general" else 1]["content"],
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        .input_ids.ne(tokenizer.pad_token_id)
        .sum()
        .item()
        >= the_least_response_token_num
    ]

    # Split the filtered data back into two lists (instances and task names)
    filtered_instances, filtered_task_names = (
        zip(*filtered_data) if filtered_data else ([], [])
    )

    # Warning message indicating how many instances are kept
    warning_msg = (
        f"Only {len(filtered_instances)} instances are kept.\n"
        f"Response token num >= {the_least_response_token_num}"
    )

    warning_rank_0(warning_msg)

    return list(filtered_instances), list(filtered_task_names)


def filter_task_instances_gt_response_token_num(
    instances, task_names, the_most_query_token_num, tokenizer
):
    # Assert that instances and task names match in length
    assert len(instances) == len(
        task_names
    ), "The number of instances and task names should be the same."

    # List comprehension to filter based on response token count
    filtered_data = [
        (instance, task_name)
        for instance, task_name in zip(instances, task_names)
        if tokenizer(
            instance[1 if task_name != "general" else 0]["content"],
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        .input_ids.ne(tokenizer.pad_token_id)
        .sum()
        .item()
        <= the_most_query_token_num
    ]

    # Split the filtered data back into two lists (instances and task names)
    filtered_instances, filtered_task_names = (
        zip(*filtered_data) if filtered_data else ([], [])
    )

    # Warning message indicating how many instances are kept
    warning_msg = (
        f"Only {len(filtered_instances)} instances are kept.\n"
        f"Query token num <= {the_most_query_token_num}"
    )

    warning_rank_0(warning_msg)

    return list(filtered_instances), list(filtered_task_names)


class LazyCRiticDataset(Dataset):
    """Dataset for CRitic.

    Why using a customized dataset instead of .map?
    A: We have to establish a CR sample with multi-column fields.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        task_names,
        is_eval,
        data_args: DataArguments,
        cache_path: str = "",
    ):
        super(LazyCRiticDataset, self).__init__()

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.data_path = Path(data_path)
        self.task_names = task_names
        self.is_eval = is_eval
        self.adaptive_masking = data_args.adaptive_masking
        self.flexible_wording = data_args.flexible_wording
        self.accepted_tasks = data_args.accepted_tasks
        self.removed_tasks = data_args.removed_tasks
        self.data_counter = 0
        self.remove_short_responses_le = data_args.remove_short_responses_le
        self.remove_long_queries_ge = data_args.remove_long_queries_ge
        self.epoch = None
        self.total_epoch = data_args.num_train_epochs
        self.last_leap = data_args.last_leap
        self.personalized = data_args.personalized
        self.pre_clip = data_args.pre_clip
        self.runtime_clip = data_args.runtime_clip
        self.runtime_ppl_filter = data_args.runtime_ppl_filter
        self.data_aug = data_args.data_aug
        self.perturb_degree = data_args.perturb_degree
        self.general_mixup = data_args.general_mixup
        self.replicate_num = data_args.replicate_num
        self.remove_sys = data_args.remove_sys
        self.random_introduction_formatting = data_args.random_introduction_formatting
        self.balancing_replicates = data_args.balancing_replicates
        self.train_with_cpt = data_args.train_with_cpt

        # load the system instruction wording
        if self.flexible_wording and os.path.exists(self.flexible_wording):
            self._load_flexible_wordings()
            print_rank_0("Flexible Wordings are loaded.")
        else:
            self.system_inst_wording_maps = None

        # Each source is a conversation, composed of three turns:
        # [system, user, assistant]

        self.dataset_identifier = self._get_dataset_identifier()

        if cache_path == "" or self.is_eval:
            cache_path = (
                Path(CACHE_DATASET_MESSAGES_DIR) / f"{self.dataset_identifier}.jsonl"
            )

        if (
            self.data_args.cache_path and not self.is_eval
        ):  # Manually set cache_path for the training set
            cache_path = self.data_args.cache_path

        if not data_args.enable_cache or not self._check_cache_existing(cache_path):
            if self.data_path.is_file() and self.data_path.suffix == ".jsonl":
                self.sources_list, self.task_names = self._load_from_processed()
                type_str = "Preprocessed Convs"
            else:
                self.sources_list, self.task_names = self._load_from_raw_json()
                type_str = "Raw CRs"

            if len(self.accepted_tasks) != 0 and not self.is_eval:
                self.sources_list, self.task_names = filter_task_instances_via_task(
                    self.sources_list, self.task_names, self.accepted_tasks
                )

            if len(self.removed_tasks) != 0 and not self.is_eval:
                self.sources_list, self.task_names = remove_task_instances_via_task(
                    self.sources_list, self.task_names, self.removed_tasks
                )

            if self.remove_short_responses_le != -1 and not self.is_eval:
                self.sources_list, self.task_names = (
                    filter_task_instances_lt_response_token_num(
                        self.sources_list,
                        self.task_names,
                        self.remove_short_responses_le,
                        self.tokenizer,
                    )
                )

            if self.remove_long_queries_ge != -1 and not self.is_eval:
                self.sources_list, self.task_names = (
                    filter_task_instances_gt_response_token_num(
                        self.sources_list,
                        self.task_names,
                        self.remove_long_queries_ge,
                        self.tokenizer,
                    )
                )

            if self.pre_clip and not self.is_eval:
                print_rank_0(f"Start to clip data")
                # Clipping long samples
                clipped_counter = 0
                for sample in tqdm(self.sources_list):
                    data_dict = preprocess([sample], self.tokenizer, False)
                    if data_dict["token_num"][0] > self.tokenizer.model_max_length:
                        clipped_counter += 1
                        clip_ratio = (
                            data_dict["token_num"][0]
                            + 100
                            - self.tokenizer.model_max_length
                        ) / (self.tokenizer.model_max_length + 100)
                        sample[0 if len(sample) == 2 else 1]["content"] = (
                            self._random_clip(
                                sample[0 if len(sample) == 2 else 1]["content"],
                                clip_ratio,
                            )
                        )
                print_rank_0(f"Clipped {len(self.sources_list)} samples.")

            if self.balancing_replicates and not self.is_eval:
                print_rank_0(
                    f"Balancing redundant replicates (maximum set to {self.replicate_num})"
                )
                num_before_balancing = len(self.sources_list)
                self.sources_list, self.tasks = self._remove_redundant_replicates()
                num_after_balancing = len(self.sources_list)
                print_rank_0(
                    f"{num_after_balancing - num_before_balancing} samples are removed. (Before: {num_before_balancing}, After: {num_after_balancing})"
                )

            if data_args.enable_cache:
                # Save both sources_list and task_names to the cache
                cache_data = {
                    "magic": MAGIC_NUM,
                    "sources_list": self.sources_list,
                    "task_names": self.task_names,
                }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)
                print_rank_0(f"Saved {len(self.sources_list)} samples to {cache_path}")

        # Loading from cache (if needed elsewhere in your code)
        else:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
                self.sources_list = cache_data.get("sources_list", [])
                self.task_names = cache_data.get("task_names", [])
            type_str = "Cached Convs"

        print_rank_0(f"[Type: {type_str}] Loaded {len(self.sources_list)} samples.")

        self.masking_or_not = (
            [True] * len(self.sources_list)
            if self.adaptive_masking
            else [False] * len(self.sources_list)
        )
        self.sample_lens = [0] * len(self.sources_list)
        self.sample_utilities = [True] * len(self.sources_list)

        if not self.is_eval:
            self.general_sources_index = [
                i
                for i in range(len(self.sources_list))
                if self.task_names[i] == "general"
            ]

        if (
            self.general_mixup
            and not self.is_eval
            and len(self.general_sources_index) == 0
        ):
            warning_msg = "No general task samples are found for mixup."
            warning_rank_0(warning_msg)
            exit(0)

        # [Flexible wording related]
        self.flexible_wording_counter = Counter()

    def communicate_with_trainer(
        self, epoch, start=0, model=None, args=None, collate_fn=None
    ):
        self.epoch = epoch

        # if self.general_mixup and not self.is_eval and

        if self.data_aug and not self.is_eval and start == 1:  # at the end of epoch
            self._dataset_evolving_inplace()

        if self.runtime_ppl_filter and not self.is_eval and start == 1:
            bs = args.per_device_eval_batch_size
            self._runtime_ppl_filter(model, bs, collate_fn)

        if int(self.epoch) == int(self.total_epoch) - 1 and self.last_leap:
            self.system_inst_wording_maps = None
            print_rank_0(
                f"Current training progress: {int(self.epoch)} / {self.total_epoch - 1} epoches.\n Disable flexible_wording due to the last_leap config"
            )

    def _remove_redundant_replicates(self):

        sig_counter = Counter()

        def sig_generate(source):  # User query as sig
            return source[1]["content"][:250] + source[1]["content"][-250:]

        new_sources = []
        new_tasks = []
        for source, task in zip(self.sources_list, self.task_names):
            sig = sig_generate(source)

            # Check sig_counter
            if sig_counter[sig] < self.replicate_num:  # keep it
                new_sources.append(source)
                new_tasks.append(task)
                sig_counter[sig] += 1
            else:
                pass

        return new_sources, new_tasks

    def _runtime_ppl_filter(self, ref_model, bs, collate_fn):
        ref_model.eval()
        all_ppl = []
        updated_sample_utilities = copy.deepcopy(self.sample_utilities)

        # Only rank 0 processes all data since with ZeRO-2, model parameters are sharded
        if torch.distributed.get_rank() == 0:
            with torch.no_grad():
                # Process data in batches
                for i in range(0, len(self), bs):
                    batch = [self[j] for j in range(i, min(i + bs, len(self)))]
                    collated = collate_fn(batch)
                    input_ids = collated["input_ids"].to(ref_model.device)
                    labels = collated["labels"].to(ref_model.device)

                    outputs = ref_model(input_ids, labels=labels)
                    batch_ppl = torch.exp(
                        outputs.loss / labels.ne(self.tokenizer.pad_token_id).sum(dim=1)
                    )
                    all_ppl.extend(batch_ppl.tolist())

                # Calculate threshold and update utilities
                all_ppl_tensor = torch.tensor(all_ppl)
                num_to_filter = int(0.2 * len(all_ppl))
                threshold = torch.topk(
                    all_ppl_tensor, num_to_filter, largest=False
                ).values.max()

                for i, ppl in enumerate(all_ppl):
                    updated_sample_utilities[i] = False if ppl <= threshold else True

        # Synchronize the updated utilities across all ranks
        torch.distributed.barrier()
        torch.distributed.broadcast_object_list([updated_sample_utilities], src=0)

        self.sample_utilities = updated_sample_utilities[0]
        report_msg = f"[Runtime PPL Filter] Filtered {len(all_ppl) - sum(self.sample_utilities)} samples.\n Remaining: {sum(self.sample_utilities)}"
        print_rank_0(report_msg)

    def _random_clip(self, input_str, clip_ratio):
        sentences = input_str.split(".")
        num_sentences_to_remove = int(clip_ratio * len(sentences))
        if num_sentences_to_remove == 0:
            return input_str
        else:
            sentences_to_remove = random.sample(sentences, num_sentences_to_remove)
            return ".".join(
                [
                    sentence
                    for sentence in sentences
                    if sentence not in sentences_to_remove
                ]
            )

    def _random_combine(self, source_str, ref_str, clip_ratio):
        sentences = source_str.split(".")
        ref_str_sentences = ref_str.split(".")
        num_sentences_to_insert = int(clip_ratio * len(ref_str_sentences))
        if num_sentences_to_insert == 0:
            return source_str
        else:
            sentences_to_insert = random.sample(
                ref_str_sentences, num_sentences_to_insert
            )
            # randomly choose one point to insert the ref_sentences
            insert_point = random.randint(0, len(sentences))
            new_sentences = (
                sentences[:insert_point]
                + sentences_to_insert
                + sentences[insert_point:]
            )
            return ".".join(new_sentences)

    def _check_cache_existing(self, cache_path):
        """
        Checks if the cache file exists and validates the magic number.

        Args:
            cache_path (str): The path to the cache file.
            expected_magic (str): The expected magic number to validate against.

        Returns:
            bool: True if the cache file exists and the magic number is valid, False otherwise.
        """
        expected_magic = MAGIC_NUM

        if os.path.exists(cache_path):
            print_rank_0(f"Cache file found at {cache_path}")

            # Load the cache data
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)

                # Check the magic number
                if cache_data.get("magic") == expected_magic:
                    print_rank_0("Magic number is valid.")
                    return True
                else:
                    print_rank_0(
                        f"Invalid magic number: {cache_data.get('magic')}. Expected {expected_magic}."
                    )
                    return False

            except Exception as e:
                print_rank_0(f"Failed to read or parse cache file: {e}")
                return False
        else:
            print_rank_0(f"No cache file found at {cache_path}")
            return False

    def _get_dataset_identifier(self) -> Optional[str]:
        """
        Generates a dataset identifier hash based on important dataset configurations.

        Returns:
            Optional[str]: A hash representing the dataset identifier.
        """
        # List of things to include in the hash for a unique identifier
        hash_of_config = generate_16_char_hash(
            "".join(
                [
                    generate_16_char_hash(str(something))
                    for something in (
                        self.data_args.task_names
                        + self.accepted_tasks
                        + self.removed_tasks
                        + [self.is_eval]
                        + list(self.data_args.__dict__.values())
                    )
                ]
            )
        )

        # Combine all hashes to form a unique identifier
        return hash_of_config

    def _load_from_processed(self):
        """In some cases, we have already processed the data and saved it in a cache.
        We directly load from a jsonl file.
        """
        if not self.is_eval:
            sources_list = []
            task_names = []
            with open(self.data_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        sources_list.append(data["messages"])
                        task_names.append(data["task"])
                    elif isinstance(data, list):
                        sources_list.append(data)
                        task_names.append("unknown")
                    else:
                        raise NotImplementedError(
                            f"Unsopported data type: {type(data)}"
                        )
            return sources_list, task_names
        else:
            sources_list = []
            with open(self.data_path, "r") as f:
                for line in f:
                    messages = json.loads(line)
                    sources_list.append(messages)

            return sources_list, None

    def _load_from_raw_json(self):

        list_data_dict = load_crs_from_dir(self.data_path)

        tokenizer = self.tokenizer
        num_workers = self.data_args.dataloader_num_workers

        sources = list()
        task_names = list()
        for task_name in self.task_names:
            new_sources = establish_messages(
                task_name, list_data_dict, tokenizer, num_workers
            )
            sources.extend(new_sources)
            task_names.extend([task_name] * len(new_sources))

        del list_data_dict

        return sources, task_names

    def _load_flexible_wordings(self):
        # will be a json file in the formt {"task_name": [wording1, wording2, ...]}
        with open(self.flexible_wording, "r") as f:
            system_inst_wording_maps = json.loads(f.read())
            self.system_inst_wording_maps = system_inst_wording_maps
            info_to_print = "Successfully loaded the following wordings\n"
            for task, wordings in system_inst_wording_maps.items():
                info_to_print += f"Task ({task}): {len(wordings)} wordings\n"
            warning_rank_0(info_to_print)

    def _alter_inst(self, task_belonging, original_inst):
        inst_pool = [original_inst]

        if self.system_inst_wording_maps and not self.is_eval:
            inst_pool_a = self.system_inst_wording_maps.get(task_belonging, None)
            # add the original one
            inst_pool = [original_inst] + inst_pool_a if inst_pool_a else inst_pool
            inst_pool = inst_pool[: self.replicate_num]
        else:
            pass

        selected_index = random.choice(range(len(inst_pool)))

        global_index_key = f"{task_belonging}_{selected_index}"

        self.flexible_wording_counter[global_index_key] += 1

        return inst_pool[selected_index]

    def _dataset_evolving_inplace(self):
        """
        The dataset is evolving. We need to re-construct the dataset in place.

        This method should be called in the training loop when num_epochs > 1.

        We assume that the dataset belongs to the same task, and the data is in the same format.

        Two trends:
            1. If the data is too long, randomly remove segments from it;
            2. If the data is too short, randomly combine them.
        """
        print_rank_0("The dataset is evolving.")
        perturb_degree = self.perturb_degree
        updated_sources_list = copy.deepcopy(self.sources_list)

        if torch.distributed.get_rank() == 0:
            sample_lens = [
                preprocess([source], self.tokenizer)["token_num"][0]
                for source in tqdm(updated_sources_list)
            ]

            # Fetch the top 10% of the longest samples
            top_10_percent = int(len(sample_lens) * 0.2)
            top_10_percent_indices = np.argsort(sample_lens)[-top_10_percent:]
            others_indices = np.argsort(sample_lens)[:-top_10_percent]

            for i in tqdm(
                range(len(sample_lens)),
                desc=f"Dataset evolving {torch.distributed.get_rank()}",
            ):
                if i in top_10_percent_indices:
                    # Modify the content by clipping it if it's too long
                    updated_sources_list[i][1]["content"] = self._random_clip(
                        updated_sources_list[i][1]["content"], perturb_degree
                    )
                elif i in others_indices:
                    # Randomly sample another source from the whole updated_sources_list for combination
                    another_idx = random.choice(range(len(sample_lens)))
                    combined_content = self._random_combine(
                        updated_sources_list[i][1]["content"],
                        updated_sources_list[another_idx][1]["content"],
                        perturb_degree,
                    )
                    updated_sources_list[i][1]["content"] = combined_content
        # The dataset is modified in place, so no need to extend or reassign the lists.

        wait_for_everyone()

        torch.distributed.broadcast_object_list(updated_sources_list, src=0)

        self.sources_list = updated_sources_list

    def __len__(self):
        return len(self.sources_list)

    def __getitem__(self, idx):

        sources = self.sources_list[idx]

        # We only use the single-turn conversation
        cut_point = None
        for i, message in enumerate(sources):
            if i >= 2 and message["role"] == "user":
                cut_point = i
                break
        sources = sources[:cut_point]

        if self.flexible_wording and not self.is_eval:
            sources[0]["content"] = self._alter_inst(
                self.task_names[idx], sources[0]["content"]
            )

        if self.random_introduction_formatting and not self.is_eval:
            sources[-1]["content"] = alter_introduction_formatting(
                sources[-1]["content"]
            )

        if self.personalized:
            if sources[0]["content"].startswith("You are"):
                sources[0]["content"] = PERSONALIZATION_ANCHOR + ".".join(
                    sources[0]["content"].split(".")[1:]
                )
            else:
                sources[0]["content"] = PERSONALIZATION_ANCHOR + sources[0]["content"]

        if isinstance(idx, int):
            sources = [sources]

        if not self.is_eval and self.train_with_cpt:
            data_dict = preprocess_plain(sources, self.tokenizer)
        else:
            data_dict = preprocess(sources, self.tokenizer, adaptive_masking=False)

        self.sample_lens[idx] = data_dict["token_num"][0]

        if (
            data_dict["token_num"][0] > self.tokenizer.model_max_length
            and not self.is_eval
        ):
            clip_ratio = (
                data_dict["token_num"][0] - self.tokenizer.model_max_length + 100
            ) / (data_dict["token_num"][0] + 100)
            sources[0][1]["content"] = self._random_clip(
                sources[0][1]["content"], clip_ratio
            )
            data_dict = preprocess(sources, self.tokenizer, adaptive_masking=False)
        else:
            pass

        if self.data_counter % 100000 == 0 and not self.is_eval:
            eval_flag = "Eval" if self.is_eval else "Training"
            print_rank_0(f"{eval_flag} {self.task_names[idx]} {data_dict}")

        self.data_counter += 1

        if isinstance(idx, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

            if self.remove_sys:
                first_occurrence_idx = (
                    (data_dict["input_ids"] == 128009)
                    .nonzero(as_tuple=True)[0]
                    .min()
                    .item()
                )  # Remove the whole system message plus the bos
                # Keep only the elements starting after the first occurrence of 12009
                data_dict["labels"] = data_dict["labels"][first_occurrence_idx + 1 :]
                data_dict["input_ids"] = data_dict["input_ids"][
                    first_occurrence_idx + 1 :
                ]

            # Remove useless samples (although the compute will be wasted)
            if not self.is_eval and not self.sample_utilities[idx]:
                data_dict["labels"][:] = IGNORE_INDEX

        return data_dict

    def save_cache(self):

        if len(self.cached_data_dict) == 0:
            # Read through the dataset
            for idx in range(len(self)):
                _ = self[idx]

        cache_path = (
            Path(CACHE_DATASET_TOKENIZED_DIR) / f"{self.dataset_identifier}.jsonl"
        )

        with open(cache_path, "w") as f:
            for cached_data in self.cached_data_dict.values():
                serializable_data = self.make_serializable(cached_data)
                f.write(json.dumps(serializable_data) + "\n")
        print_rank_0(f"Saved {len(self.cached_data_dict)} samples to {cache_path}")

    def make_serializable(self, data):

        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Convert Tensor to list or other serializable format
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
        return serializable_data

    def on_train_end(self):
        if self.flexible_wording and not self.is_eval:
            warning_rank_0(
                f"Flexible Wordings Counter:\n{self.flexible_wording_counter}"
            )


def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


class CPTCRiticDataset(LazyCRiticDataset):
    """Using the same training strategy with SFT. But the data is from CPT."""

    def __init__(
        self,
        data_path_list: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        cache_path: str = "",
    ):

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.args = data_args
        self.adaptive_masking = data_args.adaptive_masking
        self.flexible_wording = data_args.flexible_wording
        self.accepted_tasks = data_args.accepted_tasks
        self.removed_tasks = data_args.removed_tasks
        self.data_counter = 0
        self.remove_short_responses_le = data_args.remove_short_responses_le
        self.remove_long_queries_ge = data_args.remove_long_queries_ge
        self.epoch = None
        self.total_epoch = data_args.num_train_epochs
        self.last_leap = data_args.last_leap
        self.personalized = data_args.personalized
        self.pre_clip = data_args.pre_clip
        self.runtime_clip = data_args.runtime_clip
        self.runtime_ppl_filter = data_args.runtime_ppl_filter
        self.data_aug = data_args.data_aug
        self.perturb_degree = data_args.perturb_degree
        self.general_mixup = data_args.general_mixup
        self.replicate_num = data_args.replicate_num
        self.remove_sys = data_args.remove_sys

        cache_identifier = self._get_dataset_identifier(data_path_list)

        if self.args.cache_path != "":
            cache_path = self.args.cache_path

        self.dataset_identifier = cache_identifier

        if cache_path == "":
            cache_path = Path(CACHE_DATASET_MESSAGES_DIR) / f"{cache_identifier}.jsonl"

        if not self.args.enable_cache or not self._check_cache_existing(cache_path):

            # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
            # hence building up memory
            semaphore = Semaphore(10000 + data_args.dataloader_num_workers)

            # use multiprocessing to iterate over input documents
            #! text should be stored in the `text` field of the json objects
            fin = yield_from_files(data_path_list, semaphore)

            selected_chunking_fn = self._construct_chat_with_overlap

            if data_args.dataloader_num_workers > 1:
                pool = multiprocessing.Pool(data_args.dataloader_num_workers)
                chunked_sources = pool.imap(selected_chunking_fn, fin)
            else:
                chunked_sources = (selected_chunking_fn(text) for text in fin)

            sources_list = []

            # actually do tokenization
            proc_start = time.time()
            total_chunks = 0
            pbar = tqdm()
            for i, chunks in enumerate(chunked_sources, start=1):
                total_chunks += len(chunks)

                # release semaphore so `yield_from_files` can add another file to the buffer
                semaphore.release()

                sources_list.extend(chunks)

                # log progress
                current = time.time()
                elapsed = current - proc_start
                pbar.set_description(
                    f"Processed {i}{''} docs ({i / elapsed :.2f} docs/s)."
                )

            if self.args.enable_cache:
                cache_data = {
                    "magic": MAGIC_NUM,
                    "sources_list": sources_list,
                }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)
                print_rank_0(f"Saved {len(sources_list)} samples to {cache_path}")
        else:  # load from cache
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
                sources_list = cache_data.get("sources_list", [])

            print_rank_0(f"Loaded {len(sources_list)} samples from {cache_path}")

        # Each source should be a ``[{"content": "xx"}]``''
        self.sources_list = sources_list
        self.task_names = ["knowledge-learning"] * len(sources_list)

    def __getitem__(self, idx):

        sources = self.sources_list[idx]

        # We only use the single-turn conversation
        cut_point = None
        for i, message in enumerate(sources):
            if i >= 2 and message["role"] == "user":
                cut_point = i
                break
        sources = sources[:cut_point]

        if self.personalized:
            if sources[0]["content"].startswith("You are"):
                sources[0]["content"] = PERSONALIZATION_ANCHOR + ".".join(
                    sources[0]["content"].split(".")[1:]
                )
            else:
                sources[0]["content"] = PERSONALIZATION_ANCHOR + sources[0]["content"]

        if isinstance(idx, int):
            sources = [sources]

        data_dict = preprocess(sources, self.tokenizer, adaptive_masking=False)

        if self.data_counter % 100000 == 0:
            print_rank_0(f"{self.task_names[idx]} {data_dict}")

        self.data_counter += 1

        if isinstance(idx, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        return data_dict

    def _construct_chat_with_overlap(self, text):
        """
        Given a text string, splits it into sentences and iterates over them.
        For each tuple in the final list of tuples, each tuple contains two parts.
        The combined token number of them should be less than max_chunk_size, and
        the token number of the first element in the tuple should be more than min_overlap_size.
        For the first tuple, start filling from the beginning of the text; otherwise,
        read from the end of the previous tuple's second element. This ensures that between
        adjacent tuples, the first element of the tuple always overlaps with the second
        element of the previous tuple.
        """
        if self.args.use_ftfy:
            text = ftfy.fix_text(text)

        # Tokenize the text into sentences using NLTK
        sentences = sent_tokenize(text)

        max_chunk_size = self.args.max_chunk_size_in_pretraining
        min_overlap_size = self.args.min_overlap_size_in_pretraining

        pairs = []
        first_part = []
        second_part = []
        first_chunk = True
        first_part_size = 0
        second_part_size = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_tokens)

            if first_chunk:  # To get the first part filled
                first_part_size += sentence_length
                first_part.append(sentence)
                if first_part_size > min_overlap_size:
                    first_chunk = False
                continue

            if first_part_size + second_part_size + sentence_length > max_chunk_size:
                # Add the current chunk to the list of chunks
                pairs.append(
                    [
                        {
                            "role": "system",
                            "content": "Given an incomplete content, you are asked to use your domain knowledge and reasoning ability to complete the content.",
                        },
                        {"role": "user", "content": " ".join(first_part)},
                        {"role": "assistant", "content": " ".join(second_part)},
                    ]
                )
                # read back
                first_part = []
                first_part_size = 0
                for sentencet in reversed(second_part):
                    sentencet_tokens = self.tokenizer.encode(
                        sentencet, add_special_tokens=False
                    )
                    sentencet_length = len(sentencet_tokens)
                    first_part_size += sentencet_length
                    first_part.insert(0, sentencet)
                    if first_part_size >= min_overlap_size:
                        break
                second_part = []
                second_part_size = 0
            else:
                second_part.append(sentence)
                second_part_size += sentence_length

        return pairs

    def _check_cache_existing(self, cache_path):
        """
        Checks if the cache file exists and validates the magic number.

        Args:
            cache_path (str): The path to the cache file.
            expected_magic (str): The expected magic number to validate against.

        Returns:
            bool: True if the cache file exists and the magic number is valid, False otherwise.
        """
        expected_magic = MAGIC_NUM

        if os.path.exists(cache_path):
            print_rank_0(f"Cache file found at {cache_path}")

            # Load the cache data
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)

                # Check the magic number
                if cache_data.get("magic") == expected_magic:
                    print_rank_0("Magic number is valid.")
                    return True
                else:
                    print_rank_0(
                        f"Invalid magic number: {cache_data.get('magic')}. Expected {expected_magic}."
                    )
                    return False

            except Exception as e:
                print_rank_0(f"Failed to read or parse cache file: {e}")
                return False
        else:
            print_rank_0(f"No cache file found at {cache_path}")
            return False

    def _get_dataset_identifier(self, path_list) -> Optional[str]:
        hash_list = sorted(
            [
                generate_16_char_hash(str(something.split("/")[-1]))
                for something in path_list
                + [
                    str(self.args.max_chunk_size_in_pretraining),
                    str(self.args.min_overlap_size_in_pretraining),
                ]
                + ["knowledge-learning-method"]
            ]
        )

        return "".join(hash_list)


from datasets import Split


class DataManager(object):
    def __init__(self, tokenizer, data_config: DataArguments, is_pretraining=False):
        self._num_proc = data_config.dataloader_num_workers

        self._dataset_dct = {}

        for split, data_path in data_config.data_files.items():
            if not data_path:
                continue
            if not isinstance(data_path, str):
                raise ValueError(f"Invalid data file: {data_path}")
            if split == Split.TRAIN:
                is_eval = False
            else:
                is_eval = True
            if is_pretraining and not is_eval:
                continue
            self._dataset_dct[split] = LazyCRiticDataset(
                data_path,
                tokenizer,
                data_config.task_names,
                is_eval,
                data_config,
            )

    def get_dataset(self, split) -> Optional[LazyCRiticDataset]:
        return self._dataset_dct.get(split, None)


class RawTextDataset(Dataset):
    """
    A dataset for raw text data.
    Used for pretraining (using the plain framing version)
    The pretraining will only last for one epoch.
    > Read from raw text data (e.g., jsonl).
    > Smartly split the data into multiple parts.
    > Do not merge multiple docs into one context.
    > Split too long docs into multiple parts (decided by pretraining max_chunk_size).
    """

    def __init__(
        self,
        data_path_list: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(RawTextDataset, self).__init__()

        self.tokenizer = tokenizer
        self.args = data_args
        self.chunking_mode = data_args.pretraining_chunking_mode
        self.role_play_in_pretraining = data_args.role_play_in_pretraining
        self.data_counter = 0

        cache_identifier = self._get_dataset_identifier(data_path_list)

        if self.args.cache_path == "":
            cache_path = Path(CACHE_DATASET_MESSAGES_DIR) / f"{cache_identifier}.jsonl"
        else:
            cache_path = self.args.cache_path

        if not self.args.enable_cache or not self._check_cache_existing(cache_path):

            # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
            # hence building up memory
            semaphore = Semaphore(10000 + data_args.dataloader_num_workers)

            # use multiprocessing to iterate over input documents
            #! text should be stored in the `text` field of the json objects
            fin = yield_from_files(data_path_list, semaphore)

            if self.chunking_mode == "overlap":
                selected_chunking_fn = self._chunk_with_overlap
            elif self.chunking_mode == "even":
                selected_chunking_fn = self._evenly_split
            else:
                raise ValueError(f"Invalid chunking mode: {self.chunking_mode}")

            if data_args.dataloader_num_workers > 1:
                pool = multiprocessing.Pool(data_args.dataloader_num_workers)
                chunked_sources = pool.imap(selected_chunking_fn, fin)
            else:
                chunked_sources = (selected_chunking_fn(text) for text in fin)

            sources_list = []

            # actually do tokenization
            proc_start = time.time()
            total_tokens_processed = 0
            total_chunks = 0
            pbar = tqdm()
            for i, chunks in enumerate(chunked_sources, start=1):
                total_chunks += len(chunks)

                # release semaphore so `yield_from_files` can add another file to the buffer
                semaphore.release()

                sources_list.extend(chunks)

                # log progress
                current = time.time()
                elapsed = current - proc_start
                pbar.set_description(
                    f"Processed {i}{''} docs ({i / elapsed :.2f} docs/s)."
                )

            if self.args.enable_cache:
                cache_data = {
                    "magic": MAGIC_NUM,
                    "sources_list": sources_list,
                }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f)
                print_rank_0(f"Saved {len(sources_list)} samples to {cache_path}")
        else:  # load from cache
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
                sources_list = cache_data.get("sources_list", [])

            print_rank_0(f"Loaded {len(sources_list)} samples from {cache_path}")

        # Each source should be a ``[{"content": "xx"}]``''
        self.sources_list = sources_list

    def _check_cache_existing(self, cache_path):
        """
        Checks if the cache file exists and validates the magic number.

        Args:
            cache_path (str): The path to the cache file.
            expected_magic (str): The expected magic number to validate against.

        Returns:
            bool: True if the cache file exists and the magic number is valid, False otherwise.
        """
        expected_magic = MAGIC_NUM

        if os.path.exists(cache_path):
            print_rank_0(f"Cache file found at {cache_path}")

            # Load the cache data
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)

                # Check the magic number
                if cache_data.get("magic") == expected_magic:
                    print_rank_0("Magic number is valid.")
                    return True
                else:
                    print_rank_0(
                        f"Invalid magic number: {cache_data.get('magic')}. Expected {expected_magic}."
                    )
                    return False

            except Exception as e:
                print_rank_0(f"Failed to read or parse cache file: {e}")
                return False
        else:
            print_rank_0(f"No cache file found at {cache_path}")
            return False

    def _get_dataset_identifier(self, path_list) -> Optional[str]:
        hash_list = sorted(
            [
                generate_16_char_hash(str(something.split("/")[-1]))
                for something in path_list
            ]
        )

        return "".join(hash_list)

    def __len__(self):
        return len(self.sources_list)

    def __getitem__(self, idx):

        sources = self.sources_list[idx]
        if isinstance(idx, int):
            if isinstance(sources, dict):  # a hot patch
                sources = [sources]
            sources = [sources]

        if not self.role_play_in_pretraining:
            data_dict = preprocess_plain(
                sources, self.tokenizer
            )  # directly call the plain template
        else:
            data_dict = preprocess_plain_like_chat(sources, self.tokenizer)

        if self.data_counter % 10000 == 0:
            print_rank_0(data_dict)

        self.data_counter += 1

        if isinstance(idx, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        return data_dict

    def _chunk_with_overlap(self, text):
        """
        Splits the text into chunks with a specified maximum chunk size and a minimum overlap size.
        Both chunks and overlaps are constructed from complete sentences.
        """
        if self.args.use_ftfy:
            text = ftfy.fix_text(text)

        # Tokenize the text into sentences using NLTK
        sentences = sent_tokenize(text)

        max_chunk_size = self.args.max_chunk_size_in_pretraining
        min_overlap_size = self.args.min_overlap_size_in_pretraining

        chunks = []
        current_chunk = []
        current_chunk_size = 0

        for sentence in sentences:
            sentence_length = len(
                self.tokenizer.encode(sentence, add_special_tokens=False)
            )

            # Check if adding this sentence exceeds the max chunk size
            if current_chunk_size + sentence_length > max_chunk_size:
                if current_chunk:
                    new_chunk = " ".join(current_chunk)
                    current_joined_chunk_size = len(
                        self.tokenizer.encode(new_chunk, add_special_tokens=False)
                    )
                    if current_joined_chunk_size < max_chunk_size:
                        chunks.append(new_chunk)

                # Start a new chunk including the overlap from the previous chunk
                overlap_chunk = []
                overlap_chunk_size = 0

                # Walk backwards through the current chunk to create the overlap
                for overlap_sentence in reversed(current_chunk):
                    overlap_sentence_length = len(
                        self.tokenizer.encode(
                            overlap_sentence, add_special_tokens=False
                        )
                    )
                    if overlap_chunk_size + overlap_sentence_length > min_overlap_size:
                        break
                    overlap_chunk.insert(0, overlap_sentence)
                    overlap_chunk_size += overlap_sentence_length

                # The new chunk begins with this overlap
                current_chunk = overlap_chunk + [sentence]
                current_chunk_size = overlap_chunk_size + sentence_length
            else:
                # Add the sentence to the current chunk
                current_chunk.append(sentence)
                current_chunk_size += sentence_length

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [[{"content": chunk}] for chunk in chunks]

    def _evenly_split(self, text):
        # The input text will be a full document in string format
        # Split it according to the max_chunk_size

        if self.args.use_ftfy:
            text = ftfy.fix_text(text)

        # Tokenize the text into sentences using NLTK
        sentences = sent_tokenize(text)

        # Now, chunk the sentences into chunks based on max_chunk_size
        max_chunk_size = self.args.max_chunk_size_in_pretraining
        chunks = []
        current_chunk = []
        tokens_processed = 0

        current_chunk_size = 0
        for sentence in sentences:
            sentence_length = len(
                self.tokenizer.encode(sentence, add_special_tokens=False)
            )
            tokens_processed += sentence_length

            if current_chunk_size + sentence_length > max_chunk_size:
                # When current chunk is too large, finalize the chunk and start a new one
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_chunk_size = 0

            current_chunk.append(sentence)
            current_chunk_size += sentence_length

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [[{"content": chunk}] for chunk in chunks], tokens_processed


# [10/27] Add a class implementation of `BlendedDataset` to `crdataset.py`
# Borrowed from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/blended_dataset.py#L24


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)
    w = (w / w_sum).tolist()
    return w


def build_exhaustive_blending_indices(
    dataset_index: np.ndarray,
    dataset_sample_index: np.ndarray,
    sizes: np.ndarray,
    num_datasets: int,
):
    """
    Build blending indices by sampling exactly as many samples from dataset[i]
    as is requested by sizes[i] for all i in the range [0, num_datasets).
    """
    # Prepare pointers (np arrays can be accessed directly)
    dataset_sample_counts = np.zeros(
        num_datasets, dtype=np.int64
    )  # Similar to dataset_sample_counts array in C++
    dataset_unspent_indices = set(
        range(num_datasets)
    )  # Set of indices that still need to contribute samples

    total_size = np.sum(sizes)  # Calculate total number of samples across datasets

    # Calculate weights based on the proportion of total samples from each dataset
    weights = sizes / total_size

    index_sample = 0  # This tracks the overall index we're placing in the output arrays

    while len(dataset_unspent_indices) > 0:
        # Ensure index_sample_double is at least 1.0 to avoid division by zero
        index_sample_double = max(float(index_sample), 1.0)

        # Find dataset with the maximum error (i.e., that is underrepresented)
        error_argmax = None
        error_max = -np.inf  # Equivalent to numeric_limits<double>::lowest()

        for index_dataset in dataset_unspent_indices:
            # Calculate error for each dataset
            error = (
                weights[index_dataset] * index_sample_double
                - dataset_sample_counts[index_dataset]
            )
            if error > error_max:
                error_max = error
                error_argmax = index_dataset

        # Populate the output indices
        dataset_index[index_sample] = error_argmax
        dataset_sample_index[index_sample] = dataset_sample_counts[error_argmax]

        # Update the sample counts for the selected dataset
        dataset_sample_counts[error_argmax] += 1

        # If the selected dataset has no more samples to give, remove it from unspent indices
        if dataset_sample_counts[error_argmax] >= sizes[error_argmax]:
            dataset_unspent_indices.remove(error_argmax)

        index_sample += 1


def build_blending_indices(
    dataset_index, dataset_sample_index, weights, num_datasets, size, verbose=False
):
    """
    Given multiple datasets and a weighting array, build samples such that it follows those weights.

    Parameters:
    - dataset_index: Array where the dataset indices will be stored.
    - dataset_sample_index: Array where the sample indices will be stored.
    - weights: Array of weights for each dataset.
    - num_datasets: Number of datasets.
    - size: Total number of samples.
    - verbose: If true, prints the progress and final ratios.
    """

    if verbose:
        print("> building indices for blended datasets ...")

    # Initialize buffer for number of samples used for each dataset
    current_samples = np.zeros(num_datasets, dtype=np.int64)

    # For each sample:
    for sample_idx in range(size):
        # Determine where the max error in sampling is happening
        sample_idx_double = max(float(sample_idx), 1.0)
        max_error_index = 0
        max_error = weights[0] * sample_idx_double - current_samples[0]

        # Loop over all datasets to find the dataset with max error
        for dataset_idx in range(1, num_datasets):
            error = (
                weights[dataset_idx] * sample_idx_double - current_samples[dataset_idx]
            )
            if error > max_error:
                max_error = error
                max_error_index = dataset_idx

        # Populate the indices
        dataset_index[sample_idx] = max_error_index
        dataset_sample_index[sample_idx] = current_samples[max_error_index]

        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1

    # Optionally, print the achieved sample ratios
    if verbose:
        print(" > sample ratios:")
        for dataset_idx in range(num_datasets):
            achieved_ratio = current_samples[dataset_idx] / float(size)
            print(
                f"   dataset {dataset_idx}, input: {weights[dataset_idx]}, achieved: {achieved_ratio}"
            )


class BlendedDataset(torch.utils.data.Dataset):
    """Conjugating class for a set of MegatronDataset instances

    Args:
        datasets (List[MegatronDataset]): The MegatronDataset instances to blend

        weights (List[Union[int, float]]): The weights that determine the dataset blend ratios

        size (Optional[int]): The number of samples to draw from the blend. If None, for each dataset index idx draw exactly weights[idx] samples from datasets[idx].

        config (BlendedMegatronDatasetConfig): The config

    Raises:
        RuntimeError: When the dataset has fewer or more samples than 'size' post-initialization
    """

    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        weights: List[Union[int, float]],
        size: Optional[int],
        args: DataArguments,
    ) -> None:
        assert len(datasets) == len(weights)
        assert len(datasets) < 32767
        assert all(map(lambda _: type(_) == type(datasets[0]), datasets))
        assert all(map(lambda _: _ > 0, weights))
        assert all(map(lambda _: type(_) == type(weights[0]), weights))
        if size is None and isinstance(weights[0], float):
            assert all(map(lambda _: _ == int(_), weights))

        # Alert user to unnecessary blending
        if len(datasets) == 1:
            print_rank_0(f"Building a BlendedDataset for a single MegatronDataset")

        if size is not None:
            weights = normalize(weights)

        self.datasets = datasets
        self.weights = weights
        self.size = size
        self.args = args

        unique_identifiers = OrderedDict()
        unique_identifiers["class"] = type(self).__name__
        unique_identifiers["datasets"] = [
            dataset.dataset_identifier for dataset in self.datasets
        ]
        unique_identifiers["weights"] = self.weights
        unique_identifiers["size"] = self.size

        self.unique_description = json.dumps(
            unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()

        self.built_anew_on_cache_miss = False

        self.dataset_index, self.dataset_sample_index = self._build_indices()

        # fetch `task_name` from the original datasets
        task_names = []
        for idx in range(self.dataset_index.shape[0]):
            dataset_id = self.dataset_index[idx]
            dataset_sample_id = self.dataset_sample_index[idx]
            task_name = self.datasets[dataset_id].task_names[dataset_sample_id]
            task_names.append(task_name)
        self.task_names = task_names

    def __len__(self) -> int:
        return self.dataset_index.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Union[int, np.ndarray]]:
        dataset_id = self.dataset_index[idx]
        dataset_sample_id = self.dataset_sample_index[idx]
        return {
            "dataset_id": dataset_id,
            **self.datasets[dataset_id][dataset_sample_id],
        }

    def _build_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build and optionally cache the dataset index and the dataset sample index

        The dataset index is a 1-D mapping which determines the dataset to query. The dataset
        sample index is a 1-D mapping which determines the sample to request from the queried
        dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The dataset index and the dataset sample index
        """
        if self.args.cache_path == "":
            path_to_cache = Path(CACHE_DATASET_MESSAGES_DIR)
        else:
            path_to_cache = "/".join(self.args.cache_path.split("/")[:-1])

        if path_to_cache:
            get_path_to = lambda suffix: os.path.join(
                path_to_cache,
                f"{self.unique_description_hash}-{type(self).__name__}-{suffix}",
            )
            path_to_description = get_path_to("description.txt")
            path_to_dataset_index = get_path_to("dataset_index.npy")
            path_to_dataset_sample_index = get_path_to("dataset_sample_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [
                        path_to_description,
                        path_to_dataset_index,
                        path_to_dataset_sample_index,
                    ],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (not cache_hit and torch.distributed.get_rank() == 0):
            print_rank_0(f"Build and save the {type(self).__name__} indices")
            self.built_anew_on_cache_miss = True

            # Build the dataset and dataset sample indexes
            print_rank_0(f"\tBuild and save the dataset and dataset sample indexes")
            t_beg = time.time()

            if self.size is not None:
                dataset_index = np.zeros(self.size, dtype=np.int16)
                dataset_sample_index = np.zeros(self.size, dtype=np.int64)
                build_blending_indices(
                    dataset_index,
                    dataset_sample_index,
                    self.weights,
                    len(self.datasets),
                    self.size,
                    verbose=True,
                )
            else:
                size = sum(self.weights)
                dataset_index = np.zeros(size, dtype=np.int16)
                dataset_sample_index = np.zeros(size, dtype=np.int64)
                build_exhaustive_blending_indices(
                    dataset_index,
                    dataset_sample_index,
                    self.weights,
                    len(self.datasets),
                )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                # Save the indexes
                np.save(path_to_dataset_index, dataset_index, allow_pickle=True)
                np.save(
                    path_to_dataset_sample_index,
                    dataset_sample_index,
                    allow_pickle=True,
                )
            else:
                print_rank_0(
                    f"Unable to save the {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            print_rank_0(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            return dataset_index, dataset_sample_index

        print_rank_0(f"Load the {type(self).__name__} indices")

        print_rank_0(f"\tLoad the dataset index from {path_to_dataset_index}")
        t_beg = time.time()
        dataset_index = np.load(path_to_dataset_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        print_rank_0(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        print_rank_0(
            f"\tLoad the dataset sample index from {path_to_dataset_sample_index}",
        )
        t_beg = time.time()
        dataset_sample_index = np.load(
            path_to_dataset_sample_index, allow_pickle=True, mmap_mode="r"
        )
        t_end = time.time()
        print_rank_0(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        return dataset_index, dataset_sample_index

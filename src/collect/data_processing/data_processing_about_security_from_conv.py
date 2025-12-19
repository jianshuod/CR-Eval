import re
import json
import transformers
from tqdm import tqdm
from pathlib import Path
from datasets import Split
from typing import List, Tuple
from src.utils.logging import logging
from src.train.data.crdataset import DataManager
from src.utils import generate_16_char_hash
import src.utils.conversation as conversation_lib
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.train.args import (
    DataArguments,
    ProcessArguments,
    ModelArguments,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def match_from_end(s: str) -> str:
    match = re.search(r"(security-related|security-unrelated)", s.lower())
    if match:
        return match.group(0)
    else:
        return None


def main(conv_list):
    print("Parsing arguments")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ProcessArguments)
    )

    model_args, data_args, training_args, process_args = (
        parser.parse_args_into_dataclasses()
    )
    data_dir = data_args.data_dir

    print("Loading tokenizer")
    if "mpt" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    print(f"Configuring tokenizer for version {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            print("Adding pad token as '[PAD]'")
            special_tokens_dict = dict(pad_token="[PAD]")
            tokenizer.add_special_tokens(special_tokens_dict)
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
            print("Adding pad token as '<pad>'")
            special_tokens_dict = dict(pad_token="<pad>")
            tokenizer.add_special_tokens(special_tokens_dict)

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "llama3"
            ]
        print(
            f"Using conversation format: {conversation_lib.default_conversation.version}"
        )

    print(f"Tasks included. {data_args.task_names}")

    print("Initializing DataManager and loading dataset")
    data_manager = DataManager(data_dir, tokenizer, data_args)
    train_dataset = data_manager.get_dataset(Split.TRAIN)

    sources_list = train_dataset.sources_list

    print("Removing incomplete sentences from sources")
    for source in sources_list:
        source[1]["content"] = source[1]["content"].rsplit(".", 1)[0] + "."

    def get_token_num(string: str, tokenizer: transformers.PreTrainedTokenizer) -> int:
        tokenized = tokenizer(
            string,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        return tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()

    def check_sample_structure(source, keywords_list):
        full_sequence = "".join([sentence["content"] for sentence in source])
        for keywords in keywords_list:
            pass_flag = all([keyword in full_sequence for keyword in keywords])
            if pass_flag:
                return True
        return False

    print(f"Before filtering, {len(sources_list)} sources are available")

    def framing4validitycheck(source):
        final_text = [
            "### SYSTEM INSTRUCTION",
            source[0]["content"],
            "",
            "### USER QUERY",
            source[1]["content"],
            "",
            "### ASSISTANT RESPONSE",
            source[2]["content"],
        ]
        return "\n".join(final_text)

    if process_args.llm_checking:

        conv_list = Path(conv_list)
        max_workers = 16

        conv_lookup = dict()
        # establish a hash table for the conversation
        for json_file in conv_list.iterdir():
            with open(json_file, "r") as f:
                conv = json.load(f)
                conv_lookup[generate_16_char_hash(conv["messages"][1]["content"])] = (
                    conv["response"]["choices"][0]["message"]["content"]
                )

        def load_response(source):
            query = framing4validitycheck(source)
            query_hash = generate_16_char_hash(query)
            hit = conv_lookup.get(query_hash)
            if hit:
                conv_lookup.pop(query_hash)
                return hit

        def validity_check(source) -> Tuple[str, str, str]:
            ans = load_response(source)

            if ans == None:
                return source, None
            yes_or_no = match_from_end(ans)
            return source, yes_or_no

        def process_sources_in_batches(
            sources_list: List[str], num_workers: int, batch_size: int
        ) -> List[Tuple[str, str, str]]:
            print("Processing sources in batches using concurrent.futures")

            results = []
            total_batches = (len(sources_list) + batch_size - 1) // batch_size

            with tqdm(total=len(sources_list), desc="Processing sources") as pbar:
                for i in range(0, len(sources_list), batch_size):
                    batch = sources_list[i : i + batch_size]
                    print(f"Processing batch {i // batch_size + 1}/{total_batches}")

                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = {
                            executor.submit(validity_check, source): source
                            for source in batch
                        }

                        for future in as_completed(futures):
                            result = future.result()
                            results.append(result)
                            pbar.update(1)

            return results

        batch_size = (
            max_workers * 32
        )  # Defining the batch size to be 32 times the number of max workers
        results = process_sources_in_batches(sources_list, max_workers, batch_size)

        sr_sources_list = [
            source for source, yes_or_no in results if yes_or_no == "security-related"
        ]
        sur_sources_list = [
            source for source, yes_or_no in results if yes_or_no != "security-related"
        ]

    print(len(sr_sources_list))
    print(len(sur_sources_list))

    final_sources_list = {
        "security-related": sr_sources_list,
        "security-unrelated": sur_sources_list,
    }

    # Ensure the directory exists
    save_to = Path(process_args.save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving results to {save_to}")

    # Open the file and write the entire final_sources_list as JSON
    with open(save_to, "w") as f:
        json.dump(final_sources_list, f, indent=4)  # Pretty print with indentation

    print("Processing completed")

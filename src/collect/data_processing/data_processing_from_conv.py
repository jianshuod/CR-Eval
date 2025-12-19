import re
import json
import transformers
from tqdm import tqdm
from pathlib import Path
from datasets import Split
from typing import List, Tuple
from src.utils.logging import logging
from src.utils import generate_16_char_hash
from src.train.data.crdataset import DataManager
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
    match = re.search(r"(YES|Invalid RESPONSE|Invalid QUERY)", s)
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

    REPONSE_TO_INVALID_QUERY_CASE1 = "The query seems to be invalid. It is too short to analyze for the requested task. "
    REPONSE_TO_INVALID_QUERY_CASE2 = "The query seems to be invalid. It does not contain the necessary structure as described in the system instruction. "

    final_sources_list = list()

    print(f"Before manual checking, {len(sources_list)} sources are available")

    if process_args.manual_checking:
        print("Starting manual checking of sources")
        if process_args.retain_invalid:
            invalid_sources_case1 = list(
                filter(
                    lambda x: get_token_num(x[1]["content"], tokenizer)
                    <= process_args.min_query_token_num,
                    sources_list,
                )
            )
            for source in invalid_sources_case1:
                source[2]["content"] = REPONSE_TO_INVALID_QUERY_CASE1
            final_sources_list.extend(invalid_sources_case1)
            print(
                f"Retaining {len(invalid_sources_case1)} invalid sources based on length"
            )

        sources_list = list(
            filter(
                lambda x: get_token_num(x[1]["content"], tokenizer)
                > process_args.min_query_token_num,
                sources_list,
            )
        )
        print(f"Remaining sources after length check: {len(sources_list)}")
        print(f"Retaining {len(final_sources_list)} invalid sources.")

        fill_cr_keywords = (
            ">>> Original Specification Statements:",
            ">>> REASON FOR CHANGE",
            ">>> SUMMARY OF CHANGE",
            ">>> CONSEQUENCES IF NOT REVISED",
        )
        outline_revision_keywords = (
            ">>> Change Request:",
            "- **Reason for change**:",
            "- **Consequences if not revised**:",
            ">>> Original Specification Statement:",
            ">>> SUMMARY OF CHANGES",
        )
        diff_analysis_keywords = (
            ">>> Diffed Specification Statements:",
            ">>> SUMMARY OF CHANGE",
            ">>> REASON FOR CHANGE",
            ">>> CONSEQUENCES IF NOT REVISED",
        )
        explain_vuln_keywords = (
            ">>> Marked Specification Statements:",
            ">>> REASON FOR CHANGE",
            ">>> SUMMARY OF CHANGE",
            ">>> CONSEQUENCES IF NOT REVISED",
        )
        keywords_list = [
            fill_cr_keywords,
            outline_revision_keywords,
            diff_analysis_keywords,
            explain_vuln_keywords,
        ]

        if process_args.retain_invalid:
            invalid_sources_case2 = list(
                filter(
                    lambda x: not check_sample_structure(x, keywords_list), sources_list
                )
            )
            for source in invalid_sources_case2:
                source[2]["content"] = REPONSE_TO_INVALID_QUERY_CASE2
            final_sources_list.extend(invalid_sources_case2)
            print(
                f"Retaining {len(invalid_sources_case2)} invalid sources based on structure"
            )

        sources_list = list(
            filter(lambda x: check_sample_structure(x, keywords_list), sources_list)
        )
        print(f"Remaining sources after structure check: {len(sources_list)}")
        print(f"Retaining {len(final_sources_list)} invalid sources.")

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

        max_workers = 32

        conv_list = Path(conv_list)

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
                return None, None
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

        valid_sources_list = [
            source for source, yes_or_no in results if yes_or_no == "YES"
        ]

        if process_args.retain_invalid:
            print("Retaining invalid sources based on LLM check")
            invalid_sources_case3_filtered = [
                source for source, yes_or_no in results if yes_or_no == "Invalid QUERY"
            ]
            invalid_sources_case3 = list()
            for source in invalid_sources_case3_filtered:
                source[2][
                    "content"
                ] = f"The query seems to be invalid. It is inadequate for the requested task."
                invalid_sources_case3.append(source)
            final_sources_list.extend(invalid_sources_case3)
            print(
                f"Retaining {len(invalid_sources_case3)} invalid sources based on LLM check"
            )

        print(f"Remaining valid sources after LLM check: {len(valid_sources_list)}")
        print(f"Retaining {len(final_sources_list)} invalid sources.")

    final_sources_list.extend(valid_sources_list)

    save_to = Path(data_args.data_dir) / process_args.save_to

    print(f"Saving results to {save_to}")
    for source in final_sources_list:
        with open(save_to, "a") as f:
            f.write(json.dumps(source) + "\n")

    print("Processing completed")

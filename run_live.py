import os
import json
import time
import traceback
import dataclasses
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from collections import Counter
from src.eval import ResponseEval
from argparse import ArgumentParser
from src.deploy.models import ModelChoice
from src.analysis.task_worker import Tool
from src.deploy.models import MODEL_LIMITS
from src.utils import is_port_open, parse_dict
import src.utils.conversation as conversation_lib
from src.analysis.definition import available_tasks
from src.make_data.play_with_ftp import download_cr
from src.utils.logging import logging, setup_logging
from src.make_data.framing import apply_task_framing
from src.make_data.play_with_cr import get_cr_fields
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.make_data.create_instance import setup_single_instance
from src.make_data.spec_processing.chunking import CHUNKING_FUNCTIONS
from src.deploy.vllm_setup import start_model_servers, stop_model_servers

logger = logging.getLogger(__name__)


PERSONALIZATION_ANCHOR = (
    "You are a specialist in cellular network protocol and security analysis.\n\n"
)


def report_exp_info(
    model_name,
    using_lora,
    task_name,
    cr_indices,
    dataset_dir,
    from_jsonl,
    max_context_length,
    max_model_length,
    chunking_mode,
    document_encoding_func_str,
    eval_funcs_str,
    eval_configs_for_gpt_scorer,
    eval_repetition,
    output_dir,
    retrieval_mode,
    k_trials,
    add_shots_from_jsonl,
):
    """
    Logs and prints basic information about the current experiment setup.
    """
    print(
        f"Experiment Info:\n"
        f"Model Name: {model_name}\n"
        f"Using LoRA: {using_lora}\n"
        f"Task Name: {task_name}\n"
        f"CR Indices: {cr_indices}\n"
        f"Dataset Directory: {dataset_dir}\n"
        f"From JSONL: {from_jsonl}\n"
        f"Output Directory: {output_dir}\n"
        f"Max Context Length: {max_context_length}\n"
        f"Max Model Length: {max_model_length}\n"
        f"Chunking Mode: {chunking_mode}\n"
        f"Document Encoding Function: {document_encoding_func_str}\n"
        f"Evaluation Functions: {eval_funcs_str}\n"
        f"Evaluation Configurations for GPT Scorer: {eval_configs_for_gpt_scorer}\n"
        f"Evaluation Repetition: {eval_repetition}\n"
        f"Retrieval Mode: {retrieval_mode}\n"
        f"Number of Trials (k): {k_trials}\n"
        f"Experiment Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Add Shot From Jsonl: {add_shots_from_jsonl}"
    )


def process_single_cr_stage_1(
    cr_index,
    output_dir,
    tool,
    task_name,
    model_name,
    max_context_length,
    chunking_mode,
    document_encoding_func_str,
    retrieval_mode,
    k_trials,
    gen_conf,
):
    """
    Stage 1: Query the model and obtain completions.
    """
    try:
        stime = time.time()

        if isinstance(cr_index, str):
            if cr_index.startswith("ftp:"):
                cr_path = download_cr(cr_index, output_dir)
            elif cr_index.startswith("local:"):
                cr_path = cr_index.replace("local:", "")
                if not os.path.exists(cr_path):
                    logger.warning(f"File {cr_path} does not exist")
                    return None
            cr_fields = get_cr_fields(Path(cr_path), output_dir)

            instance = setup_single_instance(
                cr_fields,
                max_context_length,
                output_dir,
                chunking_mode,
                document_encoding_func_str,
                task_name,
                model_name,
                retrieval_mode,
            )
            data = apply_task_framing(instance, task_name)

        elif isinstance(cr_index, list) and len(cr_index) == 3:
            instance = {
                "messages": cr_index[:2],
                "gold_response": cr_index[2]["content"],
            }
            data = instance["messages"][1]["content"]
        else:
            raise ValueError(
                "Currently only support input an path or a conversation list."
            )

        if "gemini" in model_name.lower() or model_name == "oai-endpoint:qwen3-235b-a22b" or  "doubao-seed" in model_name:
            completions = []
            for idx in range(k_trials):
                completion, messages = tool.run(data, gen_conf=gen_conf)
                completions.extend(completion)
        else:
            if k_trials != 1:
                gen_conf.update({"n": k_trials, "temperature": 0.9})
            completions, messages = tool.run(data, gen_conf=gen_conf)

        instance["query_text"] = messages
        instance["completions"] = completions
        instance["time_cost"] = time.time() - stime

        return instance
    except Exception as e:
        full_traceback = traceback.format_exc()
        logger.error(f"{full_traceback}")
        return None


def process_single_cr_stage_2(instance, evalFactory):
    """
    Stage 2: Evaluate the completions obtained from Stage 1.
    """
    try:
        if instance is None:
            return None

        eval_results = dict()
        if evalFactory.get_metric_num() == 1:
            for idx, completion in enumerate(instance["completions"]):
                eval_result = evalFactory.run_eval(instance, completion)[0]
                eval_result["response_text"] = completion
                eval_results[f"completion_{idx}"] = eval_result

        gold_response = evalFactory.form_desirable_response(instance)

        instance["gold_response"] = gold_response
        instance["eval_results"] = eval_results

        return instance
    except Exception as e:
        full_traceback = traceback.format_exc()
        logger.error(f"{full_traceback}")
        return None


def main(
    model_name,
    using_lora,
    task_name,
    cr_indices: str,
    dataset_dir,
    from_jsonl,
    max_context_length,
    max_model_length,
    chunking_mode,
    document_encoding_func_str,
    output_dir,
    eval_funcs_str,
    retrieval_mode,
    eval_configs_for_gpt_scorer,
    eval_repetition,
    vllm_config,
    version,
    system_suffix,
    assistant_prefix,
    use_my_prompt,
    k_trials,
    gen_conf,
    personalized,
    num_worker,
    add_shots_from_jsonl,
    run_name,
    is_third_party_chat,
):
    # Report experiment information
    report_exp_info(
        model_name,
        using_lora,
        task_name,
        cr_indices,
        dataset_dir,
        from_jsonl,
        max_context_length,
        max_model_length,
        chunking_mode,
        document_encoding_func_str,
        eval_funcs_str,
        eval_configs_for_gpt_scorer,
        eval_repetition,
        output_dir,
        retrieval_mode,
        k_trials,
        add_shots_from_jsonl,
    )

    # Load data samples
    cr_list = list()

    if from_jsonl is not None and Path(from_jsonl).exists():
        with open(from_jsonl, "r") as f:
            for line in f:
                cr_list.append(json.loads(line))

        loaded_instruction = (
            cr_list[0][0]["content"] if use_my_prompt is None else use_my_prompt
        )
        if not personalized:
            task_info = ("custom", loaded_instruction)
        else:
            if loaded_instruction.startswith("You are"):
                task_instruction = PERSONALIZATION_ANCHOR + ".".join(
                    loaded_instruction.split(".")[1:]
                )
            else:
                task_instruction = PERSONALIZATION_ANCHOR + loaded_instruction
            task_info = ("custom", task_instruction)

        # Use the first conversation system as global task name
        logger.warning(f"The worker instruction will be set as {task_info}")
    else:  # Load from raw CRs
        for cr_index in cr_indices:
            if cr_index.startswith("ftp:") or cr_index.startswith("local:"):
                cr_list.append(cr_index)
            else:
                logger.warning(f"Invalid CR index: {cr_index}")

        if dataset_dir is not None and Path(dataset_dir).exists():
            num_cr_seton = len(cr_list)
            logger.info(f"Loading CRs from {dataset_dir}")
            for cr_file in Path(dataset_dir).iterdir():
                if cr_file.suffix in (".docx", ".json"):
                    cr_list.append(f"local:{cr_file}")
            logger.info(f"Loaded {len(cr_list) - num_cr_seton} CRs from {dataset_dir}")

        task_info = task_name

    if len(cr_list) == 0:
        logger.error("Please assign at least one change request for analysis.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tool = Tool(
        task_info,
        model_name,
        using_shot=False,
        system_suffix=system_suffix,
        assistant_prefix=assistant_prefix,
        use_my_prompt=use_my_prompt,
        add_shots_from_jsonl=add_shots_from_jsonl,
        is_third_party_chat=is_third_party_chat,
    )
    model_name = tool.model_name

    # set up local servers
    default_vllm_config = ModelChoice.from_string(model_name).get_vllm_configs()
    if default_vllm_config is None and vllm_config is not None:
        raise ValueError(f"Model {model_name} does not support local servers")
    elif vllm_config is not None:
        local_server_settings = vllm_config
        logger.info(f"Using custom VLLM configuration: {local_server_settings}")
    else:
        local_server_settings = default_vllm_config
        logger.info(f"Using default VLLM configuration: {local_server_settings}")

    if max_model_length is not None and max_model_length > 0:
        MODEL_LIMITS[model_name] = max_model_length

    if local_server_settings is not None:
        logger.info("Setting up vllm-based local servers")
        processes = start_model_servers(
            model_name=model_name, using_lora=using_lora, **local_server_settings
        )

        # Wait for up to 120 seconds for the server to start
        max_wait_time = 240
        interval = 5  # Check every 5 seconds
        waited = 0

        pids = [process.pid for process, _ in processes]

        while waited < max_wait_time:
            if is_port_open("localhost", 8000):
                logger.info("Server is up and running")
                break
            # if not all(is_process_running(pid) for pid in pids):
            #     logger.error("Some servers did not start successfully")
            #     return
            time.sleep(interval)
            waited += interval
        else:
            logger.error("Server did not start within the expected time frame")
            return
    else:
        processes = []

    if version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "llama3"
        ]
    print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    server_num = len(processes)
    max_workers = ModelChoice.from_string(model_name).get_max_workers(server_num)
    if num_worker > 0:
        max_workers = num_worker

    print(f"[Info] {max_workers} workers are ready to work.")

    outputs_stage_1 = []

    # Stage 1: Obtain completions for all CRs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_cr_stage_1,
                cr_index,
                output_dir,
                tool,
                task_name,
                model_name,
                max_context_length,
                chunking_mode,
                document_encoding_func_str,
                retrieval_mode,
                k_trials,
                gen_conf,
            ): cr_index
            for cr_index in cr_list
        }

        progress_bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Stage 1: Querying completions",
        )

        for future in progress_bar:
            instance = future.result()
            if instance is not None:
                outputs_stage_1.append(instance)

    # shut down the servers (if they exist)
    stop_model_servers(processes)

    outputs_stage_2 = []

    evalFactory = ResponseEval(
        task_name,
        eval_funcs_str,
        {
            "caveat_list": eval_configs_for_gpt_scorer,
            "eval-repetition": eval_repetition,
        },
    )

    # Stage 2: Evaluate completions for all instances from Stage 1
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(process_single_cr_stage_2, instance, evalFactory): instance
            for instance in outputs_stage_1
        }

        progress_bar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Stage 2: Evaluating completions",
        )

        for future in progress_bar:
            instance = future.result()
            if instance is not None:
                outputs_stage_2.append(instance)

    if os.path.exists(model_name):
        model_name = model_name.split("/")[-1]

    if run_name is not None:
        file_output_name = f"{run_name}.jsonl"
    else:
        file_output_name = f'{model_name}_{task_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jsonl'

    # Store in local
    output_file = Path(
        output_dir,
        task_name,
        model_name,
        file_output_name,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    for output in outputs_stage_2:
        if isinstance(output, dict):
            for key, value in output["eval_results"].items():
                if dataclasses.is_dataclass(value):
                    output["eval_results"][key] = dataclasses.asdict(value)
            with Path(output_file).open("a") as file:
                file.write(json.dumps(output) + "\n")
        else:
            output.to_jsonl(output_file)
    logger.info(f"Wrote output to {output_file}")

    for num_trials in range(1, k_trials + 1):
        max_score_counter = Counter()
        for output in outputs_stage_2:
            eval_results = output["eval_results"]
            max_score = -2
            for idx, (key, value) in enumerate(eval_results.items()):
                if idx == num_trials:
                    break
                score_for_this_compl = (
                    int(value["score"]) if value["score"] != None else -2
                )
                max_score = max(max_score, score_for_this_compl)
            max_score_counter[max_score] += 1
        print("*" * 40 + f" pass@{num_trials} " + "*" * 40)
        print(max_score_counter)
        # Score larger than 0 should be considered as positive
        pos_counter = 0
        neg_counter = 0
        for score, num in max_score_counter.items():
            if score > 0:
                pos_counter += num
            else:
                neg_counter += num
        print(f"Positive: {pos_counter}, Negative: {neg_counter}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name / local_path to use for inference",
    )
    parser.add_argument(
        "--using_lora", type=str, default=None, help="Path to lora adapter"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="llama3",
        help="Prompt version for local checkpoints",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="edit specs under cr",
        choices=available_tasks.keys(),
        help="Task name to take on",
    )
    parser.add_argument(
        "--chunking_mode",
        type=str,
        choices=CHUNKING_FUNCTIONS.keys(),
        default="section",
        help="Chunking mode to use for the input text",
    )
    parser.add_argument(
        "--cr_indices",
        type=str,
        default=[],
        nargs="+",
        help="Currently support input an address, starting with ``local:'' or``ftp:''",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to the dataset directory, composed of the 3GPP CR files",
    )
    parser.add_argument(
        "--from_jsonl",
        type=str,
        default=None,
        help="Path to a jsonl file, which contains conversations (system instruction, query messages, and gold responses). For this case, there is no need to specify task.",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=16_000,
        help="Maximum context length for the input text",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=None,
        help="Maximum model length for the input text",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="oracle",
        choices=["oracle", "bm25"],
        help="Retrieval mode to use for fetching the spec statements",
    )
    parser.add_argument(
        "--document_encoding_func_str",
        type=str,
        default="file_name_and_contents",
        help="Document encoding function to use for encoding the spec",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./live_outputs",
        help="Directory to save the output JSONL file",
    )
    parser.add_argument(
        "--system_suffix",
        type=str,
        default=None,
        help="Addtional system instruction for LLM workers",
    )
    parser.add_argument(
        "--assistant_prefix",
        type=str,
        default=None,
        help="Addtional assistant prefix to condition the generation of completion models (not supported for chat models)",
    )
    parser.add_argument(
        "--use_my_prompt", type=str, default=None, help="Set the prompt for the model"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run name")
    parser.add_argument(
        "--eval_funcs_str",
        type=str,
        nargs="+",
        default=[],
        help="Evaluation functions to use for scoring the model output",
    )
    parser.add_argument(
        "--eval_configs_for_gpt_scorer",
        nargs="+",
        default=["only-scoring"],  # only scoring as default
        help="Specify one or more configuration choices for the GPT scorer.",
    )
    parser.add_argument(
        "--eval_repetition",
        type=int,
        default=1,
        help="Number of times to repeat the evaluation.",
    )
    parser.add_argument(
        "--vllm_config",
        type=parse_dict,
        default=None,
        help="A dictionary of configuration parameters.",
    )
    parser.add_argument(
        "--gen_conf",
        type=parse_dict,
        default="{}",
        help="A dictionary of generation parameters.",
    )
    parser.add_argument("--k_trials", type=int, default=1, help="$k$ in passk")
    parser.add_argument(
        "--num_worker",
        type=int,
        default=-1,
        help="Max worker to query the LLM endpoint",
    )
    parser.add_argument(
        "--personalized",
        type=bool,
        default=False,
        help="Whether to use the personalized. The system message will be changed.",
    )
    parser.add_argument(
        "--is_third_party_chat",
        type=bool,
        default=False,
        help="Using the chatting template of the third party chat models. (For the baseline models)",
    )
    parser.add_argument(
        "--add_shots_from_jsonl",
        type=str,
        default=None,
        help="Load shots (user-assistant pairs) from a jsonl file",
    )
    args = parser.parse_args()

    setup_logging()

    main(**vars(args))

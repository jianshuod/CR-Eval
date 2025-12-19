import os
import json
import pathlib
import asyncio
from openai import OpenAI
from typing import Callable, List
from src.utils.logging import logging
from src.analysis.caveats import apply_caveat
from src.train.data.crdataset import preprocess
from src.deploy.models import ModelChoice, model_name_conversion
from src.deploy.models import MODEL_LIMITS, MODEL_RPM_LIMITS, MODEL_TPM_LIMITS
from src.analysis.utils import (
    num_tokens_from_string,
    generate_session_name,
    truncate_query,
)
from src.analysis.api_request_parallel_processor import (
    process_api_requests_from_file,
    generate_chat_completion_requests,
    total_tokens_consumed_from_file,
)
from src.analysis.utils import (
    load_tokenizer,
    convert_convs_to_completion,
    process_glm_chat,
)
from src.configs import (
    CONV_CACHE_DIR,
    RESPONSE_SPACE,
    BATCH_INPUT_DIR,
    BATCH_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


class APIChat:
    def __init__(self, model_name="gpt-3.5-turbo", key=None, base_url=None):
        self.model_name = model_name
        self.key = key
        self.base_url = base_url

        self.encoding = load_tokenizer(model_name)

        self.system_message = "You are a helpful assistant."
        self.assistant_prefix = None
        self.shots = []

        self.is_third_party_chat = False

        self.setupFlag = False

        self.use_hints = False

    def _fetch_client(self):
        modelchoice = ModelChoice.from_string(self.model_name)

        if not self.base_url or not self.key:
            key = modelchoice.get_api_key()
            base_url = modelchoice.get_base_url()
        else:
            key, base_url = self.key, self.base_url
        client = OpenAI(api_key=key, base_url=base_url)
        if self.model_name in model_name_conversion:
            dummy_name = model_name_conversion[self.model_name]
        elif os.path.exists(self.model_name):
            dummy_name = "local-model"
        elif self.model_name.startswith("oai-endpoint:"):
            dummy_name = self.model_name.replace("oai-endpoint:", "")
        else:
            dummy_name = self.model_name

        if modelchoice == ModelChoice.CLAUDE_3_5_SONNET:

            def query_api(input_unified, **gen_conf):
                responses = []
                for _ in range(gen_conf.get("n", 1)):
                    response = client.chat.completions.create(
                        model=dummy_name, messages=input_unified, **gen_conf
                    )
                    responses.append(response)
                return responses

            prompt_conversion = None

            def answer_extraction(responses):
                answers = []
                for response in responses:
                    for choice in response.choices:
                        answers.append(choice.message.content.strip())
                return answers

        elif modelchoice.is_chat() or self.is_third_party_chat:

            def query_api(input_unified, **gen_conf):
                if self.model_name in [
                    "llama-3-8b-instruct",
                    "llama-3-70b-instruct",
                ]:  ## [HOT PATCH]: add "<|eot_id|>" to end token list of LLaMA-3-8B-instruct
                    gen_conf.update({"extra_body": {"stop_token_ids": [128009]}})
                elif self.model_name in ["glm-4-9b-instruct"]:
                    gen_conf.update(
                        {"extra_body": {"stop_token_ids": [151329, 151336, 151338]}}
                    )
                return client.chat.completions.create(
                    model=dummy_name,
                    messages=input_unified[:1] + self.shots + input_unified[1:],
                    **gen_conf,
                )

            if self.model_name in ["glm-4-9b-instruct"]:
                prompt_conversion = process_glm_chat
            elif "gemma" in self.model_name.lower():

                def combine_system_into_user(messages):
                    new_user_message = {
                        "role": "user",
                        "content": messages[0]["content"]
                        + "\n\n"
                        + messages[1]["content"],
                    }
                    return [new_user_message] + messages[2:]

                prompt_conversion = combine_system_into_user
            else:
                prompt_conversion = None

            def answer_extraction(response):
                answers = []
                for choice in response.choices:
                    answers.append(choice.message.content.strip())
                return answers

        else:

            def query_api(input_unified, **gen_conf):
                max_response_token_num = gen_conf.pop(
                    "max_response_token_num", RESPONSE_SPACE
                )
                if dummy_name == "local-model":
                    # the fine-tuned model is actually a chat model
                    # from Meta-Llama-3.1-8B-Instruct/generation_config.json
                    gen_conf.update(
                        {"extra_body": {"stop_token_ids": [128001, 128008, 128009]}}
                    )
                return client.completions.create(
                    model=dummy_name,
                    prompt=input_unified,
                    max_tokens=max_response_token_num,
                    echo=False,
                    **gen_conf,
                )

            # 【Checked】 The user has no need to add BOS by himself.
            # VLLM add special tokens for completion-style model automatically
            # VLLM determines whether to add BOS token for chat-style models according to user configurations

            prefix = self.assistant_prefix
            if dummy_name == "local-model":

                def prompt_conversion(convs):
                    # The last message will be the hint if use_hints is True
                    if self.use_hints:
                        hint = convs[-1]["content"]
                        convs = convs[:-1]
                    else:
                        hint = None
                    completion = preprocess(
                        [
                            convs[:1]
                            + self.shots
                            + convs[1:]
                            + [{"role": "assistant", "content": None}]
                        ],
                        None,
                        only_inst=True,
                    )[0]
                    if prefix is not None:
                        completion = completion + prefix
                    if self.use_hints:
                        completion = completion + hint
                    return completion

            else:

                def prompt_conversion(convs):
                    return convert_convs_to_completion(prefix, convs)

            def answer_extraction(response):
                answers = []
                for choice in response.choices:
                    answers.append(choice.text.strip())
                return answers

        return prompt_conversion, query_api, answer_extraction

    def setup(
        self,
        system_message: str,
        task_name,
        chain_name,
        assistant_prefix,
        add_shots_from_jsonl,
        is_third_party_chat,
        use_hints,
    ):
        self.system_message = system_message
        self.task_name = task_name
        self.chain_name = chain_name
        self.session_name = generate_session_name(
            task_name, chain_name, self.model_name
        )
        self.service_time = 0

        self.assistant_prefix = assistant_prefix

        self.is_third_party_chat = is_third_party_chat
        self.use_hints = use_hints
        if add_shots_from_jsonl is not None:
            with open(add_shots_from_jsonl, "r") as f:
                for line in f:
                    json_obj = json.loads(line)
                    messages = json_obj["messages"]
                    for message in messages:
                        if message["role"] in ["user", "assistant"] and (
                            len(self.shots) == 0
                            or message["role"] != self.shots[-1]["role"]
                        ):
                            self.shots.append(message)

        self.setupFlag = True

    def assemble_system_instruction(self, caveat_list):
        final_system_message = self.system_message
        if caveat_list is not None:
            for caveat in caveat_list:
                final_system_message = apply_caveat(final_system_message, caveat)

        return [{"role": "system", "content": f"{final_system_message}"}]

    def chat(
        self,
        query,
        gen_conf=None,
        only_prepare=False,
        max_query_len=None,
        caveat_list=None,
    ):
        prompt_conversion, query_api, answer_extraction = self._fetch_client()

        if not self.setupFlag:
            self.setup("You are a helpful assistant.", "default", "default")

        temp_system_message = self.assemble_system_instruction(caveat_list)

        # Prepare the query message
        query_message = (
            [{"role": "user", "content": query}] if isinstance(query, str) else query
        )

        # Combine system and query messages
        messages = temp_system_message + query_message

        if prompt_conversion is not None and isinstance(prompt_conversion, Callable):
            input_unified = prompt_conversion(messages)
        else:
            input_unified = messages

        # Control the input length to leave space for the output
        reserved_response_space = RESPONSE_SPACE
        max_input_space = MODEL_LIMITS[self.model_name] - reserved_response_space
        if max_query_len is not None:
            max_input_space = min(max_input_space, max_query_len)
        if gen_conf == None or "max_response_token_num" not in gen_conf:
            max_response_token_num = MODEL_LIMITS[self.model_name] - max_input_space
        else:
            max_response_token_num = gen_conf.pop("max_response_token_num")

        if isinstance(input_unified, str):
            input_unified = truncate_query(
                input_unified, max_input_space, self.model_name, self.encoding
            )
            gen_conf.update(
                {"max_response_token_num": max_response_token_num - 10}
            )  # Special tokens may occupy some
        elif isinstance(input_unified, List):
            system_prompt_len = num_tokens_from_string(
                temp_system_message[0]["content"], self.model_name, self.encoding
            )
            # truncate the last message
            input_unified[-1]["content"] = truncate_query(
                input_unified[-1]["content"],
                max_input_space - system_prompt_len,
                self.model_name,
                self.encoding,
            )

        if only_prepare:
            return input_unified
        try:
            response = query_api(input_unified, **gen_conf)
        except Exception as e:
            logger.error(f"Error querying API: {e}")
            response=None

        answers = answer_extraction(response)

        self.service_time += 1

        self.save_chat(input_unified, response, gen_conf)

        return answers, response, input_unified

    def save_chat(self, messages, response, gen_conf, save_dir=None, prefered_id=None):

        if save_dir is None:
            save_dir = pathlib.Path(CONV_CACHE_DIR) / self.session_name
        else:
            save_dir = pathlib.Path(save_dir)
        if prefered_id is None:
            prefered_id = str(self.service_time)
        target_path = save_dir / f"{prefered_id}.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if "claude" in self.model_name:
            response_content = [
                single_response.model_dump(mode="python")
                for single_response in response
            ]
        else:
            try:
                response_content = response.model_dump(mode="python")
            except:
                response_content = response
        with open(target_path, "w") as f:
            f.write(
                json.dumps(
                    {
                        "messages": messages,
                        "response": response_content,
                        "gen_conf": gen_conf,
                    }
                )
            )
        # logger.info(f"Chat saved to {target_path}")

    def parallel_chat(self, requests_dir, save_dir, **gen_conf):

        queries = []
        request_file_list = []
        for idx, request_file in enumerate(pathlib.Path(requests_dir).iterdir()):
            request_file_list.append(request_file)
            with open(request_file, "r") as f:
                queries.append(f.read())

        requests_in_json_filepath = pathlib.Path(
            f"{BATCH_INPUT_DIR} / {self.session_name}.jsonl"
        )
        responses_in_json_filepath = pathlib.Path(
            f"{BATCH_OUTPUT_DIR} / {self.session_name}.jsonl"
        )

        requests_in_json_filepath.parent.mkdir(parents=True, exist_ok=True)
        responses_in_json_filepath.parent.mkdir(parents=True, exist_ok=True)

        messages_list = [
            self.chat(query, gen_conf, only_prepare=True) for query in queries
        ]
        generate_chat_completion_requests(
            requests_in_json_filepath, messages_list, self.model_name, **gen_conf
        )

        estimated_total_token, token_cost_list = total_tokens_consumed_from_file(
            requests_in_json_filepath, f"chat/completions", self.model_name
        )

        for token_cost, file_path in zip(token_cost_list, request_file_list):
            print(f"{file_path.name}: {token_cost} tokens")

        # exit(0)
        ACK = input(
            f"Estimated total tokens: {estimated_total_token}, continue? (y/n): "
        )
        if ACK.lower() != "y":
            return

        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=requests_in_json_filepath,
                save_filepath=responses_in_json_filepath,
                request_url=f"{self.base_url}/chat/completions",
                api_key=self.client.api_key,
                max_requests_per_minute=float(0.5 * MODEL_RPM_LIMITS[self.model_name]),
                max_tokens_per_minute=float(0.5 * MODEL_TPM_LIMITS[self.model_name]),
                token_encoding_name=self.model_name,
                max_attempts=int(5),
                logging_level=int(logging.INFO),
            )
        )

        # split the responses into individual files
        with open(responses_in_json_filepath, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                response = json.loads(line)
                self.save_chat(
                    response[0]["messages"],
                    response[1],
                    gen_conf,
                    save_dir=save_dir,
                    prefered_id=idx,
                )

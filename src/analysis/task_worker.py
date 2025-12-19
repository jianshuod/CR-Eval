import os
from tqdm import tqdm
from typing import Union
from src.utils.logging import logging
from src.analysis.query import APIChat
from src.deploy.models import ModelChoice
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.analysis.utils import local_image_to_data_url, score_extraction
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from src.analysis.definition import (
    available_tasks,
    TaskType,
    available_task_chains,
    Task,
)


available_tools = available_tasks

logger = logging.getLogger(__name__)


def show_available_tools():

    print(available_tools.keys())


class Tool:

    def __init__(
        self,
        task: Union[str, Task, tuple],
        model: str = None,
        api_key=None,
        url_base=None,
        using_shot=True,
        gen_conf=None,
        chain_name=None,
        system_suffix=None,
        assistant_prefix=None,
        use_my_prompt=None,
        add_shots_from_jsonl=None,
        is_third_party_chat=False,
        use_hints=False,
    ) -> None:

        if isinstance(task, Task):
            self.task = task
        elif isinstance(task, tuple):
            self.task = Task(
                name=task[0],
                system=task[1],
                shots=None,
                task_type=TaskType.TEXT,
                default_params=dict(temperature=0.2, frequency_penalty=0),
                model=ModelChoice.GPT_4_O,
            )
        else:
            assert task in available_tasks, f"Task {task} not found"
            self.task = available_tasks[task]

        self.task_name = self.task.name
        if model is not None:
            self.model_name = model
        else:
            self.model_name = self.task.model.get_model_string()

        self.chat_engine = APIChat(self.model_name, api_key, url_base)

        self.using_shot = using_shot

        self.system_suffix = system_suffix
        self.assistant_prefix = assistant_prefix
        self.use_hints = use_hints
        system_message_content = (
            use_my_prompt if use_my_prompt is not None else self.task.system
        )
        self.system_message = self.assemble_system_message(system_message_content)
        self.chat_engine.setup(
            self.system_message,
            self.task_name,
            chain_name,
            self.assistant_prefix,
            add_shots_from_jsonl,
            is_third_party_chat,
            use_hints,
        )

        self.gen_conf = self.task.default_params if gen_conf is None else gen_conf
        self.deprecated = self.task.deprecated
        self.is_eval = self.task.is_eval_task
        self.framing_func = self.task.framing_func

        """ The user can determine the global gen_conf when instantiating the Tool object.
            If the user does not provide a gen_conf, the default gen_conf of the task will be used.
            When running, the user can also provide a gen_conf to override the global gen_conf.
        """

    def assemble_system_message(self, system_message_content):
        system_message = f"{system_message_content}"
        if (
            self.task.default_params.get("response_format", "text") == "json_object"
            or self.task.return_json
        ):
            system_message = f"{system_message}\n\n" + f"Return the output in JSON."

        if self.task.shots is not None and self.using_shot:
            system_message = (
                f"{system_message}\n\n"
                + f"# Here are some examples for you.\n\n"
                + f"{self.task.shots}"
                + f"You should strictly follow the response formats in the examples."
                + f"You should not lazily repeat the example responses, but move on to "
                + f"actively analyze the given new samples."
            )

        if self.system_suffix is not None:
            system_message = f"{system_message}" + f"{self.system_suffix}"

        return f"{system_message}"

    @retry(
        wait=wait_random_exponential(min=5, max=15), stop=stop_after_attempt(4)
    )  # Just to retry for altering the port or API connection error
    def run(self, query, gen_conf: dict = None, max_query_len=None, caveat_list=None):
        if self.deprecated:
            raise ValueError(f"Task {self.task.name} is deprecated")

        if self.task.task_type == TaskType.IMAGE:
            if isinstance(query, str):  # only image file path
                assert os.path.exists(query), f"File {query} not found"
                user_message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url(query)},
                            }
                        ],
                    }
                ]
            elif isinstance(query, tuple):  # (text, file_path)
                assert os.path.exists(query[1]), f"File {query[1]} not found"
                user_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{query[0]}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": local_image_to_data_url(query[1])},
                            },
                        ],
                    }
                ]
            else:
                raise ValueError("Invalid query type")
        elif self.task.task_type == TaskType.TEXT:
            if isinstance(query, str):
                user_message = [{"role": "user", "content": query}]
            elif isinstance(query, list) and len(query) == 2:  # with a hint
                user_message = [
                    {"role": "user", "content": query[0]},
                    {"role": "hint", "content": query[1]},
                ]
        else:
            raise ValueError("Unsupported task type")

        gen_conf_x = self.gen_conf.copy()
        if gen_conf is not None:
            gen_conf_x.update(gen_conf)

        answers, response, input_unified = self.chat_engine.chat(
            user_message,
            gen_conf=gen_conf_x,
            max_query_len=max_query_len,
            caveat_list=caveat_list,
        )

        if self.is_eval:  # extract the score
            return answers, [
                score_extraction(self.task.system, answer) for answer in answers
            ]

        if (
            len(answers) == 0 and answers[0] == ""
        ):  # An empty response (<EOS>/<EOT> comes the first) will incur scorer error.
            return [" "], input_unified

        return answers, input_unified

    def run_after_framing(
        self,
        required_data_fields,
        gen_conf: dict = None,
        max_query_len=None,
        caveat_list=None,
    ):
        data = self.framing_func(*required_data_fields)

        return self.run(data, gen_conf, max_query_len, caveat_list)

    def batch_run(
        self, queries, is_parallel: bool = False, num_threads: int = 20, gen_conf=None
    ):
        response_result = []

        if is_parallel:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(self.run, test_case, gen_conf, idx)
                    for idx, test_case in enumerate(queries)
                ]

                for future in tqdm(
                    as_completed(futures),
                    desc="Processing test cases",
                    unit="test case",
                ):
                    try:
                        result = future.result()
                        response_result.append(result)
                    except Exception as e:
                        logger.error(f"Exception occurred in thread {future}: {e}")
        else:
            for idx, test_case in enumerate(
                tqdm(queries, desc="Processing test cases", unit="test case")
            ):
                response_result.append(self.run(test_case, gen_conf, idx))

        return response_result

    def parallel_run(self, request_dir, save_dir, **gen_conf):

        if not gen_conf:
            gen_conf = self.gen_conf

        return self.chat_engine.parallel_chat(request_dir, save_dir, **gen_conf)


class ToolChain:

    def __init__(
        self, taskchain_name: str, model_name: Union[str, list] = None
    ) -> None:
        assert (
            taskchain_name in available_task_chains
        ), f"TaskChain {taskchain_name} not found"
        self.taskchain = available_task_chains[taskchain_name]
        if isinstance(model_name, str):
            self.chain = [
                Tool(task, model_name, chain_name=taskchain_name)
                for task in self.taskchain.task_list
            ]
        elif isinstance(model_name, list):
            assert len(model_name) == len(
                self.taskchain.task_list
            ), "Length of model_name should be the same as the length of task_list"
            self.chain = [
                Tool(task, model, chain_name=taskchain_name)
                for task, model in zip(self.taskchain.task_list, model_name)
            ]
        else:
            self.chain = [
                Tool(task, chain_name=taskchain_name)
                for task in self.taskchain.task_list
            ]
        self.post_processing_funcs = available_task_chains[
            taskchain_name
        ].post_processing_funcs + [None]

    def run_chain(self, query, **kwargs):
        for tool, post_func in zip(self.chain, self.post_processing_funcs):
            response = tool.run(query)
            if response is None:
                return None

            if post_func is not None:
                query = post_func(response, **kwargs)
            else:
                query = response
        return response

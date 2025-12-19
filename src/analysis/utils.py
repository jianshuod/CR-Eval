import os
import re
import sys
import time
import json
import base64
import textwrap
import tiktoken
from mimetypes import guess_type
from modelscope import AutoTokenizer
from src.utils.logging import logging
from src.deploy.models import ModelChoice
from src.analysis.definition import available_tasks
from typing import Union, Tuple, Callable, List, Dict


logger = logging.getLogger(__name__)


def score_extraction(scoring_prompt, completion):

    try:

        if completion in ("-2", "-1", "0", "1", "2"):
            return int(completion)

        pattern = r"s[:=] (-?\d+)"
        numbers = re.findall(pattern, completion)
        numbers = [int(num) for num in numbers]

        if len(numbers) != 0:
            return numbers[-1]

        if "strongly agree" in completion.lower():
            return 2
        elif "strongly disagree" in completion.lower():
            return -2
        elif "disagree" in completion.lower():
            return -1
        elif "agree" in completion.lower():
            return 1
        elif "neutral" in completion.lower():
            return 0

        scoring_template = scoring_prompt.split('"')[1][:-2]  # remove the s,

        pattern = re.compile(re.escape(scoring_template) + r"\D*(\d+)")

        match = pattern.search(completion)

        if match:
            return int(match.group(1))
        else:
            return None
    except IndexError:

        logger.error(
            "Error: The scoring_prompt does not contain a template within quotes."
        )
        return None


def convert_convs_to_completion(prefix, convs: List[Dict]) -> str:
    t = ""
    if convs is not None:
        for conv in convs:
            content = conv.get("content", None)
            role = conv.get("role", None)
            if role == "user":
                t += "\n\n# Here comes the sample that you should analyze."
                t += f"\n\n## Input\n\n{content}\n\n## Output\n"
            elif role == "system":
                t += "# Here comes the instruction."
                t += f"\n\n{content}"
    if prefix is not None:
        t += prefix

    return t


def json_extraction(completion, instance, task_name) -> Tuple[bool, Dict]:

    expected_fields = available_tasks[task_name].json_fields

    answer_fields = available_tasks[task_name].answer_fields

    try:
        completion_fields = json.loads(completion)
        json_success = all(field in completion_fields for field in expected_fields)
    except json.JSONDecodeError:
        json_success = False

    if json_success:
        return json_success, [
            (completion_fields[p_key], instance[a_key])
            for p_key, a_key in zip(expected_fields, answer_fields)
        ]
    else:
        return json_success, {}


def generate_session_name(task_name, chain_name, model_name):

    if os.path.exists(model_name):
        saved_to = model_name.split("/")[-1]
    else:
        saved_to = model_name

    start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    command = "_".join(sys.argv)

    session_name = f"""{start_time}_{chain_name if chain_name is not None else ""}_{task_name}_{saved_to}"""

    return session_name


def print_class_values(obj):
    for attr, value in vars(obj).items():
        print(f"{attr}: {value}")


tokenizer = None


def load_tokenizer(model_choice: Union[ModelChoice, str] = "gpt-3.5-turbo"):

    if isinstance(model_choice, str):
        model_name = model_choice
    elif isinstance(model_choice, ModelChoice):
        model_name = model_choice.get_model_string()
    else:
        raise ValueError("model_choice must be a string or a ModelChoice object.")

    if "gpt" or "claude" in model_name:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
    else:
        tokenizer_id = ModelChoice.from_string(model_name).get_tokenizer_id()
        encoding = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    return encoding


def process_glm_chat(prompt):
    tokenizer = load_tokenizer("ZhipuAI/glm-4-9b-chat")
    inputs = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    return inputs


def num_tokens_from_string(
    string: str, model_choice: Union[ModelChoice, str] = "gpt-3.5-turbo", encoding=None
) -> int:
    """Returns the number of tokens in a text string."""

    if not isinstance(encoding, Callable):
        encoding = load_tokenizer(model_choice)

    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def truncate_query(query_text, max_tokens, model_choice, encoding=None):
    """Truncate a query to a maximum number of tokens."""
    if (
        num_tokens_from_string(query_text, model_choice, encoding=encoding)
        <= max_tokens
    ):
        return query_text
    else:

        if not isinstance(encoding, Callable):
            encoding = load_tokenizer(model_choice)

        tokens = encoding.encode(query_text)
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        return truncated_text


# Function to read the file in blocks and count tokens
def count_tokens_in_file(
    model_choice: ModelChoice,
    file_path: str,
    block_size: int = 1024,
    verbose: bool = False,
) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_choice.get_model_string())
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    try:
        with open(file_path, "r") as file:
            if block_size == float("inf"):
                # Read the entire file at once
                content = file.read()
                block_tokens = num_tokens_from_string(content)
                total_tokens += block_tokens
                if verbose:
                    print(f"The token number is {block_tokens}\n\n\n", content)
            else:
                # Read the file in blocks
                while True:
                    block = file.read(block_size)
                    if not block:
                        break
                    block_tokens = num_tokens_from_string(block)
                    total_tokens += block_tokens
                    if verbose:
                        print(f"The token number is {block_tokens}\n\n\n", block)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    return total_tokens


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def print_json(content):
    print(json.dumps(content, indent=4))


def print_structured(content):
    print(textwrap.dedent(content))

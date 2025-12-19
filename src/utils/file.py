import json
from pathlib import Path
from src.utils.logging import logging

logger = logging.getLogger(__name__)


# 写入列表到JSON文件
def save_to_json_file(item, filename):
    with open(filename, "w") as file:
        json.dump(item, file)


# 从JSON文件读取列表
def load_from_json_file(filename):
    with open(filename, "r") as file:
        item = json.load(file)
    return item


def get_all_files_in_directory(directory_path):
    """
    Get all files in the directory and return their absolute paths.

    :param directory_path: The path to the directory to search for files.
    :return: A list of absolute paths to all files in the directory.
    """
    # Create a Path object from the directory path
    directory = Path(directory_path)

    # Use the rglob method to recursively get all files in the directory
    all_files = [str(file.resolve()) for file in directory.rglob("*") if file.is_file()]

    return all_files


import jsonlines


def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of JSON objects using jsonlines library.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[dict]: A list of dictionaries, each representing a JSON object.
    """
    with jsonlines.open(file_path) as reader:
        json_objects = [obj for obj in reader]

    logger.info(f"Loaded {len(json_objects)} instances from {file_path}")

    return json_objects

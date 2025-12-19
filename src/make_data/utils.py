import json
import random
from src.utils.logging import logging
from argparse import ArgumentTypeError


logger = logging.getLogger(__name__)


def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except TypeError:
        logger.warning(f"Value {value} is not JSON serializable")
        return False


def coin_flip():
    return 0 if random.random() < 0.5 else 1


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


import difflib


def remove_empty_lines(text):
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    return non_empty_lines


def locate_the_revision(original_text, modified_text):
    original_paragraphs = remove_empty_lines(original_text)
    modified_paragraphs = remove_empty_lines(modified_text)

    s = difflib.SequenceMatcher(None, original_paragraphs, modified_paragraphs)

    modified_result = ""
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "replace" or tag == "insert":
            for j in range(j1, j2):
                modified_result += f"{modified_paragraphs[j]}"

    return modified_result


def get_detailed_diff_list(original_text, modified_text):

    revision_counter = 0

    original_paragraphs = remove_empty_lines(original_text)
    modified_paragraphs = remove_empty_lines(modified_text)

    s = difflib.SequenceMatcher(None, original_paragraphs, modified_paragraphs)

    detailed_diff = []

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "replace":
            original_block = original_paragraphs[i1:i2]
            modified_block = modified_paragraphs[j1:j2]
            detailed_diff.append(("[>]", original_block, modified_block))
            revision_counter += 1
        elif tag == "delete":
            original_block = original_paragraphs[i1:i2]
            detailed_diff.append(("[-]", original_block, []))
            revision_counter += 1
        elif tag == "insert":
            modified_block = modified_paragraphs[j1:j2]
            detailed_diff.append(("[+]", [], modified_block))
            revision_counter += 1
        elif tag == "equal":
            for i in range(i1, i2):
                detailed_diff.append(
                    ("[ ]", [original_paragraphs[i]], [original_paragraphs[i]])
                )

    return detailed_diff, revision_counter


def get_masked_statements(original_text, modified_text):

    original_paragraphs = remove_empty_lines(original_text)
    modified_paragraphs = remove_empty_lines(modified_text)

    s = difflib.SequenceMatcher(None, original_paragraphs, modified_paragraphs)

    detailed_diff = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "replace":
            for i in range(i1, i2):
                detailed_diff.append(("[*]", original_paragraphs[i]))
        elif tag == "delete":
            for i in range(i1, i2):
                detailed_diff.append(("[*]", original_paragraphs[i]))
        elif tag == "insert":
            detailed_diff.append(("[?]", ""))
        elif tag == "equal":
            for i in range(i1, i2):
                detailed_diff.append(("", original_paragraphs[i]))

    detailed_diff = "\n".join([f"{op} {content}" for op, content in detailed_diff])

    return detailed_diff

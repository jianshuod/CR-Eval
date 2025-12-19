import nltk
from nltk.data import find


def ensure_nltk_resource(resource_name="punkt_tab"):
    try:
        # Try to find the resource locally
        find(f"tokenizers/{resource_name}")
    except LookupError:
        # Download only if the resource is not found
        print(f"Downloading {resource_name}...")
        nltk.download(resource_name)
        print("Download complete.")


# Usage
# ensure_nltk_resource("punkt_tab")

from nltk.tokenize import sent_tokenize

import difflib


def remove_empty_lines(text):
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    return non_empty_lines


def locate_the_revision_fine_grained(original_text, modified_text):
    original_paragraphs = sent_tokenize(original_text)
    modified_paragraphs = sent_tokenize(modified_text)

    s = difflib.SequenceMatcher(None, original_paragraphs, modified_paragraphs)

    modified_result = ""
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "replace" or tag == "insert":
            for j in range(j1, j2):
                modified_result += f"{modified_paragraphs[j]}"

    return modified_result


def get_detailed_diff(original_text, modified_text):
    original_paragraphs = remove_empty_lines(original_text)
    modified_paragraphs = remove_empty_lines(modified_text)

    s = difflib.SequenceMatcher(None, original_paragraphs, modified_paragraphs)

    detailed_diff = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "replace":
            for i in range(i1, i2):
                detailed_diff.append(f"- {original_paragraphs[i]}")
            for j in range(j1, j2):
                detailed_diff.append(f"+ {modified_paragraphs[j]}")
        elif tag == "delete":
            for i in range(i1, i2):
                detailed_diff.append(f"- {original_paragraphs[i]}")
        elif tag == "insert":
            for j in range(j1, j2):
                detailed_diff.append(f"+ {modified_paragraphs[j]}")
        elif tag == "equal":
            for i in range(i1, i2):
                detailed_diff.append(f"  {original_paragraphs[i]}")

    return detailed_diff

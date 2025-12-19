import re
import json
from pathlib import Path
from src.utils.logging import logging

logger = logging.getLogger(__name__)


def load_json(file_path: Path) -> dict:

    with open(file_path, "r") as f:
        return json.load(f)


def get_cr_reasoning_fields(json_data: dict) -> str:

    return {
        "title": json_data["title"],
        "spec": json_data["spec"],
        "reason_for_change": json_data["reason_for_change"],
        "summary_of_change": json_data["summary_of_change"],
        "consequences_if_not_approved": json_data["consequences_if_not_approved"],
    }


def get_cr_reasoning_text(json_data: dict) -> str:

    extracted_reasoning_fields = get_cr_reasoning_fields(json_data)

    spec_value = extracted_reasoning_fields.pop("spec", "")

    return (
        f"The change request for 3GPP specification {spec_value} includes the following contents:\n\n"
        + "\n".join(
            [
                f"- {key.title()}: {value}"
                for key, value in extracted_reasoning_fields.items()
            ]
        )
    )


def get_cr_reasoning_fields_in_one(json_data: dict) -> str:

    extracted_reasoning_fields = get_cr_reasoning_fields(json_data)

    return " ".join([f"{value}" for key, value in extracted_reasoning_fields.items()])


def extract_reasoning_fields_from_critici_cr(critic_cr: str) -> dict:
    sections = {
        "reason_for_change": "",
        "summary_of_change": "",
        "consequences_if_not_approved": "",
        "title": "",
    }

    # Define the regex patterns for each section
    reason_pattern = r"\*\*Reason for change\*\*:\s*(.*?)(?=\*\*|$)"
    summary_pattern = r"\*\*Summary of change\*\*:\s*(.*?)(?=\*\*|$)"
    consequences_pattern = r"\*\*Consequences if not approved\*\*:\s*(.*?)(?=\*\*|$)"
    title_pattern = r"\*\*Title\*\*:\s*(.*?)(?=\*\*|$)"

    # Extract each section
    reason_match = re.search(reason_pattern, critic_cr, re.DOTALL)
    if reason_match:
        sections["reason_for_change"] = reason_match.group(1).strip()

    summary_match = re.search(summary_pattern, critic_cr, re.DOTALL)
    if summary_match:
        sections["summary_of_change"] = summary_match.group(1).strip()

    consequences_match = re.search(consequences_pattern, critic_cr, re.DOTALL)
    if consequences_match:
        sections["consequences_if_not_approved"] = consequences_match.group(1).strip()

    title_match = re.search(title_pattern, critic_cr, re.DOTALL)
    if title_match:
        sections["title"] = title_match.group(1).strip()

    return sections


def get_cr_reasoning_text_post_processing(critic_cr: str, **kwargs) -> str:

    spec_value = kwargs.get("spec", "")
    extracted_reasoning_fields = extract_reasoning_fields_from_critici_cr(critic_cr)

    return (
        f"The change request for 3GPP specification {spec_value} includes the following contents:\n\n"
        + "\n".join(
            [
                f"- {key.title()}: {value}"
                for key, value in extracted_reasoning_fields.items()
            ]
        )
    )


def get_cr_original_text(json_data: dict) -> str:

    spec_value = json_data.get("spec", "")

    return (
        f"The original text under the change request is excerpted from the 3GPP specification {spec_value}:\n\n"
        + f"""<excerpt>\n{json_data["change_list"][0].strip()}\n</excerpt>"""
    )


def get_cr_revised_text(json_data: dict) -> str:

    spec_value = json_data.get("spec", "")

    return (
        f"The  text under the change request is excerpted from the 3GPP specification {spec_value}:\n\n"
        + f"""<excerpt>\n{json_data["change_list"][0].strip()}\n</excerpt>"""
    )


def get_cr_original_text_test(original_text: str, **kwargs) -> str:

    spec_value = kwargs.get("spec", "")
    return (
        f"The original text under the change request is excerpted from the 3GPP specification {spec_value}:\n\n"
        + f"""<excerpt>\n{original_text.strip()}\n</excerpt>"""
    )

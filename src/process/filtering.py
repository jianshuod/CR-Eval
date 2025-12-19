from datetime import datetime
from collections import defaultdict
from src.utils.logging import logging
from src.utils.date import parse_and_format_date
from src.analysis.utils import num_tokens_from_string
from src.process.diff import locate_the_revision_fine_grained
from src.make_data.play_with_json import get_cr_reasoning_fields_in_one

logger = logging.getLogger(__name__)


def filter_by_validity(crs):
    """
    Check whether some necessary fields are vacant.
    """
    original_cr_count = len(crs)
    accepted_crs = []
    for cr in crs:
        if any(
            [
                cr["spec"] is None,
                cr["current_version"] is None,
                cr["meeting_id"] is None,
                cr["title"] is None,
                cr["reason_for_change"] is None,
                cr["summary_of_change"] is None,
                cr["consequences_if_not_approved"] is None,
                cr["clauses_affected"] is None,
                cr["extracted_index"] is None,
                cr["date"] is None,
                cr["category"] is None,
            ]
        ):  # bypass the invalid cr
            continue
        accepted_crs.append(cr)
    logger.info(f"{len(accepted_crs)} out of {original_cr_count} CRs are valid.")
    return accepted_crs


def check_validity(cr):
    """
    Check whether some necessary fields are vacant.
    """
    try:
        if any(
            [
                cr["spec"] is None,
                cr["current_version"] is None,
                cr["meeting_id"] is None,
                cr["title"] is None,
                cr["reason_for_change"] is None,
                cr["change_list"] is None,
                len(cr["change_list"]) != 2,
                cr["summary_of_change"] is None,
                cr["consequences_if_not_approved"] is None,
                cr["clauses_affected"] is None,
                cr["extracted_index"] is None,
                cr["date"] is None,
                cr["category"] is None,
            ]
        ):  # bypass the invalid cr
            return False
        return True
    except KeyError:
        return False


def normalize_date(cr, start_time=None, end_time=None):
    """
    Filter CRs by time range.
    """
    accepted_crs = []

    # Convert start_time and end_time to datetime objects
    if start_time is not None:
        start_obj = datetime.strptime(start_time, "%Y-%m-%d")
    else:
        start_obj = datetime.strptime("1998-01-01", "%Y-%m-%d")
    if end_time is not None:
        end_obj = datetime.strptime(end_time, "%Y-%m-%d")
    else:
        end_obj = datetime.strptime("2024-05-01", "%Y-%m-%d")

    try:
        cr_time = parse_and_format_date(cr["date"])
        if cr_time is None:
            return None

        # Convert the date string to a datetime object
        date_obj = datetime.strptime(cr_time, "%Y-%m-%d")

        # Check if the date is within the specified time range
        if start_obj <= date_obj <= end_obj:
            return cr_time
        else:
            return None
    except ValueError:
        # Ignore invalid dates
        return None


def filter_by_time(crs, start_time=None, end_time=None):
    """
    Filter CRs by time range.
    """
    accepted_crs = []

    # Convert start_time and end_time to datetime objects
    if start_time is not None:
        start_obj = datetime.strptime(start_time, "%Y-%m-%d")
    else:
        start_obj = datetime.strptime("1998-01-01", "%Y-%m-%d")
    if end_time is not None:
        end_obj = datetime.strptime(end_time, "%Y-%m-%d")
    else:
        end_obj = datetime.strptime("2024-05-01", "%Y-%m-%d")

    for cr in crs:
        cr_time = parse_and_format_date(cr["date"])
        if cr_time is None:
            continue

        try:
            # Convert the date string to a datetime object
            date_obj = datetime.strptime(cr_time, "%Y-%m-%d")
            cr["date"] = cr_time
            # Check if the date is within the specified time range
            if start_obj <= date_obj <= end_obj:
                accepted_crs.append(cr)
        except ValueError:
            # Ignore invalid dates
            continue
    logger.info(f"{len(accepted_crs)} CRs are within the specified time range.")

    return accepted_crs


def filter_by_reasoning_richness(crs, ratio: float = 1):
    """
    Filter CRs by the ratio of reasoning to the total length.
    """
    accepted_crs = []
    for cr in crs:
        # revision
        revision_part = locate_the_revision_fine_grained(
            cr["change_list"][0], cr["change_list"][1]
        )
        revision_length = num_tokens_from_string("gpt-3.5-turbo", revision_part)

        # reasoning
        reasoning_part = get_cr_reasoning_fields_in_one(cr)
        reasoning_length = num_tokens_from_string("gpt-3.5-turbo", reasoning_part)

        # use gpt-3.5-turbo to roughly estimate the token number
        if (
            revision_length != 0
            and reasoning_length != 0
            and reasoning_length / revision_length >= ratio
        ):
            accepted_crs.append(cr)

    return accepted_crs


def group_by_artifact(crs):
    """
    Divide CRs into 2 groups based on whether they are involved in artifacts.
    """

    figure_counter = 0
    table_counter = 0

    crs_impacting_no_artifact = []
    crs_impacting_artifacts = []
    for cr in crs:
        artifact_flag = False
        if cr["table_modified_flag"]:
            artifact_flag = True
            table_counter += 1

        if cr["figure_modified_flag"]:
            artifact_flag = True
            figure_counter += 1

        if artifact_flag:
            crs_impacting_artifacts.append(cr)
        else:
            crs_impacting_no_artifact.append(cr)
    logger.info(
        f"{len(crs_impacting_no_artifact)} CRs are not impacting any artifact, {len(crs_impacting_artifacts)} CRs are impacting artifacts (including {table_counter} tables and {figure_counter} figures)."
    )
    return crs_impacting_no_artifact, crs_impacting_artifacts


def group_by_impact(crs):
    """
    Divide CRs into 3 groups based on their impact.
    """
    crs_impacting_multi_specs = []
    crs_impacting_multi_clauses = []
    crs_impacting_single_clause = []
    for cr in crs:
        if cr["other_specs_affected"][0] != "":
            crs_impacting_multi_specs.append(cr)
        elif cr["clauses_affected"][0].count(",") > 0:
            crs_impacting_multi_clauses.append(cr)
        else:
            crs_impacting_single_clause.append(cr)

    logger.info(
        f"{len(crs_impacting_single_clause)} CRs are impacting a single clause, {len(crs_impacting_multi_clauses)} CRs are impacting multiple clauses, and {len(crs_impacting_multi_specs)} CRs are impacting multiple specs."
    )

    return (
        crs_impacting_single_clause,
        crs_impacting_multi_clauses,
        crs_impacting_multi_specs,
    )


def group_by_category(items):
    grouped = defaultdict(list)
    for item in items:
        category = item.get("category")
        if category is not None:
            grouped[category].append(item)
    return dict(grouped)

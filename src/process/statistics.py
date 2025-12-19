from datetime import datetime
from src.utils.logging import logging
from src.utils.date import parse_and_format_date
from src.analysis.utils import num_tokens_from_string
from src.process.diff import locate_the_revision_fine_grained
from src.make_data.play_with_json import get_cr_reasoning_fields_in_one

logger = logging.getLogger(__name__)


def collectTimeInfo(crs):
    """
    Filter by time
    """
    cr_dates = []
    failure_cases = []
    for cr in crs:
        cr_time = parse_and_format_date(cr["date"])
        if cr_time is None:
            failure_cases.append(cr["date"])
            continue

        try:
            date_obj = datetime.strptime(cr_time, "%Y-%m-%d")

            if date_obj < datetime(1990, 1, 1) or date_obj > datetime(2025, 12, 31):
                failure_cases.append(cr["date"])
            else:
                cr_dates.append(cr_time)
        except ValueError:

            failure_cases.append(cr["date"])

    return cr_dates, failure_cases


def collectReasoningRichnessInfo(crs):
    """
    Filter CRs by the ratio of reasoning to the total length.
    """
    reasoning_richness_list = []

    for cr in crs:
        # revision
        revision_part = locate_the_revision_fine_grained(
            cr["change_list"][0], cr["change_list"][1]
        )
        revision_length = num_tokens_from_string("gpt-3.5-turbo", revision_part)

        # reasoning
        reasoning_part = get_cr_reasoning_fields_in_one(cr)
        reasoning_length = num_tokens_from_string("gpt-3.5-turbo", reasoning_part)
        if revision_length == 0:
            continue
        reasoning_richness_list.append(reasoning_length / revision_length)

    return reasoning_richness_list

from src.utils.logging import logging
from src.analysis.utils import num_tokens_from_string
from src.process.diff import locate_the_revision_fine_grained
from src.make_data.play_with_json import get_cr_reasoning_fields_in_one

logger = logging.getLogger(__name__)


def labeling_reasoning_richness(crs):
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
        if revision_length != 0 and reasoning_length != 0:
            cr["reasoning_richness"] = reasoning_length / revision_length
            accepted_crs.append(cr)

    return accepted_crs


def labeling_impact(crs):
    """
    Divide CRs into 3 groups based on their impact.
    """

    accepted_crs = []

    for cr in crs:

        if cr["other_specs_affected"][0] != "":
            cr["impact"] = "MS"
        elif cr["clauses_affected"][0].count(",") > 0:
            cr["impact"] = "MC"
        else:
            cr["impact"] = "SC"

        accepted_crs.append(cr)

    return accepted_crs


def labeling_reasoning_richness_single(cr):
    """
    Filter CRs by the ratio of reasoning to the total length.
    """
    # revision
    revision_part = locate_the_revision_fine_grained(
        cr["change_list"][0], cr["change_list"][1]
    )
    revision_length = num_tokens_from_string(revision_part)

    # reasoning
    reasoning_part = get_cr_reasoning_fields_in_one(cr)
    reasoning_length = num_tokens_from_string(reasoning_part)

    # use gpt-3.5-turbo to roughly estimate the token number
    if revision_length != 0 and reasoning_length != 0:
        return reasoning_length / revision_length
    else:
        return None


def labeling_impact_single(cr):
    """
    Divide CRs into 3 groups based on their impact.
    """

    if cr["other_specs_affected"][0] != "":
        impact = "MS"
    elif cr["clauses_affected"][0].count(",") > 0:
        impact = "MC"
    else:
        impact = "SC"

    return impact

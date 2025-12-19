from src.make_data.utils import coin_flip


def random_concat(text1, text2):
    if coin_flip():
        return "\n\n".join([text1, text2])
    else:
        return "\n\n".join([text2, text1])



def framing4outline_revision(instance):
    reason_for_change = instance["reason_for_change"]
    consequence_if_not_approved = instance["consequences_if_not_approved"]
    original_spec = instance["input_text"]

    final_text = [
        ">>> Change Request:",
        "- **Reason for change**:",
        reason_for_change,
        "- **Consequences if not revised**:",
        consequence_if_not_approved,
        "",
        ">>> Original Specification Statement:",
        original_spec,
    ]
    final_text = "\n".join(final_text)
    return final_text


def framing4diff_analysis(instance):
    final_text = [
        ">>> Diffed Specification Statements:",
        instance["diffRevision"],
    ]
    final_text = "\n".join(final_text)
    return final_text



def framing4fill_cr(instance):
    original_spec = instance["input_text"]

    final_text = [
        f">>> Original Specification Statements:",
        original_spec,
    ]

    final_text = "\n".join(final_text)
    return final_text



def framing4labeling_sr_cr_reasoning(instance):

    final_text = "\n".join(
        [
            "### Consequences if not revised",
            instance["consequences_if_not_approved"],
            "### Reason for the revision",
            instance["reason_for_change"],
        ]
    )

    return final_text


available_framing_funcs = {
    "fill-cr": framing4fill_cr,
    "outline-revision": framing4outline_revision,
    "diff-analysis": framing4diff_analysis,
}


def apply_framing(instance, framing_func_str):

    framing_func = available_framing_funcs.get(framing_func_str, None)

    if framing_func is None:

        raise ValueError(f"Invalid framing function: {framing_func_str}")

    return framing_func(instance)


def apply_task_framing(instance, task_name):

    from src.analysis.definition import available_tasks

    framing_func = available_tasks[task_name].framing_func

    if framing_func is None:

        raise ValueError(f"Invalid task name: {task_name}")

    return framing_func(instance)


def concatenate_retrieved_text(hits) -> str:

    TextBuffer = ""
    for idx, hit in enumerate(hits):
        TextBuffer += f"#### Change {idx + 1} Begins ####\n"
        TextBuffer += hit.text
        TextBuffer += f"#### Change {idx + 1} Ends ####\n"

    return TextBuffer



def sequential_concat(text1, text2):
    return "\n\n".join([text1, text2])


def framing4gpt_score_outline_revision_v3(a, target):

    revision_a = "\n".join([f"## Reference Edition", target])

    revision_b = "\n".join(
        [
            f"## Drafted Edition",
            a,
        ]
    )

    final_text = sequential_concat(revision_a, revision_b)

    return final_text


def framing4gpt_score_diff_analysis_v3(a, target):

    reasoning_a = "\n".join(["## Reference Report", target])

    reasoning_b = "\n".join(
        [
            "## Drafted Report",
            a,
        ]
    )

    final_text = sequential_concat(reasoning_a, reasoning_b)

    return final_text


def framing4gpt_score_fill_cr_v3(a, target):

    reasoning_a = "\n".join(["## Reference Report", target])

    reasoning_b = "\n".join(
        [
            "## Drafted Report",
            a,
        ]
    )

    final_text = sequential_concat(reasoning_a, reasoning_b)

    return final_text
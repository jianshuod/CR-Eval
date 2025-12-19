def only_score_caveat(instruction):
    return f"{instruction}\n\nDo not do anything else other than scoring. Only the final score (x) should be returned in the form of `s: x`."


def score_after_short_explanation_caveat(instruction):
    return f"{instruction}\n\nBriefly analyze the similarities and differences between the ground-truth answer and the answer to be judged. The final score (x) should be returned in the form of `s: x`."


def explain_scoring_caveat(instruction):
    return (
        f"{instruction}\n\nA strong judgement (2 or -2) means that you are confident while "
        f"a moderate judgement (1 or -1) means that the Drafted answer cannot fully convince you. "
        f"For cases where you cannot give an accurate judgement, for example, a blank answer "
        f"is given, you should return 0 as your judgement."
    )


available_caveats = {
    "only-scoring": only_score_caveat,
    "with-scoring-explation": explain_scoring_caveat,
    "score-after-analysis": score_after_short_explanation_caveat,
}


def apply_caveat(instruction, caveat):
    if caveat in available_caveats:
        return available_caveats[caveat](instruction)
    else:
        print(
            f"{caveat} is not an available choice. It will be directly appended to the end of the instruction."
        )
        return f"{instruction}\n\n{caveat}"

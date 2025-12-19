import numpy as np
from typing import Callable, List
from src.eval.gpt_scorer import *
from src.eval.metric import bleu, rouge
from src.analysis.definition import available_tasks
from src.make_data.instance import EvaluationResult


available_tasks_keys = list(available_tasks.keys())


class ResponseEval:

    def __init__(self, task_name, eval_funs_str: List[str], eval_options) -> None:
        self.task_name = task_name
        self.metrics = []
        for eval_func_str in eval_funs_str:

            if eval_func_str:
                if eval_func_str == "semantic-cosine-similarity":
                    eval_func = self.semantic_cosine_similarity
                else:
                    eval_func = get_eval_func(eval_func_str)
                self.metrics.append((eval_func_str, eval_func))
            else:  # No evaluation is needed
                pass
        self.eval_options = eval_options
        if self.eval_options is not None:
            self.eval_repetition = self.eval_options.pop("eval-repetition")
        else:
            self.eval_repetition = 1

    def get_metric_num(self):
        return len(self.metrics)

    def semantic_cosine_similarity(self, prediction, target):

        pred_embed = get_oai_embedding(prediction)
        target_embed = get_oai_embedding(target)

        score = np.dot(pred_embed, target_embed) / (
            np.linalg.norm(pred_embed) * np.linalg.norm(target_embed)
        )

        return score

    def check_instance_validity_for_eval(self, instance):
        if instance["gold_response"] != "":
            return True

        if self.task_name == "fill-cr":
            if any(
                [
                    instance["summary_of_change"] == "",
                    instance["reason_for_change"] == "",
                    instance["consequences_if_not_approved"] == "",
                ]
            ):
                return False
        elif self.task_name == "outline-revision":
            if any(
                [
                    instance["summary_of_change"] == "",
                ]
            ):
                return False
        elif self.task_name == "diff-analysis":
            if any(
                [
                    instance["summary_of_change"] == "",
                    instance["reason_for_change"] == "",
                    instance["consequences_if_not_approved"] == "",
                ]
            ):
                return False
        else:
            raise ValueError(f"Unsupported task {self.task_name}")

        return True

    def form_desirable_response(self, instance, chat=False):

        if isinstance(instance, dict) and instance.get("gold_response") is not None:
            return instance["gold_response"]

        if self.task_name == "fill-cr":
            target = "\n".join(
                [
                    ">>> REASON FOR CHANGE",
                    instance["reason_for_change"],
                    "",
                    ">>> SUMMARY OF CHANGE",
                    instance["summary_of_change"],
                    "",
                    ">>> CONSEQUENCES IF NOT REVISED",
                    instance["consequences_if_not_approved"],
                ]
            )
        elif self.task_name == "outline-revision":
            target = "\n".join(
                [
                    f">>> SUMMARY OF CHANGES",
                    instance["summary_of_change"],
                ]
            )
        elif self.task_name == "diff-analysis":
            target = "\n".join(
                [
                    ">>> SUMMARY OF CHANGE",
                    instance["summary_of_change"],
                    "",
                    ">>> REASON FOR CHANGE",
                    instance["reason_for_change"],
                    "",
                    ">>> CONSEQUENCES IF NOT REVISED",
                    instance["consequences_if_not_approved"],
                ]
            )
        else:
            raise ValueError(f"Unknown task_name {self.task_name}")

        if chat:
            return [{"role": "assistant", "content": target}]
        else:
            return target

    def run_eval(self, instance, prediction, to_serializable=False):
        if isinstance(instance, str):
            target = instance
        else:
            target = self.form_desirable_response(instance)

        if "</think>" in prediction:
            prediction = prediction.split("</think>")[-1].strip()

        results = []
        # result should in the form of dict
        # e.g., {"bleu": score}
        for eval_func_str, eval_func in self.metrics:
            if "gpt-score" in eval_func_str:
                scoring_texts, scores = eval_func(
                    (
                        (prediction, target)
                        if "v5" in eval_func_str
                        else (prediction, instance)
                    ),
                    gen_conf={"n": self.eval_repetition},
                    **self.eval_options,
                )

                if self.eval_repetition != len(scores):
                    raise ValueError(
                        f"Number of scores is not equal to the number of repetitions: {len(scores)} != {self.eval_repetition}"
                    )

                # Choose the majority among repetitions, breaking ties by selecting the highest score
                final_score = max(set(scores), key=lambda x: (scores.count(x), x))
                # based on the final score, we calculate the variance
                var = sum((x - final_score) ** 2 for x in scores) / len(scores)

                result = EvaluationResult(
                    eval_func_str=eval_func_str,
                    scoring_text=scoring_texts,
                    score=final_score,
                    variance=var,
                )
            else:
                result = eval_func(prediction, target)

            if isinstance(result, EvaluationResult):
                results.append(result if not to_serializable else result.to_dict())
            elif isinstance(result, float):
                results.append(
                    EvaluationResult(eval_func_str, "", result)
                    if not to_serializable
                    else {eval_func_str: result}
                )
            elif isinstance(result, dict):
                for key, value in result.items():
                    results.append(EvaluationResult(key, "", value))

        return results


def get_eval_func(eval_func_str) -> Callable:

    eval_func = None

    if eval_func_str == "bleu":
        eval_func = bleu
    elif eval_func_str == "rouge":
        eval_func = rouge
    elif eval_func_str == "gpt-score-outline-revision-v5":
        eval_func = gpt_scorer_outline_revision_v5.run_after_framing
    elif eval_func_str == "gpt-score-diff-analysis-v5":
        eval_func = gpt_scorer_diff_analysis_v5.run_after_framing
    elif eval_func_str == "gpt-score-fill-cr-v5":
        eval_func = gpt_scorer_fill_cr_v5.run_after_framing
    else:
        raise ValueError(f"Unknown eval_func_str {eval_func_str}")

    return eval_func

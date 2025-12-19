# Copyright 2024 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is borrowed from the T5 library maintained by the Google Research team.
# Original source: https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/evaluation/metrics.py
# We extend our thanks to the original authors for their implementation.


"""Functions for computing metrics.

Every function must accept a list of targets and a list of predictions and
return a dict of metrics.

Functions should assume all text inputs are unicode strings.
"""

import sacrebleu
from absl import logging
from rouge_score import scoring
from rouge_score import rouge_scorer


def bleu(targets, predictions, tokenizer="intl"):
    """Computes BLEU score.

    Args:
      targets: list of strings or list of list of strings if multiple references
        are present.
      predictions: list of strings
      tokenizer: tokenizer option for corpus_bleu

    Returns:
      bleu_score across all targets and predictions
    """
    if isinstance(targets[0], list):
        targets = [[x for x in target] for target in targets]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        targets = [targets]

    bleu_score = sacrebleu.corpus_bleu(
        predictions,
        targets,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize=tokenizer,
        use_effective_order=False,
    )
    return {"bleu": bleu_score.score}


def _prepare_summary_rouge(summary):
    # Make sure the summary is not bytes-type
    # Add newlines between sentences so that rougeLsum is computed correctly.
    summary = summary.replace(" . ", " .\n")
    return summary


def rouge(
    targets,
    predictions,
    score_keys=("rouge1", "rouge2", "rougeLsum"),
    verbose=True,
    **kwargs,
):
    """Computes rouge score nondeterministically using the bootstrap.

    Args:
      targets: list of strings.
      predictions: list of strings.
      score_keys: list of strings with the keys to compute.
      verbose: whether to enable additional logging.
      **kwargs: additional keyword arguments for RougeScorer.

    Returns:
      dict with score_key: rouge score across all targets and predictions
    """

    scorer = rouge_scorer.RougeScorer(rouge_types=score_keys, **kwargs)
    aggregator = scoring.BootstrapAggregator()

    for prediction, target in zip(predictions, targets):
        target = _prepare_summary_rouge(target)
        prediction = _prepare_summary_rouge(prediction)
        aggregator.add_scores(scorer.score(target=target, prediction=prediction))
    result = aggregator.aggregate()
    if verbose:
        for key in score_keys:
            logging.info(
                "%s = %.2f, 95%% confidence [%.2f, %.2f]",
                key,
                result[key].mid.fmeasure * 100,
                result[key].low.fmeasure * 100,
                result[key].high.fmeasure * 100,
            )
    return {key: result[key].mid.fmeasure * 100 for key in score_keys}

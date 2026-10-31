"""Post-hoc analysis of a Learning by Consensus run, from its vote traces.

Everything here is derived from the per-example vote traces the clients
return (see ConsensusClient.evaluate), so it costs no extra computation on
the clients:

  - metrics of every hypothesis H_i alone on the common test set;
  - metrics of the consensus (including MCC);
  - vote margins: for each example, how far the vote was from the majority
    threshold (a margin of 0 means the example was decided by a single vote).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def confusion_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def analyse_vote_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """traces: the list stored in vote_traces.json."""
    if not traces:
        return {}

    hypothesis_names = [vote["hypothesis"] for vote in traces[0]["votes"]]
    counts = {name: Counter() for name in hypothesis_names}
    consensus = Counter()
    margins: Counter = Counter()
    margin_by_outcome = {"correct": [], "wrong": []}

    for trace in traces:
        is_positive = trace["true_label"] == "POS"

        for vote in trace["votes"]:
            predicted_positive = vote["prediction"] == "POS"
            counts[vote["hypothesis"]][
                ("t" if is_positive == predicted_positive else "f")
                + ("p" if predicted_positive else "n")
            ] += 1

        predicted_positive = trace["prediction"] == "POS"
        consensus[
            ("t" if is_positive == predicted_positive else "f")
            + ("p" if predicted_positive else "n")
        ] += 1

        # Distance from the decision threshold, in votes. Positive votes
        # needed: required_majority. A margin of 0 = decided by one vote.
        positive_votes = trace["positive_votes"]
        required = trace["required_majority"]
        margin = (
            positive_votes - required
            if predicted_positive
            else required - 1 - positive_votes
        )
        margins[margin] += 1
        margin_by_outcome["correct" if trace["correct"] else "wrong"].append(margin)

    def mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "consensus": confusion_metrics(
            consensus["tp"], consensus["fp"], consensus["tn"], consensus["fn"]
        ),
        "per_hypothesis": {
            name: confusion_metrics(c["tp"], c["fp"], c["tn"], c["fn"])
            for name, c in counts.items()
        },
        "margin_distribution": {str(k): v for k, v in sorted(margins.items())},
        "mean_margin_correct": mean(margin_by_outcome["correct"]),
        "mean_margin_wrong": mean(margin_by_outcome["wrong"]),
        "examples_decided_by_one_vote": margins.get(0, 0),
    }

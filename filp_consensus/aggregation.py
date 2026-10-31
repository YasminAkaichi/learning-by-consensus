"""Aggregation utilities for Learning by Consensus.

Consensus does not merge the rules learned by the clients.
Each local hypothesis remains an independent voter.

Given K hypotheses, an example is predicted as positive when at least

    floor(K / 2) + 1

hypotheses predict it as positive.
"""

from collections.abc import Mapping, Sequence
from typing import Hashable


ExampleId = Hashable
Predictions = Mapping[ExampleId, bool]


def majority_threshold(n_hypotheses: int) -> int:
    """Return the number of positive votes required for a strict majority."""
    if n_hypotheses <= 0:
        raise ValueError("At least one hypothesis is required.")

    return (n_hypotheses // 2) + 1


def majority_vote(
    predictions: Sequence[Predictions],
) -> dict[ExampleId, bool]:
    """Aggregate hypothesis predictions using strict majority voting.

    Parameters
    ----------
    predictions:
        One prediction mapping per hypothesis:
            predictions[hypothesis][example] -> bool

    Returns
    -------
    dict
        Final consensus prediction for every example.

    Raises
    ------
    ValueError
        If no hypothesis is supplied or hypotheses do not predict
        exactly the same examples.
    """
    if not predictions:
        raise ValueError("Cannot compute consensus without hypotheses.")

    example_ids = set(predictions[0])

    for index, hypothesis_predictions in enumerate(predictions[1:], start=1):
        if set(hypothesis_predictions) != example_ids:
            raise ValueError(
                f"Hypothesis {index} does not contain the same examples "
                "as the other hypotheses."
            )

    threshold = majority_threshold(len(predictions))

    return {
        example_id: (
            sum(
                bool(hypothesis_predictions[example_id])
                for hypothesis_predictions in predictions
            )
            >= threshold
        )
        for example_id in example_ids
    }
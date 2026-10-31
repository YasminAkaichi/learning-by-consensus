"""Generic evaluation utilities for Learning by Consensus."""

from __future__ import annotations

import re
from pathlib import Path


def read_labels(exs_file: str | Path) -> dict[str, bool]:
    """Read ground-truth labels from a Popper exs.pl file."""

    exs_file = Path(exs_file)

    labels: dict[str, bool] = {}

    pattern = re.compile(
        r"^(pos|neg)\((\w+)\(([^)]+)\)\)\.$"
    )

    for raw_line in exs_file.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line or line.startswith("%"):
            continue

        match = pattern.match(line)

        if not match:
            raise ValueError(
                f"Unsupported example: {line}"
            )

        label, _, example_id = match.groups()

        labels[example_id] = label == "pos"

    return labels


def read_andante_labels(
    dataset_file: str | Path,
) -> dict[str, bool]:
    """Read labels from a native Andante dataset."""

    dataset_file = Path(dataset_file)

    text = dataset_file.read_text(
        encoding="utf-8"
    )

    labels: dict[str, bool] = {}

    sections = (
        (
            ":- begin_in_pos.",
            ":- end_in_pos.",
            True,
        ),
        (
            ":- begin_in_neg.",
            ":- end_in_neg.",
            False,
        ),
    )

    for begin, end, label in sections:

        if begin not in text or end not in text:
            raise ValueError(
                f"Missing Andante section: {begin}"
            )

        content = (
            text.split(begin, 1)[1]
            .split(end, 1)[0]
        )

        for raw_line in content.splitlines():

            line = raw_line.strip()

            if (
                not line
                or line.startswith("%")
            ):
                continue

            if not line.endswith("."):
                raise ValueError(
                    f"Unsupported Andante example: {line}"
                )

            example = line[:-1].strip()

            labels[example] = label

    return labels
def evaluate_predictions(
    labels: dict[str, bool],
    predictions: dict[str, bool],
) -> dict[str, float | int]:

    if set(labels) != set(predictions):
        raise ValueError(
            "Ground-truth labels and predictions "
            "do not contain the same examples."
        )

    tp = tn = fp = fn = 0

    for example_id, true_label in labels.items():

        predicted = predictions[example_id]

        if true_label and predicted:
            tp += 1

        elif true_label and not predicted:
            fn += 1

        elif not true_label and predicted:
            fp += 1

        else:
            tn += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # Matthews correlation coefficient: unlike accuracy/F1, it stays
    # meaningful when the classes are unbalanced. 0 if undefined.
    denominator = (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    ) ** 0.5
    mcc = (
        (tp * tn - fp * fn) / denominator
        if denominator else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
from __future__ import annotations

import random

from filp_consensus.partitioning.dataset import Dataset


def split_train_test(
    dataset: Dataset,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Stratified train/test split.

    Positive and negative examples are split independently so that
    both train and test preserve the class distribution.

    Background knowledge, rules, and bias are initially retained.
    Client-specific/local BK is selected later by partition_dataset().
    """

    if not 0.0 < test_ratio < 1.0:
        raise ValueError(
            "test_ratio must be strictly between 0 and 1."
        )

    rng = random.Random(random_seed)

    positive = list(dataset.positive_examples)
    negative = list(dataset.negative_examples)

    rng.shuffle(positive)
    rng.shuffle(negative)

    n_positive_test = _test_size(
        len(positive),
        test_ratio,
    )

    n_negative_test = _test_size(
        len(negative),
        test_ratio,
    )

    positive_test = positive[:n_positive_test]
    positive_train = positive[n_positive_test:]

    negative_test = negative[:n_negative_test]
    negative_train = negative[n_negative_test:]

    train_dataset = Dataset(
        name=f"{dataset.name}_train",
        positive_examples=positive_train,
        negative_examples=negative_train,
        facts=list(dataset.facts),
        rules=list(dataset.rules),
        bias=dataset.bias,
    )

    test_dataset = Dataset(
        name=f"{dataset.name}_test",
        positive_examples=positive_test,
        negative_examples=negative_test,
        facts=list(dataset.facts),
        rules=list(dataset.rules),
        bias=dataset.bias,
    )

    return train_dataset, test_dataset


def _test_size(
    number_of_examples: int,
    test_ratio: float,
) -> int:
    if number_of_examples == 0:
        return 0

    if number_of_examples == 1:
        return 0

    size = round(number_of_examples * test_ratio)

    # Keep at least one example in both train and test.
    return max(
        1,
        min(size, number_of_examples - 1),
    )
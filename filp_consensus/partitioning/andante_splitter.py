"""Split a native Andante dataset into train/test and client partitions.

Expected Andante format:

    <header: set, modeh, modeb, determinations...>

    :- begin_bg.
    ...
    :- end_bg.

    :- begin_in_pos.
    ...
    :- end_in_pos.

    :- begin_in_neg.
    ...
    :- end_in_neg.

The header and background knowledge are preserved in every generated
dataset. Only positive and negative examples are partitioned.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _extract_section(
    text: str,
    begin_marker: str,
    end_marker: str,
) -> tuple[str, list[str], str]:
    """Extract one Andante section."""

    if begin_marker not in text:
        raise ValueError(f"Missing marker: {begin_marker}")

    if end_marker not in text:
        raise ValueError(f"Missing marker: {end_marker}")

    before, rest = text.split(begin_marker, 1)
    content, after = rest.split(end_marker, 1)

    examples = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("%")
    ]

    return before, examples, after


def parse_andante_dataset(
    dataset_file: str | Path,
) -> tuple[str, list[str], list[str]]:
    """Read a native Andante dataset."""

    dataset_file = Path(dataset_file).resolve()

    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"Andante dataset not found: {dataset_file}"
        )

    text = dataset_file.read_text(encoding="utf-8")

    pos_begin = ":- begin_in_pos."
    pos_end = ":- end_in_pos."
    neg_begin = ":- begin_in_neg."
    neg_end = ":- end_in_neg."

    before_pos, positives, after_pos = _extract_section(
        text,
        pos_begin,
        pos_end,
    )

    _, negatives, after_neg = _extract_section(
        after_pos,
        neg_begin,
        neg_end,
    )

    # Everything before the examples contains the native Andante
    # configuration, modes, determinations and background knowledge.
    prefix = before_pos.rstrip()

    # Preserve anything after the negative examples too.
    suffix = after_neg.strip()

    if suffix:
        prefix += "\n\n" + suffix

    return prefix, positives, negatives


def write_dataset(
    output_file: Path,
    prefix: str,
    positives: list[str],
    negatives: list[str],
) -> None:
    """Write one native Andante dataset."""

    lines = [
        prefix.rstrip(),
        "",
        ":- begin_in_pos.",
        "",
        *positives,
        "",
        ":- end_in_pos.",
        "",
        ":- begin_in_neg.",
        "",
        *negatives,
        "",
        ":- end_in_neg.",
        "",
    ]

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def split_examples(
    examples: list[str],
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    """Split examples into train and test."""

    examples = list(examples)
    rng.shuffle(examples)

    n_test = round(len(examples) * test_ratio)

    if len(examples) > 1:
        n_test = max(1, min(n_test, len(examples) - 1))

    test = examples[:n_test]
    train = examples[n_test:]

    return train, test


def partition_clients(
    examples: list[str],
    n_clients: int,
) -> list[list[str]]:
    """Distribute examples across clients."""

    partitions = [
        []
        for _ in range(n_clients)
    ]

    for index, example in enumerate(examples):
        partitions[index % n_clients].append(example)

    return partitions


def generate_consensus_dataset(
    source: str | Path,
    output_dir: str | Path,
    n_clients: int = 2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    """Generate native Andante datasets for Consensus."""

    if n_clients < 1:
        raise ValueError(
            "n_clients must be at least 1."
        )

    if not 0 < test_ratio < 1:
        raise ValueError(
            "test_ratio must be between 0 and 1."
        )

    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()

    prefix, positives, negatives = (
        parse_andante_dataset(source)
    )

    rng = random.Random(seed)

    train_pos, test_pos = split_examples(
        positives,
        test_ratio,
        rng,
    )

    train_neg, test_neg = split_examples(
        negatives,
        test_ratio,
        rng,
    )

    # Shuffle train examples before round-robin partitioning.
    rng.shuffle(train_pos)
    rng.shuffle(train_neg)

    client_pos = partition_clients(
        train_pos,
        n_clients,
    )

    client_neg = partition_clients(
        train_neg,
        n_clients,
    )

    for client_index in range(n_clients):
        client_file = (
            output_dir
            / "train"
            / f"family_part{client_index + 1}.pl"
        )

        write_dataset(
            client_file,
            prefix,
            client_pos[client_index],
            client_neg[client_index],
        )

    test_file = (
        output_dir
        / "test"
        / "family_test.pl"
    )

    write_dataset(
        test_file,
        prefix,
        test_pos,
        test_neg,
    )

    print(
        f"Original: {len(positives)} POS / "
        f"{len(negatives)} NEG"
    )

    print(
        f"Global train: {len(train_pos)} POS / "
        f"{len(train_neg)} NEG"
    )

    print(
        f"Global test: {len(test_pos)} POS / "
        f"{len(test_neg)} NEG"
    )

    for index in range(n_clients):
        print(
            f"Client {index + 1}: "
            f"{len(client_pos[index])} POS / "
            f"{len(client_neg[index])} NEG"
        )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    generate_consensus_dataset(
        source=args.source,
        output_dir=args.output,
        n_clients=args.clients,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import json
import shutil
from pathlib import Path

from filp_consensus.partitioning.dataset import Dataset
from filp_consensus.partitioning.partitioner import (
    DatasetPartition,
    _attach_background_knowledge,
)


def write_consensus_dataset(
    dataset_name: str,
    train_partitions: list[DatasetPartition],
    test_dataset: Dataset,
    strategy: str,
    random_seed: int,
    test_ratio: float,
    output_root: Path,
) -> Path:
    output_directory = (
        output_root
        / dataset_name
        / (
            f"consensus_{strategy}_"
            f"{len(train_partitions)}_"
            f"seed_{random_seed}"
        )
    )

    if output_directory.exists():
        shutil.rmtree(output_directory)

    train_directory = output_directory / "train"
    test_directory = output_directory / "test"

    train_directory.mkdir(parents=True)
    test_directory.mkdir(parents=True)

    # -------------------------
    # TRAIN CLIENTS
    # -------------------------

    clients_metadata = []

    for partition in train_partitions:
        client_directory = (
            train_directory
            / f"{dataset_name}_part{partition.client_id}"
        )

        client_directory.mkdir()

        _write_partition(
            directory=client_directory,
            partition=partition,
        )

        clients_metadata.append(
            {
                "client_id": partition.client_id,
                "positive_examples": len(
                    partition.positive_examples
                ),
                "negative_examples": len(
                    partition.negative_examples
                ),
                "facts": len(partition.facts),
                "rules": len(partition.rules),
                "directory": str(
                    client_directory.relative_to(
                        output_directory
                    )
                ),
            }
        )

    # -------------------------
    # GLOBAL TEST
    # -------------------------

    test_partition = DatasetPartition(
        client_id=0,
        positive_examples=list(
            test_dataset.positive_examples
        ),
        negative_examples=list(
            test_dataset.negative_examples
        ),
        bias=test_dataset.bias,
    )

    _attach_background_knowledge(
        dataset=test_dataset,
        partitions=[test_partition],
    )

    _write_partition(
        directory=test_directory,
        partition=test_partition,
    )

    # -------------------------
    # METADATA
    # -------------------------

    metadata = {
        "dataset": dataset_name,
        "engine": "consensus",
        "strategy": strategy,
        "number_of_clients": len(train_partitions),
        "random_seed": random_seed,
        "test_ratio": test_ratio,
        "voting": "strict_majority",
        "train": {
            "positive_examples": sum(
                len(p.positive_examples)
                for p in train_partitions
            ),
            "negative_examples": sum(
                len(p.negative_examples)
                for p in train_partitions
            ),
            "clients": clients_metadata,
        },
        "test": {
            "positive_examples": len(
                test_partition.positive_examples
            ),
            "negative_examples": len(
                test_partition.negative_examples
            ),
            "facts": len(test_partition.facts),
            "rules": len(test_partition.rules),
            "directory": "test",
        },
    }

    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return output_directory


def _write_partition(
    directory: Path,
    partition: DatasetPartition,
) -> None:
    # Examples
    example_lines = []

    for example in partition.positive_examples:
        example_lines.append(
            f"pos({example.identifier})."
        )

    for example in partition.negative_examples:
        example_lines.append(
            f"neg({example.identifier})."
        )

    (directory / "exs.pl").write_text(
        "\n".join(example_lines) + "\n",
        encoding="utf-8",
    )

    # Background knowledge
    statements = [
        fact.text
        for fact in partition.facts
    ]

    statements.extend(
        rule.text
        for rule in partition.rules
    )

    (directory / "bk.pl").write_text(
        "\n\n".join(statements) + "\n",
        encoding="utf-8",
    )

    # Bias
    (directory / "bias.pl").write_text(
        partition.bias,
        encoding="utf-8",
    )
"""Isolated Popper-v4 predictor for Consensus.

This module is executed in a fresh Python process so that the Prolog
state used during training cannot leak into evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import janus_swi as janus


def read_examples(
    exs_file: Path,
) -> tuple[list[str], str]:

    examples: list[str] = []
    predicate = None

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

        _, current_predicate, example_id = (
            match.groups()
        )

        if predicate is None:
            predicate = current_predicate

        elif predicate != current_predicate:
            raise ValueError(
                "Multiple target predicates found."
            )

        examples.append(example_id)

    if predicate is None:
        raise ValueError(
            f"No examples found in {exs_file}"
        )

    return examples, predicate


def predict(
    hypothesis: list[str],
    test_dataset: Path,
) -> tuple[dict[str, bool], dict[str, list[int]]]:
    """Returns (predictions, fired): fired[example] = 1-based indexes of the
    rules of the hypothesis that cover the example."""

    exs_file = test_dataset / "exs.pl"
    bk_file = test_dataset / "bk.pl"

    examples, predicate = read_examples(exs_file)

    # Fresh process:
    # only TEST background knowledge is loaded.
    janus.consult(str(bk_file))

    # The evaluation set only holds the facts of ITS examples, so a predicate
    # that a hypothesis learned elsewhere uses may have no fact here at all.
    # Prolog then raises "Unknown procedure" instead of simply failing. Every
    # body predicate declared in the bias that is still undefined is declared
    # dynamic: a call to it fails, which is the right answer ("no proof").
    bias_file = test_dataset / "bias.pl"

    if bias_file.is_file():
        for line in bias_file.read_text(encoding="utf-8").splitlines():
            declaration = re.match(
                r"^\s*body_pred\(\s*(\w+)\s*,\s*(\d+)\s*\)",
                line,
            )

            if declaration:
                name, arity = declaration.groups()
                # No Prolog variable is used, so janus has nothing to convert
                # back to Python. System predicates cannot be declared
                # dynamic: such errors are ignored.
                try:
                    exists = janus.query_once(
                        f"current_predicate({name}/{arity})"
                    ).get("truth", False)

                    if not exists:
                        janus.query_once(f"dynamic({name}/{arity})")
                except janus.PrologError:
                    pass

    rules = []

    for rule in hypothesis:

        rule = rule.strip()

        if not rule:
            continue

        if not rule.endswith("."):
            rule += "."

        rules.append(rule)

    # Traceability: every rule is also compiled under its own head name
    # (fired_<k>), so we can tell, for each example, WHICH rules of the
    # hypothesis cover it. The body is left untouched, so the semantics of
    # the rule (including calls to the target predicate) is unchanged.
    head_pattern = re.compile(rf"^\s*{re.escape(predicate)}\(")
    traced_rules = [
        head_pattern.sub(f"fired_{index}(", rule, count=1)
        for index, rule in enumerate(rules, start=1)
    ]

    program = (
        f":- dynamic {predicate}/1.\n"
        + "".join(
            f":- dynamic fired_{index}/1.\n"
            for index in range(1, len(rules) + 1)
        )
        + "\n".join(rules)
        + "\n"
        + "\n".join(traced_rules)
    )

    janus.consult(
        "consensus_hypothesis",
        program,
    )

    predictions = {}
    fired = {}

    for example_id in examples:

        query = f"{predicate}({example_id})"

        result = janus.query_once(query)

        predictions[example_id] = bool(
            result.get("truth", False)
        )

        fired[example_id] = [
            index
            for index in range(1, len(rules) + 1)
            if janus.query_once(
                f"fired_{index}({example_id})"
            ).get("truth", False)
        ]

    return predictions, fired


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-dataset",
        required=True,
    )

    parser.add_argument(
        "--hypothesis-file",
        required=True,
    )

    args = parser.parse_args()

    hypothesis = json.loads(
        Path(args.hypothesis_file).read_text(
            encoding="utf-8"
        )
    )

    # SWI-Prolog may print to stdout (e.g. output of the background
    # knowledge). Point file descriptor 1 at stderr while predicting, so
    # that stdout only carries the JSON written below.
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    os.dup2(2, 1)

    try:
        predictions, fired = predict(
            hypothesis=hypothesis,
            test_dataset=Path(args.test_dataset),
        )
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)

    # stdout is intentionally machine-readable.
    print(json.dumps({"predictions": predictions, "fired": fired}), flush=True)


if __name__ == "__main__":
    main()
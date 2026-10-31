"""Convert a Popper dataset directory to a monolithic Andante dataset.

Input format
------------
dataset/
    bias.pl
    bk.pl
    exs.pl

Output format
-------------
dataset.pl

The converter translates:
- Popper head/body predicates + type/direction declarations -> Andante modes
- body predicates -> Andante determinations
- bk.pl -> Andante background knowledge
- exs.pl -> Andante positive/negative examples
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PRED_RE = re.compile(
    r"^(head_pred|body_pred)\((\w+),(\d+)\)\.$"
)

TYPE_RE = re.compile(
    r"^type\((\w+),\((.*?)\)\)\.$"
)

DIRECTION_RE = re.compile(
    r"^direction\((\w+),\((.*?)\)\)\.$"
)

EXAMPLE_RE = re.compile(
    r"^(pos|neg)\((.+)\)\.$"
)


def _clean_items(text: str) -> list[str]:
    """Parse comma-separated Prolog tuple content."""
    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def parse_bias(
    bias_file: Path,
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    """Extract predicate, type and direction declarations."""

    head_preds: dict[str, int] = {}
    body_preds: dict[str, int] = {}
    types: dict[str, list[str]] = {}
    directions: dict[str, list[str]] = {}

    for raw_line in bias_file.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line or line.startswith("%"):
            continue

        pred_match = PRED_RE.match(line)

        if pred_match:
            kind, predicate, arity = pred_match.groups()

            if kind == "head_pred":
                head_preds[predicate] = int(arity)
            else:
                body_preds[predicate] = int(arity)

            continue

        type_match = TYPE_RE.match(line)

        if type_match:
            predicate, values = type_match.groups()
            types[predicate] = _clean_items(values)
            continue

        direction_match = DIRECTION_RE.match(line)

        if direction_match:
            predicate, values = direction_match.groups()
            directions[predicate] = _clean_items(values)

    return (
        head_preds,
        body_preds,
        types,
        directions,
    )


def build_mode(
    predicate: str,
    arity: int,
    types: dict[str, list[str]],
    directions: dict[str, list[str]],
) -> str:
    """Translate Popper type/direction declarations to Andante mode."""

    predicate_types = types.get(predicate)
    predicate_directions = directions.get(predicate)

    if predicate_types is None:
        raise ValueError(
            f"Missing type declaration for {predicate}/{arity}"
        )

    if predicate_directions is None:
        # Popper accepts a bias without directions (every mode is allowed),
        # but Andante needs +/- modes. Default used by Progol-style
        # biases: first argument is an input, the others are outputs
        # (e.g. has_car(+train,-car), short(+car)).
        predicate_directions = ["in"] + ["out"] * (arity - 1)

    if len(predicate_types) != arity:
        raise ValueError(
            f"Type arity mismatch for {predicate}/{arity}: "
            f"{predicate_types}"
        )

    if len(predicate_directions) != arity:
        raise ValueError(
            f"Direction arity mismatch for "
            f"{predicate}/{arity}: "
            f"{predicate_directions}"
        )

    arguments = []

    for direction, argument_type in zip(
        predicate_directions,
        predicate_types,
    ):
        if direction == "in":
            prefix = "+"
        elif direction == "out":
            prefix = "-"
        else:
            raise ValueError(
                f"Unsupported direction '{direction}' "
                f"for {predicate}/{arity}"
            )

        arguments.append(
            f"{prefix}{argument_type}"
        )

    return (
        f"{predicate}("
        + ",".join(arguments)
        + ")"
    )


def read_examples(
    exs_file: Path,
) -> tuple[list[str], list[str]]:
    """Read Popper pos(...) and neg(...) examples."""

    positives: list[str] = []
    negatives: list[str] = []

    for raw_line in exs_file.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if not line or line.startswith("%"):
            continue

        match = EXAMPLE_RE.match(line)

        if not match:
            raise ValueError(
                f"Unsupported example declaration: {line}"
            )

        label, example = match.groups()

        if label == "pos":
            positives.append(example + ".")
        else:
            negatives.append(example + ".")

    return positives, negatives


# Andante's own Prolog-like solver has no cut and no arithmetic comparison.
# Some Popper background-knowledge files (Mutagenesis, Carcinogenesis) have
# SWI-Prolog helper rules that use them, e.g. `gteq(X,Y) :- ..., X >= Y, !.`,
# and Andante's parser then rejects the WHOLE file ("Failed to parse rule").
_UNSUPPORTED_IN_RULE = re.compile(
    r"!|>=|=<|=:=|=\\=|\bis\b|(?<![-=:<>])[<>](?![-=:<>])"
)


def _drop_unsupported_rules(background: str) -> str:
    """Remove the rules that Andante cannot parse (cut, arithmetic
    comparison). Facts and the other rules are kept unchanged."""

    statements: list[str] = []
    current: list[str] = []

    for raw_line in background.splitlines():
        stripped = raw_line.strip()

        if not stripped or stripped.startswith("%"):
            continue

        current.append(raw_line)

        if stripped.endswith("."):
            statements.append("\n".join(current))
            current = []

    if current:
        statements.append("\n".join(current))

    kept = [
        statement
        for statement in statements
        if not (
            ":-" in statement
            and _UNSUPPORTED_IN_RULE.search(statement)
        )
    ]

    return "\n".join(kept)


def convert_popper_to_andante(
    source_dir: str | Path,
    output_file: str | Path,
) -> Path:
    """Convert one Popper dataset to Andante format."""

    source_dir = Path(source_dir).resolve()
    output_file = Path(output_file).resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Popper dataset not found: {source_dir}"
        )

    bias_file = source_dir / "bias.pl"
    bk_file = source_dir / "bk.pl"
    exs_file = source_dir / "exs.pl"

    for required in (
        bias_file,
        bk_file,
        exs_file,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing required file: {required}"
            )

    (
        head_preds,
        body_preds,
        types,
        directions,
    ) = parse_bias(bias_file)

    if not head_preds:
        raise ValueError(
            "No head predicate found in bias.pl"
        )

    positives, negatives = read_examples(
        exs_file
    )

    lines: list[str] = []

    # --------------------------------------------------
    # Options
    # --------------------------------------------------

    # Andante's default search budget is only 100 states per clause, which
    # is too small: on Zendo it returns an EMPTY hypothesis even with enough
    # examples. Same values as the hand-converted datasets
    # (datasets/andante/zendo1.pl).
    lines.append("set(max_search_states,5000).")
    lines.append("set(i,3).")
    lines.append("set(verbose,0).")
    lines.append("")

    # --------------------------------------------------
    # Modes
    # --------------------------------------------------

    for predicate, arity in head_preds.items():

        mode = build_mode(
            predicate,
            arity,
            types,
            directions,
        )

        lines.append(
            f"modeh(1,{mode})."
        )

    for predicate, arity in body_preds.items():

        mode = build_mode(
            predicate,
            arity,
            types,
            directions,
        )

        lines.append(
            f"modeb(*,{mode})."
        )

    lines.append("")

    # --------------------------------------------------
    # Determinations
    # --------------------------------------------------

    for head_predicate, head_arity in head_preds.items():

        for body_predicate, body_arity in body_preds.items():

            lines.append(
                "determination("
                f"{head_predicate}/{head_arity},"
                f"{body_predicate}/{body_arity}"
                ")."
            )

    lines.append("")

    # --------------------------------------------------
    # Background knowledge
    # --------------------------------------------------

    lines.append(":- begin_bg.")
    lines.append("")

    background_lines = []

    for raw_line in bk_file.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        # SWI-Prolog-specific directives used by the Popper dataset.
        # They are not part of Andante's input grammar.
        if line.startswith(":-style_check("):
            continue

        background_lines.append(raw_line)

    background = _drop_unsupported_rules(
        "\n".join(background_lines)
    ).strip()

    lines.append(background)

    lines.append("")
    lines.append(":- end_bg.")
    lines.append("")

    # --------------------------------------------------
    # Positive examples
    # --------------------------------------------------

    lines.append(":- begin_in_pos.")
    lines.append("")

    lines.extend(positives)

    lines.append("")
    lines.append(":- end_in_pos.")
    lines.append("")

    # --------------------------------------------------
    # Negative examples
    # --------------------------------------------------

    lines.append(":- begin_in_neg.")
    lines.append("")

    lines.extend(negatives)

    lines.append("")
    lines.append(":- end_in_neg.")
    lines.append("")

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_file


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Convert a Popper dataset directory "
            "to an Andante .pl file."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Popper dataset directory containing "
            "bias.pl, bk.pl and exs.pl"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output Andante .pl file",
    )

    args = parser.parse_args()

    output = convert_popper_to_andante(
        args.source,
        args.output,
    )

    print(
        f"Andante dataset written to: {output}"
    )


if __name__ == "__main__":
    main()
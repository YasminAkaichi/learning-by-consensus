"""Isolated Popper-v4 trainer for Consensus.

Each invocation runs Popper in a fresh Python/SWI-Prolog process,
preventing the Prolog state of one client from leaking into another.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from popper.loop import learn_solution
from popper.util import Settings, format_rule


def _declare_missing_predicates_after_loading(dataset_path: Path) -> None:
    """A client only holds the facts of ITS examples, so a body predicate of
    the bias can have no fact at all in its background knowledge (e.g. a rare
    clinical feature that none of its patients has). SWI-Prolog then raises
    "Unknown procedure" when Popper tests a rule using it, instead of simply
    failing. Right after Popper has loaded the background knowledge, every
    body predicate of the bias that is still undefined is declared dynamic: a
    call to it then fails, which is the correct answer.

    Popper's own code is left untouched: only its Tester constructor is
    wrapped.
    """
    import re

    import janus_swi as janus
    from popper import tester as popper_tester

    bias_file = dataset_path / "bias.pl"
    pattern = re.compile(r"^\s*body_pred\(\s*(\w+)\s*,\s*(\d+)\s*\)")
    predicates = []

    if bias_file.is_file():
        for line in bias_file.read_text(encoding="utf-8").splitlines():
            declaration = pattern.match(line)
            if declaration:
                predicates.append(declaration.groups())

    original_init = popper_tester.Tester.__init__

    def init_with_declarations(self, settings):
        original_init(self, settings)
        for name, arity in predicates:
            try:
                exists = janus.query_once(
                    f"current_predicate({name}/{arity})"
                ).get("truth", False)
                if not exists:
                    janus.query_once(f"dynamic({name}/{arity})")
            except janus.PrologError:
                pass

    popper_tester.Tester.__init__ = init_with_declarations


def train(
    dataset_path: Path,
    timeout: int = 600,
    debug: bool = False,
    show_stats: bool = False,
) -> dict:

    dataset_path = dataset_path.resolve()

    required_files = ("bias.pl", "bk.pl", "exs.pl")

    missing = [
        filename
        for filename in required_files
        if not (dataset_path / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"Invalid Popper dataset {dataset_path}. "
            f"Missing: {', '.join(missing)}"
        )

    settings = Settings(
        kbpath=str(dataset_path),
        cmd_line=False,
        timeout=timeout,
        show_stats=show_stats,
        debug=debug,
    )

    # Popper 4 orders the literals of a rule with a table of "recalls"
    # computed from the background knowledge, but it first removes the
    # relations it judges pointless (for instance a predicate with a single
    # fact) from that table, while the generator can still produce them.
    # The lookup then raises KeyError (e.g. ('ring_subst_6', (1, 0)) on the
    # Alzheimer data). Unknown entries default to a large recall, which only
    # puts such literals last in the ordering.
    settings.recall = defaultdict(lambda: 10**9, settings.recall)

    _declare_missing_predicates_after_loading(dataset_path)

    program, score, stats = learn_solution(settings)

    if program is None:
        hypothesis = []
    else:
        hypothesis = [
            format_rule(rule)
            for rule in program
        ]

    return {
        "hypothesis": hypothesis,
        "score": list(score) if score is not None else None,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
    )

    args = parser.parse_args()

    # Popper and SWI-Prolog may print to stdout (progress, warnings, output
    # of the background knowledge...), including from C code that bypasses
    # sys.stdout. While training, file descriptor 1 is pointed at stderr so
    # that stdout carries nothing but the JSON result written below.
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    os.dup2(2, 1)

    try:
        result = train(
            dataset_path=Path(args.dataset),
            timeout=args.timeout,
        )
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)

    # IMPORTANT:
    # stdout must contain only machine-readable JSON.
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
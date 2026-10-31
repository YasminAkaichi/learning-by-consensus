"""Isolated Andante trainer for Consensus.

Runs in its own process so that the caller can enforce a HARD wall-clock
limit by killing it: Andante's internal time limit is only checked between
search states, and a single state can take a very long time on large
datasets (e.g. Alzheimer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from andante.program import AndanteProgram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--time-limit", type=int, default=None)
    args = parser.parse_args()

    # Andante logs on stdout; keep stdout for the JSON result only.
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    os.dup2(2, 1)

    try:
        program = AndanteProgram.build_from(args.dataset)
        learned = program.induce(
            update_knowledge=True,
            logging=True,
            verbose=1,
            time_limit=args.time_limit,
        )
        hypothesis = (
            [str(rule).strip() for rule in learned if str(rule).strip()]
            if learned
            else []
        )
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)

    print(json.dumps({"hypothesis": hypothesis}), flush=True)


if __name__ == "__main__":
    main()

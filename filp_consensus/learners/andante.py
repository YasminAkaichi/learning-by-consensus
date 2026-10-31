"""Andante adapter for the Consensus engine.

The learned hypothesis is converted to the canonical Consensus
representation: one string per rule.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any



class AndanteConsensusLearner:
    """Local Andante learner used by Learning by Consensus."""

    def __init__(
        self,
        dataset_path: str | Path,
        timeout: int | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        # Wall-clock limit (seconds) for learn(); when reached, Andante
        # returns the clauses it has learned so far (possibly none).
        self.timeout = timeout

        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"Andante dataset not found: {self.dataset_path}"
            )

        self.ilp: Any = None
        self.raw_hypothesis: Any = None

    def learn(self) -> tuple[list[str], None, None]:
        """Learn one local hypothesis with Andante."""

        if not self.timeout:
            # No limit requested: learn in this process, as before.
            from andante.program import AndanteProgram

            self.ilp = AndanteProgram.build_from(
                str(self.dataset_path)
            )

            self.raw_hypothesis = self.ilp.induce(
                update_knowledge=True,
                logging=True,
                verbose=1,
            )

            if not self.raw_hypothesis:
                return [], None, None

            hypothesis = [
                str(rule).strip()
                for rule in self.raw_hypothesis
                if str(rule).strip()
            ]

            return hypothesis, None, None

        # With a limit, Andante runs in a separate process. Its own limit
        # (`--time-limit`) is only checked between search states, so the
        # process is also killed after a short grace period: a client can
        # no longer run for ever on a large dataset. A client that is
        # killed returns an EMPTY hypothesis (it then votes "no").
        grace_seconds = 15

        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "filp_consensus.learners.andante_trainer",
                    "--dataset",
                    str(self.dataset_path),
                    "--time-limit",
                    str(int(self.timeout)),
                ],
                capture_output=True,
                text=True,
                timeout=int(self.timeout) + grace_seconds,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            print(
                f"[Andante] Timeout: stopped after {self.timeout}s on "
                f"{self.dataset_path.name}; empty hypothesis.",
                file=sys.stderr,
            )
            return [], None, None

        if process.returncode != 0:
            raise RuntimeError(
                f"Andante failed on {self.dataset_path} "
                f"(exit code {process.returncode}).\n"
                f"--- stderr (tail) ---\n{process.stderr[-3000:]}"
            )

        for line in reversed(process.stdout.strip().splitlines()):
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue
            return list(result.get("hypothesis", [])), None, None

        raise RuntimeError(
            "Andante returned no result for "
            f"{self.dataset_path}.\n"
            f"stdout (tail): {process.stdout[-1500:]!r}"
        )
    def predict_with_rules(
        self,
        hypothesis: list[str],
        test_dataset: str | Path,
    ) -> tuple[dict[str, bool], dict[str, list[int]]]:
        """Predictions plus, per example, the 1-based indexes of the rules
        that cover it (each rule is evaluated alone)."""

        predictions = self.predict(hypothesis, test_dataset)
        fired: dict[str, list[int]] = {example: [] for example in predictions}

        for index, rule in enumerate(hypothesis, start=1):
            for example, covered in self.predict([rule], test_dataset).items():
                if covered:
                    fired.setdefault(example, []).append(index)

        return predictions, fired

    def predict(
    self,
    hypothesis: list[str],
    test_dataset: str | Path,
) -> dict[str, bool]:
        """Evaluate a learned hypothesis on an Andante test dataset."""

        test_dataset = Path(test_dataset).resolve()

        if not test_dataset.is_file():
            raise FileNotFoundError(
                f"Andante test dataset not found: {test_dataset}"
            )

        # Load the common test dataset.
        from andante.program import AndanteProgram

        test_program = AndanteProgram.build_from(
            str(test_dataset)
        )

        # Add the learned hypothesis to the test background knowledge.
        #for rule in hypothesis:
        #    test_program.add_background_knowledge_from_text(
        #        rule
        #    )

        if hypothesis:
            hypothesis_text = "\n".join(hypothesis)

            background_text = (
                ":- begin_bg.\n"
                f"{hypothesis_text}\n"
                ":- end_bg."
            )

            test_program.add_background_knowledge_from_text(
                background_text
            )

        predictions: dict[str, bool] = {}

        # Evaluate the examples explicitly contained in the test set.
        for label in ("pos", "neg"):
            for example in test_program.examples[label]:

                # An example is represented as a bodyless Clause.
                target = example.head

                predicted_positive = test_program.solver.succeeds_on(
                    target,
                    test_program.knowledge,
                )

                example_id = (
                repr(target)
                .replace(", ", ",")
                )

                predictions[example_id] = bool(
                    predicted_positive
                )

        return predictions
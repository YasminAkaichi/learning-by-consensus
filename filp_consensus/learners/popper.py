"""Popper-v4 adapter for the Consensus engine.

Training and prediction are executed in isolated subprocesses so that
each client has an independent SWI-Prolog state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from pathlib import Path
from typing import Any



def _run_with_retry(command, env, attempts: int = 3):
    """subprocess.run, repeated when the child is killed by a signal.

    A negative return code means the child died by a signal (e.g. -13,
    SIGPIPE) before doing its job, which happens at random when a process
    holding gRPC threads starts a child. A real Popper error (positive exit
    code) is NOT retried: it is raised at once.
    """
    last_error = None
    for _ in range(attempts):
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as error:
            if error.returncode >= 0:
                raise
            last_error = error
    raise last_error


class PopperConsensusLearner:

    def __init__(
        self,
        dataset_path: str | Path,
        timeout: int = 600,
        debug: bool = False,
        show_stats: bool = False,
    ) -> None:

        self.dataset_path = Path(dataset_path).resolve()
        self.timeout = timeout
        self.debug = debug
        self.show_stats = show_stats

        self._validate_dataset()


    def _validate_dataset(self) -> None:

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: "
                f"{self.dataset_path}"
            )

        required_files = (
            "bias.pl",
            "bk.pl",
            "exs.pl",
        )

        missing = [
            filename
            for filename in required_files
            if not (
                self.dataset_path / filename
            ).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Invalid Popper dataset "
                f"{self.dataset_path}. "
                f"Missing: {', '.join(missing)}"
            )


    def learn(
        self,
    ) -> tuple[list[str], Any, Any]:

        from filp_consensus.paths import child_environment

        env = child_environment("popper")

        try:
            process = _run_with_retry(
                [
                    sys.executable,
                    "-m",
                    "filp_consensus.learners.popper_trainer",
                    "--dataset",
                    str(self.dataset_path),
                    "--timeout",
                    str(self.timeout),
                ],
                env,
            )
        except subprocess.CalledProcessError as error:
            # The real cause (Popper's traceback) is in the captured stderr,
            # which CalledProcessError does not show by itself.
            raise RuntimeError(
                f"Popper failed on {self.dataset_path} "
                f"(exit code {error.returncode}).\n"
                f"--- stderr (tail) ---\n{(error.stderr or '')[-3000:]}\n"
                f"--- stdout (tail) ---\n{(error.stdout or '')[-1000:]}"
            ) from error

        # Popper logs may be written to stderr.
        if process.stderr:
            print(
                process.stderr,
                file=sys.stderr,
                end="",
            )

        # The trainer prints its JSON result as the last line of stdout.
        # Anything printed before it (progress, Prolog output) is ignored.
        result = None

        for line in reversed(process.stdout.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

        if result is None:
            raise RuntimeError(
                "Popper did not return a result for "
                f"{self.dataset_path}.\n"
                f"stdout (tail): {process.stdout[-1500:]!r}\n"
                f"stderr (tail): {process.stderr[-1500:]!r}"
            )

        hypothesis = result.get(
            "hypothesis",
            [],
        )

        score = result.get(
            "score",
        )

        # Stats are not serialized yet.
        stats = None

        return hypothesis, score, stats


    def predict(
        self,
        hypothesis: list[str],
        test_dataset: str | Path,
    ) -> dict[str, bool]:
        return self.predict_with_rules(hypothesis, test_dataset)[0]

    def predict_with_rules(
        self,
        hypothesis: list[str],
        test_dataset: str | Path,
    ) -> tuple[dict[str, bool], dict[str, list[int]]]:
        """Predictions plus, per example, the 1-based indexes of the rules
        of `hypothesis` that cover it."""

        test_dataset = Path(
            test_dataset
        ).resolve()

        if not test_dataset.is_dir():
            raise FileNotFoundError(
                f"Test dataset not found: "
                f"{test_dataset}"
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as file:

            json.dump(
                hypothesis,
                file,
                ensure_ascii=False,
            )

            hypothesis_file = file.name

        try:

            from filp_consensus.paths import child_environment

            env = child_environment("popper")

            try:
                process = _run_with_retry(
                    [
                        sys.executable,
                        "-m",
                        "filp_consensus.learners.popper_predictor",
                        "--test-dataset",
                        str(test_dataset),
                        "--hypothesis-file",
                        hypothesis_file,
                    ],
                    env,
                )
            except subprocess.CalledProcessError as error:
                # Show the real cause (Prolog / Python error) instead of
                # just the exit status.
                raise RuntimeError(
                    f"Popper prediction failed on {test_dataset} "
                    f"(exit code {error.returncode}).\n"
                    f"--- stderr (tail) ---\n{(error.stderr or '')[-3000:]}\n"
                    f"--- stdout (tail) ---\n{(error.stdout or '')[-1000:]}"
                ) from error

            if process.stderr:
                print(
                    process.stderr,
                    file=sys.stderr,
                    end="",
                )

            # The JSON result is the last line of stdout; anything printed
            # before it is ignored.
            output = None
            for line in reversed(process.stdout.strip().splitlines()):
                try:
                    output = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

            if output is None:
                raise RuntimeError(
                    "Popper prediction returned no result.\n"
                    f"stdout (tail): {process.stdout[-1500:]!r}\n"
                    f"stderr (tail): {process.stderr[-1500:]!r}"
                )

            predictions = {
                example_id: bool(prediction)
                for example_id, prediction
                in output["predictions"].items()
            }
            fired = {
                example_id: [int(i) for i in indexes]
                for example_id, indexes
                in output["fired"].items()
            }

            return predictions, fired

        finally:

            Path(
                hypothesis_file
            ).unlink(
                missing_ok=True
            )
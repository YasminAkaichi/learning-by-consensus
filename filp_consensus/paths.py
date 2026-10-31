"""Locations used by the standalone Consensus package.

The (modified) Popper-v4 and Andante used in the paper are bundled in
``filp_consensus/learners/popper-v4`` and ``filp_consensus/learners/andante-ilp``.
To use other copies, set the environment variables below:

    FILP_POPPER_PATH   folder containing the ``popper`` package (Popper v4)
    FILP_ANDANTE_PATH  folder containing the ``andante`` package
"""

from __future__ import annotations

import os
from pathlib import Path

# .../consensus (the folder that contains `filp_consensus/`)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PACKAGE_ROOT / "datasets"
LEARNERS_DIR = Path(__file__).resolve().parent / "learners"


def popper_path() -> Path:
    return Path(os.environ.get("FILP_POPPER_PATH", LEARNERS_DIR / "popper-v4"))


def andante_path() -> Path:
    return Path(os.environ.get("FILP_ANDANTE_PATH", LEARNERS_DIR / "andante-ilp"))


def child_environment(learner: str | None = None) -> dict[str, str]:
    """Environment for the server / client / learner subprocesses."""
    env = os.environ.copy()
    paths = [str(PACKAGE_ROOT)]
    if learner in (None, "popper") and popper_path().is_dir():
        paths.insert(0, str(popper_path()))
    if learner in (None, "andante") and andante_path().is_dir():
        paths.append(str(andante_path()))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
    return env

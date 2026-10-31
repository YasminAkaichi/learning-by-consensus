"""End-to-end Learning by Consensus run, without any user interface.

    prepare()      read the dataset, split train/test, partition the training
                   examples among the clients, write everything to disk;
    run()          prepare + start the Flower server and the clients, wait,
                   and return the path of the result folder;
    centralized()  run the learner once on the whole training set of the SAME
                   split and score it on the same test set.

The split only depends on (dataset, seed), so a consensus run and a
centralized run with the same seed use exactly the same test set.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from filp_consensus.partitioning.consensus_writer import write_consensus_dataset
from filp_consensus.partitioning.dataset_reader import read_dataset
from filp_consensus.partitioning.partitioner import partition_dataset
from filp_consensus.partitioning.splitter import split_train_test
from filp_consensus.paths import DATASETS_DIR, PACKAGE_ROOT, child_environment

TEST_RATIO = 0.2


@dataclass
class Split:
    train_dirs: list[Path]   # one per client
    test_dir: Path
    root: Path
    partitions: list         # DatasetPartition objects (for the summary)


def resolve_dataset(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path.resolve()
    candidate = DATASETS_DIR / name_or_path
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Dataset '{name_or_path}' not found (looked in {DATASETS_DIR})."
    )


def run_name(dataset: Path, learner: str, clients: int, partition: str, seed: int) -> str:
    return f"{dataset.name}_{learner}_k{clients}_{partition}_seed{seed}"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _to_andante(path: Path) -> Path:
    from filp_consensus.partitioning.popper_to_andante import convert_popper_to_andante

    if path.is_file():
        return path
    return convert_popper_to_andante(path, path.parent / f"{path.name}.andante.pl")


def prepare(
    dataset: Path,
    clients: int,
    partition: str,
    seed: int,
    output_root: Path,
    learner: str = "popper",
) -> Split:
    data = read_dataset(dataset)
    train, test = split_train_test(data, test_ratio=TEST_RATIO, random_seed=seed)
    parts = partition_dataset(
        dataset=train, number_of_clients=clients, strategy=partition, random_seed=seed
    )
    root = write_consensus_dataset(
        dataset_name=dataset.name,
        train_partitions=parts,
        test_dataset=test,
        strategy=partition,
        random_seed=seed,
        test_ratio=TEST_RATIO,
        output_root=output_root,
    )
    train_dirs = [root / "train" / f"{dataset.name}_part{i}" for i in range(1, clients + 1)]
    test_dir = root / "test"
    if learner == "andante":
        train_dirs = [_to_andante(p) for p in train_dirs]
        test_dir = _to_andante(test_dir)
    return Split(train_dirs, test_dir, root, parts)


def describe(split: Split) -> str:
    lines = ["Client   positives  negatives"]
    for p in split.partitions:
        lines.append(
            f"{p.client_id:>6}   {len(p.positive_examples):>9}  {len(p.negative_examples):>9}"
        )
    return "\n".join(lines)


def run(
    dataset: str,
    learner: str = "popper",
    clients: int = 3,
    partition: str = "iid",
    seed: int = 42,
    timeout: int = 600,
    output: str | Path = "results",
    port: int | None = None,
    show_predictions: bool = False,
) -> Path:
    source = resolve_dataset(dataset)
    out = Path(output).resolve() / run_name(source, learner, clients, partition, seed)
    result_dir = out / "consensus"
    result_dir.mkdir(parents=True, exist_ok=True)

    split = prepare(source, clients, partition, seed, out / "data", learner)
    print(f"[consensus] dataset={source.name} learner={learner} K={clients} "
          f"partition={partition} seed={seed}")
    print(describe(split))

    env = child_environment(learner)
    address = f"127.0.0.1:{port or _free_port()}"
    processes: list[subprocess.Popen] = []
    try:
        server = subprocess.Popen(
            [sys.executable, "-m", "filp_consensus.server",
             "--num-clients", str(clients),
             "--server-address", address,
             "--output-dir", str(result_dir)],
            cwd=PACKAGE_ROOT, env=env,
        )
        processes.append(server)
        time.sleep(2)
        if server.poll() is not None:
            raise RuntimeError("The Consensus server stopped during start-up.")

        client_processes = []
        for i, train_dir in enumerate(split.train_dirs, start=1):
            proc = subprocess.Popen(
                [sys.executable, "-m", "filp_consensus.client",
                 "--client-id", str(i),
                 "--learner", learner,
                 "--dataset", str(train_dir),
                 "--test-dataset", str(split.test_dir),
                 "--server-address", address,
                 "--timeout", str(timeout)],
                cwd=PACKAGE_ROOT, env=env,
            )
            processes.append(proc)
            client_processes.append(proc)

        if server.wait() != 0:
            raise RuntimeError("The Consensus server failed (see messages above).")
        for i, proc in enumerate(client_processes, start=1):
            if proc.wait(timeout=60) != 0:
                raise RuntimeError(f"Client {i} failed (see messages above).")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()

    (out / "run_config.json").write_text(json.dumps({
        "dataset": source.name, "learner": learner, "clients": clients,
        "partition": partition, "seed": seed, "timeout": timeout,
        "test_ratio": TEST_RATIO,
    }, indent=2), encoding="utf-8")
    print_summary(result_dir, show_predictions)
    return out


def centralized(
    dataset: str,
    learner: str = "popper",
    seed: int = 42,
    timeout: int = 600,
    output: str | Path = "results",
    clients: int = 3,
    partition: str = "iid",
) -> Path:
    """Centralized baseline on the split of the matching consensus run."""
    source = resolve_dataset(dataset)
    out = Path(output).resolve() / run_name(source, learner, clients, partition, seed)
    result_dir = out / "centralized"
    result_dir.mkdir(parents=True, exist_ok=True)

    # One client holding the whole training set; same seed => same test set.
    split = prepare(source, 1, "iid", seed, out / "data_centralized", learner)
    subprocess.run(
        [sys.executable, "-m", "filp_consensus.centralized_baseline",
         "--learner", learner,
         "--dataset", str(split.train_dirs[0]),
         "--test-dataset", str(split.test_dir),
         "--output-dir", str(result_dir),
         "--timeout", str(timeout)],
        cwd=PACKAGE_ROOT, env=child_environment(learner), check=True,
    )
    print_summary(result_dir)
    return out


def print_summary(result_dir: Path, show_predictions: bool = False) -> None:
    from filp_consensus.report import print_report

    print_report(result_dir, show_predictions)

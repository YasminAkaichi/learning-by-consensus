"""Command line: python -m filp_consensus <command> [options]"""

from __future__ import annotations

import argparse
from pathlib import Path


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True,
                   help="dataset name (in datasets/) or path to a Popper folder "
                        "containing bias.pl, bk.pl and exs.pl")
    p.add_argument("--learner", choices=["popper", "andante"], default="popper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=600,
                   help="local learner timeout in seconds")
    p.add_argument("--output", default="results", help="results folder")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="filp_consensus",
        description="Learning by Consensus for federated ILP",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run Learning by Consensus (server + clients)")
    _common(run)
    run.add_argument("--clients", type=int, default=3)
    run.add_argument("--partition", choices=["iid", "non_iid"], default="iid")
    run.add_argument("--show-predictions", action="store_true",
                     help="print the vote of every client on every test example")
    run.add_argument("--port", type=int, default=None,
                     help="server port (default: a free one)")

    cen = sub.add_parser(
        "centralized",
        help="centralized baseline on the same split as the matching consensus run",
    )
    _common(cen)
    cen.add_argument("--clients", type=int, default=3,
                     help="K of the consensus run it is compared with (names the folder)")
    cen.add_argument("--partition", choices=["iid", "non_iid"], default="iid")

    prep = sub.add_parser(
        "prepare",
        help="only split and partition the dataset (no ILP system needed)",
    )
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--clients", type=int, default=3)
    prep.add_argument("--partition", choices=["iid", "non_iid"], default="iid")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--output", default="results")

    args = parser.parse_args(argv)

    from filp_consensus import pipeline

    if args.command == "run":
        pipeline.run(args.dataset, args.learner, args.clients, args.partition,
                     args.seed, args.timeout, args.output, args.port,
                     args.show_predictions)
    elif args.command == "centralized":
        pipeline.centralized(args.dataset, args.learner, args.seed, args.timeout,
                             args.output, args.clients, args.partition)
    else:
        source = pipeline.resolve_dataset(args.dataset)
        split = pipeline.prepare(source, args.clients, args.partition, args.seed,
                                 Path(args.output).resolve() / "data")
        print(pipeline.describe(split))
        print(f"written to {split.root}")


if __name__ == "__main__":
    main()

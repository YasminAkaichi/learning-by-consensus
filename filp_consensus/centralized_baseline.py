"""Centralized baseline for Learning by Consensus.

Runs the learner (Popper-v4 or Andante) ONCE on the whole training set and
scores the learned hypothesis on the common test set. No Flower server, no
client waiting: it is the plain centralized learner, evaluated exactly like a
consensus run (same split, same predictor, same metrics, same output files),
so the Analytics tab can compare the two directly.

python -m filp_consensus.centralized_baseline \
  --learner popper \
  --dataset datasets/generated/zendo1/consensus_iid_1_seed_42/train/zendo1_part1 \
  --test-dataset datasets/generated/zendo1/consensus_iid_1_seed_42/test \
  --output-dir artifacts/experiment_X/consensus
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from filp_consensus.analysis import analyse_vote_traces
from filp_consensus.client import ConsensusClient, encode_json


def parse_args():
    parser = argparse.ArgumentParser(description="Centralized baseline")
    parser.add_argument("--learner", required=True, choices=["popper", "andante"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--test-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    started = time.perf_counter()

    learner = ConsensusClient(
        client_id="1",
        learner_name=args.learner,
        dataset_path=args.dataset,
        test_dataset=args.test_dataset,
        timeout=args.timeout,
    )

    # 1. Learn on the whole training set.
    _, _, fit_metrics = learner.fit([], {})
    hypothesis = learner.local_hypothesis

    # 2. Score on the common test set (same code path as the consensus).
    ensemble = encode_json(
        {
            "type": "hypothesis_ensemble",
            "hypotheses": [
                {
                    "client_id": "1",
                    "learner": args.learner,
                    "hypothesis": hypothesis,
                }
            ],
        }
    )
    _, _, metrics = learner.evaluate(ensemble, {})
    total_seconds = time.perf_counter() - started

    # 3. Same files as a consensus run.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = ("tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1", "mcc")
    result = {
        "learner": args.learner,
        "number_of_clients": 1,
        "number_of_hypotheses": 1,
        "centralized": True,
        "hypotheses": [hypothesis],
        "metrics": {
            **{key: metrics[key] for key in keys},
            "n_hypotheses": 1,
        },
    }
    (output_dir / "server_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    traces = json.loads(metrics.get("vote_traces", "[]"))
    (output_dir / "vote_traces.json").write_text(
        json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    analysis = analyse_vote_traces(traces)
    analysis["communication"] = {
        "upload_bytes_per_client": {"1": 0},
        "ensemble_bytes": 0,
        "total_bytes": 0,
    }
    analysis["induction_time_per_client"] = {
        "1": float(fit_metrics.get("induction_time", 0.0))
    }
    analysis["federated_round_seconds"] = total_seconds
    try:
        analysis["local_coverage"] = {
            "1": json.loads(metrics.get("local_coverage", "[]"))
        }
    except json.JSONDecodeError:
        analysis["local_coverage"] = {}

    (output_dir / "consensus_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Centralized baseline written to {output_dir}")


if __name__ == "__main__":
    main()

"""Console report and CSV of a finished run: one line per client, plus the
prediction of every hypothesis on every test example."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _rule_stats(hypothesis: list[str]) -> tuple[int, float]:
    """(number of rules, mean number of body literals)."""
    lengths = []
    for rule in hypothesis:
        body = rule.split(":-", 1)[1] if ":-" in rule else ""
        lengths.append(body.count("),") + 1 if body.strip() else 0)
    return len(hypothesis), (sum(lengths) / len(lengths) if lengths else 0.0)


def write_predictions_csv(result_dir: Path) -> Path | None:
    traces = _read(result_dir / "vote_traces.json")
    if not traces:
        return None
    names = [v["hypothesis"] for v in traces[0]["votes"]]
    path = result_dir / "predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["example", "true_label", *names, "positive_votes",
                         "consensus", "correct"])
        for t in traces:
            votes = {v["hypothesis"]: v["prediction"] for v in t["votes"]}
            writer.writerow([t["example_id"], t["true_label"],
                             *[votes[n] for n in names], t["positive_votes"],
                             t["prediction"], t["correct"]])
    return path


def print_report(result_dir: Path, show_predictions: bool = False) -> None:
    result = _read(result_dir / "server_result.json")
    analysis = _read(result_dir / "consensus_analysis.json")
    traces = _read(result_dir / "vote_traces.json")
    if not result or not analysis:
        return

    hypotheses = result["hypotheses"]
    per_h = analysis.get("per_hypothesis", {})
    times = analysis.get("induction_time_per_client", {})
    comm = analysis.get("communication", {})
    uploads = comm.get("upload_bytes_per_client", {})
    # H1..Hk follow the clients sorted by id (as a string), like the server.
    client_ids = sorted(times, key=str) or [str(i) for i in range(1, len(hypotheses) + 1)]

    print(f"\n=== {result_dir.parent.name} ===")
    header = (f"{'Client':>6} {'rules':>5} {'lits':>5} {'time(s)':>8} {'up(KB)':>7} "
              f"{'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'MCC':>6}")
    print(header)
    print("-" * len(header))
    for index, cid in enumerate(client_ids, start=1):
        m = per_h.get(f"H{index}", {})
        rules, lits = _rule_stats(hypotheses[index - 1] if index <= len(hypotheses) else [])
        print(f"{cid:>6} {rules:>5} {lits:>5.1f} {float(times.get(cid, 0)):>8.2f} "
              f"{uploads.get(cid, 0) / 1024:>7.2f} "
              f"{m.get('accuracy', 0):>6.3f} {m.get('precision', 0):>6.3f} "
              f"{m.get('recall', 0):>6.3f} {m.get('f1', 0):>6.3f} {m.get('mcc', 0):>6.3f}")
    c = analysis["consensus"]
    print("-" * len(header))
    print(f"{'VOTE':>6} {'':>5} {'':>5} {'':>8} {'':>7} "
          f"{c['accuracy']:>6.3f} {c['precision']:>6.3f} {c['recall']:>6.3f} "
          f"{c['f1']:>6.3f} {c['mcc']:>6.3f}   "
          f"(tp={c['tp']} fp={c['fp']} tn={c['tn']} fn={c['fn']})")

    print("\nHypotheses:")
    for index, cid in enumerate(client_ids, start=1):
        rules = hypotheses[index - 1] if index <= len(hypotheses) else []
        print(f"  client {cid}: " + (" | ".join(rules) if rules else "(empty)"))

    csv_path = write_predictions_csv(result_dir)
    if csv_path:
        print(f"\nPredictions per example: {csv_path}")

    if show_predictions and traces:
        names = [v["hypothesis"] for v in traces[0]["votes"]]
        print(f"\n{'example':<14} {'true':>5} " + " ".join(f"{n:>4}" for n in names)
              + f" {'votes':>5} {'vote':>5}  ok")
        for t in traces:
            votes = {v["hypothesis"]: v["prediction"] for v in t["votes"]}
            print(f"{t['example_id']:<14} {t['true_label']:>5} "
                  + " ".join(f"{votes[n]:>4}" for n in names)
                  + f" {t['positive_votes']:>5} {t['prediction']:>5}  "
                  + ("✓" if t["correct"] else "✗"))

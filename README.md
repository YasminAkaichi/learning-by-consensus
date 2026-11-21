# Learning by Consensus

Command-line implementation of **Learning by Consensus** for federated ILP,
extracted from the FILP platform (no user interface). Each client learns a
local hypothesis $H_i$ with an ILP system (**Popper v4** or **Andante**),
sends it to the server, and the final prediction is the strict-majority vote
(`n//2 + 1`) of all hypotheses. Communication uses [Flower](https://flower.ai):
one fit round and one evaluate round.

> **Relation to the published paper.** FILP (Federated Inductive Logic
> Programming) was introduced in
> [*Federated Inductive Logic Programming for Explainable Artificial Intelligence*](https://ceur-ws.org/Vol-4059/paper1.pdf) (Akaichi, Jacquet, Linden and Vanhoof, EXPLIMED @ ECAI 2025). This repository goes beyond the
> published version: it adds Learning by Consensus with two local ILP systems
> (Popper v4 and Andante), IID and non-IID partitions, a centralized baseline
> evaluated on the same train/test split, and a per-client report (accuracy,
> MCC, hypothesis size, learning time, communication cost, vote of every
> client on every test example). Results obtained with this code may therefore
> differ from, or complement, those reported in the paper.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{akaichi2025filp,
  title     = {Federated Inductive Logic Programming for Explainable Artificial Intelligence},
  author    = {Akaichi, Yasmine and Jacquet, Jean-Marie and Linden, Isabelle and Vanhoof, Wim},
  booktitle = {EXPLIMED @ ECAI 2025},
  year      = {2025},
  url       = {https://ceur-ws.org/Vol-4059/paper1.pdf}
}
```

## Install

```bash
cd learning-by-consensus
pip install -r requirements.txt
```

The modified **Popper v4** and **Andante** used in this work are bundled in
`filp_consensus/learners/popper-v4` and `filp_consensus/learners/andante-ilp`
(see their `LICENSE` files); nothing else to clone. The original Popper is
available at [logic-and-learning-lab/Popper](https://github.com/logic-and-learning-lab/Popper);
the bundled copy contains small modifications needed by this code (for
example, it no longer depends on `pkg_resources`). Popper needs
[SWI-Prolog](https://www.swi-prolog.org) installed on the machine.
To use other copies, set `FILP_POPPER_PATH` / `FILP_ANDANTE_PATH`.

## Docker (easiest way to try it)

Nothing to install except [Docker](https://www.docker.com): SWI-Prolog, Python,
Flower, Popper and Andante are all inside the image. The default demo
(`docker run --rm learning-by-consensus`) has been tested.

```bash
docker build -t learning-by-consensus .

# default demo: Zendo1, Popper, 3 clients, IID
docker run --rm learning-by-consensus

# any other run: the arguments are those of `python -m filp_consensus`
docker run --rm -v "$PWD/results:/app/results" learning-by-consensus \
       run --dataset trains1000 --learner andante --clients 5 --partition non_iid --show-predictions

# your own dataset (a folder with bias.pl, bk.pl, exs.pl)
docker run --rm -v "$PWD/mydata:/data" -v "$PWD/results:/app/results" \
       learning-by-consensus run --dataset /data --clients 3
```

With `-v "$PWD/results:/app/results"` the results (JSON files, `predictions.csv`)
are kept on your machine.

## Quick start

```bash
python -m filp_consensus run --dataset zendo1 --learner popper --clients 3 --partition iid --seed 42 --show-predictions
```

Expected output (Zendo1, Popper, 3 clients, IID, seed 42):

```
=== zendo1_popper_k3_iid_seed42 ===
Client rules  lits  time(s)  up(KB)    acc   prec    rec     F1    MCC
----------------------------------------------------------------------
     1     1   3.0     0.46    0.26  1.000  1.000  1.000  1.000  1.000
     2     1   2.0     0.37    0.25  0.875  0.800  1.000  0.889  0.775
     3     1   5.0     1.05    0.28  1.000  1.000  1.000  1.000  1.000
----------------------------------------------------------------------
  VOTE                               1.000  1.000  1.000  1.000  1.000   (tp=4 fp=0 tn=4 fn=0)

Hypotheses:
  client 1: zendo(V0):- contact(V1,V2),red(V2),piece(V0,V1).
  client 2: zendo(V0):- piece(V0,V1),contact(V1,V2).
  client 3: zendo(V0):- size(V3,V1),contact(V2,V3),small(V1),red(V2),piece(V0,V2).

Predictions per example: results/zendo1_popper_k3_iid_seed42/consensus/predictions.csv

example         true   H1   H2   H3 votes  vote  ok
19               POS  POS  POS  POS     3   POS  ✓
5                POS  POS  POS  POS     3   POS  ✓
14               POS  POS  POS  POS     3   POS  ✓
4                POS  POS  POS  POS     3   POS  ✓
34               NEG  NEG  NEG  NEG     0   NEG  ✓
28               NEG  NEG  POS  NEG     1   NEG  ✓
23               NEG  NEG  NEG  NEG     0   NEG  ✓
31               NEG  NEG  NEG  NEG     0   NEG  ✓
```

Each client row gives the size of its local hypothesis (`rules`, `lits` = mean
body literals), its learning time, the size of the message it sends, and the
metrics of its hypothesis $H_i$ alone on the common test set. The `VOTE` row
is the consensus (strict majority). Here client 2 alone misclassifies one
negative example, and the vote corrects it.

**The dataset is a parameter**: replace `zendo1` with any folder of `datasets/`
(`trains`, `trains1000`, `iggp-rps`, `mutagenesis`, `carcinogenesis`,
`noisy-alzheimer_acetyl`, `intubation`) or with the path to your own Popper
dataset (`bias.pl`, `bk.pl`, `exs.pl`). The number of clients, the partition
(`iid` / `non_iid`), the learner (`popper` / `andante`) and the seed can be
changed the same way.

## Example: the vote under label skew (non-IID, 10 clients)

```bash
python -m filp_consensus run --dataset trains1000 --learner popper --clients 10 --partition non_iid
```

Output (Trains1000, Popper, 10 clients, non-IID, seed 42; the results are written
to `results/trains1000_popper_k10_non_iid_seed42/`):

```
=== trains1000_popper_k10_non_iid_seed42 ===
Client rules  lits  time(s)  up(KB)    acc   prec    rec     F1    MCC
----------------------------------------------------------------------
     1     1   3.0     0.38    0.26  0.885  1.000  0.709  0.830  0.772
    10     1   4.0     0.44    0.28  0.910  0.814  1.000  0.898  0.833
     2     1   5.0     0.64    0.29  1.000  1.000  1.000  1.000  1.000
     3     1   5.0     0.72    0.29  1.000  1.000  1.000  1.000  1.000
     4     1   5.0     0.73    0.29  1.000  1.000  1.000  1.000  1.000
     5     1   5.0     0.65    0.29  1.000  1.000  1.000  1.000  1.000
     6     1   5.0     0.76    0.29  1.000  1.000  1.000  1.000  1.000
     7     1   5.0     0.61    0.29  1.000  1.000  1.000  1.000  1.000
     8     1   5.0     0.58    0.29  1.000  1.000  1.000  1.000  1.000
     9     1   5.0     0.62    0.29  1.000  1.000  1.000  1.000  1.000
----------------------------------------------------------------------
  VOTE                               1.000  1.000  1.000  1.000  1.000   (tp=79 fp=0 tn=121 fn=0)

Hypotheses:
  client 1: f(V0):- roof_closed(V1),three_wheels(V1),has_car(V0,V1).
  client 10: f(V0):- has_car(V0,V1),roof_closed(V2),has_car(V0,V2),three_wheels(V1).
  client 2: f(V0):- long(V1),has_car(V0,V2),roof_closed(V1),three_wheels(V2),has_car(V0,V1).
  client 3: f(V0):- long(V1),three_wheels(V2),has_car(V0,V1),roof_closed(V1),has_car(V0,V2).
  ...                                    (clients 4 to 9: the same rule as client 3,
                                          with the literals in another order)

Predictions per example: results/trains1000_popper_k10_non_iid_seed42/consensus/predictions.csv
```

With `non_iid` the label skew is strong when there are many clients: client 1 gets
only 6 positive and 88 negative training examples, client 10 gets 57 positive and
9 negative ones. Their local hypotheses are wrong in opposite ways:

- **client 1** (few positives) learns a rule that is too *specific*: one car that
  is both closed-roof and three-wheeled, without `long`. It misses positives
  (recall 0.709).
- **client 10** (few negatives) learns a rule that is too *general*: no negative
  example forces the `long` literal, so it accepts false positives (precision 0.814).
- **clients 2 to 9** find the exact rule (a long closed-roof car and a
  three-wheeled car), written with the literals in different orders.

A strict majority needs 6 votes out of 10. Eight clients are right and the two
errors are of opposite kinds, so the vote cancels them: the consensus reaches
accuracy 1.000 although the worst client alone has 0.885. This is one run (one
seed) meant to illustrate the mechanism; on this noise-free synthetic dataset the
vote is not needed with IID partitions, where all clients find the same rule.

## Other commands

```bash
# Same data, non-IID partition, Andante
python -m filp_consensus run --dataset trains1000 --learner andante \
       --clients 3 --partition non_iid

# Centralized baseline on the SAME train/test split (same seed)
python -m filp_consensus centralized --dataset zendo1 --learner popper \
       --clients 3 --partition iid --seed 42

# Only split and partition the data (no ILP system needed)
python -m filp_consensus prepare --dataset trains1000 --clients 10 --partition non_iid
```

## What happens

1. The dataset is split once, stratified, 80 % train / 20 % test (depends only
   on the dataset and `--seed`, so consensus and centralized share the test set).
2. The training examples are partitioned among `--clients` clients:
   `iid` (stratified random) or `non_iid` (label-skewed).
3. A Flower server and one process per client are started on a free local port.
4. Each client learns $H_i$ (timeout `--timeout`), the server collects the
   hypotheses, broadcasts them, and every client votes on the common test set.

## Output

`results/<dataset>_<learner>_k<K>_<partition>_seed<S>/`

| Path | Content |
| --- | --- |
| `consensus/server_result.json` | hypotheses and consensus metrics (accuracy, precision, recall, F1, MCC) |
| `consensus/consensus_analysis.json` | each $H_i$ alone, vote margins, communication cost, learning time per client |
| `consensus/vote_traces.json` | per-example votes and the rules that fired |
| `centralized/` | the same three files for the centralized baseline |
| `consensus/predictions.csv` | for every test example: true label, the prediction of each $H_i$, number of positive votes, consensus prediction |
| `data/` | the generated client partitions and the common test set |

## Layout

```
filp_consensus/
  cli.py, pipeline.py      command line and orchestration
  server.py, client.py     Flower server and client
  strategy/                consensus strategy (collection, voting, metrics)
  learners/                Popper v4 and Andante adapters
  partitioning/            split, IID / non-IID partitioning, dataset writers
  centralized_baseline.py  centralized run on the same split
```

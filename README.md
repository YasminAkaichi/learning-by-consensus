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

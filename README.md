# Platonic

Minimal, standalone reproduction code for the paper's Multilingual Random
Hierarchy Model (MRHM) experiments.

The repository can:

- generate code-switched MRHM data;
- train the decoder-only transformers used in the paper;
- measure Information Imbalance (II) and mutual kNN (mKNN);
- run shared-latent probes and layerwise novelty analysis on checkpoints; and
- reproduce the settled causal belief-propagation (BP) baselines.

Pretrained-LLM feature extraction is intentionally out of scope. The `arrays`
command can apply II and mKNN to already-extracted, aligned representations.

## Install

Python 3.10 or newer is required.

```bash
git clone https://github.com/d-doshi/platonic.git
cd platonic
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Five-minute check

Run the complete tiny CPU experiment:

```bash
python -m platonic smoke
```

This generates data, trains a small transformer, reloads its checkpoint, runs
all checkpoint analyses, and evaluates both BP metrics. Outputs are written to
`runs/smoke/`. This checks the software; it is not a paper-scale experiment.

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Reproduce the paper sweep

All paper settings live in [`configs/paper.json`](configs/paper.json). Run the
three shared depths and four random seeds with:

```bash
python -m platonic reproduce \
  --config configs/paper.json \
  --output runs/paper
```

This launches 12 large training jobs sequentially and normally requires a
CUDA GPU. The exact sweep is:

| item | value |
|---|---:|
| training examples (`P`) | `2^20 = 1,048,576` |
| validation examples | `32,768` |
| random seeds | `42, 420, 4200, 42000` |
| shared depths | `1, 2, 3` |
| code-switch probability | `0.1` |
| hierarchy | `L=4, s=2, m=4, v=16, classes=16` |
| transformer | 8 layers, width 512, 4 heads, ReLU |
| attention | causal softmax with learned absolute positions |
| parameterization | bias-free maps and norms, tied token/output embedding |
| batch size / epochs | 1024 / 128 |
| optimizer | AdamW, lr `1e-3`, weight decay `1e-2`, betas `(0.9, 0.999)` |
| schedule | 2% warmup, stable phase, 10% final cosine decay |
| selected checkpoint | minimum last-token validation loss |

To train a single configuration, remove the `sweep` field from a copy of the
configuration and choose `random_seed` and `data.shared_depth`:

```bash
python -m platonic train --config one_run.json --output runs/one_run
```

## Analyze a checkpoint

```bash
python -m platonic analyze \
  --checkpoint runs/one_run/checkpoint.pt \
  --config one_run.json \
  --output runs/one_run/transformer
```

The default paper analysis uses independent translated-pair splits:

- 4,096 fit pairs with seed `1,000,045`;
- 256 validation pairs with seed `2,000,046`;
- 256 measurement pairs with seed `3,000,048`;
- `k=10` for mKNN; and
- ridge penalties `10^-8, 10^-7.5, ..., 10^8` for novelty selection.

It writes:

```text
metrics.csv             raw, novelty, and direct-subtraction II/mKNN
probes.csv              within- and cross-language latent-probe accuracy
novelty_selection.csv   validation-selected ridge penalties
manifest.json           checkpoint, representation stages, and analysis seeds
```

Each training run also contains `run_manifest.json`, which records the fully
resolved configuration, derived random seeds, selected training step, software
versions, and Git commit.

## Belief-propagation baselines

```bash
python -m platonic bp \
  --config one_run.json \
  --output runs/one_run/bp
```

The settled encoder-decoder trajectory is
`u1,u2,u3,d3,d2,d1,d0`, with no root-posterior block. The II output contains
the complete encoding/algorithm/retention sweep. The mKNN output contains the
paper's raw-probability, efficient-retention curve and its input stage `u0`.
Sentence representations are means over autoregressive prefix positions.

## Analyze external representation arrays

```bash
python -m platonic arrays \
  --language-a language_a.npy \
  --language-b language_b.npy \
  --k 10 \
  --output runs/language/layer_12.json
```

Rows in the two arrays must identify the same objects in the same order.

## Code map

```text
configs/                  exact paper and CPU-smoke configurations
src/platonic/data.py      MRHM rules, training data, and translation pairs
src/platonic/model.py     decoder-only transformer
src/platonic/training.py  WSD training and portable checkpoints
src/platonic/features.py  residual-stream feature extraction
src/platonic/analysis.py  metric, novelty, and probe orchestration
src/platonic/metrics/     II, mKNN, and layerwise novelty definitions
src/platonic/bp/          settled causal BP baseline
tests/                    deterministic and end-to-end checks
```

No external research repository, pretrained checkpoint, or private data is
required.

## Citation

Citation details will be added when the manuscript is public. The paper will
be listed as the preferred citation for this repository. Until then, a
specific software version can be identified by the repository URL and Git
commit hash.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).

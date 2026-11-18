# Reproducing these numbers

Everything needed to re-run this project is in this directory: the code, the
dataset, the per-seed predictions and the run logs. Nothing lives in a temp
folder and nothing is fetched at run time.

## Environment

`requirements-lock.txt` is the exact set of versions the reported numbers were
produced with. `requirements.txt` keeps the loose ranges for a fresh install.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt   # Windows
# .venv/bin/python -m pip install -r requirements-lock.txt     # Linux/macOS
```

Arms 1 and 2 are CPU-only and were run on Windows 11, Python 3.13.9, torch
2.12.0+cpu. Arm 3 (PNA) needs a GPU: either install the CUDA build of the same
torch version, or run `notebooks/colab_run.ipynb` on a Colab T4 and let Colab
supply torch.

Two version-sensitive traps, both already handled in the code. They are worth
knowing because each one fails loudly or silently in a different way:

| Trap | Symptom | Where it is handled |
|---|---|---|
| torch >= 2.6 defaults `torch.load(weights_only=True)`; ogb 1.3.6 never passes the argument | `UnpicklingError: Weights only load failed ... DataEdgeAttr` — the dataset cannot be loaded at all | `src/data.py` allowlists the three PyG classes involved. No-op on torch < 2.6 |
| scikit-learn's `HistGradientBoostingClassifier` defaults to `early_stopping="auto"`, which above 10k rows holds out a *random* 10% of training rows | silent: a scaffold benchmark selects its iteration count on a random split that shares scaffolds with its training rows | `src/run_wegl.py` disables it and selects on the official scaffold validation split |

## The dataset

`dataset/ogbg_molhiv/` is OGB's `ogbg-molhiv`, downloaded once from
`http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip`
(NCI AIDS Antiviral Screen data, routed through MoleculeNet, repackaged by OGB).

```
41,127 molecules
32,901 train / 4,113 valid / 4,113 test   (official scaffold split, shipped with the data)
 1,443 positives (3.509%)
 2 / 25.5 / 222 atoms per molecule (min/mean/max), max node degree 9
```

The split is read with `dataset.get_idx_split()`. No splitting logic in this
repo touches it.

What is in there, and what you can throw away:

| Subfolder | Size | Keep? |
|---|---|---|
| `raw/`, `split/`, `mapping/` | ~2 MB | **yes** — the actual data, the official split and the SMILES. Keep these and the project never needs the network again |
| `processed/` | ~160 MB | optional cache. Delete it before zipping; ogb rebuilds it from `raw/` in about 10 seconds |

That second row is tested, not assumed: with all outbound sockets blocked, a
copy containing only `raw/`, `split/` and `mapping/` reloads to 41,127 graphs,
the same 32,901/4,113/4,113 split and the same 1,443 positives.

`mapping/mol.csv.gz` carries the SMILES and has columns `HIV_active`, `smiles`,
`mol_id`. `HIV_active` is element-wise identical to `dataset.y` across all
41,127 rows, which is what lets `src/baseline_fp.py` and
`src/error_analysis.py` read labels and structures from the CSV while
`src/train.py` reads them from the graph objects.

## Running it

From this directory:

```bash
python -m src.baseline_fp --seeds 0 1 2                  # arm 1, CPU, minutes
python -m src.run_wegl    --seeds 0 1 2 --classifier hgb # arm 2, CPU
python -m src.train       --seeds 0 1 2 --epochs 50      # arm 3, GPU, ~25 min/seed
python -m src.report                                     # comparison table
python -m src.error_analysis --preds results/test_preds_seed0.npz
```

`run_all.sh` runs the same sequence. Arms 1 and 2 are CPU and arm 3 is GPU, so
starting arm 3 first and running the others alongside it costs nothing.

## What lands in `results/`

| File | From |
|---|---|
| `fp_rf_preds_seed*.npz` | arm 1 — test labels and probabilities |
| `wegl_preds_seed*.npz` | arm 2 — test labels and probabilities |
| `test_preds_seed*.npz` | arm 3 — test labels, logits and probabilities |
| `pna_seed*.pt` | arm 3 checkpoints (state dict, degree histogram, args), also what `app/main.py` serves |
| `calibration_seed*.json` | arm 3 Brier score and reliability curve |
| `wegl_cache_s*_p*_r*_seed*.pkl` | arm 2 embeddings, cached per hyperparameter set *and* per seed |
| `logs/` | stdout of each run |

Every seed is fixed, so re-running reproduces the same numbers. `src/report.py`
rebuilds the comparison table from the `.npz` files alone — you do not need to
retrain to regenerate the table.

The embedding cache filename carries the diffusion steps, PCA dimension,
reference size and seed. This matters: an earlier version cached to one fixed
path, so changing `--pca-dim` silently reloaded the old embedding and reported
a sweep that never ran.

## Checking that the comparison is still honest

`python -m src.report` prints the test-set size and positive count for every
arm and refuses to present the table as comparable if they differ. That guard
exists because of a real failure: RDKit cannot parse 7 of molhiv's SMILES
(3 train, 2 valid, 2 test, all inactive), and arm 1 originally dropped them
from training *and* test, scoring itself on 4,111 molecules while the other two
arms used 4,113. Unparseable molecules are now dropped from training only; the
2 test molecules are kept with an all-zero fingerprint.

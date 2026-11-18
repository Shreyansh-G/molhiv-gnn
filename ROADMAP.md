# Roadmap

The question this project answers:

> On scaffold-split molhiv, does learned message passing actually beat
> non-learned graph embeddings and plain substructure counting - by more than
> the run-to-run variation?

Three arms, one split, same evaluator. Any outcome is a result: if the GNN wins
clearly, that is a finding; if the confidence intervals overlap, that is a more
interesting one.

| Arm | Method | Learning | Script |
|---|---|---|---|
| 1 | Morgan fingerprints + Random Forest | none (chemistry prior) | `src/baseline_fp.py` |
| 2 | WEGL: diffusion + optimal-transport embedding + Random Forest | none (structure only) | `src/run_wegl.py` |
| 3 | PNA graph neural network | trained end to end | `src/train.py` |

## Day plan

| Hours | Step | Command | Check before moving on |
|---|---|---|---|
| 0.0-0.5 | Environment + data | `notebooks/colab_run.ipynb` cells 1-2 | `ogbg-molhiv` loads (41127 graphs, 32901/4113/4113), `torch.cuda.is_available()` is True |
| 0.5-1.0 | Sanity baseline from the OGB repo | `python main_pyg.py --gnn gin-virtual --epochs 5` | loss drops, valid ROC-AUC > 0.6 |
| 1.0-1.5 | Arm 1 | `python -m src.baseline_fp --seeds 0 1 2` | ~0.7-0.8 ROC-AUC, runs on CPU |
| 1.5-2.0 | Arm 2 | `python -m src.run_wegl --seeds 0 1 2` | ~2 min embedding + ~90s forest per seed. Add `--classifier hgb` only if you can afford hours: boosting costs ~12s/iteration on the 960-dim embedding |
| 3.0-6.0 | Arm 3 | `python -m src.train --seeds 0 1 2 --epochs 50` | valid ROC-AUC climbing past 0.75 |
| 6.0-6.5 | Imbalance experiment | `python -m src.train --seeds 0 --pos-weight 10 --out results_pw` | does weighting help or hurt? |
| 6.5-7.0 | Comparison table | `python -m src.report` | table fills in |
| 7.0-7.5 | Error analysis | `python -m src.error_analysis` | two PNG grids + CSV |
| 7.5-8.5 | Serving | `uvicorn app.main:app` then the curl in the README | `/predict` returns probability + spread |
| 8.5-9.0 | Write-up | fill the README table, 5 lines of discussion | someone else could rerun it |

Start the arm-3 training early and do arms 1-2 while it runs; they use different
hardware (CPU vs GPU).

## What to record while running

For every arm: mean +/- std ROC-AUC over 3 seeds, a bootstrap 95% CI on the test
molecules, and wall-clock training time. The timing matters: if WEGL lands within
touching distance of PNA at a fraction of the cost, that is the headline.

## If something breaks

| Symptom | Fix |
|---|---|
| `UnpicklingError: Weights only load failed ... DataEdgeAttr` when loading the dataset | already handled: `src/data.py` allowlists the three PyG classes before ogb's `torch.load`. torch 2.6 flipped `weights_only` to `True` and ogb 1.3.6 never passes it, so an unpatched checkout cannot load molhiv at all |
| `torch_geometric` import errors on Colab | install `torch_geometric` only, after torch; the compiled extras are not needed |
| CUDA out of memory | `--batch-size 64` |
| WEGL step is slow | `--reference-size 20 --pca-dim 16`; embeddings cache per hyperparameter set and per seed |
| Changed `--pca-dim` but the numbers did not move | fixed: the cache filename now carries the hyperparameters, so a stale embedding cannot be reused silently |
| `ot` not found | `pip install POT` |
| BatchNorm error on a 1-node graph | guarded in `run_epoch`, training only. molhiv's smallest molecule has 2 atoms so it never fires; the guard must not run in eval or it would drop a test molecule |

## Stretch, only if time remains

- 10 seeds instead of 3, matching how the leaderboard reports.
- Calibration plot from the JSON that `train.py` already writes.
- A Chemprop arm (`pip install chemprop`) as an industry-standard reference.

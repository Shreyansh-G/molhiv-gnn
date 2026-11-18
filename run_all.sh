#!/usr/bin/env bash
# Full run. On a Colab T4 the PNA seeds dominate (~25 min each).
set -e
echo "=== arm 1: substructure counting ==="
python -m src.baseline_fp --seeds 0 1 2

echo "=== arm 2: graph structure, no learning ==="
# Random Forest by default: the 960-dim WEGL embedding makes histogram boosting
# cost ~12s per iteration (hours per seed), while the forest fits in ~90s. Add
# --classifier hgb to run the boosted ablation.
python -m src.run_wegl --seeds 0 1 2 --classifier rf

echo "=== arm 3: learned message passing ==="
python -m src.train --seeds 0 1 2 --epochs 50

echo "=== comparison + error analysis ==="
python -m src.report
python -m src.error_analysis --preds results/test_preds_seed0.npz

"""Collect every results/*.npz into the comparison table for the README.

Also checks the thing the whole comparison rests on: that every arm was scored
on the same test molecules. If one arm quietly dropped a few (an unparseable
SMILES, a skipped batch), the table says so instead of comparing numbers that
are not comparable.
"""
import glob
import os
import re

import numpy as np

from src.metrics import bootstrap_ci, rocauc

MODELS = {
    "Morgan FP + RF": "fp_rf_preds_seed*.npz",
    "WEGL + classifier": "wegl_preds_seed*.npz",
    "PNA (trained GNN)": "test_preds_seed*.npz",
}


def _seed_of(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main(results="results"):
    rows, shapes = [], {}
    for name, pattern in MODELS.items():
        files = sorted(glob.glob(os.path.join(results, pattern)), key=_seed_of)
        if not files:
            continue
        aucs, seeds, first = [], [], None
        for f in files:
            d = np.load(f)
            # ROC-AUC is rank-based, so a logit and its sigmoid give the same
            # score; use whichever the arm saved.
            pred = d["logit"] if "logit" in d else d["prob"]
            y = d["y"]
            aucs.append(rocauc(y, pred))
            seeds.append(_seed_of(f))
            if first is None:
                first = (y, pred)
        shapes[name] = (int(np.asarray(first[0]).size), int(np.asarray(first[0]).sum()))
        # The CI is a property of one set of predictions, not of a mean over
        # seeds, so it is reported for a single named seed.
        lo, hi = bootstrap_ci(*first)
        rows.append((name, seeds, np.mean(aucs), np.std(aucs), lo, hi, aucs))

    if not rows:
        print(f"no results found in {results}/ - run the arms first")
        return

    print(f"{'model':<22}{'seeds':>6}{'ROC-AUC':>10}{'std':>8}"
          f"   95% CI (seed {rows[0][1][0]})      per-seed")
    for name, seeds, mean, std, lo, hi, aucs in rows:
        per = " ".join(f"{a:.4f}" for a in aucs)
        print(f"{name:<22}{len(seeds):>6}{mean:>10.4f}{std:>8.4f}"
              f"   [{lo:.4f}, {hi:.4f}]   {per}")

    print("\ntest set actually scored (must be identical across arms):")
    for name, (n, pos) in shapes.items():
        print(f"  {name:<22} n={n:<6} positives={pos}")
    if len(set(shapes.values())) > 1:
        print("  !! MISMATCH - these arms were not scored on the same molecules,")
        print("     so the ROC-AUC values above are not directly comparable.")
    else:
        print("  all arms scored on the same test molecules.")


if __name__ == "__main__":
    main()

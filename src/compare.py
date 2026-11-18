"""Paired comparison between arms: is the gap bigger than the noise?

Why this file exists, and why `src/report.py` is not enough.

`report.py` gives each arm a bootstrap CI over the test molecules. Those CIs are
wide (~0.09 on molhiv) because the test set has only 4,113 molecules and ~144
actives, and they overlap heavily between arms. It is tempting to read the
overlap as "the arms are indistinguishable". That inference is wrong.

Those are *marginal* CIs. Both arms are scored on exactly the same molecules, so
most of what makes an arm's CI wide - an unlucky draw of hard molecules - moves
both arms in the same direction at once. The question "is A better than B" is
about the *difference*, and the difference has its own, much narrower,
distribution. The right procedure is a paired bootstrap: resample molecules
once, score both arms on that same resample, and record the difference. The
shared draw noise cancels.

So two different statements come out of this, and they are both true:
  - each arm's absolute score is uncertain to about +/- 0.04 (report.py)
  - the gap between two arms can still be resolved far more tightly (here)

Also reported: the fraction of resamples in which A beats B. That is a direct,
assumption-light answer to "would this ordering survive a different test set?",
and it is not a p-value - no null hypothesis is being tested.

    python -m src.compare
    python -m src.compare --n-boot 2000
"""
import argparse
import glob
import os
import re

import numpy as np

from src.metrics import rocauc
from src.report import MODELS, _seed_of


def load_arm(results, pattern):
    """Returns {seed: (y, pred)} for one arm, or {} if it has not been run."""
    out = {}
    for f in sorted(glob.glob(os.path.join(results, pattern)), key=_seed_of):
        d = np.load(f)
        pred = d["logit"] if "logit" in d else d["prob"]
        out[_seed_of(f)] = (np.asarray(d["y"]).ravel(), np.asarray(pred).ravel())
    return out


def paired_bootstrap(y, pred_a, pred_b, n_boot=1000, alpha=0.05, seed=0):
    """Bootstrap the DIFFERENCE in ROC-AUC, resampling molecules once per draw.

    Both arms are scored on the identical resample, which is the whole point:
    the "which molecules did we happen to draw" noise is common to both and
    cancels out of the difference.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue  # a one-class resample has no ROC-AUC
        diffs.append(rocauc(y[idx], pred_a[idx]) - rocauc(y[idx], pred_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    arms = {name: load_arm(args.results, pat) for name, pat in MODELS.items()}
    arms = {k: v for k, v in arms.items() if v}
    if len(arms) < 2:
        print(f"need at least two arms in {args.results}/; found {list(arms)}")
        return

    names = list(arms)
    print("arms found:", ", ".join(f"{n} ({len(arms[n])} seeds)" for n in names))

    # Every arm must have been scored on the same molecules or none of this is
    # meaningful. report.py checks sizes; here we check the labels themselves.
    ref_name = names[0]
    ref_y = arms[ref_name][sorted(arms[ref_name])[0]][0]
    for n in names:
        for s, (y, _) in arms[n].items():
            if y.shape != ref_y.shape or not np.array_equal(y, ref_y):
                print(f"\n!! {n} seed {s} has different test labels from "
                      f"{ref_name}. Arms are not comparable; stopping.")
                return
    print(f"all arms scored on the same {len(ref_y)} molecules "
          f"({int(ref_y.sum())} active)\n")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            sa = {s: rocauc(*arms[a][s]) for s in arms[a]}
            sb = {s: rocauc(*arms[b][s]) for s in arms[b]}
            print(f"=== {a}  vs  {b} ===")
            print(f"  {a:<22} {np.mean(list(sa.values())):.4f} "
                  f"+/- {np.std(list(sa.values())):.4f}")
            print(f"  {b:<22} {np.mean(list(sb.values())):.4f} "
                  f"+/- {np.std(list(sb.values())):.4f}")
            print(f"  difference of means    "
                  f"{np.mean(list(sa.values())) - np.mean(list(sb.values())):+.4f}")

            shared = sorted(set(arms[a]) & set(arms[b]))
            print(f"  paired bootstrap on the difference, per seed "
                  f"({args.n_boot} resamples):")
            crossings = 0
            for s in shared:
                y, pa = arms[a][s]
                _, pb = arms[b][s]
                diffs, lo, hi = paired_bootstrap(y, pa, pb, args.n_boot, seed=s)
                frac = float((diffs > 0).mean())
                excl = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"
                crossings += (lo <= 0 <= hi)
                print(f"    seed {s}: {sa[s] - sb[s]:+.4f}  "
                      f"95% CI [{lo:+.4f}, {hi:+.4f}]  {excl}   "
                      f"{a} ahead in {frac * 100:.1f}% of resamples")
            if crossings == 0:
                print(f"  -> the gap is resolved in every seed: {a} is ahead by "
                      f"more than test-set sampling noise.")
            elif crossings == len(shared):
                print(f"  -> no seed resolves the gap: on this test set these two "
                      f"arms are statistically indistinguishable.")
            else:
                print(f"  -> mixed: {crossings} of {len(shared)} seeds leave 0 "
                      f"inside the interval. Treat the gap as unresolved.")
            print()


if __name__ == "__main__":
    main()

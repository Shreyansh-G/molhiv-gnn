"""Generate the figures used in the README, from the saved predictions.

    python -m src.figures                 # writes results/figures/*.png

Everything here is derived from the .npz files in results/, so the figures
regenerate without retraining anything. One number is not: the WEGL + Random
Forest variant, whose per-molecule predictions were overwritten before they were
saved. It is carried below as a clearly-marked constant read off that run's log,
and it is drawn as an open marker with no interval so it never looks like it has
the same backing as the rest.

Colours are the validated categorical slots 1-3, assigned per *arm* rather than
per row, so the two WEGL variants share a hue - they are the same method with a
different classifier bolted on, and the figure should say so.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

from src.metrics import rocauc

# --- palette (validated categorical slots, light surface) -------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ARM_COLOR = {"fp": "#2a78d6", "wegl": "#eb6834", "pna": "#1baf7a"}

# Point estimate for WEGL + Random Forest, read from that run's log. Its
# predictions were not retained, so it gets no interval anywhere.
WEGL_RF_FROM_LOG = {"mean": 0.7673, "std": 0.0057}

ARMS = [
    ("fp", "Morgan FP + Random Forest", "fp_rf_preds_seed*.npz"),
    ("wegl", "WEGL + boosted trees", "wegl_preds_seed*.npz"),
    ("pna", "PNA (trained GNN)", "test_preds_seed*.npz"),
]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "legend.frameon": False,
        "figure.dpi": 160,
    })


def load(results):
    """{arm: {seed: (y, prob)}} from the saved predictions."""
    import glob, re
    out = {}
    for key, _label, pattern in ARMS:
        per_seed = {}
        for f in sorted(glob.glob(os.path.join(results, pattern))):
            seed = int(re.search(r"seed(\d+)", os.path.basename(f)).group(1))
            d = np.load(f)
            prob = d["prob"] if "prob" in d else d["logit"]
            per_seed[seed] = (np.asarray(d["y"]).ravel(),
                              np.asarray(prob).ravel().astype(float))
        if per_seed:
            out[key] = per_seed
    return out


def fast_auc_ok(y, p):
    """Confirm sklearn matches the official OGB evaluator before using it in
    tight bootstrap loops - they agree for a single binary task with no missing
    labels, and the loops run thousands of times."""
    return abs(rocauc(y, p) - roc_auc_score(y, p)) < 1e-12


def boot_ci(y, p, n_boot, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def paired_diffs(y, pa, pb, n_boot, seed=0):
    """Resample molecules ONCE per draw and score both arms on that same
    resample, so the noise the two arms share cancels out of the difference."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    return np.asarray(d)


# --------------------------------------------------------------------------
def fig_arms(data, out, n_boot):
    """The headline: point estimates with BOTH uncertainties, to scale."""
    rows = []
    for key, label, _ in ARMS:
        aucs = [roc_auc_score(*data[key][s]) for s in sorted(data[key])]
        y0, p0 = data[key][0]
        lo, hi = boot_ci(y0, p0, n_boot)
        rows.append((key, label, np.mean(aucs), np.std(aucs), lo, hi, aucs))
    rows.sort(key=lambda r: r[2], reverse=True)

    # insert the log-only WEGL+RF point in its ranked place
    extra = ("wegl", "WEGL + Random Forest", WEGL_RF_FROM_LOG["mean"],
             WEGL_RF_FROM_LOG["std"], None, None, None)
    allrows = sorted(rows + [extra], key=lambda r: r[2], reverse=True)

    from matplotlib.transforms import blended_transform_factory
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    # labels are pinned to the left edge in axes coords, rows in data coords,
    # and offset in points so nothing collides with the title or the ticks
    ys = np.arange(len(allrows))[::-1] * 1.0
    for y, (key, label, mean, std, lo, hi, aucs) in zip(ys, allrows):
        c = ARM_COLOR[key]
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=c, lw=2, alpha=0.30,
                    solid_capstyle="round", zorder=2)
            ax.plot([mean - std, mean + std], [y, y], color=c, lw=6,
                    solid_capstyle="round", zorder=3)
            ax.scatter(aucs, [y] * len(aucs), s=26, facecolor=SURFACE,
                       edgecolor=c, linewidth=1.2, zorder=4)
            ax.scatter([mean], [y], s=90, color=c, zorder=5,
                       edgecolor=SURFACE, linewidth=1.5)
        else:
            ax.scatter([mean], [y], s=90, facecolor=SURFACE, edgecolor=c,
                       linewidth=2, zorder=5)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.annotate(label, (0.010, y), xycoords=trans, textcoords="offset points",
                    xytext=(0, 10), fontsize=10.5, color=INK, va="bottom")
        note = "   (point estimate from the run log)" if lo is None else ""
        ax.annotate(f"{mean:.4f} ± {std:.4f}{note}", (0.010, y), xycoords=trans,
                    textcoords="offset points", xytext=(0, -12), fontsize=9,
                    color=INK_2, va="top")

    ax.set_yticks([])
    ax.set_ylim(-0.62, len(allrows) - 1 + 0.72)
    ax.set_xlim(0.695, 0.858)
    ax.set_xlabel("Test ROC-AUC  (higher is better)")
    ax.set_title("Three ways to represent a molecule, and two kinds of uncertainty",
                 loc="left", pad=26)
    ax.grid(axis="y", visible=False)

    # legend explaining the two interval types
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=MUTED, lw=6, label="± 1 s.d. across 3 seeds"),
        Line2D([], [], color=MUTED, lw=2, alpha=0.35,
               label="95% bootstrap CI over test molecules (seed 0)"),
        Line2D([], [], marker="o", color=SURFACE, markeredgecolor=MUTED,
               lw=0, label="individual seeds"),
    ], loc="lower right", fontsize=8.5)

    fig.text(0.013, 0.025,
             "The thin bars are ~20x the thick ones: which molecules landed in the "
             "test set matters far more than which seed was used.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out, "fig1_arms.png"))
    plt.close(fig)
    return allrows


def fig_paired(data, out, n_boot):
    """Does any gap survive a paired test? Three panels, zero marked."""
    pairs = [("fp", "pna", "Morgan FP  −  PNA"),
             ("fp", "wegl", "Morgan FP  −  WEGL"),
             ("wegl", "pna", "WEGL  −  PNA")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, (a, b, title) in zip(axes, pairs):
        resolved = 0
        for s in sorted(data[a]):
            y, pa = data[a][s]
            _, pb = data[b][s]
            d = paired_diffs(y, pa, pb, n_boot, seed=s)
            lo, hi = np.percentile(d, [2.5, 97.5])
            resolved += (lo > 0 or hi < 0)
            if s == 0:
                ax.hist(d, bins=44, color=ARM_COLOR[a], alpha=0.55,
                        edgecolor=SURFACE, linewidth=0.5)
                ax.axvspan(lo, hi, color=ARM_COLOR[a], alpha=0.10, zorder=0)
                ahead = float((d > 0).mean())
        ax.axvline(0, color=INK, lw=1.4, zorder=5)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("difference in ROC-AUC")
        ax.text(0.03, 0.94,
                f"seed 0: ahead in {ahead*100:.0f}% of resamples\n"
                f"{resolved} of 3 seeds exclude zero",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
        ax.set_yticks([])
    axes[0].set_ylabel("bootstrap resamples")
    fig.suptitle("Resample the molecules, score both arms on the same resample, "
                 "take the difference", x=0.012, ha="left", fontsize=12)
    fig.text(0.012, 0.015,
             "Zero sits inside almost every interval: on 4,113 molecules with 130 "
             "actives, these gaps are not resolvable.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(os.path.join(out, "fig2_paired.png"))
    plt.close(fig)


def fig_screening(data, out):
    """What ROC-AUC hides at a 3.16% hit rate: precision-recall and enrichment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    y0 = data["fp"][0][0]
    base = y0.mean()

    for key, label, _ in ARMS:
        y, p = data[key][0]
        c = ARM_COLOR[key]
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax1.plot(rec, prec, color=c, lw=2, label=f"{label}  (AP {ap:.3f})")

        order = np.argsort(-p)
        yy = y[order]
        k = np.arange(1, len(yy) + 1)
        ef = (np.cumsum(yy) / k) / base
        frac = k / len(yy) * 100
        m = frac <= 25
        ax2.plot(frac[m], ef[m], color=c, lw=2, label=label)

    ax1.axhline(base, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax1.text(0.52, base + 0.012, f"random guessing ({base*100:.2f}% active)",
             color=INK_2, fontsize=8.5)
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_ylim(0, 0.75)
    ax1.set_title("Precision-recall (seed 0)", loc="left")
    ax1.legend(fontsize=8.5, loc="upper right")

    ax2.axhline(1, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax2.text(12, 1.12, "no better than random", color=INK_2, fontsize=8.5)
    ax2.set_xlabel("% of the library screened, best-ranked first")
    ax2.set_ylabel("enrichment factor")
    ax2.set_title("Enrichment: actives found per unit of screening effort",
                  loc="left")
    ax2.legend(fontsize=8.5, loc="upper right")

    fig.text(0.012, 0.015,
             "ROC-AUC near 0.78 looks healthy; at a 3.16% hit rate the "
             "precision-recall view shows how little of the ranking is usable.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out, "fig3_screening.png"))
    plt.close(fig)


def fig_agreement(data, out):
    """Do the arms find the SAME actives? Rank-vs-rank, actives highlighted."""
    from scipy.stats import spearmanr
    pairs = [("fp", "pna"), ("fp", "wegl"), ("wegl", "pna")]
    names = {k: l for k, l, _ in ARMS}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    for ax, (a, b) in zip(axes, pairs):
        y, pa = data[a][0]
        _, pb = data[b][0]
        ra = (-pa).argsort().argsort() / len(pa) * 100
        rb = (-pb).argsort().argsort() / len(pb) * 100
        inact = y == 0
        ax.scatter(ra[inact], rb[inact], s=3, color=AXIS, alpha=0.35,
                   linewidths=0, zorder=1)
        ax.scatter(ra[~inact], rb[~inact], s=18, color="#d03b3b", alpha=0.85,
                   linewidths=0.5, edgecolor=SURFACE, zorder=3)
        rho = spearmanr(pa, pb).statistic
        top = int(np.intersect1d(np.argsort(-pa)[:100],
                                 np.argsort(-pb)[:100]).size)
        ax.set_title(f"{names[a]}\nvs {names[b]}", loc="left", fontsize=9.5)
        ax.text(0.04, 0.95, f"Spearman ρ = {rho:.2f}\ntop-100 shared: {top}/100",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
        ax.set_xlabel("percentile rank"); ax.set_ylabel("percentile rank")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    from matplotlib.lines import Line2D
    axes[-1].legend(handles=[
        Line2D([], [], marker="o", lw=0, color="#d03b3b", label="active (130)"),
        Line2D([], [], marker="o", lw=0, color=AXIS, label="inactive"),
    ], fontsize=8.5, loc="lower right")
    fig.suptitle("The arms score the same and rank differently",
                 x=0.012, ha="left", fontsize=12)
    fig.text(0.012, 0.015,
             "If two arms agreed, actives would lie on the diagonal. They scatter: "
             "the arms are finding partly different molecules.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.91))
    fig.savefig(os.path.join(out, "fig4_agreement.png"))
    plt.close(fig)


def fig_metric_choice(data, out, n_boot):
    """The same models under two metrics, with the paired verdict under each.

    ROC-AUC cannot separate WEGL from PNA. Average precision - the metric that
    suits a 3.16% positive rate - separates them in every seed. Same
    predictions, same molecules, same test; opposite conclusions.
    """
    metrics = [("ROC-AUC", roc_auc_score, "the benchmark's official metric"),
               ("Average precision", average_precision_score,
                "appropriate at a 3.16% hit rate")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, (mname, fn, sub) in zip(axes, metrics):
        rows = []
        for key, label, _ in ARMS:
            vals = [fn(*data[key][s]) for s in sorted(data[key])]
            rows.append((key, label, np.mean(vals), np.std(vals), vals))
        rows.sort(key=lambda r: r[2], reverse=True)
        ys = np.arange(len(rows))[::-1]
        for yy, (key, label, mean, std, vals) in zip(ys, rows):
            c = ARM_COLOR[key]
            ax.plot([mean - std, mean + std], [yy, yy], color=c, lw=6,
                    solid_capstyle="round", zorder=3)
            ax.scatter(vals, [yy] * len(vals), s=24, facecolor=SURFACE,
                       edgecolor=c, linewidth=1.2, zorder=4)
            ax.scatter([mean], [yy], s=85, color=c, zorder=5,
                       edgecolor=SURFACE, linewidth=1.5)
            ax.annotate(f"{label}  {mean:.4f}", (mean, yy),
                        textcoords="offset points", xytext=(0, 13),
                        ha="center", fontsize=9, color=INK)
        # paired verdict for WEGL vs PNA under this metric
        resolved = 0
        for s in sorted(set(data["wegl"]) & set(data["pna"])):
            y, pa = data["wegl"][s]
            _, pb = data["pna"][s]
            rng = np.random.default_rng(s)
            d = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(y), len(y))
                if len(np.unique(y[idx])) < 2:
                    continue
                d.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
            lo, hi = np.percentile(d, [2.5, 97.5])
            resolved += (lo > 0 or hi < 0)
        ax.set_yticks([])
        ax.set_ylim(-0.9, len(rows) - 0.2)
        # leave room so an outlying seed marker is not clipped by the spine
        ax.margins(x=0.10)
        ax.set_title(f"{mname}\n{sub}", loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
        verdict = f"WEGL vs PNA: {resolved} of 3 seeds resolve the gap"
        colour = "#0ca30c" if resolved == 3 else "#d03b3b"
        ax.text(0.5, 0.03, verdict, transform=ax.transAxes, ha="center",
                fontsize=9.5, color=colour, fontweight="bold")

    fig.suptitle("Same predictions, same molecules, same test — opposite conclusions",
                 x=0.012, ha="left", fontsize=12.5)
    fig.text(0.012, 0.015,
             "At a 3.16% hit rate ROC-AUC is dominated by the easy negatives, and "
             "hides a difference that average precision resolves in every seed.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(os.path.join(out, "fig6_metric_choice.png"))
    plt.close(fig)


def fig_calibration(data, out, n_bins=8):
    """Ranking well and being honest about confidence are different skills.

    A reliability curve: group molecules by what the model claimed, and plot
    that against what actually happened. On the diagonal means "when it says
    20%, 20% of them really are active".
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    base = data["fp"][0][0].mean()
    for key, label, _ in ARMS:
        y, p = data[key][0]
        frac, mean_p = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        br = np.mean([brier_score_loss(*data[key][s]) for s in sorted(data[key])])
        ax1.plot(mean_p, frac, color=ARM_COLOR[key], lw=2, marker="o", ms=7,
                 label=f"{label}  (Brier {br:.4f})")
    lim = 0.62
    ax1.plot([0, lim], [0, lim], color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax1.text(lim * 0.62, lim * 0.70, "perfectly honest", color=INK_2,
             fontsize=8.5, rotation=34)
    ax1.set_xlim(-0.02, lim); ax1.set_ylim(-0.02, lim)
    ax1.set_xlabel("what the model claimed")
    ax1.set_ylabel("what actually happened")
    ax1.set_title("Reliability (seed 0)", loc="left")
    ax1.legend(fontsize=8.5, loc="upper left")

    # Brier against the do-nothing baselines
    names, vals, cols = [], [], []
    for key, label, _ in ARMS:
        names.append(label.replace(" + ", "\n+ "))
        vals.append(np.mean([brier_score_loss(*data[key][s])
                             for s in sorted(data[key])]))
        cols.append(ARM_COLOR[key])
    const = float(np.mean((data["fp"][0][0] - base) ** 2))
    names.append("always guess\nthe base rate")
    vals.append(const)
    cols.append(AXIS)
    xs = np.arange(len(names))
    ax2.bar(xs, vals, color=cols, width=0.62, edgecolor=SURFACE, linewidth=2)
    for x, v in zip(xs, vals):
        ax2.annotate(f"{v:.4f}", (x, v), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=9, color=INK_2)
    ax2.axhline(const, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_ylabel("Brier score  (lower is better)")
    ax2.set_title("Anything above the dashed line is worse\nthan not trying",
                  loc="left")
    ax2.grid(axis="x", visible=False)

    fig.text(0.012, 0.015,
             "WEGL ranks molecules well and still scores worse than guessing the "
             "base rate: a good ranking and an honest probability are not the "
             "same achievement.", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(out, "fig7_calibration.png"))
    plt.close(fig)


def fig_posweight(data, out, results):
    """Validation went up, test went down."""
    pw = os.path.join(os.path.dirname(results.rstrip("/\\")), "results_posweight",
                      "test_preds_seed0.npz")
    if not os.path.exists(pw):
        print("  (skipping pos-weight figure: results_posweight not found)")
        return
    d = np.load(pw)
    y_pw, p_pw = np.asarray(d["y"]).ravel(), np.asarray(d["prob"]).ravel()
    base_auc = roc_auc_score(*data["pna"][0])
    pw_auc = roc_auc_score(y_pw, p_pw)
    # validation figures come from the training logs, not the saved predictions
    valid = {"unweighted": 0.8154, "weighted": 0.8308}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [0, 1]
    ax.plot(xs, [valid["unweighted"], valid["weighted"]], color="#6250d6",
            lw=2, marker="o", ms=9, label="validation")
    ax.plot(xs, [base_auc, pw_auc], color=ARM_COLOR["pna"], lw=2, marker="o",
            ms=9, label="test")
    for x, v in zip(xs, [valid["unweighted"], valid["weighted"]]):
        ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, color=INK_2)
    for x, v in zip(xs, [base_auc, pw_auc]):
        ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points",
                    xytext=(0, -17), ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(xs); ax.set_xticklabels(["unweighted loss", "--pos-weight 10"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Weighting the positive class: validation up, test down",
                 loc="left", pad=14)
    ax.legend(fontsize=9, loc="center right")
    fig.text(0.013, 0.02,
             "Selecting on validation, as one normally does, would have shipped "
             "the worse model.\nScaffold validation and test are different "
             "chemistry, so a gain on one need not transfer.",
             color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(out, "fig5_posweight.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out = args.out or os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)
    style()

    data = load(args.results)
    missing = [k for k, _, _ in ARMS if k not in data]
    if missing:
        print(f"missing predictions for {missing}; run those arms first")
        return
    y0, p0 = data["fp"][0]
    print("sklearn matches the OGB evaluator:", fast_auc_ok(y0, p0))
    print(f"{len(y0)} test molecules, {int(y0.sum())} active")

    rows = fig_arms(data, out, args.n_boot);   print("  fig1_arms.png")
    fig_paired(data, out, args.n_boot);        print("  fig2_paired.png")
    fig_screening(data, out);                  print("  fig3_screening.png")
    fig_agreement(data, out);                  print("  fig4_agreement.png")
    fig_posweight(data, out, args.results);    print("  fig5_posweight.png")
    fig_metric_choice(data, out, args.n_boot); print("  fig6_metric_choice.png")
    fig_calibration(data, out);                print("  fig7_calibration.png")
    print(f"\nwrote figures to {out}")


if __name__ == "__main__":
    main()

"""Scoring helpers: official OGB ROC-AUC plus the uncertainty reporting."""
import numpy as np
from ogb.graphproppred import Evaluator
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

_evaluator = Evaluator(name="ogbg-molhiv")


def rocauc(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    return _evaluator.eval({"y_true": y_true, "y_pred": y_pred})["rocauc"]


def bootstrap_ci(y_true, y_pred, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for ROC-AUC over the test molecules.

    Resamples molecules (not seeds): it answers "how much does this score move
    if the test set had been a slightly different draw of molecules?".
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # a resample with one class has no ROC-AUC
        scores.append(rocauc(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def calibration(y_true, y_prob, n_bins=10):
    """Does a predicted 0.8 behave like 0.8? Brier score + reliability curve."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }

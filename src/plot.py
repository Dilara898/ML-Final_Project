"""
plots.py — Person D figures: confusion matrices, ROC, error analysis,
feature importance and SVM weights. Every function saves <name>.pdf and
<name>.png into report/figures/ and closes the figure.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")                      # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np

from . import evaluate

FIG_DIR = "report/figures"


def _save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def confusion_matrix_fig(cm, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["standard", "premium"])
    ax.set_yticks([0, 1], ["standard", "premium"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    _save(fig, name)


def roc_fig(y_true, score_by_model: dict, name: str = "roc_curves") -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for label, s in score_by_model.items():
        fpr, tpr = evaluate.roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{label} (AUC={evaluate.roc_auc(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — price tier (test)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, name)


def pred_vs_actual_fig(y_true, y_pred, title: str, name: str,
                       xlabel="True price (AZN)", ylabel="Predicted price (AZN)") -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lim, lim, "r--", lw=0.8, label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, name)


def residuals_fig(y_pred_log, resid_log, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.scatter(y_pred_log, resid_log, s=4, alpha=0.3)
    ax.axhline(0, color="r", lw=0.8)
    ax.set_xlabel("Predicted log(price)")
    ax.set_ylabel("Residual (log scale)")
    ax.set_title(title, fontsize=9)
    _save(fig, name)


def error_by_price_decile(y_true, y_pred, n_bins: int = 10):
    """
    Group listings by TRUE-price quantile and report MAE and median absolute
    percentage error per group (price scale). Shows where the model fails
    (luxury tail). Returns (labels, mae_per_bin, mape_per_bin, counts).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    edges = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, y_true, side="right") - 1, 0, n_bins - 1)
    maes, mapes, counts = [], [], []
    for b in range(n_bins):
        m = bins == b
        counts.append(int(m.sum()))
        maes.append(float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else np.nan)
        mapes.append(float(np.median(np.abs(y_true[m] - y_pred[m]) / y_true[m]) * 100)
                     if m.any() else np.nan)
    labels = [f"D{b + 1}" for b in range(n_bins)]
    return labels, np.array(maes), np.array(mapes), np.array(counts)


def error_by_decile_fig(y_true, preds_by_model: dict, name: str = "error_by_price_decile") -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for label, p in preds_by_model.items():
        labels, _, mape, _ = error_by_price_decile(y_true, p)
        ax.plot(labels, mape, marker="o", ms=3, label=label)
    ax.set_xlabel("True-price decile (D10 = most expensive)")
    ax.set_ylabel("Median abs. % error")
    ax.set_title("Where the models fail (test)", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, name)


def importance_fig(names, values, title: str, name: str, top: int = 15,
                   signed: bool = False) -> None:
    """Horizontal bar plot of the `top` largest |values| (importances or SVM weights)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(3.8, 0.28 * len(order) + 0.9))
    colors = (["tab:red" if v < 0 else "tab:blue" for v in values[order]]
              if signed else "tab:blue")
    ax.barh([str(names[i]) for i in order], values[order], color=colors)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    if signed:
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("weight (standardized features)")
    else:
        ax.set_xlabel("impurity-decrease importance")
    _save(fig, name)

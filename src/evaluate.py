"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH, NumPy only).

scikit-learn is used ONLY in tests/test_evaluate.py to sanity-check these
numbers. Nothing in this file imports sklearn.
"""

from __future__ import annotations

import time

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    """sqrt(mean((y_true - y_pred)^2))"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    """mean(|y_true - y_pred|)"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """1 - SS_res / SS_tot (1.0 if y_true is constant and perfectly predicted)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def regression_report(y_true, y_pred) -> dict:
    """RMSE / MAE / R2 in one dict."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


# --- Classification (Task B: price tier) -----------------------------------
def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]] (labels 0/1, positive = 1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class.
    A 0/0 case returns 0.0 (same convention as sklearn's zero_division=0)."""
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def _average_ranks(x):
    """1-based ranks of x; tied values get the average of their ranks."""
    x = np.asarray(x, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)               # last rank of each tie group
    starts = ends - counts + 1             # first rank of each tie group
    return ((starts + ends) / 2.0)[inv]


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (SVM margins or tree probabilities).
    Rank-based (Mann-Whitney U) computation; ties get average ranks.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes present in y_true")
    ranks = _average_ranks(scores)
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(y_true, scores):
    """Return (fpr, tpr) arrays for plotting, thresholds descending."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = scores[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    # keep only the last index of each distinct score (handles ties)
    distinct = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tpr = np.r_[0.0, tps[distinct] / tps[-1]]
    fpr = np.r_[0.0, fps[distinct] / fps[-1]]
    return fpr, tpr


def classification_report(y_true, y_pred, scores=None) -> dict:
    """accuracy / precision / recall / f1 (+ auc if scores are given)."""
    p, r, f = precision_recall_f1(y_true, y_pred)
    out = {"accuracy": accuracy(y_true, y_pred),
           "precision": p, "recall": r, "f1": f}
    if scores is not None:
        out["auc"] = roc_auc(y_true, scores)
    return out


# --- Cross-validation helpers (own folds, shared by ours AND sklearn) -------
def kfold_indices(n, k=5, seed=42, y=None):
    """
    Deterministic k-fold. Returns a list of (train_idx, val_idx).
    If y is given, folds are stratified by class.
    The SAME folds must be reused for our models and the sklearn baselines.
    """
    rng = np.random.default_rng(seed)
    if y is None:
        idx = rng.permutation(n)
        folds = [np.sort(f) for f in np.array_split(idx, k)]
    else:
        y = np.asarray(y)
        folds = [[] for _ in range(k)]
        for c in np.unique(y):
            members = rng.permutation(np.where(y == c)[0])
            for i, part in enumerate(np.array_split(members, k)):
                folds[i].extend(part.tolist())
        folds = [np.array(sorted(f)) for f in folds]
    all_idx = np.arange(n)
    return [(np.setdiff1d(all_idx, f), np.asarray(f)) for f in folds]


def timed(fn, *args, **kwargs):
    """Run fn and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def mean_std(values) -> tuple[float, float]:
    """(mean, sample std) of a list of per-fold scores."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


# --- Comparison helper ------------------------------------------------------
def _fmt(v) -> str:
    if isinstance(v, (tuple, list)) and len(v) == 2:       # (mean, std)
        return f"{v[0]:.4f} ± {v[1]:.4f}"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}"
    return str(v)


def compare_table(name_to_metrics: dict) -> str:
    """Build the comparison table as a string (columns = union of metrics)."""
    cols: list[str] = []
    for m in name_to_metrics.values():
        for k in m:
            if k not in cols:
                cols.append(k)
    rows = [[name] + [_fmt(m[c]) if c in m else "-" for c in cols]
            for name, m in name_to_metrics.items()]
    header = ["model"] + cols
    widths = [max(len(str(r[i])) for r in [header] + rows)
              for i in range(len(header))]
    line = "  ".join("-" * w for w in widths)
    out = ["  ".join(str(h).ljust(w) for h, w in zip(header, widths)), line]
    out += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def compare(name_to_metrics: dict, title: str | None = None) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    name_to_metrics = {"Ours: tree": {"rmse": 0.41, "f1": (0.88, 0.01)}, ...}
    A (mean, std) tuple is printed as 'mean ± std'.
    """
    if title:
        print(f"\n=== {title} ===")
    print(compare_table(name_to_metrics))


def to_latex_rows(name_to_metrics: dict, cols: list[str], digits: int = 3) -> str:
    """LaTeX rows for report.tex (paste between \\midrule and \\bottomrule)."""
    def f(v):
        if isinstance(v, (tuple, list)):
            return f"{v[0]:.{digits}f} $\\pm$ {v[1]:.{digits}f}"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{digits}f}"
        return str(v)
    lines = []
    for name, m in name_to_metrics.items():
        cells = [f(m[c]) if c in m else "n/a" for c in cols]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)

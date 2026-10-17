from __future__ import annotations

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x).ravel()


def _check_same_length(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true).astype(float), _arr(y_pred).astype(float)
    _check_same_length(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true).astype(float), _arr(y_pred).astype(float)
    _check_same_length(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _arr(y_true).astype(float), _arr(y_pred).astype(float)
    _check_same_length(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    y_true, y_pred = _arr(y_true).astype(int), _arr(y_pred).astype(int)
    _check_same_length(y_true, y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation with average ranks for ties.
    """
    y_true = _arr(y_true).astype(int)
    scores = _arr(scores).astype(float)
    _check_same_length(y_true, scores)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc needs both classes present in y_true")

    # Average ranks (1-based), ties get the mean of the ranks they span.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg_rank_per_group = cum - (counts - 1) / 2.0
    ranks = avg_rank_per_group[inverse]

    sum_pos_ranks = ranks[y_true == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.

    name_to_metrics: {"My Ridge": {"RMSE": 1.2, "MAE": 0.9, "R2": 0.8}, ...}
    """
    if not name_to_metrics:
        print("(nothing to compare)")
        return

    cols = []
    for m in name_to_metrics.values():
        for k in m:
            if k not in cols:
                cols.append(k)

    name_w = max(len("Model"), max(len(str(n)) for n in name_to_metrics))
    col_w = max(10, max(len(c) for c in cols))

    header = f"{'Model':<{name_w}} | " + " | ".join(f"{c:>{col_w}}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in name_to_metrics.items():
        cells = []
        for c in cols:
            v = m.get(c)
            cells.append(f"{'—':>{col_w}}" if v is None else f"{float(v):>{col_w}.4f}")
        print(f"{str(name):<{name_w}} | " + " | ".join(cells))
"""Evaluation metrics written from scratch (Problem 6).
Each function is unit-tested against scikit-learn in tests/test_metrics.py.
Labels for classification are 0/1 (1 = "premium")."""
import numpy as np


# ---------- regression (Task A) ----------
def rmse(y_true, y_pred):
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true, y_pred):
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.mean(np.abs(d)))


def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


# ---------- classification (Task B) ----------
def confusion_matrix(y_true, y_pred):
    """[[TN, FP], [FN, TP]] for labels 0/1."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    (_, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return float(p), float(r), float(f1)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true).astype(int) == np.asarray(y_pred).astype(int)))


def roc_auc(y_true, scores):
    """ROC-AUC via the trapezoidal rule over the ROC curve."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    thresholds = np.unique(scores)[::-1]
    tpr, fpr = [0.0], [0.0]
    for t in thresholds:
        pred = scores >= t
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)

    return float(np.trapezoid(tpr, fpr))


def pr_auc(y_true, scores):
    """Average precision (same result as sklearn.average_precision_score)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")

    thresholds = np.unique(scores)[::-1]   

    ap = 0.0
    prev_recall = 0.0
    for t in thresholds:
        pred = scores >= t                    
        tp = np.sum(pred & (y_true == 1))       
        fp = np.sum(pred & (y_true == 0))      

        precision = tp / (tp + fp)
        recall = tp / n_pos

        ap += (recall - prev_recall) * precision   
        prev_recall = recall

    return float(ap)

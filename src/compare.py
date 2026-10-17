"""Benchmark helpers (Problem 5/6): CV mean±std, timing, results table."""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from . import metrics as M


class ScaledWrapper:
    """Standardises with TRAIN statistics only, then delegates to the model."""
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        self.model.fit((X - self.mu_) / self.sd_, y)
        return self

    def __getattr__(self, name):
        if name in ("predict", "predict_proba", "decision_function"):
            f = getattr(self.model, name)  # AttributeError -> hasattr() is False
            return lambda X: f((X - self.mu_) / self.sd_)
        if name in ("coef_", "w_", "w", "feature_importances_"):
            return getattr(self.model, name)
        raise AttributeError(name)


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.decision_function(X))


def metric_dict(task, model, X, y):
    """Task A: y is log(price) -> metrics on log and on price. Task B: y in {0,1}."""
    pred = np.asarray(model.predict(X))
    if task == "reg":
        out = {"RMSE_log": M.rmse(y, pred), "MAE_log": M.mae(y, pred), "R2_log": M.r2(y, pred),
               "RMSE_price": M.rmse(np.exp(y), np.exp(pred)),
               "MAE_price": M.mae(np.exp(y), np.exp(pred)),
               "R2_price": M.r2(np.exp(y), np.exp(pred))}
    else:
        p, r, f1 = M.precision_recall_f1(y, pred)
        s = _scores(model, X)
        out = {"Acc": M.accuracy(y, pred), "Precision": p, "Recall": r, "F1": f1,
               "ROC_AUC": M.roc_auc(y, s), "PR_AUC": M.pr_auc(y, s)}
    return out


def cv_summary(factory, task, X, y, k=5, seed=42):
    """Cross-validated mean±std on train+val (never touches the test set)."""
    split = (StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y) if task == "cls"
             else KFold(k, shuffle=True, random_state=seed).split(X))
    rows = []
    for tr, va in split:
        m = factory().fit(X[tr], y[tr])
        rows.append(metric_dict(task, m, X[va], y[va]))
    df = pd.DataFrame(rows)
    return df.mean().to_dict(), df.std(ddof=1).to_dict()


def evaluate_model(name, kind, factory, task, d, k=5, seed=42):
    """CV on train+val; final fit on train+val; test metrics computed ONCE."""
    Xtv = np.vstack([d["X_train"], d["X_val"]])
    ytv = np.r_[d[f"y{task}_train"], d[f"y{task}_val"]]
    cv_mean, cv_std = cv_summary(factory, task, Xtv, ytv, k, seed)
    m = factory()
    t0 = time.perf_counter(); m.fit(Xtv, ytv); fit_t = time.perf_counter() - t0
    t0 = time.perf_counter(); m.predict(d["X_test"]); pred_t = time.perf_counter() - t0
    test = metric_dict(task, m, d["X_test"], d[f"y{task}_test"])
    row = {"model": name, "kind": kind, "task": task, "fit_s": fit_t, "predict_s": pred_t}
    for key in test:
        row[f"test_{key}"] = test[key]
        row[f"cv_{key}"] = f"{cv_mean[key]:.4f} ± {cv_std[key]:.4f}"
    return row, m


def results_table(rows):
    return pd.DataFrame(rows)


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)

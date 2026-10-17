"""
run_all.py — ONE command reproduces every headline number and figure.

    python -m src.run_all                 # full run (tables + figures + CV)
    python -m src.run_all --no-cv         # skip cross-validation (fast)
    python -m src.run_all --check-seeds   # run twice, assert identical metrics

Stages (each is its own function so each can be its own commit):
  1. stage_data       load -> clean -> features -> split -> tier label -> scale
  2. stage_own        our DecisionTree (A, B) and our PegasosSVM (B) on TRAIN
  3. stage_baselines  scikit-learn baselines in a Pipeline, same split
  4. stage_cv         k-fold CV (mean ± std) on TRAIN+VAL, same folds for all
  5. stage_figures    confusion matrices, ROC, error analysis, importances, weights
  6. stage_report     comparison tables (console + LaTeX rows) + results.json

The test set is touched ONLY for the final numbers. Hyperparameters in CONFIG
must come from validation evidence (Person B / Person C sweeps) — never test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import data_prep, evaluate, plot as plots
from .decision_tree import DecisionTree
from .svm import PegasosSVM

SEED = data_prep.SEED
N_FOLDS = 5
SVC_MAX_ROWS = 12_000          # sklearn SVC(RBF) is O(n^2)+: skip above this
RESULTS_DIR = "results"

# ---- Final configurations (paste the VALIDATION winners here) ---------------
CONFIG = {
    "tree_reg": dict(criterion="mse", max_depth=10, min_samples_split=10,
                     min_samples_leaf=5, min_impurity_decrease=0.0),
    "tree_clf": dict(criterion="gini", max_depth=8, min_samples_split=10,
                     min_samples_leaf=5, min_impurity_decrease=0.0),
    "svm": dict(lambda_=3e-3, n_iters=500_000),
}


# =============================================================================
# 1. DATA
# =============================================================================
def stage_data() -> dict:
    """Load, clean, featurize, split, derive tier (TRAIN median), standardize."""
    np.random.seed(SEED)
    p = data_prep.prepare_data(seed=SEED)   # split first, preprocessing fitted on TRAIN only

    X_tr = np.asarray(p["X_train"], dtype=float)
    X_val = np.asarray(p["X_val"], dtype=float)
    X_te = np.asarray(p["X_test"], dtype=float)
    Xs_tr = np.asarray(p["X_train_scaled"], dtype=float)
    Xs_val = np.asarray(p["X_val_scaled"], dtype=float)
    Xs_te = np.asarray(p["X_test_scaled"], dtype=float)
    thr = p["tier_threshold"]

    d = dict(
        names=list(p["feature_names"]), threshold=float(thr),
        X_tr=X_tr, X_val=X_val, X_te=X_te,
        Xs_tr=Xs_tr, Xs_val=Xs_val, Xs_te=Xs_te,
        price_tr=np.asarray(p["y_train"], dtype=float),
        price_val=np.asarray(p["y_val"], dtype=float),
        price_te=np.asarray(p["y_test"], dtype=float),
        tier_tr=np.asarray(p["tier_train"]).astype(int),
        tier_val=np.asarray(p["tier_val"]).astype(int),
        tier_te=np.asarray(p["tier_test"]).astype(int),
    )
    d["info"] = dict(
        rows=int(len(X_tr) + len(X_val) + len(X_te)), n_features=int(X_tr.shape[1]),
        n_train=int(len(X_tr)), n_val=int(len(X_val)), n_test=int(len(X_te)),
        tier_threshold=float(thr),
        premium_share_train=float(d["tier_tr"].mean()),
        premium_share_val=float(d["tier_val"].mean()),
        premium_share_test=float(d["tier_te"].mean()),
    )
    print("[data]", d["info"])
    return d


# =============================================================================
# Model factories (fresh instance per call -> no state leaks between folds)
# =============================================================================
def own_tree_reg():
    return DecisionTree(task="regression", random_state=SEED, **CONFIG["tree_reg"])


def own_tree_clf():
    return DecisionTree(task="classification", random_state=SEED, **CONFIG["tree_clf"])


def own_svm():
    return PegasosSVM(random_state=SEED, **CONFIG["svm"])


def _sk_models(n_train: int) -> dict:
    """scikit-learn baselines with MATCHED settings, scaling inside a Pipeline."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    tr, tc, sv = CONFIG["tree_reg"], CONFIG["tree_clf"], CONFIG["svm"]
    lam = sv["lambda_"]
    C = 1.0 / (n_train * lam)          # Pegasos lambda  <->  SVM C
    clf = {
        "sklearn DecisionTree (clf)": Pipeline([("m", DecisionTreeClassifier(
            criterion=tc["criterion"], max_depth=tc["max_depth"],
            min_samples_split=tc["min_samples_split"],
            min_samples_leaf=tc["min_samples_leaf"],
            min_impurity_decrease=tc["min_impurity_decrease"],
            random_state=SEED))]),
        "sklearn SGD hinge": Pipeline([("s", StandardScaler()), ("m", SGDClassifier(
            loss="hinge", alpha=lam, max_iter=1000, tol=1e-4, random_state=SEED))]),
        "sklearn LinearSVC": Pipeline([("s", StandardScaler()), ("m", LinearSVC(
            C=C, max_iter=20_000, random_state=SEED))]),
    }
    if n_train <= SVC_MAX_ROWS:
        clf["sklearn SVC (RBF)"] = Pipeline([("s", StandardScaler()),
                                             ("m", SVC(C=C, kernel="rbf", gamma="scale"))])
    reg = {
        "sklearn DecisionTree (reg)": Pipeline([("m", DecisionTreeRegressor(
            criterion="squared_error", max_depth=tr["max_depth"],
            min_samples_split=tr["min_samples_split"],
            min_samples_leaf=tr["min_samples_leaf"],
            min_impurity_decrease=tr["min_impurity_decrease"],
            random_state=SEED))]),
    }
    return {"clf": clf, "reg": reg}


# =============================================================================
# Shared fit / score helpers
# =============================================================================
def _scores(model, X) -> np.ndarray:
    """Continuous score for ROC-AUC: P(premium) if available, else the margin."""
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, -1] if p.ndim == 2 else p
    return np.asarray(model.decision_function(X))


def reg_metrics(ylog_true, ylog_pred) -> dict:
    """Task A metrics on log(price) AND on the original price scale."""
    m = {f"{k}_log": v for k, v in evaluate.regression_report(ylog_true, ylog_pred).items()}
    m.update(evaluate.regression_report(np.exp(ylog_true), np.exp(ylog_pred)))
    return m


def fit_eval_reg(make, X_tr, ylog_tr, X_ev, ylog_ev):
    model = make()
    _, t_fit = evaluate.timed(model.fit, X_tr, ylog_tr)
    pred, t_pred = evaluate.timed(model.predict, X_ev)
    pred = np.asarray(pred, dtype=float)
    return reg_metrics(ylog_ev, pred), pred, model, dict(fit_s=t_fit, predict_s=t_pred)


def fit_eval_clf(make, X_tr, y_tr, X_ev, y_ev):
    model = make()
    _, t_fit = evaluate.timed(model.fit, X_tr, y_tr)
    pred, t_pred = evaluate.timed(model.predict, X_ev)
    pred = np.asarray(pred).astype(int)
    score = _scores(model, X_ev)
    return (evaluate.classification_report(y_ev, pred, score),
            dict(pred=pred, score=score), model, dict(fit_s=t_fit, predict_s=t_pred))


# =============================================================================
# 2. OUR MODELS  (fit on TRAIN, evaluate on TEST)
# =============================================================================
def stage_own(d: dict) -> dict:
    out = {"metrics": {}, "timing": {}, "preds": {}, "models": {}}
    ylog_tr, ylog_te = np.log(d["price_tr"]), np.log(d["price_te"])

    m, pred, model, t = fit_eval_reg(own_tree_reg, d["X_tr"], ylog_tr, d["X_te"], ylog_te)
    out["metrics"]["Ours: tree (reg)"] = m
    out["timing"]["Ours: tree (reg)"] = t
    out["preds"]["Ours: tree (reg)"] = pred
    out["models"]["Ours: tree (reg)"] = model

    m, r, model, t = fit_eval_clf(own_tree_clf, d["X_tr"], d["tier_tr"], d["X_te"], d["tier_te"])
    out["metrics"]["Ours: tree (clf)"], out["timing"]["Ours: tree (clf)"] = m, t
    out["preds"]["Ours: tree (clf)"], out["models"]["Ours: tree (clf)"] = r, model

    m, r, model, t = fit_eval_clf(own_svm, d["Xs_tr"], d["tier_tr"], d["Xs_te"], d["tier_te"])
    out["metrics"]["Ours: SVM (Pegasos)"], out["timing"]["Ours: SVM (Pegasos)"] = m, t
    out["preds"]["Ours: SVM (Pegasos)"], out["models"]["Ours: SVM (Pegasos)"] = r, model
    for k, v in out["metrics"].items():
        print(f"[own] {k}: " + ", ".join(f"{a}={b:.4f}" for a, b in v.items()))
    return out


# =============================================================================
# 3. SKLEARN BASELINES  (same split, scaling inside a Pipeline)
# =============================================================================
def stage_baselines(d: dict) -> dict:
    sk = _sk_models(len(d["X_tr"]))
    out = {"metrics": {}, "timing": {}, "preds": {}, "models": {}}
    ylog_tr, ylog_te = np.log(d["price_tr"]), np.log(d["price_te"])

    for name, pipe in sk["reg"].items():
        from sklearn.base import clone
        m, pred, model, t = fit_eval_reg(lambda p=pipe: clone(p), d["X_tr"], ylog_tr,
                                         d["X_te"], ylog_te)
        out["metrics"][name], out["timing"][name] = m, t
        out["preds"][name], out["models"][name] = pred, model

    for name, pipe in sk["clf"].items():
        from sklearn.base import clone
        m, r, model, t = fit_eval_clf(lambda p=pipe: clone(p), d["X_tr"], d["tier_tr"],
                                      d["X_te"], d["tier_te"])
        out["metrics"][name], out["timing"][name] = m, t
        out["preds"][name], out["models"][name] = r, model
    for k, v in out["metrics"].items():
        print(f"[sklearn] {k}: " + ", ".join(f"{a}={b:.4f}" for a, b in v.items()))
    return out


# =============================================================================
# 4. CROSS-VALIDATION  (TRAIN+VAL only; identical folds for every model)
# =============================================================================
def stage_cv(d: dict, k: int = N_FOLDS) -> dict:
    from sklearn.base import clone

    X = np.vstack([d["X_tr"], d["X_val"]])
    price = np.concatenate([d["price_tr"], d["price_val"]])
    ylog = np.log(price)
    sk = _sk_models(len(X))
    # Folds for Task B are stratified on the TRAIN-median tier so classes stay balanced.
    tier_strat, _ = data_prep.make_tier_label(price, d["threshold"])
    folds = evaluate.kfold_indices(len(X), k=k, seed=SEED, y=np.asarray(tier_strat).astype(int))

    per_fold: dict[str, list] = {}
    for f, (tr, va) in enumerate(folds):
        # --- Task A ------------------------------------------------------
        for name, make in [("Ours: tree (reg)", own_tree_reg)] + \
                          [(n, (lambda p=p: clone(p))) for n, p in sk["reg"].items()]:
            m, *_ = fit_eval_reg(make, X[tr], ylog[tr], X[va], ylog[va])
            per_fold.setdefault(name, []).append(m)

        # --- Task B: threshold recomputed from THIS fold's training rows --
        thr_f = float(np.median(price[tr]))
        t_tr = (price[tr] > thr_f).astype(int)
        t_va = (price[va] > thr_f).astype(int)
        Xs_tr, Xs_va = data_prep.standardize(X[tr], X[va])
        for name, make, Xa, Xb in (
            [("Ours: tree (clf)", own_tree_clf, X[tr], X[va]),
             ("Ours: SVM (Pegasos)", own_svm, Xs_tr, Xs_va)]
            + [(n, (lambda p=p: clone(p)), X[tr], X[va]) for n, p in sk["clf"].items()]
        ):
            m, *_ = fit_eval_clf(make, Xa, t_tr, Xb, t_va)
            per_fold.setdefault(name, []).append(m)
        print(f"[cv] fold {f + 1}/{k} done")

    cv = {name: {key: evaluate.mean_std([m[key] for m in ms]) for key in ms[0]}
          for name, ms in per_fold.items()}
    return cv


# =============================================================================
# 5. FIGURES
# =============================================================================
def stage_figures(d: dict, own: dict, base: dict) -> None:
    y_te = d["tier_te"]
    price_te = d["price_te"]
    ylog_te = np.log(price_te)

    # --- confusion matrices (Task B) ---
    for name, tag in [("Ours: tree (clf)", "tree"), ("Ours: SVM (Pegasos)", "svm")]:
        cm = evaluate.confusion_matrix(y_te, own["preds"][name]["pred"])
        plots.confusion_matrix_fig(cm, f"{name} — test", f"cm_{tag}")

    # --- ROC (ours + sklearn) ---
    scores = {n: r["score"] for n, r in own["preds"].items() if isinstance(r, dict)}
    scores.update({n: r["score"] for n, r in base["preds"].items() if isinstance(r, dict)})
    plots.roc_fig(y_te, scores)

    # --- Task A error analysis ---
    pred_log = own["preds"]["Ours: tree (reg)"]
    plots.pred_vs_actual_fig(price_te, np.exp(pred_log), "Ours: tree — test", "pred_vs_actual_tree")
    plots.residuals_fig(pred_log, ylog_te - pred_log, "Residuals — our tree (test)", "residuals_tree")
    plots.error_by_decile_fig(price_te, {
        "Ours: tree": np.exp(pred_log),
        "sklearn tree": np.exp(base["preds"]["sklearn DecisionTree (reg)"]),
    })

    # --- tree feature importance (impurity decrease) ---
    for name, tag in [("Ours: tree (reg)", "reg"), ("Ours: tree (clf)", "clf")]:
        imp = getattr(own["models"][name], "feature_importances_", None)
        if imp is not None:
            plots.importance_fig(d["names"], imp, f"Tree importances ({tag})", f"importance_tree_{tag}")
        else:
            print(f"[figures] {name} has no feature_importances_ yet (Person B) — skipped")

    # --- SVM weights + margin/support-vector report ---
    svm = own["models"]["Ours: SVM (Pegasos)"]
    if svm.w is not None:
        plots.importance_fig(d["names"], svm.w, "SVM weights (standardized)",
                             "svm_weights", signed=True)


def svm_margin_report(d: dict, svm) -> dict:
    """||w||, margin width 2/||w|| and support-vector count (y*f(x) <= 1) on TRAIN."""
    w_norm = float(np.linalg.norm(svm.w))
    ypm = np.where(d["tier_tr"] == 1, 1.0, -1.0)
    margins = ypm * svm.decision_function(d["Xs_tr"])
    return dict(w_norm=w_norm, margin_width=2.0 / w_norm if w_norm > 0 else float("inf"),
                n_support_vectors=int(np.sum(margins <= 1.0)),
                share_support_vectors=float(np.mean(margins <= 1.0)))


# =============================================================================
# 6. REPORT TABLES
# =============================================================================
def stage_report(d, own, base, cv) -> dict:
    test = {**own["metrics"], **base["metrics"]}
    timing = {**own["timing"], **base["timing"]}

    reg_names = [n for n in test if "(reg)" in n]
    clf_names = [n for n in test if "(reg)" not in n]

    evaluate.compare({n: {k: test[n][k] for k in ("rmse", "mae", "r2", "rmse_log", "r2_log")}
                      for n in reg_names}, "Task A — regression (TEST)")
    evaluate.compare({n: {k: test[n][k] for k in ("accuracy", "precision", "recall", "f1", "auc")}
                      for n in clf_names}, "Task B — price tier (TEST)")
    evaluate.compare({n: {"fit_s": round(timing[n]["fit_s"], 3),
                          "predict_s": round(timing[n]["predict_s"], 4)}
                      for n in test}, "Timing (train rows = %d)" % len(d["X_tr"]))
    if cv:
        evaluate.compare({n: {k: cv[n][k] for k in ("rmse", "r2_log")} for n in cv if "(reg)" in n},
                         f"Task A — {N_FOLDS}-fold CV mean ± std (TRAIN+VAL)")
        evaluate.compare({n: {k: cv[n][k] for k in ("f1", "auc", "accuracy")}
                          for n in cv if "(reg)" not in n},
                         f"Task B — {N_FOLDS}-fold CV mean ± std (TRAIN+VAL)")
        print("\nLaTeX rows (Task A RMSE / Task B F1, CV mean±std):")
        rows = {n: {"rmse": cv[n].get("rmse", "n/a"), "f1": cv[n].get("f1", "n/a")} for n in cv}
        print(evaluate.to_latex_rows(rows, ["rmse", "f1"]))

    svm_rep = svm_margin_report(d, own["models"]["Ours: SVM (Pegasos)"])
    print("\n[svm margin]", svm_rep)

    result = dict(info=d["info"], config=CONFIG, test=test, timing=timing,
                  cv=cv, svm_margin=svm_rep)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


# =============================================================================
# ENTRY POINT
# =============================================================================
def run(cv_enabled: bool = True, figures: bool = True) -> dict:
    d = stage_data()
    own = stage_own(d)
    base = stage_baselines(d)
    cv = stage_cv(d) if cv_enabled else {}
    if figures:
        stage_figures(d, own, base)
    return stage_report(d, own, base, cv)


def _flat_test_metrics(result: dict) -> dict:
    return {f"{m}/{k}": v for m, ks in result["test"].items() for k, v in ks.items()}


def check_seeds() -> bool:
    """Run the pipeline twice; every TEST metric must match exactly."""
    a = _flat_test_metrics(run(cv_enabled=False, figures=False))
    b = _flat_test_metrics(run(cv_enabled=False, figures=False))
    bad = [k for k in a if not np.isclose(a[k], b[k], rtol=0, atol=1e-12)]
    if bad:
        print("\nSEED CHECK FAILED — non-deterministic:", bad)
        return False
    print("\nSEED CHECK OK — two runs gave identical metrics.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    ap.add_argument("--check-seeds", action="store_true", help="run twice and compare")
    args = ap.parse_args()
    if args.check_seeds:
        sys.exit(0 if check_seeds() else 1)
    run(cv_enabled=not args.no_cv)


if __name__ == "__main__":
    main()

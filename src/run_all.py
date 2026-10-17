"""
run_all.py — ONE command reproduces every headline number and figure.

    python -m src.run_all                 # full run (tables + figures + CV)
    python -m src.run_all --no-cv         # skip cross-validation (fast)
    python -m src.run_all --check-seeds   # run twice, assert identical metrics

The test set is touched ONLY for the final numbers. Hyperparameters in CONFIG
must come from validation evidence (Person B / Person C sweeps) — never test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import data_prep, evaluate, plots
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
    "svm": dict(lambda_=1e-3, n_iters=50_000),
}


# =============================================================================
# 1. DATA
# =============================================================================
def stage_data() -> dict:
    """Load, clean, featurize, split, derive tier (TRAIN median), standardize."""
    np.random.seed(SEED)
    df = data_prep.load_raw()
    df = data_prep.clean(df)
    X, y, names = data_prep.make_features(df)         # y = price (raw, AZN)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_prep.train_val_test_split(X, y)

    tier_tr, thr = data_prep.make_tier_label(y_tr)                 # threshold: TRAIN only
    tier_val, _ = data_prep.make_tier_label(y_val, thr)
    tier_te, _ = data_prep.make_tier_label(y_te, thr)

    Xs_tr, Xs_val, Xs_te = data_prep.standardize(X_tr, X_val, X_te)  # TRAIN stats only

    d = dict(
        names=list(names), threshold=float(thr),
        X_tr=X_tr, X_val=X_val, X_te=X_te,
        Xs_tr=Xs_tr, Xs_val=Xs_val, Xs_te=Xs_te,
        price_tr=y_tr, price_val=y_val, price_te=y_te,
        tier_tr=np.asarray(tier_tr).astype(int),
        tier_val=np.asarray(tier_val).astype(int),
        tier_te=np.asarray(tier_te).astype(int),
    )
    d["info"] = dict(
        rows=int(len(X)), n_features=int(X.shape[1]),
        n_train=int(len(X_tr)), n_val=int(len(X_val)), n_test=int(len(X_te)),
        tier_threshold=float(thr),
        premium_share_train=float(d["tier_tr"].mean()),
        premium_share_val=float(d["tier_val"].mean()),
        premium_share_test=float(d["tier_te"].mean()),
    )
    print("[data]", d["info"])
    return d


# TEMPORARY entry point — replaced in part 7
def main() -> None:
    stage_data()


if __name__ == "__main__":
    main()
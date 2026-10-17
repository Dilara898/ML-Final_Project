'''
tree_experiments.py - Part B experiments on the REAL cleaned data.

Run from the repo root:   python -m src.tree_experiments

This first version does a 'first look' at the data and compares our tree with simple
baselines and with scikit-learn (comparison only). It never touches the TEST set.
NOTE: the small metric helpers below are temporary; once Person D's evaluate.py is
merged we can switch to it (and cross-check the numbers).
'''

from __future__ import annotations

import time

import numpy as np

from . import data_prep
from .decision_tree import DecisionTree


# ---------- tiny metric helpers (temporary) ----------------------------------
def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


def _r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def _accuracy(y, p):
    return float(np.mean(y == p))


def _f1(y, p):
    tp = np.sum((y == 1) & (p == 1))
    fp = np.sum((y == 0) & (p == 1))
    fn = np.sum((y == 1) & (p == 0))
    return float(2 * tp / (2 * tp + fp + fn)) if tp > 0 else 0.0


def _auc(y, scores):
    '''Rank-based AUC (Mann-Whitney), ties get the average rank.'''
    n = len(scores)
    order = np.argsort(scores, kind='mergesort')
    s = scores[order]
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------- data --------------------------------------------------------------
def load_splits():
    '''Use Person A's pipeline. prepare_data() returns a dict with unscaled features
    (trees do not need scaling), prices, tier labels and feature names.'''
    d = data_prep.prepare_data()
    f = lambda a: np.asarray(a, dtype=float)
    return {
        'X_tr': f(d['X_train']), 'X_val': f(d['X_val']), 'X_te': f(d['X_test']),
        'y_tr': f(d['y_train']), 'y_val': f(d['y_val']),
        'tier_tr': np.asarray(d['tier_train']).astype(int),
        'tier_val': np.asarray(d['tier_val']).astype(int),
        'threshold': float(d['tier_threshold']),
        'names': list(d['feature_names']),
    }


def describe(X_tr, y_tr, X_val, y_val, X_te, names):
    print('=== 1. FIRST LOOK AT THE DATA ===')
    print(f'train {X_tr.shape}  val {X_val.shape}  test {X_te.shape}   (rows, features)')
    print(f'NaN in X_train: {int(np.isnan(X_tr).sum())}   NaN in y_train: {int(np.isnan(y_tr).sum())}')
    print(f'price (train): min {y_tr.min():.1f}  median {np.median(y_tr):.1f}  '
          f'mean {y_tr.mean():.1f}  max {y_tr.max():.1f}')
    print(f'features ({len(names)}): {names[:12]}{" ..." if len(names) > 12 else ""}')
    print()


# ---------- Task A: regression on log(price) ----------------------------------
def task_a(X_tr, y_tr, X_val, y_val, names, depth=6):
    print('=== 2. TASK A: regression on log(price), validation set ===')
    assert (y_tr > 0).all() and (y_val > 0).all(), 'log needs positive prices'
    ytr, yval = np.log(y_tr), np.log(y_val)

    base = np.full_like(yval, ytr.mean())
    print(f'baseline (always train mean): RMSE {_rmse(yval, base):.4f}  '
          f'MAE {_mae(yval, base):.4f}  R2 {_r2(yval, base):.4f}')

    t0 = time.time()
    tree = DecisionTree(task='regression', max_depth=depth).fit(X_tr, ytr)
    fit_s = time.time() - t0
    p = tree.predict(X_val)
    print(f'OUR tree depth {depth}:           RMSE {_rmse(yval, p):.4f}  '
          f'MAE {_mae(yval, p):.4f}  R2 {_r2(yval, p):.4f}   '
          f'(fit {fit_s:.2f}s, {tree.get_n_leaves()} leaves)')

    try:
        from sklearn.tree import DecisionTreeRegressor
        t0 = time.time()
        ref = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(X_tr, ytr)
        sk_s = time.time() - t0
        q = ref.predict(X_val)
        print(f'sklearn tree depth {depth}:       RMSE {_rmse(yval, q):.4f}  '
              f'MAE {_mae(yval, q):.4f}  R2 {_r2(yval, q):.4f}   (fit {sk_s:.2f}s)')
    except ImportError:
        print('(scikit-learn not installed: skipping comparison)')

    order = np.argsort(tree.feature_importances_)[::-1][:5]
    print('top 5 features:', [(names[i], round(float(tree.feature_importances_[i]), 3)) for i in order])
    print()


# ---------- Task B: premium vs standard ----------------------------------------
def task_b(X_tr, tier_tr, X_val, tier_val, threshold, names, depth=6):
    print('=== 3. TASK B: premium vs standard, validation set ===')
    ytr, yval = np.asarray(tier_tr).astype(int), np.asarray(tier_val).astype(int)
    print(f'threshold (train median price): {threshold:.1f}   premium share: '
          f'train {ytr.mean():.3f}  val {yval.mean():.3f}')

    major = int(ytr.mean() >= 0.5)
    print(f'baseline (always class {major}):  accuracy {_accuracy(yval, np.full_like(yval, major)):.4f}')

    for crit in ('gini', 'entropy'):
        t0 = time.time()
        tree = DecisionTree(criterion=crit, max_depth=depth).fit(X_tr, ytr)
        fit_s = time.time() - t0
        p = tree.predict(X_val)
        auc = _auc(yval, tree.predict_proba(X_val)[:, 1])
        print(f'OUR tree {crit:7s} depth {depth}: accuracy {_accuracy(yval, p):.4f}  '
              f'F1 {_f1(yval, p):.4f}  AUC {auc:.4f}   (fit {fit_s:.2f}s)')

    try:
        from sklearn.tree import DecisionTreeClassifier
        ref = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, ytr)
        q = ref.predict(X_val)
        print(f'sklearn tree gini    depth {depth}: accuracy {_accuracy(yval, q):.4f}  '
              f'F1 {_f1(yval, q):.4f}  AUC {_auc(yval, ref.predict_proba(X_val)[:, 1]):.4f}')
    except ImportError:
        pass
    print()


def main():
    d = load_splits()
    describe(d['X_tr'], d['y_tr'], d['X_val'], d['y_val'], d['X_te'], d['names'])
    if d['y_tr'].max() < 25:
        print('WARNING: prices look small - is y_train already log(price)? Ask Person A.\n')
    task_a(d['X_tr'], d['y_tr'], d['X_val'], d['y_val'], d['names'])
    task_b(d['X_tr'], d['tier_tr'], d['X_val'], d['tier_val'], d['threshold'], d['names'])


if __name__ == '__main__':
    main()
'''
forest_experiment.py - Bonus 9 on the REAL data (validation set only).

Run from the repo root:   python -m src.forest_experiment     (takes a few minutes)

Compares, for Task A (log price) and Task B (premium vs standard):
  - our best single tree        - our random forest        - scikit-learn's random forest (check only)
and plots the validation error as a function of the number of trees.
Outputs: report/figures/forest_trees_vs_error.png and report/forest_results.md.
The TEST set is not used here.
'''

from __future__ import annotations

import os
import time

import numpy as np

from .decision_tree import DecisionTree
from .random_forest import RandomForest
from .tree_experiments import (_accuracy, _auc, _f1, _mae, _r2, _rmse,
                               load_splits)
from .tree_final import BEST_A, BEST_B, FOREST_A, FOREST_B

FIG_DIR = 'report/figures'
OUT_MD = 'report/forest_results.md'


def _timed(fn):
    t0 = time.time()
    out = fn()
    return out, round(time.time() - t0, 1)


def plot_curves(curve_a, curve_b, ref_a, ref_b, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, curve, ref, ylabel, title in (
            (axes[0], curve_a, ref_a, 'validation RMSE on log(price)', 'Task A'),
            (axes[1], curve_b, ref_b, 'validation accuracy', 'Task B')):
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, marker='o', markersize=3, label='our random forest')
        ax.axhline(ref['tree'], linestyle='--', color='gray', label='our best single tree')
        if ref.get('sk') is not None:
            ax.axhline(ref['sk'], linestyle=':', color='black', label='scikit-learn forest')
        ax.set_xlabel('number of trees')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


def main():
    d = load_splits()
    X_tr, X_val = d['X_tr'], d['X_val']
    ytr, yval = np.log(d['y_tr']), np.log(d['y_val'])
    ctr, cval = d['tier_tr'], d['tier_val']
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---------------- Task A ----------------
    print('Task A: single tree ...', flush=True)
    tree_a, t_tree_a = _timed(lambda: DecisionTree(task='regression', **BEST_A).fit(X_tr, ytr))
    p_tree_a = tree_a.predict(X_val)
    print('Task A: our forest ...', flush=True)
    forest_a, t_forest_a = _timed(lambda: RandomForest(task='regression', **FOREST_A).fit(X_tr, ytr))
    staged = forest_a.staged_predict(X_val)
    curve_a = [_rmse(yval, p) for p in staged]
    p_forest_a = staged[-1]

    sk_a = None
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        print('Task A: scikit-learn forest ...', flush=True)
        ref_a, t_sk_a = _timed(lambda: RandomForestRegressor(
            n_estimators=FOREST_A['n_estimators'], max_features='sqrt',
            min_samples_leaf=FOREST_A['min_samples_leaf'], random_state=0, n_jobs=1).fit(X_tr, ytr))
        p_sk_a = ref_a.predict(X_val)
        sk_a = (p_sk_a, t_sk_a)
    except ImportError:
        RandomForestClassifier = None

    # ---------------- Task B ----------------
    print('Task B: single tree ...', flush=True)
    tree_b, t_tree_b = _timed(lambda: DecisionTree(task='classification', **BEST_B).fit(X_tr, ctr))
    p_tree_b = tree_b.predict(X_val)
    proba_tree_b = tree_b.predict_proba(X_val)[:, 1]
    print('Task B: our forest ...', flush=True)
    forest_b, t_forest_b = _timed(lambda: RandomForest(task='classification', **FOREST_B).fit(X_tr, ctr))
    staged_b = forest_b.staged_predict(X_val)
    curve_b = [_accuracy(cval, p) for p in staged_b]
    p_forest_b = staged_b[-1]
    proba_forest_b = forest_b.predict_proba(X_val)[:, 1]

    sk_b = None
    if RandomForestClassifier is not None:
        print('Task B: scikit-learn forest ...', flush=True)
        ref_b, t_sk_b = _timed(lambda: RandomForestClassifier(
            n_estimators=FOREST_B['n_estimators'], criterion='entropy', max_features='sqrt',
            min_samples_leaf=FOREST_B['min_samples_leaf'], random_state=0, n_jobs=1).fit(X_tr, ctr))
        sk_b = (ref_b.predict(X_val), ref_b.predict_proba(X_val)[:, 1], t_sk_b)

    # ---------------- results ----------------
    rows = []

    def add(model, task, p, proba, t):
        if task == 'A':
            rows.append({'task': 'A', 'model': model, 'RMSE': round(_rmse(yval, p), 4),
                         'MAE': round(_mae(yval, p), 4), 'R2': round(_r2(yval, p), 4),
                         'acc': '', 'F1': '', 'AUC': '', 'fit_s': t})
        else:
            rows.append({'task': 'B', 'model': model, 'RMSE': '', 'MAE': '', 'R2': '',
                         'acc': round(_accuracy(cval, p), 4), 'F1': round(_f1(cval, p), 4),
                         'AUC': round(_auc(cval, proba), 4), 'fit_s': t})

    add('our single tree', 'A', p_tree_a, None, t_tree_a)
    add('our random forest', 'A', p_forest_a, None, t_forest_a)
    if sk_a:
        add('sklearn random forest', 'A', sk_a[0], None, sk_a[1])
    add('our single tree', 'B', p_tree_b, proba_tree_b, t_tree_b)
    add('our random forest', 'B', p_forest_b, proba_forest_b, t_forest_b)
    if sk_b:
        add('sklearn random forest', 'B', sk_b[0], sk_b[1], sk_b[2])

    cols = ['task', 'model', 'RMSE', 'MAE', 'R2', 'acc', 'F1', 'AUC', 'fit_s']
    table = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n'
    for r in rows:
        table += '| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n'
        print(r, flush=True)

    plot_curves(curve_a, curve_b,
                {'tree': _rmse(yval, p_tree_a), 'sk': _rmse(yval, sk_a[0]) if sk_a else None},
                {'tree': _accuracy(cval, p_tree_b), 'sk': _accuracy(cval, sk_b[0]) if sk_b else None},
                'forest_trees_vs_error.png')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('# Random forest (Bonus 9), validation set\n\n' + table + '\n')
        f.write(f'Forest settings: {FOREST_A["n_estimators"]} trees, max_features=sqrt, '
                f'min_samples_leaf={FOREST_A["min_samples_leaf"]}.\n')
    print(f'\nSaved {OUT_MD} and {FIG_DIR}/forest_trees_vs_error.png', flush=True)


if __name__ == '__main__':
    main()
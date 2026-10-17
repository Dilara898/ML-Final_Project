'''
tree_tuning.py - Part B: tune our decision tree on the VALIDATION set and make the figures.

Run from the repo root:   python -m src.tree_tuning     (takes several minutes)

What it does (for Task A = regression on log(price) and Task B = premium vs standard):
  1. depth sweep            -> train vs validation error plot (bias-variance / overfitting)
  2. min_samples_leaf sweep -> shows how a leaf-size limit fights overfitting
  3. Gini vs entropy        -> Task B only
  4. best settings vs scikit-learn with the SAME settings (comparison only)
  5. feature importance bar charts
Outputs: report/figures/tree_*.png and report/tree_tuning_results.md.
The TEST set is never used here (it is used once, at the very end, by the final run).
'''

from __future__ import annotations

import os
import time

import numpy as np

from .decision_tree import DecisionTree
from .tree_experiments import (_accuracy, _auc, _f1, _mae, _r2, _rmse,
                               load_splits)

FIG_DIR = 'report/figures'
OUT_MD = 'report/tree_tuning_results.md'
DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20]
LEAVES = [5, 10, 20, 50, 100]


# ---------- one fit + evaluation --------------------------------------------
def run_reg(X_tr, y_tr, X_val, y_val, **kw):
    t0 = time.time()
    tree = DecisionTree(task='regression', **kw).fit(X_tr, y_tr)
    fit_s = time.time() - t0
    p_tr, p_val = tree.predict(X_tr), tree.predict(X_val)
    return tree, {
        'train_rmse': round(_rmse(y_tr, p_tr), 4), 'val_rmse': round(_rmse(y_val, p_val), 4),
        'val_mae': round(_mae(y_val, p_val), 4), 'val_r2': round(_r2(y_val, p_val), 4),
        'depth': tree.get_depth(), 'leaves': tree.get_n_leaves(), 'fit_s': round(fit_s, 1),
    }


def run_clf(X_tr, y_tr, X_val, y_val, **kw):
    t0 = time.time()
    tree = DecisionTree(task='classification', **kw).fit(X_tr, y_tr)
    fit_s = time.time() - t0
    p_tr, p_val = tree.predict(X_tr), tree.predict(X_val)
    return tree, {
        'train_acc': round(_accuracy(y_tr, p_tr), 4), 'val_acc': round(_accuracy(y_val, p_val), 4),
        'val_f1': round(_f1(y_val, p_val), 4),
        'val_auc': round(_auc(y_val, tree.predict_proba(X_val)[:, 1]), 4),
        'depth': tree.get_depth(), 'leaves': tree.get_n_leaves(), 'fit_s': round(fit_s, 1),
    }


# ---------- helpers ------------------------------------------------------------
def md_table(rows, cols):
    out = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n'
    for r in rows:
        out += '| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n'
    return out + '\n'


def plot_depth(rows, train_key, val_key, ylabel, title, fname, best_depth):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = [r['max_depth'] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, [r[train_key] for r in rows], marker='o', label='train')
    ax.plot(xs, [r[val_key] for r in rows], marker='o', label='validation')
    ax.axvline(best_depth, linestyle='--', color='gray', label=f'best max_depth = {best_depth}')
    ax.set_xlabel('max_depth')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


def plot_importance(trees_and_titles, names, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(trees_and_titles), figsize=(6 * len(trees_and_titles), 4))
    axes = np.atleast_1d(axes)
    for ax, (tree, title) in zip(axes, trees_and_titles):
        imp = tree.feature_importances_
        top = np.argsort(imp)[::-1][:10][::-1]
        ax.barh([names[i] for i in top], imp[top])
        ax.set_title(title)
        ax.set_xlabel('importance (sum of weighted impurity decrease)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)


def tune(label, runner, key, maximize, train_key, ylabel, fig_name, md):
    '''Depth sweep, then min_samples_leaf sweep. Returns (best settings, best row).'''
    print(f'\n##### {label}: depth sweep (min_samples_leaf=1) #####', flush=True)
    rows = []
    for dpt in DEPTHS:
        _, m = runner(max_depth=dpt)
        row = {'max_depth': dpt, 'min_samples_leaf': 1, **m}
        rows.append(row)
        print('  ', row, flush=True)
    pick = max if maximize else min
    best_depth_row = pick(rows, key=lambda r: r[key])
    plot_depth(rows, train_key, key, ylabel, f'{label}: error vs tree depth', fig_name,
               best_depth_row['max_depth'])
    md.append(f'## {label}\n\n### Depth sweep (min_samples_leaf = 1)\n\n')
    md.append(md_table(rows, ['max_depth', 'depth', 'leaves', train_key, key, 'fit_s']))

    print(f'\n##### {label}: min_samples_leaf sweep (max_depth={DEPTHS[-1]}) #####', flush=True)
    leaf_rows = []
    for leaf in LEAVES:
        _, m = runner(max_depth=DEPTHS[-1], min_samples_leaf=leaf)
        row = {'max_depth': DEPTHS[-1], 'min_samples_leaf': leaf, **m}
        leaf_rows.append(row)
        print('  ', row, flush=True)
    md.append(f'### min_samples_leaf sweep (max_depth = {DEPTHS[-1]})\n\n')
    md.append(md_table(leaf_rows, ['min_samples_leaf', 'depth', 'leaves', train_key, key, 'fit_s']))

    best = pick(rows + leaf_rows, key=lambda r: r[key])
    print(f'\nBEST {label}: max_depth={best["max_depth"]} min_samples_leaf={best["min_samples_leaf"]} '
          f'{key}={best[key]}', flush=True)
    md.append(f'**Best on validation:** max_depth = {best["max_depth"]}, '
              f'min_samples_leaf = {best["min_samples_leaf"]}, {key} = {best[key]}\n\n')
    return best


def main():
    d = load_splits()
    X_tr, X_val, names = d['X_tr'], d['X_val'], d['names']
    ytr, yval = np.log(d['y_tr']), np.log(d['y_val'])
    ctr, cval = d['tier_tr'], d['tier_val']
    os.makedirs(FIG_DIR, exist_ok=True)
    md = ['# Decision tree tuning (all numbers on the VALIDATION set)\n\n']

    # ----- Task A: regression -----
    reg_runner = lambda **kw: run_reg(X_tr, ytr, X_val, yval, **kw)
    best_a = tune('Task A (regression, log price)', reg_runner, 'val_rmse', False,
                  'train_rmse', 'RMSE on log(price)', 'tree_depth_sweep_regression.png', md)
    tree_a, row_a = reg_runner(max_depth=best_a['max_depth'], min_samples_leaf=best_a['min_samples_leaf'])

    # ----- Task B: classification -----
    clf_runner = lambda **kw: run_clf(X_tr, ctr, X_val, cval, criterion='gini', **kw)
    best_b = tune('Task B (classification, gini)', clf_runner, 'val_acc', True,
                  'train_acc', 'accuracy', 'tree_depth_sweep_classification.png', md)

    print('\n##### Task B: Gini vs entropy at the best settings #####', flush=True)
    crit_rows = []
    for crit in ('gini', 'entropy'):
        _, m = run_clf(X_tr, ctr, X_val, cval, criterion=crit,
                       max_depth=best_b['max_depth'], min_samples_leaf=best_b['min_samples_leaf'])
        crit_rows.append({'criterion': crit, **m})
        print('  ', crit_rows[-1], flush=True)
    md.append('### Gini vs entropy (best settings)\n\n')
    md.append(md_table(crit_rows, ['criterion', 'val_acc', 'val_f1', 'val_auc', 'fit_s']))
    best_crit = max(crit_rows, key=lambda r: r['val_acc'])['criterion']
    tree_b, row_b = run_clf(X_tr, ctr, X_val, cval, criterion=best_crit,
                            max_depth=best_b['max_depth'], min_samples_leaf=best_b['min_samples_leaf'])

    # ----- comparison with scikit-learn, same settings (comparison only) -----
    md.append('## Comparison with scikit-learn (same settings)\n\n')
    try:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        t0 = time.time()
        ref_a = DecisionTreeRegressor(max_depth=best_a['max_depth'],
                                      min_samples_leaf=best_a['min_samples_leaf'],
                                      random_state=0).fit(X_tr, ytr)
        sk_a_s = time.time() - t0
        q = ref_a.predict(X_val)
        t0 = time.time()
        ref_b = DecisionTreeClassifier(criterion=best_crit, max_depth=best_b['max_depth'],
                                       min_samples_leaf=best_b['min_samples_leaf'],
                                       random_state=0).fit(X_tr, ctr)
        sk_b_s = time.time() - t0
        qb = ref_b.predict(X_val)
        cmp_rows = [
            {'task': 'A', 'model': 'ours', 'val_rmse / val_acc': row_a['val_rmse'], 'val_r2 / val_f1': row_a['val_r2'], 'fit_s': row_a['fit_s']},
            {'task': 'A', 'model': 'sklearn', 'val_rmse / val_acc': round(_rmse(yval, q), 4), 'val_r2 / val_f1': round(_r2(yval, q), 4), 'fit_s': round(sk_a_s, 1)},
            {'task': 'B', 'model': 'ours', 'val_rmse / val_acc': row_b['val_acc'], 'val_r2 / val_f1': row_b['val_f1'], 'fit_s': row_b['fit_s']},
            {'task': 'B', 'model': 'sklearn', 'val_rmse / val_acc': round(_accuracy(cval, qb), 4), 'val_r2 / val_f1': round(_f1(cval, qb), 4), 'fit_s': round(sk_b_s, 1)},
        ]
        for r in cmp_rows:
            print('  ', r, flush=True)
        md.append(md_table(cmp_rows, ['task', 'model', 'val_rmse / val_acc', 'val_r2 / val_f1', 'fit_s']))
    except ImportError:
        md.append('(scikit-learn not installed)\n\n')

    # ----- importance figure -----
    plot_importance([(tree_a, 'Task A: feature importance'), (tree_b, f'Task B ({best_crit}): feature importance')],
                    names, 'tree_feature_importance.png')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(''.join(md))
    print(f'\nSaved figures to {FIG_DIR}/ and tables to {OUT_MD}', flush=True)


if __name__ == '__main__':
    main()
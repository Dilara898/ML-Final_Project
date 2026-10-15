import numpy as np
import pytest

from src.decision_tree import DecisionTree


def test_gini_values():
    t = DecisionTree(criterion='gini')
    assert t._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    assert t._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)


def test_entropy_values():
    t = DecisionTree(criterion='entropy')
    assert t._impurity(np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert t._impurity(np.array([1, 1, 1])) == pytest.approx(0.0)


def test_mse_values():
    t = DecisionTree(task='regression', criterion='mse')
    assert t._impurity(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0 / 3.0)
def brute_force_best_split(tree, X, y):
    '''Slow but obviously-correct reference: try every feature and every midpoint.'''
    n, d = X.shape
    parent = tree._impurity(y)
    best = None
    for j in range(d):
        values = np.unique(X[:, j])
        for a, b in zip(values[:-1], values[1:]):
            thr = (a + b) / 2.0
            left = X[:, j] <= thr
            if left.sum() < tree.min_samples_leaf or (~left).sum() < tree.min_samples_leaf:
                continue
            w = (left.sum() * tree._impurity(y[left]) + (~left).sum() * tree._impurity(y[~left])) / n
            dec = parent - w
            if best is None or dec > best[2] + 1e-12:
                best = (j, thr, dec)
    return best


@pytest.mark.parametrize('task,crit', [('classification', 'gini'),
                                       ('classification', 'entropy'),
                                       ('regression', 'mse')])
def test_fast_split_matches_brute_force(task, crit):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = rng.integers(0, 3, 60) if task == 'classification' else rng.normal(size=60)
    t = DecisionTree(task=task, criterion=crit)
    fast = t._best_split(X, y)
    slow = brute_force_best_split(t, X, y)
    assert fast[0] == slow[0]
    assert fast[1] == pytest.approx(slow[1])
    assert fast[2] == pytest.approx(slow[2])
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
def test_tiny_example_by_hand():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTree().fit(X, y)
    assert t.root.feature == 0
    assert t.root.threshold == pytest.approx(2.5)
    assert list(t.predict(X)) == [0, 0, 1, 1]


def test_string_labels_work():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array(['standard', 'standard', 'premium', 'premium'])
    t = DecisionTree().fit(X, y)
    assert list(t.predict(X)) == ['standard', 'standard', 'premium', 'premium']


def test_regression_fits_noiseless_data_perfectly():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2))
    y = 3 * X[:, 0] - X[:, 1]
    pred = DecisionTree(task='regression', criterion='mse').fit(X, y).predict(X)
    assert np.allclose(pred, y)
def leaf_sizes(node):
    '''Return the n_samples of every leaf under node.'''
    if node.is_leaf():
        return [node.n_samples]
    return leaf_sizes(node.left) + leaf_sizes(node.right)


def test_max_depth_is_respected():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y = rng.integers(0, 2, 200)
    for depth in (1, 2, 3):
        assert DecisionTree(max_depth=depth).fit(X, y).get_depth() <= depth
    assert DecisionTree(max_depth=1).fit(X, y).get_n_leaves() == 2


def test_min_samples_leaf_is_respected():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3))
    y = rng.integers(0, 2, 200)
    t = DecisionTree(min_samples_leaf=15).fit(X, y)
    assert min(leaf_sizes(t.root)) >= 15


def test_min_samples_split_is_respected():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 3))
    y = rng.integers(0, 2, 200)
    t = DecisionTree(min_samples_split=50).fit(X, y)
    big = DecisionTree().fit(X, y)
    assert t.get_n_leaves() < big.get_n_leaves()
'''Tests for src/random_forest.py (Bonus 9). Run with:  python -m pytest -q'''
import numpy as np
import pytest

from src.decision_tree import DecisionTree
from src.random_forest import RandomForest


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_regression_data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = 2 * X[:, 0] - X[:, 1] ** 2 + X[:, 2] * X[:, 3] + rng.normal(scale=1.0, size=n)
    return X[: n // 2], y[: n // 2], X[n // 2:], y[n // 2:]


# ---------- max_features inside the tree -------------------------------------
def test_tree_with_max_features_still_works():
    Xtr, ytr, _, _ = make_regression_data()
    t = DecisionTree(task='regression', max_depth=4, max_features=2).fit(Xtr, ytr)
    assert t.feature_importances_.sum() == pytest.approx(1.0)


def test_max_features_none_is_the_old_behaviour():
    Xtr, ytr, Xte, _ = make_regression_data()
    a = DecisionTree(task='regression', max_depth=4).fit(Xtr, ytr).predict(Xte)
    b = DecisionTree(task='regression', max_depth=4, max_features=None).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


# ---------- the forest --------------------------------------------------------
def test_one_tree_without_randomness_equals_single_tree():
    Xtr, ytr, Xte, _ = make_regression_data()
    forest = RandomForest(task='regression', n_estimators=1, bootstrap=False,
                          max_features=None, max_depth=5).fit(Xtr, ytr)
    tree = DecisionTree(task='regression', max_depth=5).fit(Xtr, ytr)
    assert np.allclose(forest.predict(Xte), tree.predict(Xte))


def test_forest_beats_a_single_deep_tree():
    Xtr, ytr, Xte, yte = make_regression_data()
    tree = DecisionTree(task='regression').fit(Xtr, ytr)
    forest = RandomForest(task='regression', n_estimators=40, max_features=3).fit(Xtr, ytr)
    assert rmse(forest.predict(Xte), yte) < rmse(tree.predict(Xte), yte)


def test_classification_proba_shape_and_sum():
    Xtr, ytr, Xte, _ = make_regression_data()
    y = (ytr > np.median(ytr)).astype(int)
    p = RandomForest(n_estimators=15).fit(Xtr, y).predict_proba(Xte)
    assert p.shape == (len(Xte), 2)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_class_missing_from_a_bootstrap_sample_is_handled():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    forest = RandomForest(n_estimators=25, max_features=None).fit(X, y)
    assert forest.predict_proba(X).shape == (4, 2)


def test_same_seed_gives_same_forest():
    Xtr, ytr, Xte, _ = make_regression_data()
    a = RandomForest(task='regression', n_estimators=5, max_features=3, random_state=1).fit(Xtr, ytr)
    b = RandomForest(task='regression', n_estimators=5, max_features=3, random_state=1).fit(Xtr, ytr)
    assert np.array_equal(a.predict(Xte), b.predict(Xte))


def test_staged_predict_last_step_equals_predict():
    Xtr, ytr, Xte, _ = make_regression_data()
    forest = RandomForest(task='regression', n_estimators=8, max_features=3).fit(Xtr, ytr)
    staged = forest.staged_predict(Xte)
    assert len(staged) == 8
    assert np.allclose(staged[-1], forest.predict(Xte))


# ---------- comparison with scikit-learn (CHECK ONLY) -------------------------
def test_close_to_sklearn_random_forest():
    sk = pytest.importorskip('sklearn.ensemble')
    Xtr, ytr, Xte, yte = make_regression_data()
    ours = RandomForest(task='regression', n_estimators=50, max_features=3).fit(Xtr, ytr)
    ref = sk.RandomForestRegressor(n_estimators=50, max_features=3, random_state=0).fit(Xtr, ytr)
    assert rmse(ours.predict(Xte), yte) < 1.2 * rmse(ref.predict(Xte), yte)
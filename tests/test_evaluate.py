"""
Checks src/evaluate.py (our NumPy metrics) against scikit-learn.
sklearn is used here ONLY as a reference — never inside src/.

Run:  python -m pytest tests/test_evaluate.py -q
"""

import numpy as np
import pytest
from sklearn import metrics as skm

from src import evaluate as ev

RNG = np.random.default_rng(0)


def _reg_data():
    y = RNG.normal(10, 3, size=200)
    p = y + RNG.normal(0, 1, size=200)
    return y, p


def _clf_data(ties=False):
    y = (RNG.random(300) < 0.4).astype(int)
    s = y * 0.8 + RNG.normal(0, 1, size=300)
    if ties:
        s = np.round(s, 1)          # force many tied scores
    return y, s


def test_rmse():
    y, p = _reg_data()
    assert ev.rmse(y, p) == pytest.approx(skm.root_mean_squared_error(y, p))


def test_mae():
    y, p = _reg_data()
    assert ev.mae(y, p) == pytest.approx(skm.mean_absolute_error(y, p))


def test_r2():
    y, p = _reg_data()
    assert ev.r2(y, p) == pytest.approx(skm.r2_score(y, p))


def test_r2_constant_target():
    y = np.ones(5)
    assert ev.r2(y, y) == 1.0


def test_confusion_matrix():
    y, s = _clf_data()
    pred = (s > 0.4).astype(int)
    assert np.array_equal(ev.confusion_matrix(y, pred),
                          skm.confusion_matrix(y, pred, labels=[0, 1]))


def test_precision_recall_f1():
    y, s = _clf_data()
    pred = (s > 0.4).astype(int)
    p, r, f = ev.precision_recall_f1(y, pred)
    assert p == pytest.approx(skm.precision_score(y, pred))
    assert r == pytest.approx(skm.recall_score(y, pred))
    assert f == pytest.approx(skm.f1_score(y, pred))


def test_prf_zero_division():
    y = np.array([0, 0, 1, 1])
    pred = np.zeros(4, dtype=int)         # never predicts positive
    assert ev.precision_recall_f1(y, pred) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("ties", [False, True])
def test_roc_auc(ties):
    y, s = _clf_data(ties)
    assert ev.roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))


def test_roc_auc_needs_both_classes():
    with pytest.raises(ValueError):
        ev.roc_auc(np.ones(4), np.arange(4))


@pytest.mark.parametrize("ties", [False, True])
def test_roc_curve_matches_sklearn_auc(ties):
    y, s = _clf_data(ties)
    fpr, tpr = ev.roc_curve(y, s)
    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0)   # works on numpy 1.x and 2.x
    assert area == pytest.approx(skm.roc_auc_score(y, s))
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1


def test_kfold_partition_and_determinism():
    y = (RNG.random(103) < 0.3).astype(int)
    a = ev.kfold_indices(103, k=5, seed=1, y=y)
    b = ev.kfold_indices(103, k=5, seed=1, y=y)
    all_val = np.concatenate([v for _, v in a])
    assert np.array_equal(np.sort(all_val), np.arange(103))   # each row once
    for (tr, va), (tr2, va2) in zip(a, b):
        assert np.array_equal(va, va2)
        assert len(np.intersect1d(tr, va)) == 0
    # stratified: class share per fold close to global share
    for _, va in a:
        assert abs(y[va].mean() - y.mean()) < 0.06


def test_compare_table_formats_mean_std():
    txt = ev.compare_table({"A": {"f1": (0.9, 0.01), "auc": 0.95},
                            "B": {"f1": 0.8}})
    assert "0.9000 ± 0.0100" in txt and "-" in txt

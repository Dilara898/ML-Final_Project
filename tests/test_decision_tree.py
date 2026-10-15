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
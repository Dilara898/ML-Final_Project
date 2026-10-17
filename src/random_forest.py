'''
random_forest.py - Bonus 9: a random forest built from OUR OWN decision trees.

Two sources of randomness make the trees different from each other:
  1. bootstrap: every tree is trained on a random sample (with repetition) of the rows
  2. feature subsampling: at every split a tree only looks at a random subset of columns
The forest averages the trees (regression) or averages their class probabilities
(classification). Averaging many different trees lowers the variance of one deep tree.
'''

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        task: str = 'classification',
        criterion: str = 'gini',
        n_estimators: int = 50,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features='sqrt',
        bootstrap: bool = True,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees_: list[DecisionTree] = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        if self.task == 'classification':
            self.classes_ = np.unique(y)

        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            tree = DecisionTree(
                task=self.task,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)

        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        return self

    # --- what every single tree says ---------------------------------------
    def _tree_outputs(self, X):
        '''Regression: (n_trees, n) predictions. Classification: (n_trees, n, K) probabilities.'''
        X = np.asarray(X, dtype=float)
        if self.task == 'regression':
            return np.array([t.predict(X) for t in self.trees_])
        out = np.zeros((len(self.trees_), len(X), len(self.classes_)))
        for i, t in enumerate(self.trees_):
            cols = np.searchsorted(self.classes_, t.classes_)   # a bootstrap may miss a class
            out[i][:, cols] = t.predict_proba(X)
        return out

    def _finish(self, mean_out):
        if self.task == 'regression':
            return mean_out
        return self.classes_[np.argmax(mean_out, axis=-1)]

    def predict(self, X):
        return self._finish(self._tree_outputs(X).mean(axis=0))

    def predict_proba(self, X):
        if self.task != 'classification':
            raise ValueError('predict_proba is only for classification')
        return self._tree_outputs(X).mean(axis=0)

    def staged_predict(self, X):
        '''Predictions using the first 1, 2, ..., n_estimators trees (for the error-vs-trees plot).'''
        outs = self._tree_outputs(X)
        cum = np.cumsum(outs, axis=0) / np.arange(1, len(outs) + 1).reshape((-1,) + (1,) * (outs.ndim - 1))
        return [self._finish(c) for c in cum]
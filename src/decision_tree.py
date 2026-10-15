"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------
    
    def _crit(self) -> str:
        '''Regression always uses MSE; classification uses gini or entropy.'''
        return 'mse' if self.task == 'regression' else self.criterion

    def _impurity(self, y) -> float:
        '''Gini / entropy (classification, y = integer labels) or variance (regression).'''
        y = np.asarray(y)
        if len(y) == 0:
            return 0.0
        crit = self._crit()
        if crit == 'mse':
            return float(np.var(y))
        p = np.bincount(y.astype(int)) / len(y)          # class fractions
        if crit == 'gini':
            return float(1.0 - np.sum(p ** 2))
        p = p[p > 0]                                      # entropy: skip log(0)
        return float(-np.sum(p * np.log2(p)))

        
    def _best_split(self, X, y):
        '''
        Return (feature, threshold, decrease) for the split with the largest
        impurity decrease, or None if no valid split exists.

        Fast method: sort each feature ONCE, then use cumulative sums so every
        candidate threshold is scored in O(1) instead of re-scanning the data.
        '''
        n, d = X.shape
        crit = self._crit()
        parent_imp = self._impurity(y)

        if crit == 'mse':
            yc = y.astype(float) - y.mean()               # centred -> stable variance
            tot, tot2 = yc.sum(), (yc ** 2).sum()
        else:
            k = int(y.max()) + 1
            onehot = np.eye(k)[y.astype(int)]             # (n, k) 0/1 matrix
            tot_counts = onehot.sum(axis=0)

        best = None
        best_dec = -np.inf

        for j in range(d):
            order = np.argsort(X[:, j], kind='mergesort')
            xs = X[order, j]

            n_left = np.arange(1, n)                      # left size if we cut after row i
            n_right = n - n_left
            valid = xs[:-1] != xs[1:]                     # can only cut between different values
            if not valid.any():
                continue

            if crit == 'mse':
                ys = yc[order]
                cs = np.cumsum(ys)[:-1]
                cs2 = np.cumsum(ys ** 2)[:-1]
                left_imp = cs2 / n_left - (cs / n_left) ** 2
                right_imp = (tot2 - cs2) / n_right - ((tot - cs) / n_right) ** 2
                left_imp = np.maximum(left_imp, 0.0)
                right_imp = np.maximum(right_imp, 0.0)
            else:
                left_c = np.cumsum(onehot[order], axis=0)[:-1]
                right_c = tot_counts - left_c
                pl = left_c / n_left[:, None]
                pr = right_c / n_right[:, None]
                if crit == 'gini':
                    left_imp = 1.0 - (pl ** 2).sum(axis=1)
                    right_imp = 1.0 - (pr ** 2).sum(axis=1)
                else:  # entropy
                    with np.errstate(divide='ignore', invalid='ignore'):
                        left_imp = -np.where(pl > 0, pl * np.log2(pl), 0.0).sum(axis=1)
                        right_imp = -np.where(pr > 0, pr * np.log2(pr), 0.0).sum(axis=1)

            weighted = (n_left * left_imp + n_right * right_imp) / n
            decrease = parent_imp - weighted
            decrease = np.where(valid, decrease, -np.inf)

            i = int(np.argmax(decrease))                  # first best cut for this feature
            if decrease[i] > best_dec:
                thr = (xs[i] + xs[i + 1]) / 2.0           # midpoint between neighbours
                if thr >= xs[i + 1]:                      # float safety
                    thr = xs[i]
                best = (j, float(thr), float(decrease[i]))
                best_dec = decrease[i]

        return best

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        raise NotImplementedError("TODO: predict_proba (classification)")

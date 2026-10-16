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
        self.n_samples = 0       # how many training rows reached this node
        self.impurity = 0.0      # impurity of the rows at this node

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
            valid = valid & (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
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

    def _leaf_value(self, y):
        if self._crit() == 'mse':
            return float(np.mean(y))
        return np.bincount(y.astype(int), minlength=self._n_classes) / len(y)

    def _grow(self, X, y, depth: int) -> Node:
        '''Recursively build the tree, respecting the stopping rules.'''
        node = Node()
        node.n_samples = len(y)
        node.impurity = self._impurity(y)
        node.value = self._leaf_value(y)

        # stopping rules -> this node becomes a leaf
        if node.impurity <= 1e-12:                                    # already pure
            return node
        if len(y) < self.min_samples_split:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, decrease = split
        # weighted decrease must be big enough (same definition as sklearn)
        if len(y) / self._n_total * decrease < self.min_impurity_decrease:
            return node

        self._importances[feature] += len(y) / self._n_total * decrease

        mask = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

 
    def fit(self, X, y):
        if self.task not in ('classification', 'regression'):
            raise ValueError('task must be classification or regression')
        if self.task == 'classification' and self.criterion not in ('gini', 'entropy'):
            raise ValueError('classification needs criterion gini or entropy')

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == 'classification':
            self.classes_, y = np.unique(y, return_inverse=True)   # labels -> 0..k-1
            self._n_classes = len(self.classes_)
        else:
            y = y.astype(float)

        self._n_total = len(y)
        self._importances = np.zeros(X.shape[1])
        self.root = self._grow(X, y, depth=0)

        total = self._importances.sum()
        self.feature_importances_ = self._importances / total if total > 0 else self._importances
        return self

 
    def _predict_one(self, x, node: Node):
        '''Walk from the root to a leaf and return the leaf's value.'''
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        values = [self._predict_one(x, self.root) for x in X]
        if self.task == 'regression':
            return np.array(values, dtype=float)
        return self.classes_[np.argmax(np.vstack(values), axis=1)]

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""

    def get_depth(self) -> int:
        def depth(node):
            return 0 if node.is_leaf() else 1 + max(depth(node.left), depth(node.right))
        return depth(self.root)

    def get_n_leaves(self) -> int:
        def leaves(node):
            return 1 if node.is_leaf() else leaves(node.left) + leaves(node.right)
        return leaves(self.root)

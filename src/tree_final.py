'''
tree_final.py - ONE place for the final tree / forest settings chosen on the VALIDATION set
(see report/tree_tuning_results.md). run_all.py should import these, so that every script
uses exactly the same settings.

Task A predicts log(price): compare predictions with np.log(y_test).
Task B predicts the tier label (0 = standard, 1 = premium).
'''

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTree
from .random_forest import RandomForest

BEST_A = {'max_depth': 20, 'min_samples_leaf': 20}
BEST_B = {'criterion': 'entropy', 'max_depth': 20, 'min_samples_leaf': 5}

# The forest settings are fixed choices (not tuned): many deep-ish trees, sqrt(d) features per split.
FOREST_A = {'n_estimators': 50, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'random_state': 42}
FOREST_B = {'n_estimators': 50, 'max_features': 'sqrt', 'min_samples_leaf': 5,
            'criterion': 'entropy', 'random_state': 42}


def fit_trees(X_train, price_train, tier_train):
    '''Fit the two final trees on the TRAINING set only. Returns (tree_a, tree_b).'''
    tree_a = DecisionTree(task='regression', **BEST_A).fit(X_train, np.log(np.asarray(price_train, float)))
    tree_b = DecisionTree(task='classification', **BEST_B).fit(X_train, tier_train)
    return tree_a, tree_b


def fit_forests(X_train, price_train, tier_train):
    '''Fit the two random forests (Bonus 9) on the TRAINING set only. Returns (forest_a, forest_b).'''
    forest_a = RandomForest(task='regression', **FOREST_A).fit(X_train, np.log(np.asarray(price_train, float)))
    forest_b = RandomForest(task='classification', **FOREST_B).fit(X_train, tier_train)
    return forest_a, forest_b
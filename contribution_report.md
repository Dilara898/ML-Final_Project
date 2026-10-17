# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Askerov, Agil                   | ---           | Data loading, cleaning, feature engineering, and EDA pipeline                   |
| Rzazada, Dilara                   | ---           | From-scratch Decision Tree, depth tuning, and Random Forest (Bonus 9)                   |
|Aliyeva, Leyla                  | ---           |From-scratch Pegasos SVM solver, RFF mapping, and hyperparameter sweeps                |
| Gurbanzade, Mahammad                    | ---           |Evaluation metrics, run_all.py pipeline integration, and report coordination                  |

## Per-member summary
**Askerov, Agil** — Designed and implemented the complete data preprocessing pipeline in `src/data_prep.py`. Handled column translations, numerical parsing of floor/area features, target leakage removal, and missing value imputations. Created the main EDA figures in `src/eda.py` and built the stratified train/validation/test split logic.

**Rzazade, Dilara** — Implemented the decision tree algorithm from scratch in `src/decision_tree.py` with support for Gini, Entropy, and MSE criteria alongside stopping rules (`max_depth`, `min_samples_split`, `min_samples_leaf`). Developed `src/tree_experiments.py` and `src/tree_tuning.py` to produce validation depth sweeps and learning curves. Implemented Bonus 9 in `src/random_forest.py` and `src/forest_experiment.py`, demonstrating an improved ROC-AUC of 0.9750.

**Aliyeva, Leyla** — Developed the soft-margin Pegasos SVM solver from scratch in `src/svm.py` using sub-gradient descent on the regularized hinge loss objective. Implemented learning-rate schedules, Random Fourier Features (RFF) for non-linear mapping, and objective logging. Executed hyperparameter sweeps over $\lambda$ and iterations (`src/svm_sweep.py`), analyzing support vector shares and margin widths.

**Gurbanzade, Mahammad** — Implemented all evaluation functions (`rmse`, `mae`, `r2`, `confusion_matrix`, `roc_auc`) in `src/evaluate.py` and verified them against Scikit-Learn baselines. Integrated the master execution pipeline in `src/run_all.py` with CLI flags (`--no-cv`, `--check-seeds`) and 5-fold cross-validation. Led the compilation of the IEEE LaTeX report (`report/report.tex`) and team presentation slides.

## Rough effort split
Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member | Approx. % |
|--------|-----------|
| Askerov, Agil  | 25%      |
| Rzazade, Dilara   | 25%       |
| Aliyeva, Leyla  | 25%       |
| Gurbanzade, Mahammad    | 25%       |

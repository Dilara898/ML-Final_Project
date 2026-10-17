# MLE-AI-201 Machine Learning — Final Project

Two models written **from scratch in NumPy** — a decision tree and a Pegasos
SVM — benchmarked honestly against scikit-learn on the **bina.az** sale
listings.

- **Task A — regression:** predict `price` (modelled as `log(price)`).
- **Task B — classification:** premium vs standard price tier; the threshold
  is the **training-set median** (never val/test).

scikit-learn is used only for the baselines and in `tests/` as a reference.

## Reproduce everything (one command)
```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.10-3.12
pip install -r requirements.txt

# 1) put the dataset in data/ (see data/README.md), e.g. data/bina_az_sale.csv
# 2) run:
python -m src.run_all
```
This prints every headline table (test metrics, 5-fold CV mean ± std, timing,
SVM margin report) and writes all figures to `report/figures/` and the raw
numbers to `results/results.json`.

| Command | What it does |
|---|---|
| `python -m src.run_all` | full run: tables + figures + cross-validation |
| `python -m src.run_all --no-cv` | same without CV (fast check) |
| `python -m src.run_all --check-seeds` | runs the pipeline twice, fails if any test metric differs |
| `python -m pytest -q` | unit tests (metrics vs sklearn, tree, SVM) |

Dataset: Kaggle `sehriyarmemmedli/binaaz-sale-project` — downloaded on
`<DATE>`, version `<VERSION/hash>` *(fill in; the raw archive is not committed)*.

## Reproducibility
- All randomness is seeded from `data_prep.SEED = 42` (split, folds, SGD,
  Pegasos sampling, sklearn `random_state`).
- Dependencies are pinned in `requirements.txt`.
- The test set is used **only** for final numbers. Tier threshold,
  standardization statistics and hyperparameters come from training /
  validation data only. In cross-validation they are recomputed per fold.
- Fair comparison: identical split and identical CV folds for our models and
  the sklearn baselines; sklearn scaling lives inside a `Pipeline`.

## Layout
```
src/data_prep.py      loading, cleaning, split, tier label, standardize
src/decision_tree.py  DecisionTree (Gini / entropy / MSE)
src/svm.py            PegasosSVM
src/evaluate.py       metrics, ROC, k-fold, compare table (NumPy only)
src/plots.py          confusion / ROC / error analysis / importance figures
src/run_all.py        one-command pipeline
tests/                unit tests
report/               IEEE report + figures/
presentation/         slides PDF
contribution_report.md
```

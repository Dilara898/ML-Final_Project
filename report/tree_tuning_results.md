# Decision tree tuning (all numbers on the VALIDATION set)

## Task A (regression, log price)

### Depth sweep (min_samples_leaf = 1)

| max_depth | depth | leaves | train_rmse | val_rmse | fit_s |
|---|---|---|---|---|---|
| 2 | 2 | 4 | 0.6776 | 0.6817 | 0.7 |
| 4 | 4 | 16 | 0.5342 | 0.5443 | 1.7 |
| 6 | 6 | 64 | 0.455 | 0.4702 | 2.1 |
| 8 | 8 | 236 | 0.394 | 0.4321 | 3.6 |
| 10 | 10 | 746 | 0.3339 | 0.4265 | 5.2 |
| 12 | 12 | 1856 | 0.2825 | 0.4271 | 10.3 |
| 15 | 15 | 5194 | 0.2099 | 0.4343 | 20.4 |
| 20 | 20 | 15093 | 0.116 | 0.4441 | 55.8 |

### min_samples_leaf sweep (max_depth = 20)

| min_samples_leaf | depth | leaves | train_rmse | val_rmse | fit_s |
|---|---|---|---|---|---|
| 5 | 20 | 5185 | 0.2568 | 0.3978 | 32.6 |
| 10 | 20 | 2925 | 0.3019 | 0.392 | 20.9 |
| 20 | 20 | 1583 | 0.3437 | 0.3875 | 12.8 |
| 50 | 20 | 663 | 0.3824 | 0.4165 | 7.5 |
| 100 | 20 | 346 | 0.4121 | 0.4354 | 4.7 |

**Best on validation:** max_depth = 20, min_samples_leaf = 20, val_rmse = 0.3875

## Task B (classification, gini)

### Depth sweep (min_samples_leaf = 1)

| max_depth | depth | leaves | train_acc | val_acc | fit_s |
|---|---|---|---|---|---|
| 2 | 2 | 4 | 0.8183 | 0.8165 | 1.8 |
| 4 | 4 | 16 | 0.861 | 0.8605 | 3.3 |
| 6 | 6 | 55 | 0.8849 | 0.8795 | 4.4 |
| 8 | 8 | 170 | 0.905 | 0.898 | 5.1 |
| 10 | 10 | 439 | 0.9232 | 0.9069 | 6.7 |
| 12 | 12 | 872 | 0.9413 | 0.9143 | 8.7 |
| 15 | 15 | 1706 | 0.9679 | 0.9181 | 11.7 |
| 20 | 20 | 2830 | 0.9927 | 0.9175 | 15.7 |

### min_samples_leaf sweep (max_depth = 20)

| min_samples_leaf | depth | leaves | train_acc | val_acc | fit_s |
|---|---|---|---|---|---|
| 5 | 20 | 1775 | 0.958 | 0.9186 | 13.1 |
| 10 | 20 | 1197 | 0.9427 | 0.916 | 10.8 |
| 20 | 20 | 759 | 0.9283 | 0.9112 | 8.6 |
| 50 | 20 | 387 | 0.9124 | 0.9032 | 7.8 |
| 100 | 18 | 243 | 0.9023 | 0.8941 | 4.3 |

**Best on validation:** max_depth = 20, min_samples_leaf = 5, val_acc = 0.9186

### Gini vs entropy (best settings)

| criterion | val_acc | val_f1 | val_auc | fit_s |
|---|---|---|---|---|
| gini | 0.9186 | 0.9175 | 0.9519 | 14.5 |
| entropy | 0.9221 | 0.9211 | 0.9551 | 14.4 |

## Comparison with scikit-learn (same settings)

| task | model | val_rmse / val_acc | val_r2 / val_f1 | fit_s |
|---|---|---|---|---|
| A | ours | 0.3875 | 0.7739 | 11.2 |
| A | sklearn | 0.3876 | 0.7738 | 0.6 |
| B | ours | 0.9221 | 0.9211 | 14.2 |
| B | sklearn | 0.9189 | 0.9178 | 0.5 |


# Random forest (Bonus 9), validation set

| task | model | RMSE | MAE | R2 | acc | F1 | AUC | fit_s |
|---|---|---|---|---|---|---|---|---|
| A | our single tree | 0.3875 | 0.2008 | 0.7739 |  |  |  | 24.6 |
| A | our random forest | 0.4334 | 0.2303 | 0.7172 |  |  |  | 161.9 |
| A | sklearn random forest | 0.4349 | 0.2274 | 0.7153 |  |  |  | 6.9 |
| B | our single tree |  |  |  | 0.9221 | 0.9211 | 0.9551 | 35.6 |
| B | our random forest |  |  |  | 0.915 | 0.9142 | 0.975 | 204.5 |
| B | sklearn random forest |  |  |  | 0.9177 | 0.9167 | 0.9767 | 7.0 |

Forest settings: 50 trees, max_features=sqrt, min_samples_leaf=5.

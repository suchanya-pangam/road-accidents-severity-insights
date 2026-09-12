# Modeling Methodology

## Scope

This is an academic project. The main result is the Power BI dashboard, which helps users explore road-accident patterns. The notebook and model code are included to show the analysis and machine-learning experiments behind the project. They are not a deployed prediction system.

## Reproducibility settings

`random_state=42` is used in data splitting, cross-validation, sampling, and stochastic models. The number 42 is not claimed to improve performance. It is a fixed seed that makes the experimental procedure reproducible. Any fixed integer would serve this purpose if it were used consistently.

## Evaluation protocol

The notebook uses a stratified 80:20 train-test split. The training portion is evaluated with stratified 10-fold cross-validation, and the test portion remains unseen until final evaluation. Stratification is used because accident-severity classes are imbalanced. Weighted F1-score is the primary selection metric because it combines precision and recall while reflecting overall performance across the dataset.

SMOTE, random oversampling, and random undersampling are applied only inside training folds. This prevents information from validation or test observations entering the resampling step.

## Evidence used to design the XGBoost search

The parameter values below are **search candidates**, not universal best settings. They must be selected using cross-validation on this project's data rather than copied unchanged from another study.

| Parameter | Project search candidates | Purpose |
|---|---:|---|
| `sampler__k_neighbors` | 3, 5, 7 | Controls the neighbourhood used by SMOTE. |
| `model__n_estimators` | 200, 300, 500 | Controls the number of boosting rounds. |
| `model__max_depth` | 3, 4, 5, 6 | Limits tree complexity to reduce overfitting. |
| `model__learning_rate` | 0.03, 0.05, 0.10 | Controls the step size of each boosting update. |
| `model__min_child_weight` | 1, 3, 5 | Limits splits supported by too little data. |
| `model__subsample` | 0.70, 0.85, 1.00 | Uses a fraction of training observations per tree. |
| `model__colsample_bytree` | 0.70, 0.85, 1.00 | Uses a fraction of features per tree. |
| `model__gamma` | 0, 0.1, 0.3 | Requires a minimum improvement before splitting. |
| `model__reg_lambda` | 1, 5, 10 | Applies L2 regularisation. |

The notebook includes a small XGBoost parameter search for learning purposes. The reproducible baseline recorded in `reports/model_results.csv` compares the models using the settings in `src/model_training.py` and selects the model with the highest mean weighted F1-score.

## Related road-accident-severity evidence

Chen et al. (2025) studied road-traffic-accident severity prediction with XGBoost. They applied SMOTE to the training set only, used stratified cross-validation during optimisation, and searched XGBoost parameters including `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` [2]. Their reported optimum (`n_estimators=42`, `max_depth=35`, and `learning_rate=0.4643`) is documented for transparency but is **not adopted** here, because their Chinese accident dataset, DART booster, optimisation algorithm, and feature set differ from this project.

This project uses the research as supporting context instead of copying a reported optimum. The final choice is based on this project's own cross-validation results: XGBoost without resampling had the highest mean weighted F1-score (80.09%). Random Forest with Random Oversampling had the highest mean accuracy (84.94%), and XGBoost with SMOTE had the highest mean macro F1-score (42.55%). SMOTE is included as one way to handle class imbalance, while XGBoost is a tree-boosting method for supervised learning.

## Limitations

- Twelve random-search trials are a limited computational budget, so the chosen setting should be described as the best configuration found within the defined search space, not as the global optimum.
- Findings from another road-accident dataset may motivate a search strategy but do not prove that the same parameters are optimal for this dataset.
- The modelling appendix supports methodological learning; dashboard insights remain the project's primary deliverable.

## References

1. Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research, 13*(10), 281-305. https://jmlr.org/papers/v13/bergstra12a.html
2. Chen, F., Liu, X. Q., Yang, J. J., Liu, X. K., Ma, J. H., Chen, J., & Xiao, H. Y. (2025). Traffic accident severity prediction based on an enhanced MSCPO-XGBoost hybrid model. *Scientific Reports, 15*, 25729. https://doi.org/10.1038/s41598-025-00797-7
3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research, 16*, 321-357. https://doi.org/10.1613/jair.953
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

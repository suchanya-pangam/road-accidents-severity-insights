# Modeling Methodology

## Scope

This is an academic project. The main result is the Power BI dashboard, which helps users explore road-accident patterns. The notebook and model code are included to show the analysis and machine-learning experiments behind the project. They are not a deployed prediction system.

## Reproducibility settings

`random_state=42` is used in data splitting, cross-validation, sampling, and stochastic models. The number 42 is not claimed to improve performance. It is a fixed seed that makes the experimental procedure reproducible. Any fixed integer would serve this purpose if it were used consistently.

## Evaluation protocol

The notebook uses a stratified 80:20 train-test split. The training portion is evaluated with stratified 10-fold cross-validation, and the test portion remains unseen until final evaluation. Stratification is used because accident-severity classes are imbalanced. Weighted F1-score is the primary selection metric because it combines precision and recall while reflecting overall performance across the dataset.

SMOTE, random oversampling, and random undersampling are applied only inside training folds. This prevents information from validation or test observations entering the resampling step.

## Related research

Muktar and Fono (2024) compared XGBoost, CatBoost, Random Forest, and Gradient Boosting for traffic-accident severity prediction. Their study reported XGBoost as the best-performing model. This supports including XGBoost in this project's comparison, but the final model is still selected from this project's own results because the datasets and evaluation settings are different.

The reproducible baseline recorded in `reports/model_results.csv` selects the model with the highest mean weighted F1-score. XGBoost without resampling achieved the highest mean weighted F1-score (80.09%) and was selected as the final model. Random Forest with Random Oversampling had the highest mean accuracy (84.94%), while XGBoost with SMOTE had the highest mean macro F1-score (42.55%).

## Limitations

- Findings from another road-accident dataset support including a model in the comparison, but do not prove that it will be best for this dataset.
- The modelling appendix supports methodological learning; dashboard insights remain the project's primary deliverable.

## References

1. Muktar, B., & Fono, V. (2024). *Toward Safer Roads: Predicting the Severity of Traffic Accidents in Montreal Using Machine Learning*. Electronics, 13(15), 3036. https://doi.org/10.3390/electronics13153036

# Road Accident Severity Prediction and Dashboard

This is an academic Data Science project. I used exploratory data analysis, a Power BI dashboard, and machine-learning experiments to study factors related to road accident severity. The project looks at road conditions, vehicle types, driver characteristics, and casualty information.

## Project Highlights

- Cleaned and explored more than 12,000 road accident records.
- Built an interactive Power BI dashboard to explore accident-severity patterns.
- Compared Decision Tree, XGBoost, and Random Forest with different resampling methods. The cross-validation and held-out test results are in `reports/model_results.csv`.

## Project Layout

- `notebooks/notebook.ipynb` - Primary record of exploratory analysis and model experiments.
- `src/data_processing.py` - Loads, cleans, and exports the dataset.
- `src/feature_engineering.py` - Encodes categorical variables and prepares analysis fields.
- `src/visualization.py` - Creates correlation and dashboard-supporting visualizations.
- `src/model_training.py` and `src/models/` - Defines and evaluates the machine learning models.
- `tests/` - Automated tests for preprocessing and feature engineering.
- `reports/model_results.csv` - Cross-validation and held-out test results from the reproducible baseline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/notebook.ipynb
```

## Data

Place the original dataset at `data/raw/RTA Dataset.csv`.

The project uses the [Road Accidents Severity Prediction Kaggle notebook](https://www.kaggle.com/code/kanuriviveknag/road-accidents-severity-prediction) as its dataset reference. The raw and processed data directories are ignored by Git because the dataset may be subject to redistribution restrictions.

## Workflow

1. Clean the dataset by removing selected columns with extensive missing data and dropping remaining incomplete rows.
2. Prepare categorical features for analysis and modeling.
3. Explore accident-severity patterns and relationships between relevant variables.
4. Compare Decision Tree, XGBoost, and Random Forest models with different resampling methods.
5. Present findings through an interactive dashboard for clear and accessible analysis.

## Model Evaluation

The model experiments use a stratified 80:20 train-test split and evaluate Decision Tree, XGBoost, and Random Forest classifiers. Random Under-Sampling, Random Over-Sampling, and SMOTE are used to address class imbalance.

In the current reproducible baseline run, XGBoost with SMOTE had the highest mean macro F1-score in 10-fold cross-validation. Its held-out test accuracy was 73.22% and its held-out macro F1-score was 39.23%. The full comparison is in `reports/model_results.csv`.

The modeling work is an academic experiment and is not presented as a deployed prediction system.

Methodology details are available in [`docs/modeling-methodology.md`](docs/modeling-methodology.md).

## Live Dashboard

Explore the interactive Power BI dashboard here: [Road Accident Severity Dashboard](https://app.powerbi.com/view?r=eyJrIjoiMDRiOWEwNGUtYmNiNC00NzUzLWFkOWYtZDAxOTMzOTkwNjQ5IiwidCI6ImNmODFmMWRmLWRlNTktNGMyOS05MWRhLWEyZGZkMDRhYTc1MSIsImMiOjEwfQ%3D%3D).

The dashboard supports interactive exploration of accident-severity patterns, road conditions, vehicle types, driver characteristics, and casualty information.

## License

The project code uses the MIT License. The road-accident dataset is not included in Git because it may have separate redistribution terms. Please check the dataset source and its terms before using it.

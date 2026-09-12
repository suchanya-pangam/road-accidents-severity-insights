# Road Accident Severity Prediction and Dashboard

This academic Data Science project combines exploratory data analysis, an interactive Power BI dashboard, and machine learning experiments to study factors associated with road accident severity. It examines patterns in road conditions, vehicle types, driver characteristics, and casualties, then compares classification models for severity prediction.

## Project Highlights

- Cleaned and analyzed more than 12,000 road accident records.
- Built an interactive Power BI dashboard for exploring accident-severity patterns and related factors.
- Compared Decision Tree, XGBoost, and Random Forest models using Random Under-Sampling, Random Over-Sampling, and SMOTE.
- The reproducible baseline compares Decision Tree, XGBoost, and Random Forest with several resampling methods. Detailed cross-validation and held-out test results are in `reports/model_results.csv`.

## Project Layout

- `notebooks/notebook.ipynb` - Primary record of exploratory analysis and model experiments.
- `src/data_processing.py` - Loads, cleans, and exports the dataset.
- `src/feature_engineering.py` - Encodes categorical variables and prepares analysis fields.
- `src/visualization.py` - Creates correlation and dashboard-supporting visualizations.
- `src/model_training.py` and `src/models/` - Defines and evaluates the machine learning models.
- `tests/` - Automated tests for preprocessing and feature engineering.
- `reports/model_results.csv` - Template for recording model results.

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

On the current reproducible baseline run, XGBoost with SMOTE achieved the highest mean macro F1-score during 10-fold cross-validation. Its held-out test accuracy was 73.22% and held-out macro F1-score was 39.23%. The full comparison is recorded in `reports/model_results.csv`.

The modeling work is an academic experiment and is not presented as a deployed prediction system.

Methodology details are available in [`docs/modeling-methodology.md`](docs/modeling-methodology.md).

## Live Dashboard

Explore the interactive Power BI dashboard here: [Road Accident Severity Dashboard](https://app.powerbi.com/view?r=eyJrIjoiMDRiOWEwNGUtYmNiNC00NzUzLWFkOWYtZDAxOTMzOTkwNjQ5IiwidCI6ImNmODFmMWRmLWRlNTktNGMyOS05MWRhLWEyZGZkMDRhYTc1MSIsImMiOjEwfQ%3D%3D).

The dashboard supports interactive exploration of accident-severity patterns, road conditions, vehicle types, driver characteristics, and casualty information.

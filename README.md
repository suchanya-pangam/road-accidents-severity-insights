# Road Accident Severity Prediction

A machine-learning project that predicts accident severity from road, vehicle, driver, and casualty information.

## Project layout

- `notebooks/notebook.ipynb` — exploratory analysis and the original experiments.
- `src/data_processing.py` — data loading, inspection, cleaning, and export steps.
- `src/feature_engineering.py` — categorical encoding, feature selection, and feature importance.
- `src/visualization.py` — correlation heatmap generation.
- `src/models/` — Decision Tree, Random Forest, XGBoost, training setup, and evaluation code.
- `data/raw/` — source data. Put `Dataset.csv` here.
- `data/processed/` — generated cleaned data.
- `reports/` — generated figures and model comparison results.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Data

Place the original dataset at `data/raw/Dataset.csv`. Do not commit data unless its license permits redistribution.

## Important note

The code in `src/` was copied verbatim from the original notebook, as requested. It has not yet been refactored into standalone modules. Run the notebook to reproduce the original workflow.

## Models evaluated

- Decision Tree
- Random Forest
- XGBoost

The notebook compares random oversampling, random undersampling, and SMOTE with 10-fold cross-validation.

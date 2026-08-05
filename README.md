# Road Accident Severity Dashboard

This project analyses road-accident data through an interactive dashboard. It is structured for a third-year university data-analysis project and presents patterns in accident severity, road conditions, vehicle types, driver characteristics, and casualties.

## Project layout

- `notebooks/notebook.ipynb` — the primary record of exploratory analysis.
- `src/data_processing.py` — loading, cleaning, and exporting the dataset.
- `src/feature_engineering.py` — categorical encoding and preparation of analysis fields.
- `src/visualization.py` — correlation and dashboard-supporting visualisations.
- `src/model_training.py` and `src/models/` — archived notebook-derived scripts; they are not part of the dashboard project scope.
- `tests/` — automated tests for preprocessing and feature engineering.
- `reports/model_results.csv` — header-only template for recorded model results.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/notebook.ipynb
```

## Data

Place the original dataset at `data/raw/RTA Dataset.csv`. The project uses the [Road Accidents Severity Prediction Kaggle notebook](https://www.kaggle.com/code/kanuriviveknag/road-accidents-severity-prediction) by `kanuriviveknag` as its dataset reference. The raw and processed data directories are ignored by Git because the dataset may be subject to redistribution restrictions.

## Project scope

This project focuses on exploratory data analysis and dashboard development. It does not present or evaluate a predictive model. The dashboard helps users explore patterns in accident severity, road conditions, vehicle types, driver characteristics, and casualty information.

## Workflow

1. Clean the dataset by removing selected columns with extensive missing data and dropping remaining incomplete rows.
2. Prepare the data for analysis by encoding categorical variables where necessary.
3. Explore accident-severity patterns and relationships between relevant variables.
4. Present the findings through an interactive dashboard designed for clear and accessible analysis.

## Live dashboard

Explore the interactive Power BI dashboard here: [Road Accident Severity Dashboard](https://app.powerbi.com/view?r=eyJrIjoiMDRiOWEwNGUtYmNiNC00NzUzLWFkOWYtZDAxOTMzOTkwNjQ5IiwidCI6ImNmODFmMWRmLWRlNTktNGMyOS05MWRhLWEyZGZkMDRhYTc1MSIsImMiOjEwfQ%3D%3D).

The dashboard is the project’s main deliverable. It supports interactive exploration of accident-severity patterns, road conditions, vehicle types, driver characteristics, and casualty information.

## Methodology evidence

The modelling appendix and its research-supported parameter-search approach are documented in [`docs/modeling-methodology.md`](docs/modeling-methodology.md).

## Dashboard capabilities

- Explore the distribution of accident-severity levels.
- Examine relationships between severity, road conditions, vehicle types, and driver characteristics.
- Filter data interactively to investigate specific accident patterns.
- Support clear, evidence-based discussion of road-safety insights.

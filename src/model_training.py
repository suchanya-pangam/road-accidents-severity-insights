"""Compare, select, and evaluate accident-severity classification models."""

from pathlib import Path

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from src.feature_engineering import SELECTED_FEATURES
except ModuleNotFoundError:
    # Support running this file directly from the src directory.
    from feature_engineering import SELECTED_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "reports" / "model_results.csv"
TARGET_COLUMN = "Accident_severity"


def compare_models(data: pd.DataFrame) -> tuple[pd.DataFrame, object, object, tuple]:
    """Compare notebook candidates and select the highest weighted F1-score."""
    features = data[SELECTED_FEATURES]
    target = data[TARGET_COLUMN]
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    candidates = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=42),
    }
    samplers = {
        "No sampling": None,
        "Random oversampling": RandomOverSampler(random_state=42),
        "Random undersampling": RandomUnderSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
    }
    validator = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    for sampler_name, sampler in samplers.items():
        for model_name, candidate in candidates.items():
            scores = []
            for train_index, validation_index in validator.split(train_features, train_target):
                fold_features = train_features.iloc[train_index]
                fold_target = train_target.iloc[train_index]
                if sampler is not None:
                    fold_features, fold_target = clone(sampler).fit_resample(fold_features, fold_target)
                predictions = clone(candidate).fit(fold_features, fold_target).predict(train_features.iloc[validation_index])
                validation_target = train_target.iloc[validation_index]
                scores.append({
                    "Mean Accuracy": accuracy_score(validation_target, predictions),
                    "Mean Weighted F1-score": f1_score(validation_target, predictions, average="weighted"),
                    "Mean Macro F1-score": f1_score(validation_target, predictions, average="macro"),
                })
            results.append({"Model": model_name, "Sampling method": sampler_name, **pd.DataFrame(scores).mean().to_dict()})
    results_frame = pd.DataFrame(results).sort_values("Mean Weighted F1-score", ascending=False).reset_index(drop=True)
    chosen_model = candidates[results_frame.loc[0, "Model"]]
    chosen_sampler = samplers[results_frame.loc[0, "Sampling method"]]
    return results_frame, chosen_model, chosen_sampler, (train_features, test_features, train_target, test_target)


def train_and_evaluate_selected_model(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Select a model by weighted F1-score and evaluate it on held-out data."""
    results, model, sampler, split_data = compare_models(data)
    train_features, test_features, train_target, test_target = split_data
    if sampler is not None:
        train_features, train_target = clone(sampler).fit_resample(train_features, train_target)
    predictions = clone(model).fit(train_features, train_target).predict(test_features)
    metrics = {
        "Final test accuracy": accuracy_score(test_target, predictions),
        "Final test macro F1-score": f1_score(test_target, predictions, average="macro"),
    }
    return results, metrics

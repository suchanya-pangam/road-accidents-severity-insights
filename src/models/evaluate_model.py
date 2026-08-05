"""Evaluate classifiers with the metrics used in the notebook."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Return accuracy, weighted metrics, and macro F1-score."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Weighted precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Weighted F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Macro F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

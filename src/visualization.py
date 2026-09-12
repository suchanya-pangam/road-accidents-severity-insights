"""Create and save visualisations used in the dashboard analysis."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split

try:
    from src.feature_engineering import SELECTED_FEATURES
except ModuleNotFoundError:
    from feature_engineering import SELECTED_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset encoded.csv"
FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "xgboost_feature_importance.csv"
TARGET_COLUMN = "Accident_severity"


def plot_correlation_heatmap(
    data: pd.DataFrame, output_path: Optional[Path] = None
) -> None:
    """Display and optionally save a heatmap of numeric-variable correlations."""
    correlation_matrix = data.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axis)
    axis.set_title("Correlation Matrix of Variables")
    figure.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_feature_importance(
    feature_importance: pd.DataFrame, top_n: int = 10, output_path: Optional[Path] = None
) -> None:
    """Display and optionally save the most important features from a fitted model."""
    top_features = feature_importance.head(top_n)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
    axis.set_xlabel("Feature importance")
    axis.set_title("Top Feature Importances")
    figure.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def calculate_xgboost_feature_importance(data: pd.DataFrame) -> pd.DataFrame:
    """Fit the selected XGBoost baseline and return gain-based feature importance."""
    features = data[SELECTED_FEATURES]
    target = data[TARGET_COLUMN]
    train_features, _, train_target, _ = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = xgb.XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", random_state=42
    )
    model.fit(train_features, train_target)
    gain_scores = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame({
        "feature": SELECTED_FEATURES,
        "average_gain": [gain_scores.get(feature, 0.0) for feature in SELECTED_FEATURES],
    }).sort_values("average_gain", ascending=False, ignore_index=True)
    total_gain = importance["average_gain"].sum()
    importance["relative_gain"] = (
        importance["average_gain"] / total_gain if total_gain else 0.0
    )
    return importance


def save_xgboost_feature_importance(
    feature_importance: pd.DataFrame,
    csv_path: Path = FEATURE_IMPORTANCE_PATH,
    image_path: Optional[Path] = None,
) -> None:
    """Save a reusable CSV and chart for the selected XGBoost model."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(csv_path, index=False)
    if image_path is not None:
        figure, axis = plt.subplots(figsize=(9, 6))
        top_features = feature_importance.head(10).iloc[::-1]
        axis.barh(top_features["feature"], top_features["relative_gain"], color="#2E75B6")
        axis.set_xlabel("Relative gain")
        axis.set_title("XGBoost Feature Importance")
        figure.tight_layout()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    """Create the correlation heatmap and XGBoost feature-importance outputs."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Encoded data was not found at: {INPUT_PATH}. "
            "Run data_processing.py and feature_engineering.py first."
        )

    data = pd.read_csv(INPUT_PATH)
    output_path = FIGURES_DIRECTORY / "correlation_heatmap.png"
    plot_correlation_heatmap(data, output_path)
    feature_importance = calculate_xgboost_feature_importance(data)
    feature_image_path = FIGURES_DIRECTORY / "xgboost_feature_importance.png"
    save_xgboost_feature_importance(feature_importance, FEATURE_IMPORTANCE_PATH, feature_image_path)
    print(f"Saved correlation heatmap to: {output_path}")
    print(f"Saved XGBoost feature importance to: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()

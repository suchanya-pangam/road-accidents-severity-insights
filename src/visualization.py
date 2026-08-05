"""Create and save visualisations used in the dashboard analysis."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset encoded.csv"
FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"


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


def main() -> None:
    """Create the correlation heatmap from the encoded dataset."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Encoded data was not found at: {INPUT_PATH}. "
            "Run data_processing.py and feature_engineering.py first."
        )

    data = pd.read_csv(INPUT_PATH)
    output_path = FIGURES_DIRECTORY / "correlation_heatmap.png"
    plot_correlation_heatmap(data, output_path)
    print(f"Saved correlation heatmap to: {output_path}")


if __name__ == "__main__":
    main()

"""Train and evaluate a Random Forest accident-severity classifier."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset encoded.csv"
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_importance.csv"
TARGET_COLUMN = "Accident_severity"


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load encoded data and separate predictors from the target."""
    df = pd.read_csv(INPUT_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN]
    return X, y


def train_random_forest(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    """Split data, train Random Forest, and return the model and test data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def create_feature_importance(
    model: RandomForestClassifier, X: pd.DataFrame
) -> pd.DataFrame:
    """Return feature importance ordered from highest to lowest."""
    return pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)


if __name__ == "__main__":
    X, y = load_training_data()
    model, X_test, y_test = train_random_forest(X, y)
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Weighted F1-score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    feature_importance = create_feature_importance(model, X)
    FEATURE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print("\nTop 5 features:")
    print(feature_importance.head())
    print(f"Saved feature importance to: {FEATURE_IMPORTANCE_PATH}")

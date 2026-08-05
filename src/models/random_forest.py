"""Create the Random Forest model used in the notebook experiments."""

from sklearn.ensemble import RandomForestClassifier


def create_model() -> RandomForestClassifier:
    """Return the Random Forest configuration used for final comparison."""
    return RandomForestClassifier(n_estimators=300, random_state=42)

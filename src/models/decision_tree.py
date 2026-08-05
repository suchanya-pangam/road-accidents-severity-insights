"""Create the Decision Tree model used in the notebook experiments."""

from sklearn.tree import DecisionTreeClassifier


def create_model() -> DecisionTreeClassifier:
    """Return a reproducible Decision Tree classifier."""
    return DecisionTreeClassifier(random_state=42)

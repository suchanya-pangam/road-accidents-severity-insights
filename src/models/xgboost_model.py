"""Create the XGBoost model used in the notebook experiments."""

import xgboost as xgb


def create_model() -> xgb.XGBClassifier:
    """Return the multiclass XGBoost configuration used for final comparison."""
    return xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", random_state=42)

"""Shared stratified cross-validation configuration from the notebook."""

from sklearn.model_selection import StratifiedKFold

N_SPLITS = 10
RANDOM_STATE = 42


def create_cross_validator() -> StratifiedKFold:
    """Create the stratified 10-fold validator used in every experiment."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

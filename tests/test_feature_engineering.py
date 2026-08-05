"""Tests for categorical feature encoding."""

import pandas as pd

from src.feature_engineering import encode_categorical_features


def test_encode_categorical_features_encodes_text_columns_and_returns_mapping():
    """Text columns should be numeric after encoding and retain their labels."""
    data = pd.DataFrame({"Road_surface": ["Dry", "Wet"], "Vehicles": [1, 2]})

    encoded_data, mappings = encode_categorical_features(data)

    assert encoded_data["Road_surface"].dtype.kind in "iu"
    assert mappings["Road_surface"] == {"Dry": 0, "Wet": 1}

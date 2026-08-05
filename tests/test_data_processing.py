"""Tests for dataset cleaning."""

import pandas as pd

from src.data_processing import COLUMNS_TO_DROP, load_and_clean_data


def test_load_and_clean_data_removes_selected_columns_and_missing_rows(tmp_path):
    """Cleaning should remove configured columns and incomplete rows."""
    source = tmp_path / "data.csv"
    pd.DataFrame({
        "Defect_of_vehicle": ["Unknown", "None"],
        "Road_condition": ["Dry", None],
        "Accident_severity": ["Slight", "Serious"],
    }).to_csv(source, index=False)

    cleaned_data = load_and_clean_data(source)

    assert COLUMNS_TO_DROP[0] not in cleaned_data.columns
    assert cleaned_data.to_dict("records") == [{"Road_condition": "Dry", "Accident_severity": "Slight"}]

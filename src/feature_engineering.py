"""Encode categorical variables and define the notebook feature set."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset encoded.csv"
SELECTED_FEATURES = [
    "Area_accident_occured", "Day_of_week", "Lanes_or_Medians",
    "Road_surface_conditions", "Age_band_of_driver", "Light_conditions",
    "Type_of_vehicle", "Number_of_casualties", "Cause_of_accident",
    "Number_of_vehicles_involved", "Age_band_of_casualty",
    "Driving_experience", "Type_of_collision",
]


def encode_categorical_features(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Encode text columns and return the encoded data with label mappings."""
    encoded_data = data.copy()
    label_mappings: dict[str, dict[str, int]] = {}
    categorical_columns = encoded_data.select_dtypes(include=["object", "category"]).columns

    for column in categorical_columns:
        encoder = LabelEncoder()
        encoded_data[column] = encoder.fit_transform(encoded_data[column])
        label_mappings[column] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    return encoded_data, label_mappings


def save_encoded_data(data: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save encoded data as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    cleaned_data = pd.read_csv(INPUT_PATH)
    encoded_data, mappings = encode_categorical_features(cleaned_data)
    save_encoded_data(encoded_data)
    print(f"Saved encoded data to: {OUTPUT_PATH}")
    print("Accident severity label mapping:", mappings["Accident_severity"])

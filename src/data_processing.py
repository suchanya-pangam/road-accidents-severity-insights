"""Load, clean, and save the road-traffic-accident dataset."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "RTA Dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset cleaned.csv"
COLUMNS_TO_DROP = [
    "Defect_of_vehicle",
    "Service_year_of_vehicle",
    "Fitness_of_casuality",
    "Work_of_casuality",
]


def load_and_clean_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the dataset, remove selected columns, and drop incomplete rows."""
    data = pd.read_csv(input_path)
    data = data.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    return data.dropna().reset_index(drop=True)


def save_cleaned_data(data: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save cleaned data as a CSV file, creating its directory if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    cleaned_data = load_and_clean_data()
    save_cleaned_data(cleaned_data)
    print(f"Saved cleaned data to: {OUTPUT_PATH}")

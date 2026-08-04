from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "RTA Dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset cleaned.csv"

# Define columns to drop due to high missing values
COLUMNS_TO_DROP = [
    "Time",
    "Defect_of_vehicle",
    "Service_year_of_vehicle",
    "Fitness_of_casuality",
    "Work_of_casuality",
]


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)

    # Remove columns with many missing values
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Remove remaining rows with missing values
    df = df.dropna().reset_index(drop=True)

    return df


def save_cleaned_data(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    cleaned_df = load_and_clean_data()
    save_cleaned_data(cleaned_df)
    print(f"Saved cleaned data to: {OUTPUT_PATH}")

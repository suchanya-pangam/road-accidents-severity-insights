"""Encode categorical variables in the cleaned accident-severity dataset."""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "RTA Dataset encoded.csv"


def encode_categorical_features() -> pd.DataFrame:
    """Load cleaned data, encode text columns, and return the result."""
    df = pd.read_csv(INPUT_PATH)
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])

    return df


def save_encoded_data(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    encoded_df = encode_categorical_features()
    save_encoded_data(encoded_df)
    print(f"Saved encoded data to: {OUTPUT_PATH}")

import pandas as pd
import os
from sklearn.datasets import fetch_california_housing

RAW_DATA_PATH = "app/data/raw/housing.csv"

# Save the dataset to CSV
def save_housing_data_to_csv():
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ California housing dataset saved to {RAW_DATA_PATH}")

# Load dataset from CSV (used for training)
def load_housing_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at {RAW_DATA_PATH}. Run save_housing_data_to_csv() first.")
    df = pd.read_csv(RAW_DATA_PATH)
    return df.drop("MedHouseVal", axis=1), df["MedHouseVal"]

# Optional script usage
if __name__ == "__main__":
    save_housing_data_to_csv()

from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

# Load and save California Housing dataset
def save_housing_data_to_csv():
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    print("Current working directory:", os.getcwd())
    # Ensure the directory exists
    os.makedirs("app/data/raw", exist_ok=True)

    # Save to CSV
    df.to_csv("app/data/raw/housing.csv", index=False)
    print("California housing dataset saved to data/raw/housing.csv")

# Load dataset from sklearn (used for training in memory)
def load_housing_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df, data.target

if __name__ == "__main__":
    save_housing_data_to_csv()
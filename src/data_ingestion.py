import pandas as pd
import os

RAW_DATA_PATH = "data/raw/dataset"
PROCESSED_DATA_PATH = "data/processed/"

def load_sample(filename, n_rows=100000):
    file_path = os.path.join(RAW_DATA_PATH, filename)
    print(f"Loading first {n_rows} rows from {file_path}")
    df = pd.read_csv(file_path, nrows=n_rows)
    return df

def basic_inspection(df):
    print("Shape of dataset:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

def save_processed_data(df, filename):
    output_path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved processed file to {output_path}")

if __name__ == "__main__":
    df = load_sample("flights.csv", n_rows=100000)
    basic_inspection(df)
    save_processed_data(df, "flights_sample.csv")
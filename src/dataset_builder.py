import pandas as pd
import os

DATA_PATH = "data/processed/"

def load_data():
    return pd.read_csv(os.path.join(DATA_PATH, "flights_cleaned.csv"))

def build_pre_departure(df):

    pre_cols = [
        "YEAR",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_TIME",
        "DISTANCE",
        "SCHEDULED_ARRIVAL",
        "DELAYED"
    ]

    return df[pre_cols]

def build_post_departure(df):

    post_cols = [
        "DEPARTURE_DELAY",
        "TAXI_OUT",
        "DISTANCE",
        "ARRIVAL_DELAY"
    ]

    return df[post_cols]

def save_dataset(df, filename):
    df.to_csv(os.path.join(DATA_PATH, filename), index=False)

if __name__ == "__main__":
    df = load_data()

    pre_df = build_pre_departure(df)
    post_df = build_post_departure(df)

    save_dataset(pre_df, "pre_departure_data.csv")
    save_dataset(post_df, "post_departure_data.csv")

    print("Pre-departure shape:", pre_df.shape)
    print("Post-departure shape:", post_df.shape)
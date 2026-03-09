import pandas as pd
import os

DATA_PATH = "data/processed/"

def load_pre_data():
    return pd.read_csv(os.path.join(DATA_PATH, "pre_departure_data.csv"))

def add_airline_delay_rate(df):

    airline_delay_rate = (
        df.groupby("AIRLINE")["DELAYED"]
        .mean()
        .reset_index()
        .rename(columns={"DELAYED": "AIRLINE_DELAY_RATE"})
    )

    df = df.merge(airline_delay_rate, on="AIRLINE", how="left")

    return df

def add_origin_delay_rate(df):

    origin_delay_rate = (
        df.groupby("ORIGIN_AIRPORT")["DELAYED"]
        .mean()
        .reset_index()
        .rename(columns={"DELAYED": "ORIGIN_DELAY_RATE"})
    )

    df = df.merge(origin_delay_rate, on="ORIGIN_AIRPORT", how="left")

    return df

def add_destination_delay_rate(df):

    destination_delay_rate = (
        df.groupby("DESTINATION_AIRPORT")["DELAYED"]
        .mean()
        .reset_index()
        .rename(columns={"DELAYED": "DEST_DELAY_RATE"})
    )

    df = df.merge(destination_delay_rate, on="DESTINATION_AIRPORT", how="left")

    return df

def add_departure_hour(df):

    # Convert to 4-digit string with leading zeros
    df["SCHEDULED_DEPARTURE_STR"] = df["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4)

    # Extract first two characters as hour
    df["DEPARTURE_HOUR"] = df["SCHEDULED_DEPARTURE_STR"].str[:2].astype(int)

    return df

def add_hour_delay_rate(df):

    hour_delay_rate = (
        df.groupby("DEPARTURE_HOUR")["DELAYED"]
        .mean()
        .reset_index()
        .rename(columns={"DELAYED": "HOUR_DELAY_RATE"})
    )

    df = df.merge(hour_delay_rate, on="DEPARTURE_HOUR", how="left")

    return df

if __name__ == "__main__":
    df = load_pre_data()

    df = add_airline_delay_rate(df)
    df = add_origin_delay_rate(df)
    df = add_destination_delay_rate(df)
    df = add_departure_hour(df)
    df = add_hour_delay_rate(df)

    df.to_csv(os.path.join(DATA_PATH, "pre_departure_featured.csv"), index=False)

    print("New shape:", df.shape)
    print(df[["DEPARTURE_HOUR", "HOUR_DELAY_RATE"]].head())
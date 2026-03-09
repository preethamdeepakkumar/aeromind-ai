import pandas as pd
import os

DATA_PATH = "data/processed/"


def load_data():
    return pd.read_csv(os.path.join(DATA_PATH, "pre_departure_data.csv"))


def create_rolling_feature(df):

    # 🔥 RECREATE DEPARTURE_HOUR (IMPORTANT FIX)

    df["SCHEDULED_DEPARTURE_STR"] = df["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4)
    df["DEPARTURE_HOUR"] = df["SCHEDULED_DEPARTURE_STR"].str[:2].astype(int)

    # Sort by time
    df = df.sort_values(by=["MONTH", "DAY", "SCHEDULED_DEPARTURE"])

    # Rolling airline delay
    df["AIRLINE_ROLLING_DELAY"] = (
        df.groupby("AIRLINE")["DELAYED"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    # Fill first values
    df["AIRLINE_ROLLING_DELAY"] = df["AIRLINE_ROLLING_DELAY"].fillna(df["DELAYED"].mean())

    return df


if __name__ == "__main__":

    df = load_data()

    df = create_rolling_feature(df)

    df.to_csv(os.path.join(DATA_PATH, "pre_departure_rolling.csv"), index=False)

    print(" Rolling feature created successfully.")
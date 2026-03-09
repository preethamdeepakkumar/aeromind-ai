import pandas as pd
import os

RAW_DATA_PATH = "data/processed/"
PROCESSED_DATA_PATH = "data/processed/"

def load_data(filename):
    file_path = os.path.join(RAW_DATA_PATH,filename)
    df =pd.read_csv(file_path)
    return df

def clean_data(df):
    print("initial shape:",df.shape)
    df = df[df["CANCELLED"] == 0].copy()
    print("after removing cancelled flights",df.shape)

    delay_columns = [
        "WEATHER_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "AIRLINE_DELAY",
        "SECURITY_DELAY",
        "AIR_SYSTEM_DELAY"
    ]

    df[delay_columns]= df[delay_columns].fillna(0)

    df["DELAYED"] = df["ARRIVAL_DELAY"].apply(lambda x : 1 if x>15 else 0)

    df = df.dropna(subset=["ARRIVAL_DELAY"])
    print("final cleaned shape",df.shape)
    return df

def save_cleaned_data(df,filename):
    output_path = os.path.join(PROCESSED_DATA_PATH,filename)
    df.to_csv(output_path,index=False)
    print(f"saved cleaned file to {output_path}")

if __name__ == "__main__":
    df = load_data("flights_sample.csv")
    cleaned_df = clean_data(df)
    save_cleaned_data(cleaned_df, "flights_cleaned.csv")

    '''print(cleaned_df["DELAYED"].value_counts())
    print(cleaned_df["ARRIVAL_DELAY"].describe())
    print(cleaned_df["DEPARTURE_DELAY"].describe())
    print(cleaned_df["DISTANCE"].describe())'''

    df = load_data("flights_cleaned.csv")

'''departure_arrival_corr = df["DEPARTURE_DELAY"].corr(df["ARRIVAL_DELAY"])
print("Correlation (Departure vs Arrival):", departure_arrival_corr)

distance_corr = df["DISTANCE"].corr(df["ARRIVAL_DELAY"])
print("Correlation (Distance vs Arrival):", distance_corr)

delay_columns = [
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY"
]

for col in delay_columns:
    corr = df[col].corr(df["ARRIVAL_DELAY"])
    print(f"Correlation ({col} vs Arrival):", corr)'''

print(df.columns.tolist())
import pandas as pd
import numpy as np
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

DATA_PATH = "data/processed/"

def load_data():
    return pd.read_csv("data/processed/pre_departure_rolling.csv")

def prepare_features(df):

    feature_cols = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "DISTANCE",
        "DEPARTURE_HOUR",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "AIRLINE_ROLLING_DELAY"
    ]

    X = df[feature_cols]
    y = df["DELAYED"]

    return X, y

def train_models(X, y):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []

    for train_index, val_index in kf.split(X):

        X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Combine features + target
        train_data = X_train.copy()
        train_data["DELAYED"] = y_train

        # Compute aggregation ONLY on training fold
        airline_rate = train_data.groupby("AIRLINE")["DELAYED"].mean()
        origin_rate = train_data.groupby("ORIGIN_AIRPORT")["DELAYED"].mean()
        dest_rate = train_data.groupby("DESTINATION_AIRPORT")["DELAYED"].mean()

        # Map to training
        X_train["AIRLINE_RATE"] = X_train["AIRLINE"].map(airline_rate)
        X_train["ORIGIN_RATE"] = X_train["ORIGIN_AIRPORT"].map(origin_rate)
        X_train["DEST_RATE"] = X_train["DESTINATION_AIRPORT"].map(dest_rate)

        # Map to validation
        X_val["AIRLINE_RATE"] = X_val["AIRLINE"].map(airline_rate)
        X_val["ORIGIN_RATE"] = X_val["ORIGIN_AIRPORT"].map(origin_rate)
        X_val["DEST_RATE"] = X_val["DESTINATION_AIRPORT"].map(dest_rate)

        # Fill unseen categories
        global_mean = y_train.mean()
        X_train.fillna(global_mean, inplace=True)
        X_val.fillna(global_mean, inplace=True)

        # Drop categorical columns
        drop_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
        X_train = X_train.drop(columns=drop_cols)
        X_val = X_val.drop(columns=drop_cols)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(max_iter=1000, C=0.1)
        model.fit(X_train_scaled, y_train)

        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        auc = roc_auc_score(y_val, y_prob)
        auc_scores.append(auc)

    print("Fold-aware CV ROC-AUC:", np.mean(auc_scores))

if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)
    train_models(X, y)
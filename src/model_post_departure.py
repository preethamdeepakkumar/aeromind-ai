import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/processed/"


# ---------------------------
# Load Data + Step 3 (Recovery Insight)
# ---------------------------
def load_data():
    df = pd.read_csv(os.path.join(DATA_PATH, "post_departure_data.csv"))

    # STEP 3 — Recovery Analysis
    df["RECOVERY"] = df["ARRIVAL_DELAY"] - df["DEPARTURE_DELAY"]

    print("\n========== Recovery Insights ==========")
    print("Average Recovery:", df["RECOVERY"].mean())
    print("Best Recovery (most recovered minutes):", df["RECOVERY"].min())
    print("Worst Case (delay increased most):", df["RECOVERY"].max())

    return df


# ---------------------------
# Prepare Features
# ---------------------------
def prepare_features(df):

    feature_cols = [
        "DEPARTURE_DELAY",
        "TAXI_OUT",
        "DISTANCE"
    ]

    X = df[feature_cols]
    y = df["ARRIVAL_DELAY"]

    return X, y


# ---------------------------
# Train Models
# ---------------------------
def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # Linear Regression
    # =========================
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)

    print("\n========== Linear Regression Results ==========")
    print("MAE:", mean_absolute_error(y_test, y_pred_lr))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    print("R²:", r2_score(y_test, y_pred_lr))

    # -------------------------
    # STEP 1 — Feature Impact
    # -------------------------
    print("\n========== Feature Impact (Linear Regression) ==========")

    feature_names = X.columns
    coefficients = lr_model.coef_

    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.3f}")

    # -------------------------
    # STEP 2 — Residual Analysis
    # -------------------------
    residuals = y_test - y_pred_lr

    print("\n========== Residual Analysis ==========")
    print("Mean Residual:", residuals.mean())
    print("Max Error:", residuals.abs().max())



    # =========================
    # Random Forest Regressor
    # =========================
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)

    print("\n========== Random Forest Results ==========")
    print("MAE:", mean_absolute_error(y_test, y_pred_rf))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    print("R²:", r2_score(y_test, y_pred_rf))


# ---------------------------
# Run Script
# ---------------------------
if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)
    train_models(X, y)
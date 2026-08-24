import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

def create_mock_data():
    print("Generating mock test data...")
    # Generate dates from 2023-01-01 to 2026-01-31
    dates = pd.date_range(start="2023-01-01", end="2026-01-31", freq="B") # Business days
    
    n_samples = len(dates)
    np.random.seed(42)
    
    # Generate mock features
    df = pd.DataFrame({
        "Date": dates,
        "Price_Lag1": np.random.uniform(130000, 150000, n_samples),
        "Volume_Lag1": np.random.randint(10000, 100000, n_samples),
        "Vol_7d": np.random.uniform(0.5, 2.0, n_samples),
        "Vol_30d": np.random.uniform(0.5, 2.0, n_samples),
        "Exact_Return_Lag1": np.random.normal(0, 1, n_samples),
        "Is_Anomaly": np.random.choice([0, 1], p=[0.95, 0.05], size=n_samples)
    })
    
    # Generate actual return (chg%)
    df["actual_chg%"] = np.random.normal(0.05, 1.2, n_samples)
    
    # Generate model predictions
    # Ridge: near zero
    df["pred_chg%_ridge"] = np.random.normal(0.01, 0.05, n_samples)
    # XGBoost + TCN (Best): closely tracks actual
    df["pred_chg%_xgboost_tcn"] = df["actual_chg%"] + np.random.normal(0, 0.3, n_samples)
    # XGBoost: tracks actual but higher variance
    df["pred_chg%_xgboost"] = df["actual_chg%"] + np.random.normal(0, 0.6, n_samples)
    # SVR: flat
    df["pred_chg%_svr"] = np.random.normal(0.02, 0.08, n_samples)
    # LSTM: erratic
    df["pred_chg%_lstm"] = np.random.normal(0, 2.0, n_samples)
    # MLP: somewhere in between
    df["pred_chg%_mlp"] = np.random.normal(0, 1.0, n_samples)
    
    df.to_csv("test_predictions.csv", index=False)
    print("Created test_predictions.csv")

def create_mock_models():
    print("Generating mock models (placeholders for teammates)...")
    
    # We already have ridge_model.pkl but let's create placeholders for transformation pipeline for Ridge/SVR
    scaler = StandardScaler()
    scaler.fit(np.random.normal(0, 1, (100, 6)))
    joblib.dump(scaler, "scaler.pkl")
    print("Created scaler.pkl placeholder")
    
    # Creating a dummy Ridge model for the pipeline expectation (6 features)
    dummy_ridge = Ridge()
    dummy_ridge.fit(np.random.normal(0, 1, (100, 6)), np.random.normal(0, 1, 100))
    joblib.dump(dummy_ridge, "ridge_model.pkl") # Overwrite with one that has n_features_in_ = 6
    print("Overwrote ridge_model.pkl with dummy 6-feature model for testing")

if __name__ == "__main__":
    create_mock_data()
    create_mock_models()
    print("Done!")

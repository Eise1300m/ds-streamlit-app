"""
data_loader.py
Orchestrates model and data loading, runs batch predictions, and returns a unified payload.
"""

import pandas as pd
import numpy as np

from models.loader        import load_models, load_datasets
from models.ridge         import predict_batch as ridge_batch
from models.xgboost_model import predict_batch as xgb_batch
from models.ensemble      import predict_batch as ens_batch
from models.svr           import predict_batch as svr_batch
from models.lstm_model    import predict_batch as lstm_batch
from models.mlp           import predict_batch as mlp_batch


def load_all():
    """
    Main entry point called by app.py.

    Returns
    -------
    dict with keys:
        df              - Test-set DataFrame (actual + all model predictions)
        ridge_coefs     - np.ndarray Ridge coefficients (for feature importance)
        X_test_scaled   - Toolbox B scaled features
        X_test_raw      - Toolbox A raw features
        y_test_df       - y_test with Exact_Return column
        X_train_scaled  - Toolbox B training features (for heatmap)
        X_train_raw     - Toolbox A training features (for heatmap + Ensemble context)
        y_train_df      - y_train
        ridge_model     - Fitted Ridge model object
        xgb_model       - Fitted XGBoost model object (or None)
        ensemble_model  - Fitted Ensemble model object (or None)
        svr_model       - Fitted SVR model object (or None)
        preprocessors   - Dict of fitted scalers / transformers
    """
    # 1. Load objects from disk
    models   = load_models()
    datasets = load_datasets()

    ridge_model    = models["ridge_model"]
    xgb_model      = models["xgb_model"]
    ensemble_model = models["ensemble_model"]
    svr_model      = models["svr_model"]
    lstm_model     = models["lstm_model"]
    mlp_model      = models["mlp_model"]
    preprocessors  = models["preprocessors"]

    X_test_scaled  = datasets["X_test_scaled"]
    X_test_raw     = datasets["X_test_raw"]
    y_test_df      = datasets["y_test_df"]
    X_train_scaled = datasets["X_train_scaled"]
    X_train_raw    = datasets["X_train_raw"]
    y_train_df     = datasets["y_train_df"]

    # 2. Build test DataFrame
    n     = len(X_test_raw)
    dates = pd.date_range(end=pd.Timestamp('2026-01-01'), periods=n, freq='B')

    actual_returns = y_test_df['Exact_Return'].values
    actual_prices  = X_test_raw['Price_Lag1'].values * (1 + actual_returns / 100)

    df = pd.DataFrame({
        "Date":              dates,
        "Actual_Price":      actual_prices,
        "Actual_Return_%":   actual_returns,
        "Volume":            X_test_raw['Volume_Lag1'].values,
        "Vol_7d":            X_test_raw['Vol_7d'].values,
        "Vol_30d":           X_test_raw['Vol_30d'].values,
        "Is_Anomaly":        X_test_raw['Is_Anomaly'].values,
    })

    # 3. Run batch predictions
    ridge_returns, ridge_prices = ridge_batch(ridge_model, X_test_scaled, X_test_raw)
    df["Ridge_Pred_Return_%"] = ridge_returns
    df["Ridge_Pred_Price"]    = ridge_prices

    if xgb_model is not None:
        xgb_returns, xgb_prices = xgb_batch(xgb_model, X_test_raw)
        df["XGBoost_Pred_Return_%"] = xgb_returns
        df["XGBoost_Pred_Price"]    = xgb_prices

    if ensemble_model is not None:
        ens_returns, ens_prices = ens_batch(ensemble_model, X_train_raw, X_test_raw)
        df["Ensemble Model: XGBoost + TCN_Pred_Return_%"] = ens_returns
        df["Ensemble Model: XGBoost + TCN_Pred_Price"]    = ens_prices

    if svr_model is not None:
        svr_returns, svr_prices = svr_batch(svr_model, X_test_scaled, X_test_raw)
        df["Support Vector Regression (SVR)_Pred_Return_%"] = svr_returns
        df["Support Vector Regression (SVR)_Pred_Price"]    = svr_prices

    if mlp_model is not None:
        mlp_returns, mlp_prices = mlp_batch(mlp_model, X_test_scaled, X_test_raw)
        df["Multilayer Perceptron (MLP)_Pred_Return_%"] = mlp_returns
        df["Multilayer Perceptron (MLP)_Pred_Price"]    = mlp_prices

    if lstm_model is not None:
        lstm_returns, lstm_prices = lstm_batch(lstm_model, X_train_scaled, X_test_scaled, X_test_raw)
        df["LSTM_Pred_Return_%"] = lstm_returns
        df["LSTM_Pred_Price"]    = lstm_prices

    # 4. Extract Ridge coefficients
    ridge_coefs = ridge_model.coef_
    if len(ridge_coefs.shape) > 1:
        ridge_coefs = ridge_coefs.flatten()

    return {
        "df":             df,
        "ridge_coefs":    ridge_coefs,
        "X_test_scaled":  X_test_scaled,
        "X_test_raw":     X_test_raw,
        "y_test_df":      y_test_df,
        "X_train_scaled": X_train_scaled,
        "X_train_raw":    X_train_raw,
        "y_train_df":     y_train_df,
        "ridge_model":    ridge_model,
        "xgb_model":      xgb_model,
        "ensemble_model": ensemble_model,
        "svr_model":      svr_model,
        "lstm_model":     lstm_model,
        "mlp_model":      mlp_model,
        "preprocessors":  preprocessors,
    }

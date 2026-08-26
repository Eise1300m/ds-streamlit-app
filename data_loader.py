"""
data_loader.py
==============
Responsible for all data ingestion, model loading, and pre-computation of
predictions. app.py calls load_all() once at startup and receives everything
it needs via a clean dictionary payload.

Separating this logic from app.py keeps the UI file focused purely on display,
and makes it easy to debug data/model issues in isolation.
"""

import os
import pandas as pd
import numpy as np
import joblib


def load_all():
    """
    Load all trained models and datasets, run batch predictions on the test
    set, and return a structured dict.

    Returns
    -------
    dict with keys:
        df              - Main test-set DataFrame with actual + predicted columns
        ridge_coefs     - np.ndarray of Ridge Regression coefficients
        X_test_scaled   - Scaled test features (Toolbox B)
        X_test_raw      - Raw test features   (Toolbox A)
        y_test_df       - y_test DataFrame (Exact_Return)
        X_train_scaled  - Scaled train features (for heatmap)
        X_train_raw     - Raw train features   (for heatmap)
        y_train_df      - y_train DataFrame
        ridge_model     - Loaded Ridge model object
        xgb_model       - Loaded XGBoost model object (or None)
        ensemble_model  - Loaded Ensemble model object (or None)
        preprocessors   - Dict of fitted scalers/transformers
    """
    import model_architecture
    import __main__

    # ------------------------------------------------------------------
    # 1. Resolve base directory so paths work regardless of CWD
    # ------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # 2. Register custom classes in __main__ so joblib can unpickle them
    # ------------------------------------------------------------------
    __main__.EnsembleModel      = model_architecture.EnsembleModel
    __main__.TCN                = model_architecture.TCN
    __main__.TemporalBlock      = model_architecture.TemporalBlock
    __main__.preprocess_for_tcn = model_architecture.preprocess_for_tcn

    # ------------------------------------------------------------------
    # 3. Load models (XGBoost and Ensemble are optional, graceful fallback)
    # ------------------------------------------------------------------
    ridge_model = joblib.load(os.path.join(BASE_DIR, 'Model_PKL', 'ridge_model.pkl'))

    try:
        xgb_model = joblib.load(os.path.join(BASE_DIR, 'Model_PKL', 'xgboost_tuned.pkl'))
    except Exception as e:
        xgb_model = None
        print(f"[data_loader] WARNING: Could not load XGBoost model: {e}")

    try:
        ensemble_model = joblib.load(os.path.join(BASE_DIR, 'Model_PKL', 'ensemble_model.pkl'))
    except Exception as e:
        ensemble_model = None
        print(f"[data_loader] WARNING: Could not load Ensemble model: {e}")

    preprocessors = joblib.load(os.path.join(BASE_DIR, 'Model_PKL', 'preprocessors.pkl'))

    # ------------------------------------------------------------------
    # 4. Load datasets
    # ------------------------------------------------------------------
    X_test_scaled  = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'X_test_trans_scaled.csv'))
    X_test_raw     = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'X_test_raw.csv'))
    y_test_df      = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'y_test.csv'))

    X_train_scaled = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'X_train_trans_scaled.csv'))
    X_train_raw    = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'X_train_raw.csv'))
    y_train_df     = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'y_train.csv'))

    # ------------------------------------------------------------------
    # 5. Generate dates (test CSVs do not include a Date column)
    # ------------------------------------------------------------------
    n     = len(X_test_raw)
    dates = pd.date_range(end=pd.Timestamp('2026-01-01'), periods=n, freq='B')

    # ------------------------------------------------------------------
    # 6. Compute actual prices and Ridge predictions
    # ------------------------------------------------------------------
    actual_returns = y_test_df['Exact_Return'].values
    actual_prices  = X_test_raw['Price_Lag1'].values * (1 + actual_returns / 100)

    ridge_preds  = ridge_model.predict(X_test_scaled).flatten()
    ridge_prices = X_test_raw['Price_Lag1'].values * (1 + ridge_preds / 100)

    df = pd.DataFrame({
        "Date":              dates,
        "Actual_Price":      actual_prices,
        "Actual_Return_%":   actual_returns,
        "Volume":            X_test_raw['Volume_Lag1'].values,
        "Vol_7d":            X_test_raw['Vol_7d'].values,
        "Vol_30d":           X_test_raw['Vol_30d'].values,
        "Is_Anomaly":        X_test_raw['Is_Anomaly'].values,
        "Ridge_Pred_Price":  ridge_prices,
        "Ridge_Pred_Return_%": ridge_preds,
    })

    # ------------------------------------------------------------------
    # 7. XGBoost batch predictions (Toolbox A, raw features)
    # ------------------------------------------------------------------
    if xgb_model is not None:
        xgb_returns = xgb_model.predict(X_test_raw).flatten()
        xgb_prices  = X_test_raw['Price_Lag1'].values * (1 + xgb_returns / 100)
        df["XGBoost_Pred_Price"]    = xgb_prices
        df["XGBoost_Pred_Return_%"] = xgb_returns

    # ------------------------------------------------------------------
    # 8. Ensemble (XGBoost + TCN) batch predictions
    #    Needs a 30-day sliding context from the end of the training set
    # ------------------------------------------------------------------
    if ensemble_model is not None:
        seq_len     = ensemble_model.seq_len
        context_raw = pd.concat([X_train_raw.iloc[-seq_len:], X_test_raw])
        ens_returns = ensemble_model.predict(context_raw)
        ens_prices  = X_test_raw['Price_Lag1'].values * (1 + ens_returns / 100)
        df["Ensemble Model: XGBoost + TCN_Pred_Price"]    = ens_prices
        df["Ensemble Model: XGBoost + TCN_Pred_Return_%"] = ens_returns

    # ------------------------------------------------------------------
    # 9. Extract Ridge coefficients for feature importance visualisation
    # ------------------------------------------------------------------
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
        "preprocessors":  preprocessors,
    }

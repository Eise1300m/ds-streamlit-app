"""
models/xgboost_model.py
=======================
XGBoost inference logic.

  predict_batch()   - Run predictions over the full test set.
  predict_sandbox() - Run a single live prediction from user inputs.

XGBoost uses Toolbox A (raw, scale-invariant features) so no scaling
is needed for the inputs.
"""

import numpy as np
import pandas as pd


def predict_batch(xgb_model, X_test_raw):
    """
    Batch XGBoost predictions over the full test set.

    Parameters
    ----------
    xgb_model  : fitted XGBoost model
    X_test_raw : DataFrame — Toolbox A raw test features

    Returns
    -------
    xgb_returns : np.ndarray of predicted daily returns (%)
    xgb_prices  : np.ndarray of predicted next-day prices
    """
    xgb_returns = xgb_model.predict(X_test_raw).flatten()
    xgb_prices  = X_test_raw['Price_Lag1'].values * (1 + xgb_returns / 100)
    return xgb_returns, xgb_prices


def predict_sandbox(xgb_model, X_test_raw, sandbox_price, sandbox_volume, sandbox_return):
    """
    Single-row XGBoost prediction from manually entered user inputs.

    Uses the last row of X_test_raw for context features (Vol_7d, Vol_30d,
    Is_Anomaly) that are not available in the UI, then overwrites the three
    user-supplied raw features.

    Returns
    -------
    pred_return : float — predicted exact return (%)
    pred_price  : float — predicted next-day price
    """
    last_raw_row = X_test_raw.iloc[-1].copy()
    last_raw_row['Price_Lag1']        = sandbox_price
    last_raw_row['Volume_Lag1']       = sandbox_volume
    last_raw_row['Exact_Return_Lag1'] = sandbox_return

    pred_return = xgb_model.predict(pd.DataFrame([last_raw_row]))[0]
    pred_price  = sandbox_price * (1 + pred_return / 100)
    return pred_return, pred_price

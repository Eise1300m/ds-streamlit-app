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


def predict_sandbox(xgb_model, X_test_raw, sandbox_price, sandbox_volume, sandbox_return,
                    sandbox_vol7d, sandbox_vol30d, sandbox_anomaly):
    """
    Single-row XGBoost prediction from manually entered user inputs.
    """
    last_raw_row = X_test_raw.iloc[-1].copy()
    last_raw_row['Price_Lag1']        = sandbox_price
    last_raw_row['Volume_Lag1']       = sandbox_volume
    last_raw_row['Exact_Return_Lag1'] = sandbox_return
    last_raw_row['Vol_7d']            = sandbox_vol7d
    last_raw_row['Vol_30d']           = sandbox_vol30d
    last_raw_row['Is_Anomaly']        = sandbox_anomaly

    pred_return = xgb_model.predict(pd.DataFrame([last_raw_row]))[0]
    pred_price  = sandbox_price * (1 + pred_return / 100)
    return pred_return, pred_price

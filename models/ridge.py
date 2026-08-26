"""
models/ridge.py
===============
Ridge Regression inference logic.

  predict_batch()   - Run predictions over the full test set.
  predict_sandbox() - Run a single live prediction from user inputs.
"""

import numpy as np
import pandas as pd


def predict_batch(ridge_model, X_test_scaled, X_test_raw):
    """
    Batch Ridge predictions over the full test set.

    Parameters
    ----------
    ridge_model   : fitted RidgeCV model
    X_test_scaled : DataFrame — Toolbox B scaled test features
    X_test_raw    : DataFrame — Toolbox A raw test features (for Price_Lag1)

    Returns
    -------
    ridge_returns : np.ndarray of predicted daily returns (%)
    ridge_prices  : np.ndarray of predicted next-day prices
    """
    ridge_returns = ridge_model.predict(X_test_scaled).flatten()
    ridge_prices  = X_test_raw['Price_Lag1'].values * (1 + ridge_returns / 100)
    return ridge_returns, ridge_prices


def predict_sandbox(ridge_model, X_test_scaled, preprocessors,
                    sandbox_price, sandbox_volume, sandbox_return):
    """
    Single-row Ridge prediction from manually entered user inputs.

    Scales price, volume, and return using the saved preprocessors,
    injects them into the last row of the test set (for the remaining
    features like Vol_7d, Vol_30d), then runs ridge_model.predict().

    Returns
    -------
    pred_return : float — predicted exact return (%)
    pred_price  : float — predicted next-day price
    """
    last_scaled_row = X_test_scaled.iloc[-1].copy()

    log_price    = np.log(sandbox_price)
    price_scaled = (log_price - preprocessors['price_mean']) / preprocessors['price_std']
    vol_scaled   = preprocessors['pt_vol'].transform([[sandbox_volume]])[0, 0]
    ret_scaled   = preprocessors['scaler_return'].transform([[sandbox_return]])[0, 0]

    last_scaled_row['Log_Price_Lag1']    = price_scaled
    last_scaled_row['Yeo_Volume_Lag1']   = vol_scaled
    last_scaled_row['Exact_Return_Lag1'] = ret_scaled

    pred_return = ridge_model.predict(pd.DataFrame([last_scaled_row]))[0]
    pred_price  = sandbox_price * (1 + pred_return / 100)
    return pred_return, pred_price

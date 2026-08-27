"""
models/svr.py
=============
Support Vector Regression (SVR) inference logic.

  predict_batch()   - Run predictions over the full test set.
  predict_sandbox() - Run a single live prediction from user inputs.

SVR uses Toolbox B (transformed / scaled features).
"""

import numpy as np
import pandas as pd


def predict_batch(svr_model, X_test_scaled, X_test_raw):
    """
    Batch SVR predictions over the full test set.

    Parameters
    ----------
    svr_model     : fitted SVR model
    X_test_scaled : DataFrame — Toolbox B scaled test features
    X_test_raw    : DataFrame — Toolbox A raw test features (for Price_Lag1)

    Returns
    -------
    svr_returns : np.ndarray of predicted daily returns (%)
    svr_prices  : np.ndarray of predicted next-day prices
    """
    svr_returns = svr_model.predict(X_test_scaled).flatten()
    svr_prices  = X_test_raw['Price_Lag1'].values * (1 + svr_returns / 100)
    return svr_returns, svr_prices


def predict_sandbox(svr_model, X_test_scaled, preprocessors,
                    sandbox_price, sandbox_volume, sandbox_return,
                    sandbox_vol7d, sandbox_vol30d, sandbox_anomaly):
    """
    Single-row SVR prediction from manually entered user inputs.
    """
    last_scaled_row = X_test_scaled.iloc[-1].copy()

    log_price    = np.log(sandbox_price)
    price_scaled = (log_price - preprocessors['price_mean']) / preprocessors['price_std']
    vol_scaled   = preprocessors['pt_vol'].transform([[sandbox_volume]])[0, 0]
    ret_scaled   = preprocessors['scaler_return'].transform([[sandbox_return]])[0, 0]
    vol7d_scaled = preprocessors['pt_vol7'].transform([[sandbox_vol7d]])[0, 0]
    vol30d_scaled = preprocessors['pt_vol30'].transform([[sandbox_vol30d]])[0, 0]

    last_scaled_row['Log_Price_Lag1']    = price_scaled
    last_scaled_row['Yeo_Volume_Lag1']   = vol_scaled
    last_scaled_row['Exact_Return_Lag1'] = ret_scaled
    last_scaled_row['Yeo_Vol_7d']        = vol7d_scaled
    last_scaled_row['Yeo_Vol_30d']       = vol30d_scaled
    last_scaled_row['Is_Anomaly']        = sandbox_anomaly

    pred_return = svr_model.predict(pd.DataFrame([last_scaled_row]))[0]
    pred_price  = sandbox_price * (1 + pred_return / 100)
    return pred_return, pred_price

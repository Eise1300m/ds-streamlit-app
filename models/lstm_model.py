"""
models/lstm_model.py
LSTM (Keras) inference logic. Uses Toolbox B (scaled features).
Requires a 20-step sliding window (seq_len=20).
"""

import numpy as np
import pandas as pd

LSTM_SEQ_LEN = 20


def predict_batch(lstm_model, X_train_scaled, X_test_scaled, X_test_raw):
    """
    Batch LSTM predictions over the full test set.
    Concatenates training context so the first test rows have enough history.
    """
    context = pd.concat([X_train_scaled.iloc[-LSTM_SEQ_LEN:], X_test_scaled])
    values = context.values
    n_test = len(X_test_scaled)

    preds = []
    for i in range(n_test):
        window = values[i : i + LSTM_SEQ_LEN].reshape(1, LSTM_SEQ_LEN, -1)
        pred = lstm_model.predict(window, verbose=0).flatten()[0]
        preds.append(pred)

    preds = np.array(preds)
    prices = X_test_raw['Price_Lag1'].values * (1 + preds / 100)
    return preds, prices


def predict_sandbox(lstm_model, X_test_scaled, preprocessors,
                    sandbox_price, sandbox_volume, sandbox_return,
                    sandbox_vol7d, sandbox_vol30d, sandbox_anomaly):
    """
    Single-row LSTM prediction from user inputs.
    Takes the last (seq_len - 1) rows from test set as context,
    appends 1 user-simulated scaled row, then feeds the 20-step window.
    """
    # Build the simulated row from user inputs
    sim_row = X_test_scaled.iloc[-1].copy()

    log_price     = np.log(sandbox_price)
    price_scaled  = (log_price - preprocessors['price_mean']) / preprocessors['price_std']
    vol_scaled    = preprocessors['pt_vol'].transform([[sandbox_volume]])[0, 0]
    ret_scaled    = preprocessors['scaler_return'].transform([[sandbox_return]])[0, 0]
    vol7d_scaled  = preprocessors['pt_vol7'].transform([[sandbox_vol7d]])[0, 0]
    vol30d_scaled = preprocessors['pt_vol30'].transform([[sandbox_vol30d]])[0, 0]

    sim_row['Log_Price_Lag1']    = price_scaled
    sim_row['Yeo_Volume_Lag1']   = vol_scaled
    sim_row['Exact_Return_Lag1'] = ret_scaled
    sim_row['Yeo_Vol_7d']        = vol7d_scaled
    sim_row['Yeo_Vol_30d']       = vol30d_scaled
    sim_row['Is_Anomaly']        = sandbox_anomaly

    # Build the 20-step context window
    context = X_test_scaled.iloc[-(LSTM_SEQ_LEN - 1):].copy()
    context = pd.concat([context, pd.DataFrame([sim_row])])
    window = context.values.reshape(1, LSTM_SEQ_LEN, -1)

    pred_return = lstm_model.predict(window, verbose=0).flatten()[0]
    pred_price = sandbox_price * (1 + pred_return / 100)
    return pred_return, pred_price

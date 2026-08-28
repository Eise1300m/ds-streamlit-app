"""
models/lstm_model.py
LSTM (Keras) inference logic. Uses Toolbox B (scaled features).
Configured for 5-day lookback matching LSTM_final_model.keras.
"""

import numpy as np
import pandas as pd

# Set to 5 to match WINNING_LOOKBACK = 5 from LSTM.ipynb
LSTM_SEQ_LEN = 5

def predict_batch(lstm_model, X_train_scaled, X_test_scaled, X_test_raw):
    """
    Batch LSTM predictions over the full test set.
    Replicates the exact create_sequences() logic from LSTM.ipynb.
    """
    # 1. Build test context (5 train rows + 611 test rows = 616 rows)
    X_test_context = pd.concat([X_train_scaled.tail(LSTM_SEQ_LEN), X_test_scaled], ignore_index=True)
    X_arr = X_test_context.to_numpy(dtype=np.float32)

    # 2. Build 3D sequences identically to Colab
    X_seq = []
    for i in range(LSTM_SEQ_LEN, len(X_arr)):
        X_seq.append(X_arr[i - LSTM_SEQ_LEN + 1 : i + 1])
    X_test_seq = np.asarray(X_seq)

    # 3. Vectorized batch prediction
    preds = lstm_model.predict(X_test_seq, verbose=0).flatten()
    prices = X_test_raw['Price_Lag1'].values * (1 + preds / 100)
    return preds, prices


def predict_sandbox(lstm_model, X_test_scaled, preprocessors,
                    sandbox_price, sandbox_volume, sandbox_return,
                    sandbox_vol7d, sandbox_vol30d, sandbox_anomaly):
    """
    Single-row LSTM prediction from user inputs.
    """
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
    is_anomaly_scaled = 4.287301293465667 if sandbox_anomaly else -0.2332469615616036
    sim_row['Is_Anomaly']        = is_anomaly_scaled

    # Context window: last 4 rows + 1 user simulated row = 5 timesteps
    context = X_test_scaled.tail(LSTM_SEQ_LEN - 1).copy()
    context = pd.concat([context, pd.DataFrame([sim_row])], ignore_index=True)
    window = context.to_numpy(dtype=np.float32).reshape(1, LSTM_SEQ_LEN, -1)

    pred_return = lstm_model.predict(window, verbose=0).flatten()[0]
    pred_price = sandbox_price * (1 + pred_return / 100)
    return pred_return, pred_price
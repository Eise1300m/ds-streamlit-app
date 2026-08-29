"""
models/ensemble.py
==================
Ensemble (XGBoost + TCN) inference logic.

  predict_batch()   - Run batch predictions over the full test set using a
                      30-day sliding context window.
  predict_sandbox() - Run a single live prediction using the last 29 rows of
                      test data as context + 1 user-supplied simulated row.

Because the TCN component uses a fixed-length sliding window (seq_len=30),
both functions need historical context from X_train_raw or X_test_raw to
warm-start the sequence.
"""

import numpy as np
import pandas as pd


def predict_batch(ensemble_model, X_train_raw, X_test_raw):
    """
    Batch Ensemble predictions over the full test set.

    Concatenates the last `seq_len` rows of training data with the test set
    so the TCN has enough historical context for the very first prediction.

    Parameters
    ----------
    ensemble_model : fitted EnsembleModel (XGBoost + TCN)
    X_train_raw    : DataFrame — Toolbox A raw training features
    X_test_raw     : DataFrame — Toolbox A raw test features

    Returns
    -------
    ens_returns : np.ndarray of predicted daily returns (%)
    ens_prices  : np.ndarray of predicted next-day prices
    """
    seq_len     = ensemble_model.seq_len
    context_raw = pd.concat([X_train_raw.iloc[-seq_len:], X_test_raw])

    ens_returns = ensemble_model.predict(context_raw)
    ens_prices  = X_test_raw['Price_Lag1'].values * (1 + ens_returns / 100)
    return ens_returns, ens_prices


def predict_sandbox(ensemble_model, X_test_raw,
                    sandbox_price, sandbox_volume, sandbox_return,
                    sandbox_vol7d, sandbox_vol30d, sandbox_anomaly):
    """
    Single-row Ensemble prediction from manually entered user inputs.
    """
    seq_len = ensemble_model.seq_len
    
    # 1. Take the last (seq_len - 1) rows from test set as background context
    context_window = X_test_raw.iloc[-(seq_len - 1):].copy()

    # 2. Build the simulated row from user inputs
    sim_row                      = context_window.iloc[-1].copy()
    sim_row['Price_Lag1']        = sandbox_price
    sim_row['Volume_Lag1']       = sandbox_volume
    sim_row['Exact_Return_Lag1'] = sandbox_return
    sim_row['Vol_7d']            = sandbox_vol7d
    sim_row['Vol_30d']           = sandbox_vol30d
    sim_row['Is_Anomaly']        = sandbox_anomaly

    # Stitch custom manual input to the end to make exactly `seq_len` rows
    window = pd.concat([context_window, pd.DataFrame([sim_row])]).values

    # --- A. XGBoost Sub-model Prediction ---
    latest_raw = window[-1].reshape(1, -1)
    
    xgb_pred = ensemble_model.xgb_model.predict(latest_raw)[0]

    # --- B. TCN Sub-model Prediction ---
    from model_architecture import preprocess_for_tcn
    import torch
    import numpy as np
    
    scaled_window = np.array([
        preprocess_for_tcn(row, ensemble_model.preprocessors) for row in window
    ])
    tcn_input = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(ensemble_model.device)
    
    ensemble_model.tcn_model.eval()
    with torch.no_grad():
        tcn_pred = ensemble_model.tcn_model(tcn_input).item()

    # --- C. Weighted Ensemble Combination ---
    tcn_weight = ensemble_model.tcn_weight
    pred_return = tcn_weight * tcn_pred + (1 - tcn_weight) * xgb_pred

    # --- D. Price Reconstruction ---
    pred_price = sandbox_price * (1 + pred_return / 100)
    
    return pred_return, pred_price

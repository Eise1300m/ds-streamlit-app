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
                    sandbox_price, sandbox_volume, sandbox_return):
    """
    Single-row Ensemble prediction from manually entered user inputs.

    Takes the last (seq_len - 1) rows of the test set as historical context,
    appends one user-simulated row, and passes the full seq_len window
    through the ensemble to obtain a single prediction.

    Returns
    -------
    pred_return : float — predicted exact return (%)
    pred_price  : float — predicted next-day price
    """
    seq_len        = ensemble_model.seq_len
    context_window = X_test_raw.iloc[-(seq_len - 1):].copy()

    # Build the simulated "today" row using the last known context row
    # as a template (fills in Vol_7d, Vol_30d, Is_Anomaly automatically)
    sim_row                      = context_window.iloc[-1].copy()
    sim_row['Price_Lag1']        = sandbox_price
    sim_row['Volume_Lag1']       = sandbox_volume
    sim_row['Exact_Return_Lag1'] = sandbox_return

    context_window = pd.concat([context_window, pd.DataFrame([sim_row])])

    pred_return = ensemble_model.predict(context_window)[0]
    pred_price  = sandbox_price * (1 + pred_return / 100)
    return pred_return, pred_price

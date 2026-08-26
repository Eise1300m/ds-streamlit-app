"""
models/loader.py
================
Handles all disk I/O:
  - Loading .pkl model files via joblib
  - Reading train/test CSVs
  - Registering custom PyTorch classes so joblib can unpickle the Ensemble

Returns a raw dict of loaded objects. The orchestrator (data_loader.py)
calls this first, then delegates batch predictions to each model module.
"""

import os
import joblib
import pandas as pd


def _base_dir():
    """Return the project root regardless of working directory."""
    # Go one level up from models/ to reach the project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def register_custom_classes():
    """
    Bind Ensemble / TCN custom classes into __main__ so joblib can
    deserialise the ensemble_model.pkl without ImportError.
    """
    import __main__
    import model_architecture

    __main__.EnsembleModel      = model_architecture.EnsembleModel
    __main__.TCN                = model_architecture.TCN
    __main__.TemporalBlock      = model_architecture.TemporalBlock
    __main__.preprocess_for_tcn = model_architecture.preprocess_for_tcn


def load_models():
    """
    Load all trained model files.

    Returns
    -------
    dict with keys: ridge_model, xgb_model (or None), ensemble_model (or None),
                    preprocessors
    """
    register_custom_classes()
    base = _base_dir()
    pkl  = lambda name: os.path.join(base, 'Model_PKL', name)

    ridge_model   = joblib.load(pkl('ridge_model.pkl'))
    preprocessors = joblib.load(pkl('preprocessors.pkl'))

    try:
        xgb_model = joblib.load(pkl('xgboost_tuned.pkl'))
    except Exception as e:
        xgb_model = None
        print(f"[models/loader] WARNING: XGBoost not loaded: {e}")

    try:
        ensemble_model = joblib.load(pkl('ensemble_model.pkl'))
    except Exception as e:
        ensemble_model = None
        print(f"[models/loader] WARNING: Ensemble not loaded: {e}")

    return {
        "ridge_model":    ridge_model,
        "xgb_model":      xgb_model,
        "ensemble_model": ensemble_model,
        "preprocessors":  preprocessors,
    }


def load_datasets():
    """
    Load all train/test CSV datasets.

    Returns
    -------
    dict with keys: X_test_scaled, X_test_raw, y_test_df,
                    X_train_scaled, X_train_raw, y_train_df
    """
    base = _base_dir()
    ds   = lambda name: os.path.join(base, 'train_test_dataset', name)

    return {
        "X_test_scaled":  pd.read_csv(ds('X_test_trans_scaled.csv')),
        "X_test_raw":     pd.read_csv(ds('X_test_raw.csv')),
        "y_test_df":      pd.read_csv(ds('y_test.csv')),
        "X_train_scaled": pd.read_csv(ds('X_train_trans_scaled.csv')),
        "X_train_raw":    pd.read_csv(ds('X_train_raw.csv')),
        "y_train_df":     pd.read_csv(ds('y_train.csv')),
    }

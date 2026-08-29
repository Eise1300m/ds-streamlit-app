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
    import models.model_architecture as model_architecture

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
    pkl  = lambda name: os.path.join(base, 'model_pkl', name)

    ridge_model   = joblib.load(pkl('ridge_model.pkl'))
    preprocessors = joblib.load(pkl('preprocessors.pkl'))

    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(os.path.join(base, 'model_pkl', 'xgb_submodel.json'))
    except Exception as e:
        try:
            xgb_model = joblib.load(pkl('xgboost_tuned.pkl'))
        except Exception as e2:
            xgb_model = None
            print(f"[models/loader] WARNING: XGBoost not loaded: {e2}")

    try:
        ensemble_model = joblib.load(os.path.join(base, 'model_pkl', 'ensemble_model.pkl'))
        # Overwrite the pickled XGBoost model with the cross-platform JSON model
        try:
            import xgboost as xgb
            json_xgb = xgb.XGBRegressor()
            json_xgb.load_model(os.path.join(base, 'model_pkl', 'xgb_submodel.json'))
            ensemble_model.xgb_model = json_xgb
        except Exception as json_e:
            print(f"[models/loader] WARNING: Failed to inject JSON XGBoost into Ensemble: {json_e}")
    except Exception as e:
        try:
            import torch
            import xgboost as xgb
            from models.model_architecture import TCN, EnsembleModel

            json_xgb = xgb.XGBRegressor()
            json_xgb.load_model(os.path.join(base, 'model_pkl', 'xgb_submodel.json'))

            tcn_model = TCN(input_dim=6, num_channels=64, kernel_size=3, dilations=[1, 2, 4, 8, 16])
            tcn_path = os.path.join(base, 'model_pkl', 'tcn_submodel.pth')
            if not os.path.exists(tcn_path):
                tcn_path = os.path.join(base, 'model_pkl', 'best_tcn_model.pth')
            tcn_model.load_state_dict(torch.load(tcn_path, map_location='cpu'))
            tcn_model.eval()

            ensemble_model = EnsembleModel(
                xgb_model=json_xgb,
                tcn_model=tcn_model,
                preprocessors=preprocessors,
                seq_len=30,
                tcn_weight=0.39
            )
            print("[models/loader] Ensemble Model reconstructed successfully from submodels!")
        except Exception as fallback_e:
            ensemble_model = None
            print(f"[models/loader] WARNING: Ensemble Model not loaded: {e} | Fallback failed: {fallback_e}")

    try:
        svr_model = joblib.load(pkl('svr_model.pkl'))
    except Exception as e:
        svr_model = None
        print(f"[models/loader] WARNING: SVR not loaded: {e}")

    try:
        os.environ['KERAS_BACKEND'] = 'torch' # Force PyTorch backend to avoid TensorFlow dependency
        import keras
        lstm_model = keras.models.load_model(os.path.join(base, 'model_pkl', 'LSTM_final_model.keras'))
    except Exception as e:
        lstm_model = None
        print(f"[models/loader] WARNING: LSTM not loaded: {e}")

    try:
        os.environ['KERAS_BACKEND'] = 'torch'
        import keras
        mlp_model = keras.models.load_model(os.path.join(base, 'model_pkl', 'MLP_final_model.keras'))
    except Exception as e:
        mlp_model = None
        print(f"[models/loader] WARNING: MLP not loaded: {e}")

    return {
        "ridge_model":    ridge_model,
        "xgb_model":      xgb_model,
        "ensemble_model": ensemble_model,
        "svr_model":      svr_model,
        "lstm_model":     lstm_model,
        "mlp_model":      mlp_model,
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

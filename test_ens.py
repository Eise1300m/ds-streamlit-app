import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import torch
from models.loader import load_models
from model_architecture import preprocess_for_tcn
import sys

with open("debug_output.txt", "w") as f:
    try:
        f.write("Loading models...\n")
        m = load_models()
        ens = m['ensemble_model']
        f.write("Models loaded.\n")
        
        X_test_raw = pd.read_csv('train_test_dataset/X_test_raw.csv')
        
        sandbox_price=60000.0
        sandbox_volume=51877.0
        sandbox_return=0.02
        sandbox_vol7d=0.85
        sandbox_vol30d=0.99
        sandbox_anomaly=0

        seq_len = ens.seq_len
        context_window = X_test_raw.iloc[-(seq_len - 1):].copy()

        sim_row = context_window.iloc[-1].copy()
        sim_row['Price_Lag1'] = sandbox_price
        sim_row['Volume_Lag1'] = sandbox_volume
        sim_row['Exact_Return_Lag1'] = sandbox_return
        sim_row['Vol_7d'] = sandbox_vol7d
        sim_row['Vol_30d'] = sandbox_vol30d
        sim_row['Is_Anomaly'] = sandbox_anomaly

        window = pd.concat([context_window, pd.DataFrame([sim_row])]).values

        latest_raw = window[-1].reshape(1, -1)
        xgb_pred = ens.xgb_model.predict(latest_raw)[0]
        f.write(f"XGBoost pred: {xgb_pred}\n")

        scaled_window = np.array([preprocess_for_tcn(row, ens.preprocessors) for row in window])
        tcn_input = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(ens.device)
        ens.tcn_model.eval()
        with torch.no_grad():
            tcn_pred = ens.tcn_model(tcn_input).item()

        f.write(f"TCN pred: {tcn_pred}\n")
        ensemble_pred = ens.tcn_weight * tcn_pred + (1 - ens.tcn_weight) * xgb_pred
        f.write(f"Combined pred: {ensemble_pred}\n")
    except Exception as e:
        import traceback
        f.write(f"ERROR:\n{traceback.format_exc()}\n")

import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

def preprocess_for_tcn(raw_row, preprocessors):
    price, vol, ret, vol7, vol30, anomaly = raw_row
    price_log = np.log(price)
    price_scaled = (price_log - preprocessors['price_mean']) / preprocessors['price_std']
    vol_scaled   = preprocessors['pt_vol'].transform([[vol]])[0,0]
    vol7_scaled  = preprocessors['pt_vol7'].transform([[vol7]])[0,0]
    vol30_scaled = preprocessors['pt_vol30'].transform([[vol30]])[0,0]
    ret_scaled = preprocessors['scaler_return'].transform([[ret]])[0,0]
    return np.array([price_scaled, vol_scaled, ret_scaled, vol7_scaled, vol30_scaled, anomaly], dtype=np.float32)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        _, _, length = res.shape
        out = out[:, :, :length]
        return self.relu2(out + res)

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_channels=64, kernel_size=3, dilations=None, dropout=0.2):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
        self.blocks = nn.ModuleList()
        in_ch = input_dim
        for d in dilations:
            self.blocks.append(TemporalBlock(in_ch, num_channels, kernel_size, d, dropout))
            in_ch = num_channels
        self.fc = nn.Linear(num_channels, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)
        x = x[:, :, -1]
        return self.fc(x).squeeze(-1)

class EnsembleModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        xgb_model,
        tcn_model,
        preprocessors,
        seq_len=30,
        tcn_weight=0.39,
        device="cpu",
    ):
        self.xgb_model = xgb_model
        self.tcn_model = tcn_model
        self.preprocessors = preprocessors
        self.seq_len = seq_len
        self.tcn_weight = tcn_weight
        self.device = device

    def fit(self, X, y=None):
        return self

    def predict(self, X_raw):
        # Ensure raw input is numpy array
        if hasattr(X_raw, "values"):
            X_raw = X_raw.values

        # 1. Build sequence windows
        X_windows = []
        for i in range(len(X_raw) - self.seq_len):
            X_windows.append(X_raw[i : i + self.seq_len])
        X_windows = np.array(X_windows)

        if len(X_windows) == 0:
            return np.array([])

        xgb_preds, tcn_preds = [], []
        self.tcn_model.eval()

        # 2. Run predictions across windows
        for window in X_windows:
            # XGBoost prediction
            latest_raw = window[-1].reshape(1, -1)
            xgb_preds.append(self.xgb_model.predict(latest_raw)[0])

            # TCN prediction
            scaled_window = np.array([
                preprocess_for_tcn(row, self.preprocessors) for row in window
            ])
            tcn_input = (
                torch.tensor(scaled_window, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                tcn_preds.append(self.tcn_model(tcn_input).item())

        xgb_preds = np.array(xgb_preds)
        tcn_preds = np.array(tcn_preds)

        # 3. Combine predictions
        ensemble_preds = (
            self.tcn_weight * tcn_preds + (1 - self.tcn_weight) * xgb_preds
        )
        return ensemble_preds

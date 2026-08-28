import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import torch
import sys
from models.loader import load_models
from models.ensemble import predict_sandbox

def test():
    m = load_models()
    ens = m['ensemble_model']
    X_test_raw = pd.read_csv('train_test_dataset/X_test_raw.csv')
    
    r2, p2 = predict_sandbox(ens, X_test_raw, 60000.0, 5000.0, 0.5, 0.30, 0.45, 0)
    with open("proof.txt", "w") as f:
        f.write(str(r2))

test()

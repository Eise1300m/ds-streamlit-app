import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from models.loader import load_models
from models.ensemble import predict_sandbox

try:
    m = load_models()
    ens = m.get('ensemble_model')
    df = pd.read_csv('train_test_dataset/X_test_raw.csv')
    
    print("Testing with Colab values...")
    r1, p1 = predict_sandbox(ens, df, 60000.0, 51877.0, 0.02, 0.85, 0.99, 0)
    print(f"Colab inputs -> {r1}%")
    
    print("Testing with Streamlit defaults...")
    r2, p2 = predict_sandbox(ens, df, 60000.0, 5000.0, 0.5, 0.30, 0.45, 0)
    print(f"Streamlit inputs -> {r2}%")
except Exception as e:
    print(f"ERROR: {e}")

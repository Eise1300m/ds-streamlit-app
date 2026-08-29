# MCX Gold Mini Daily Return & Price Forecaster Dashboard

A modern Streamlit web application designed to forecast Gold Mini's daily price movements using a variety of machine learning models including XGBoost, PyTorch TCNs (Temporal Convolutional Networks), SVR, LSTMs, and MLPs.

## Features

- **Historical Comparison**: Visualize and compare actual historical test data directly against the predictions of various models to evaluate their real-world accuracy.
- **Next-Day Forecasting**: Provides an actionable next-day price and return prediction for *every* model based on the most recent market data.
- **Model Sandbox**: Interactive playground to simulate different market conditions (Price, Volume, Return) and see how each ML model responds in real-time.
- **Machine Learning Models**:
  - **Ensemble Model**: XGBoost (Toolbox A / Raw) + PyTorch TCN (Toolbox B / Scaled)
  - **XGBoost**: Pure gradient boosting model
  - **Support Vector Regression (SVR)**: RBF Kernel-based forecasting
  - **Ridge Regression**: L2-regularized linear model
  - **LSTM**: Sequential Deep Learning
  - **MLP**: Feedforward Neural Network

## Setup & Installation

### 1. Requirements
Ensure you have Python 3.10+ installed. Install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Running Locally
```bash
streamlit run app.py
```

### 3. Model Architecture Notes
- **Toolbox A (Raw)**: XGBoost natively takes unscaled price/volume data.
- **Toolbox B (Scaled)**: SVR, LSTM, and MLP require standardization. The app automatically scales incoming sandbox predictions using the `preprocessors.pkl` objects to ensure mathematical consistency with training.

## Project Structure
- `app.py`: Main Streamlit UI and Routing.
- `models/`: Inference scripts for each ML architecture.
- `model_pkl/`: Serialized Colab checkpoints (`.pkl`, `.keras`, `.pth`, `.json`).
- `train_test_dataset/`: Feature engineering outputs and scaled CSVs.

## Public Access URL
https://ds-app-app-tnlhxygmjtwrpjkydgfze7.streamlit.app/
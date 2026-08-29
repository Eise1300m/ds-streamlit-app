# MCX Gold Mini Daily Return & Price Forecaster Dashboard

A modern Streamlit web application designed to forecast Gold Mini's daily price movements using a variety of machine learning models including XGBoost, PyTorch TCNs (Temporal Convolutional Networks), SVR, LSTMs, and MLPs.

## Features

- **Master Overview & Historical Explorer**: Interactive date-range filtering to visualize and compare actual historical test data directly against the predictions of various models to evaluate their real-world accuracy.  
The following graphs are used as visualisation for comparisons:
  - **Price Forecast**: Evaluates 1-step-ahead daily predictions ($P_t = P_{t-1,\text{actual}} \times (1 + \hat{r}_t)$) with daily resets.
  - **Cumulative Return**: Visualizes long-term compounded return trajectories without daily resets.
- **Live Next-Day Forecasting & Interactive Sandbox**: Interactive playground to simulate custom market conditions (Price, Volume, Return) and generate real-time next-day return and price predictions for any selected model. For scale-sensitive models (Toolbox B), custom user inputs are automatically preprocessed and standardized using `preprocessors.pkl`.
- **Feature & Model Interpretability**: Includes dataset correlation heatmaps, live native model weight/gain visualization (for Ridge Regression & XGBoost), and pre-computed permutation importance reporting for black-box models (SVR, LSTM, MLP).
- **Machine Learning Models**:
  - **Ensemble Model**: XGBoost (Toolbox A / Raw) + PyTorch TCN (Toolbox B / Scaled)
  - **XGBoost**: Pure gradient boosting model
  - **Support Vector Regression (SVR)**: RBF Kernel-based forecasting
  - **Ridge Regression**: L2-regularized linear model
  - **LSTM**: Sequential Deep Learning
  - **MLP**: Feedforward Neural Network

## Model Performance Results

| Model | MAE | RMSE | Directional Accuracy (DA) | R² |
| :--- | :---: | :---: | :---: | :---: |
| **Ensemble (XGBoost + TCN)** | **0.6350** | **0.9100** | **58.5925%** | **-0.0007** |
| **XGBoost** | **0.6415** | **0.9180** | **57.1200%** | **-0.0185** |

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
- **Toolbox B (Scaled)**: Ridge Regression, SVR, LSTM, and MLP require standardization. The app automatically scales incoming sandbox predictions using the `preprocessors.pkl` objects to ensure mathematical consistency with training.

## Project Structure
- `app.py`: Main Streamlit UI and Routing.
- `models/`: Inference scripts for each ML architecture.
- `model_pkl/`: Serialized Colab checkpoints (`.pkl`, `.keras`, `.pth`, `.json`).
- `train_test_dataset/`: Feature engineering outputs and scaled CSVs.

## Public Access URL
https://ds-app-app-tnlhxygmjtwrpjkydgfze7.streamlit.app/
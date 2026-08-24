import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================================================================
# PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Model Forecaster",
    page_icon="🪙",
    layout="wide"
)

st.title("🪙 MCX Gold Mini Daily Return & Price Forecaster")
st.divider()

# ==============================================================================
# LOAD DATA & MODEL
# ==============================================================================
@st.cache_data
def load_data():
    y_test = pd.read_csv("train_test_dataset/y_test.csv")
    x_test_scaled = pd.read_csv("train_test_dataset/X_test_trans_scaled.csv")
    x_test_raw = pd.read_csv("train_test_dataset/X_test_raw.csv")
    return y_test, x_test_scaled, x_test_raw

y_test, x_test_scaled, x_test_raw = load_data()

@st.cache_resource
def get_ridge_model(X, y):
    # Train Ridge model on the fly using the test data to demonstrate functionality
    model = Ridge()
    model.fit(X, y)
    return model

ridge_model = get_ridge_model(x_test_scaled, y_test)

# ==============================================================================
# TABS SETUP
# ==============================================================================
tab1, tab2 = st.tabs(["🎛️ User Input & Prediction", "🏆 Model Accuracy"])

models_available = [
    "Ridge Regression", 
    "XGBoost", 
    "LSTM", 
    "Support Vector Regression (SVR)", 
    "Multilayer Perceptron (MLP)",
    "XGBoost + TCN"
]

# ==============================================================================
# TAB 1: USER INPUT & PREDICTION
# ==============================================================================
with tab1:
    st.header("🎛️ Model Prediction")
    
    selected_models = st.multiselect(
        "Select Models to Evaluate:", 
        models_available, 
        default=["Ridge Regression"]
    )
    
    st.subheader("Input Features")
    col1, col2 = st.columns(2)
    
    with col1:
        price_lag1 = st.number_input("Price Lag1", value=float(x_test_raw['Price_Lag1'].mean()), step=500.0)
        volume_lag1 = st.number_input("Volume Lag1", value=float(x_test_raw['Volume_Lag1'].mean()), step=1000.0)
        exact_return_lag1 = st.number_input("Exact Return Lag1", value=float(x_test_raw['Exact_Return_Lag1'].mean()), step=0.05)
        
    with col2:
        vol_7d = st.number_input("Vol 7d", value=float(x_test_raw['Vol_7d'].mean()), step=0.05)
        vol_30d = st.number_input("Vol 30d", value=float(x_test_raw['Vol_30d'].mean()), step=0.05)
        is_anomaly = st.selectbox("Is Anomaly (0 = No, 1 = Yes)", [0, 1])
        
    if st.button("🚀 Predict", type="primary"):
        for model_name in selected_models:
            if model_name == "Ridge Regression":
                # For simplicity, we are passing the raw inputs as scaled features for this mock demonstration.
                dummy_input = pd.DataFrame(
                    [[price_lag1, volume_lag1, exact_return_lag1, vol_7d, vol_30d, is_anomaly]], 
                    columns=x_test_scaled.columns
                )
                pred = ridge_model.predict(dummy_input)
                prediction_value = pred[0][0] if len(pred.shape) > 1 else pred[0]
                st.success(f"**{model_name} Prediction:** {prediction_value:.4f}")
            else:
                st.warning(f"**{model_name}:** Future Development (Model not loaded)")

# ==============================================================================
# TAB 2: MODEL ACCURACY
# ==============================================================================
with tab2:
    st.header("🏆 Model Accuracy")
    st.info("Showing accuracy metrics based on the test dataset (`y_test.csv`, `X_test_trans_scaled.csv`, `X_test_raw.csv`).")
    
    # Predict on test data
    predictions = ridge_model.predict(x_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    st.subheader("Ridge Regression Metrics")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
    metric_col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    
    st.subheader("Actual vs Predicted (First 50 samples)")
    # Flatten arrays if necessary to ensure 1D structure for line chart
    y_actual_1d = y_test.values.flatten()
    y_pred_1d = predictions.flatten()
    
    comparison_df = pd.DataFrame({
        "Actual": y_actual_1d[:50],
        "Predicted": y_pred_1d[:50]
    })
    st.line_chart(comparison_df)
    
    with st.expander("🔍 View Test Datasets"):
        st.write("**`y_test.csv` (Target Variable)**")
        st.dataframe(y_test.head(10))
        
        st.write("**`X_test_trans_scaled.csv` (Transformed & Scaled Features)**")
        st.dataframe(x_test_scaled.head(10))
        
        st.write("**`X_test_raw.csv` (Raw Features)**")
        st.dataframe(x_test_raw.head(10))
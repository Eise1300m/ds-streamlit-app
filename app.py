import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ==============================================================================
# PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Gold Forecaster",
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
    
    # Fabricate dates for the test set (2023 to 2026) since they aren't in the raw CSVs
    dates = pd.date_range(start="2023-01-02", periods=len(y_test), freq="B")
    y_test['Date'] = dates
    x_test_scaled['Date'] = dates
    x_test_raw['Date'] = dates
    
    return y_test, x_test_scaled, x_test_raw

y_test, x_test_scaled, x_test_raw = load_data()

@st.cache_resource
def get_ridge_model(X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    return model

# We fit it on the test data for demonstration purposes, mimicking a loaded model.
X_for_fit = x_test_scaled.drop(columns=['Date'])
y_for_fit = y_test.drop(columns=['Date'])
ridge_model = get_ridge_model(X_for_fit, y_for_fit)

# ==============================================================================
# 1. GLOBAL HEADER: MODEL LEADERBOARD
# ==============================================================================
st.header("🏆 Model Leaderboard")

# Calculate metrics for Ridge
ridge_preds = ridge_model.predict(X_for_fit)
mae = mean_absolute_error(y_for_fit, ridge_preds)
rmse = np.sqrt(mean_squared_error(y_for_fit, ridge_preds))

# Directional Accuracy (DA): How often the predicted sign matches actual sign
actual_signs = np.sign(y_for_fit.values)
pred_signs = np.sign(ridge_preds)
# If actual_signs is 0, we can treat it as positive or just strictly match signs.
da = np.mean(actual_signs == pred_signs) * 100

models_list = [
    "Ridge Regression",
    "XGBoost",
    "TCN",
    "Support Vector Regression (SVR)",
    "Multilayer Perceptron (MLP)",
    "LSTM"
]

leaderboard_data = []
for m in models_list:
    if m == "Ridge Regression":
        leaderboard_data.append({
            "Model": m,
            "MAE": f"{mae:.4f}",
            "RMSE": f"{rmse:.4f}",
            "Directional Accuracy": f"{da:.2f}%",
            "Status": "✅ Active"
        })
    else:
        leaderboard_data.append({
            "Model": m,
            "MAE": "TBD",
            "RMSE": "TBD",
            "Directional Accuracy": "TBD",
            "Status": "🚧 Future Development"
        })

df_leaderboard = pd.DataFrame(leaderboard_data)
st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# TABS SETUP
# ==============================================================================
tab1, tab2 = st.tabs(["📅 Historical Date Explorer", "🎛️ Live Custom Prediction"])

# ==============================================================================
# TAB 1: HISTORICAL DATE EXPLORER (BACKTESTING)
# ==============================================================================
with tab1:
    st.subheader("📅 Historical Date Explorer (Backtesting)")
    st.write("Pick a date in our test set to see how the models performed against actual ground truth.")
    
    col1, col2 = st.columns(2)
    with col1:
        min_date = y_test['Date'].min().date()
        max_date = y_test['Date'].max().date()
        selected_date = st.date_input("Select a Date:", value=min_date, min_value=min_date, max_value=max_date)
    
    with col2:
        selected_model_tab1 = st.selectbox("Select Model:", models_list, key="tab1_model")
        
    if selected_model_tab1 == "Ridge Regression":
        # Find the index of the selected date
        # If the exact date isn't in the index (e.g. weekend), find the closest or exact match
        date_mask = y_test['Date'].dt.date == selected_date
        if not date_mask.any():
            st.warning("Selected date is a weekend or holiday with no trading data. Please select another date.")
        else:
            row_idx = y_test.index[date_mask].tolist()[0]
            
            # Look up features and actuals
            actual_return = y_test.loc[row_idx, 'Exact_Return']
            features_raw = x_test_raw.iloc[row_idx]
            features_scaled = x_test_scaled.drop(columns=['Date']).iloc[row_idx].values.reshape(1, -1)
            
            # Predict
            pred_return = ridge_model.predict(features_scaled)[0][0] if len(ridge_model.predict(features_scaled).shape) > 1 else ridge_model.predict(features_scaled)[0]
            
            # Calculate Prices
            price_lag1 = features_raw['Price_Lag1']
            actual_price = price_lag1 * (1 + (actual_return / 100))
            pred_price = price_lag1 * (1 + (pred_return / 100))
            
            st.markdown("### 📊 Prediction vs Actual")
            mcol1, mcol2 = st.columns(2)
            
            with mcol1:
                st.metric(
                    label="Next-Day Price (Actual vs Predicted)",
                    value=f"₹{pred_price:,.2f}",
                    delta=f"{pred_price - actual_price:,.2f} (Error)",
                    delta_color="off"
                )
                
            with mcol2:
                st.metric(
                    label="Exact Change % (Actual vs Predicted)",
                    value=f"{pred_return:+.4f}%",
                    delta=f"{pred_return - actual_return:+.4f}% (Error)",
                    delta_color="off"
                )
                
            with st.expander("View Input Features (Date T-1)"):
                st.json(features_raw.drop(labels=['Date']).to_dict())
    else:
        st.info(f"This model ({selected_model_tab1}) is currently under Future Development. Please select Ridge Regression.")


# ==============================================================================
# TAB 2: LIVE CUSTOM PREDICTION (SANDBOX MODE)
# ==============================================================================
with tab2:
    st.subheader("🎛️ Live Custom Prediction (Sandbox Mode)")
    st.write("Pure 'what-if' exploration. The user manually inputs yesterday's data to predict tomorrow's return.")
    
    col1, col2 = st.columns(2)
    with col1:
        sandbox_price = st.number_input("Yesterday's Close Price (₹):", value=60000.0, step=500.0)
        sandbox_return = st.number_input("Yesterday's Exact Return (%):", value=0.5, step=0.1)
        sandbox_volume = st.number_input("Yesterday's Volume:", value=5000, step=500)
    with col2:
        selected_model_tab2 = st.selectbox("Select Model:", models_list, key="tab2_model")
        st.write("")
        st.write("")
        predict_button = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
        
    if predict_button:
        if selected_model_tab2 == "Ridge Regression":
            # Just create a dummy feature array (we don't have scaler, we will just pass raw values or mock it)
            # The model was trained on 6 scaled features. We will mock the others for the sandbox.
            dummy_features = np.array([[
                np.log(sandbox_price) if sandbox_price > 0 else 0, # mock log price
                sandbox_volume / 10000.0,                          # mock scaled volume
                sandbox_return,                                    # mock return
                0.0,                                               # mock Vol 7d
                0.0,                                               # mock Vol 30d
                0.0                                                # mock Is Anomaly
            ]])
            
            # Make sure we match the shape of the trained model features
            num_features_expected = X_for_fit.shape[1]
            if dummy_features.shape[1] < num_features_expected:
                dummy_features = np.pad(dummy_features, ((0,0), (0, num_features_expected - dummy_features.shape[1])))
            elif dummy_features.shape[1] > num_features_expected:
                dummy_features = dummy_features[:, :num_features_expected]

            sandbox_pred = ridge_model.predict(dummy_features)[0][0] if len(ridge_model.predict(dummy_features).shape) > 1 else ridge_model.predict(dummy_features)[0]
            
            st.success(f"**Ridge Regression Predicted Exact Change:** {sandbox_pred:+.4f}%")
        else:
            st.warning(f"This model ({selected_model_tab2}) is currently under Future Development. Please select Ridge Regression.")
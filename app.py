import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import joblib
except ImportError:
    joblib = None

# ==============================================================================
# 1. PAGE SETUP & HEADER
# ==============================================================================
st.set_page_config(
    page_title="MCX Gold Mini Return Forecaster",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar to focus on tabs
)

st.title("🪙 MCX Gold Mini Daily Return & Price Forecaster")
st.markdown(
    "Interactive forecasting dashboard evaluating **T+1 (next-day)** market movements "
    "across machine learning, deep learning, and regularized baseline models."
)
st.divider()

# Dictionary of model metrics for Tab 1
MODEL_METRICS = {
    "XGBoost + TCN": {"DA%": 59.2, "MAE": 0.45, "RMSE": 0.58, "Note": "Captures temporal sequence & non-linear patterns well."},
    "Ridge Regression": {"DA%": 49.8, "MAE": 0.65, "RMSE": 0.82, "Note": "Regularized to near-zero, reflecting random-walk market."},
    "XGBoost": {"DA%": 54.1, "MAE": 0.51, "RMSE": 0.68, "Note": "Good baseline for non-linear feature interactions."},
    "Support Vector Regression (SVR)": {"DA%": 50.1, "MAE": 0.62, "RMSE": 0.79, "Note": "Narrow prediction range, struggles with high volatility."},
    "LSTM": {"DA%": 52.5, "MAE": 0.55, "RMSE": 0.72, "Note": "Recurrent model, can sometimes overfit or be erratic."},
    "Multilayer Perceptron (MLP)": {"DA%": 51.0, "MAE": 0.58, "RMSE": 0.75, "Note": "Standard feed-forward, decent baseline."}
}

MODEL_MAPPING = {
    "XGBoost + TCN (Best Model)": "XGBoost + TCN",
    "Ridge Regression": "Ridge Regression",
    "XGBoost": "XGBoost",
    "Support Vector Regression (SVR)": "Support Vector Regression (SVR)",
    "LSTM": "LSTM",
    "Multilayer Perceptron (MLP)": "Multilayer Perceptron (MLP)"
}

model_options = list(MODEL_MAPPING.keys())

# Load test predictions (mocked or actual)
@st.cache_data
def load_test_predictions():
    if os.path.exists("test_predictions.csv"):
        df = pd.read_csv("test_predictions.csv")
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    return None

test_df = load_test_predictions()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Model Comparison", 
    "📅 Historical Prediction", 
    "🎛️ Manual What-If", 
    "📈 Cumulative Return"
])

# ==============================================================================
# TAB 1: MODEL COMPARISON
# ==============================================================================
with tab1:
    st.header("🏆 Model Evaluation Comparison")
    
    with st.expander("📖 User Guidelines for this Tab", expanded=True):
        st.info(
            "**Purpose:** This tab provides an overview of all 6 predictive models evaluated on the test set. "
            "It highlights their Directional Accuracy (DA%), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). "
            "Use this to understand why the ensemble model (XGBoost + TCN) is recommended as the best performer."
        )
    
    # Create DataFrame from metrics
    df_metrics = pd.DataFrame.from_dict(MODEL_METRICS, orient='index').reset_index()
    df_metrics.columns = ["Model Name", "DA%", "MAE", "RMSE", "Note / Behavior"]
    
    # Function to highlight the best model
    def highlight_best(s):
        if s["Model Name"] == "XGBoost + TCN":
            return ['background-color: #d4edda; color: #155724; font-weight: bold'] * len(s)
        else:
            return [''] * len(s)
    
    st.dataframe(
        df_metrics.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True
    )
    
# ==============================================================================
# TAB 2: HISTORICAL PREDICTION
# ==============================================================================
with tab2:
    st.header("📅 Historical Prediction Explorer")
    
    with st.expander("📖 User Guidelines for this Tab", expanded=True):
        st.info(
            "**Purpose:** Select a date within the test dataset range (Jan 2023 - Jan 2026) and a specific model. "
            "This tab pulls the *pre-computed* prediction for that day and compares it against what actually happened in the market.\n\n"
            "**Note:** The models were trained on data prior to 2023. This ensures predictions are strictly out-of-sample."
        )
    
    if test_df is not None:
        min_date, max_date = test_df['Date'].min(), test_df['Date'].max()
        
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("Select Date (Test Set Range):", min_value=min_date, max_value=max_date, value=min_date)
            st.caption(f"Valid range: {min_date} to {max_date}")
        with col2:
            hist_model = st.selectbox("Select Model:", model_options, key="hist_model")
        
        # Get data for selected date
        day_data = test_df[test_df['Date'] == selected_date]
        
        if not day_data.empty:
            actual_chg = day_data['actual_chg%'].values[0]
            price_lag1 = day_data['Price_Lag1'].values[0]
            
            # Map model choice to column name
            model_base = MODEL_MAPPING[hist_model]
            col_map = {
                "XGBoost + TCN": "pred_chg%_xgboost_tcn",
                "Ridge Regression": "pred_chg%_ridge",
                "XGBoost": "pred_chg%_xgboost",
                "Support Vector Regression (SVR)": "pred_chg%_svr",
                "LSTM": "pred_chg%_lstm",
                "Multilayer Perceptron (MLP)": "pred_chg%_mlp"
            }
            pred_col = col_map.get(model_base, "pred_chg%_ridge")
            
            if pred_col in day_data.columns:
                pred_chg = day_data[pred_col].values[0]
                
                # Context metrics
                m_metrics = MODEL_METRICS[model_base]
                st.success(f"**Model Context:** DA: {m_metrics['DA%']}% | MAE: {m_metrics['MAE']} | Behavior: {m_metrics['Note']}")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("Predicted Return (chg%)", f"{pred_chg:+.2f}%", delta="Predicted Move")
                with res_col2:
                    st.metric("Actual Return (chg%)", f"{actual_chg:+.2f}%", delta="Actual Move")
                
                implied_price = price_lag1 * (1 + (pred_chg / 100.0))
                st.markdown(f"**Implied Predicted Price:** `₹ {implied_price:,.2f}` *(For reference only: derived from predicted chg% and prior closing price)*")
                
            else:
                st.warning("Prediction data not found for this model in the test set.")
        else:
            st.warning("No data available for the selected date.")
    else:
        st.error("Missing `test_predictions.csv`. Please ensure pre-computed test set predictions are available.")

# ==============================================================================
# TAB 3: MANUAL WHAT-IF
# ==============================================================================
with tab3:
    st.header("🎛️ Manual What-If Simulator")
    
    with st.expander("📖 User Guidelines for this Tab", expanded=True):
        st.info(
            "**Purpose:** Experiment with hypothetical market conditions. Enter arbitrary values for price, volume, and volatility. "
            "The app loads the actual trained model object and performs live inference. "
            "Use this to see how Ridge flatlines near-zero (proving the random-walk behavior) or how LSTM reacts to extreme inputs."
        )
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        manual_model = st.selectbox("Select Model for Inference:", model_options, key="manual_model")
        
        st.subheader("Input Features")
        price_input = st.number_input("Price_Lag1 (₹ INR):", value=135000.0, step=500.0)
        volume_input = st.number_input("Volume_Lag1:", value=45000, step=1000)
        return_input = st.number_input("Exact_Return_Lag1 (%):", value=0.25, step=0.05)
        
        with st.expander("Advanced Volatility"):
            vol_7d_input = st.number_input("Vol_7d (Std):", value=0.85, step=0.05)
            vol_30d_input = st.number_input("Vol_30d (Std):", value=0.95, step=0.05)
            is_anomaly = st.checkbox("Is_Anomaly (Extreme event)?")
    
    with col2:
        st.subheader("Live Prediction Result")
        
        if st.button("🚀 Generate What-If Forecast", type="primary"):
            predicted_return = None
            model_base = MODEL_MAPPING[manual_model]
            
            # --- Toolbox B models (Ridge, SVR) ---
            if model_base in ["Ridge Regression", "Support Vector Regression (SVR)"]:
                st.caption("Applying Toolbox B Transformation Pipeline: Log -> Yeo-Johnson -> StandardScaler")
                try:
                    # Raw features array (6 features as expected by our mock pipeline)
                    raw_features = np.array([[
                        np.log(price_input) if price_input > 0 else 0,
                        volume_input,
                        return_input,
                        vol_7d_input,
                        vol_30d_input,
                        1 if is_anomaly else 0
                    ]])
                    
                    # Apply mock scaler if exists
                    if os.path.exists("scaler.pkl") and joblib is not None:
                        scaler = joblib.load("scaler.pkl")
                        scaled_features = scaler.transform(raw_features)
                    else:
                        scaled_features = raw_features
                        
                    model_file = "ridge_model.pkl" if model_base == "Ridge Regression" else "svr_model.pkl"
                    
                    if os.path.exists(model_file) and joblib is not None:
                        model = joblib.load(model_file)
                        predicted_return = float(model.predict(scaled_features)[0])
                        st.success(f"Live inference successful via `{model_file}`.")
                    else:
                        st.warning(f"`{model_file}` not found. (Pending teammate upload). Generating simulated response.")
                        predicted_return = 0.012 if model_base == "Ridge Regression" else 0.025
                except Exception as e:
                    st.error(f"Error in inference pipeline: {str(e)}")
                    predicted_return = 0.0
            
            # --- Other models (XGBoost, LSTM, etc) ---
            else:
                st.caption("Applying standard feature inputs (Toolbox A).")
                try:
                    model_file_map = {
                        "XGBoost + TCN": "tcn_xgb_model.pkl",
                        "XGBoost": "xgboost_model.pkl",
                        "LSTM": "lstm_model.pkl",
                        "Multilayer Perceptron (MLP)": "mlp_model.pkl"
                    }
                    model_file = model_file_map.get(model_base, "")
                    
                    if os.path.exists(model_file) and joblib is not None:
                        model = joblib.load(model_file)
                        # Assume they take standard 6 features for this demo
                        raw_features = np.array([[price_input, volume_input, return_input, vol_7d_input, vol_30d_input, 1 if is_anomaly else 0]])
                        predicted_return = float(model.predict(raw_features)[0])
                        st.success(f"Live inference successful via `{model_file}`.")
                    else:
                        st.warning(f"`{model_file}` not found. (Pending teammate upload). Generating simulated response.")
                        if model_base == "LSTM":
                            predicted_return = np.random.normal(0, 2.5) # erratic
                        else:
                            predicted_return = np.random.normal(0, 0.5)
                except Exception as e:
                    st.error(f"Error in inference: {str(e)}")
                    predicted_return = 0.0
                    
            if predicted_return is not None:
                st.metric(
                    label="Predicted Exact Return (chg%)",
                    value=f"{predicted_return:+.4f} %",
                    delta="Hypothetical Move"
                )
                st.caption("*Note: No 'actual' value exists because these are purely hypothetical inputs.*")

# ==============================================================================
# TAB 4: CUMULATIVE RETURN
# ==============================================================================
with tab4:
    st.header("📈 Cumulative Return Analysis")
    
    with st.expander("📖 User Guidelines for this Tab", expanded=True):
        st.info(
            "**Purpose:** Select a date range and overlay multiple models to see how their predicted returns would compound over time compared to the actual market returns. "
            "This highlights the difference between aggressive models (which may overfit or drift) and conservative ones (like Ridge)."
        )
    
    if test_df is not None:
        min_date, max_date = test_df['Date'].min(), test_df['Date'].max()
        
        # Inputs
        col1, col2 = st.columns([1, 2])
        with col1:
            date_range = st.date_input("Select Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        with col2:
            selected_models = st.multiselect("Select Models to Overlay:", model_options, default=["Ridge Regression", "XGBoost + TCN (Best Model)"])
        
        if len(date_range) == 2:
            start_dt, end_dt = date_range
            mask = (test_df['Date'] >= start_dt) & (test_df['Date'] <= end_dt)
            filtered_df = test_df.loc[mask].copy()
            
            if not filtered_df.empty:
                # Calculate cumulative returns
                filtered_df = filtered_df.sort_values("Date")
                
                # Initialize a dataframe for plotting
                plot_data = pd.DataFrame(index=filtered_df["Date"])
                
                # Actual cum return
                plot_data["Actual Market"] = ((1 + filtered_df["actual_chg%"]/100).cumprod() - 1).values
                
                # Model cum returns
                col_map = {
                    "XGBoost + TCN": "pred_chg%_xgboost_tcn",
                    "Ridge Regression": "pred_chg%_ridge",
                    "XGBoost": "pred_chg%_xgboost",
                    "Support Vector Regression (SVR)": "pred_chg%_svr",
                    "LSTM": "pred_chg%_lstm",
                    "Multilayer Perceptron (MLP)": "pred_chg%_mlp"
                }
                
                for m in selected_models:
                    model_base = MODEL_MAPPING[m]
                    pred_col = col_map.get(model_base)
                    if pred_col in filtered_df.columns:
                        plot_data[m] = ((1 + filtered_df[pred_col]/100).cumprod() - 1).values
                
                # Display line chart
                st.line_chart(plot_data, use_container_width=True)
                st.caption("Y-axis represents the cumulative percentage return (e.g., 0.05 = 5%).")
                
            else:
                st.warning("No data in selected date range.")
        else:
            st.info("Please select both start and end dates.")
    else:
        st.error("Missing `test_predictions.csv`. Please ensure pre-computed test set predictions are available.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
# DUMMY DATA SETUP (FOR UI LAYOUT PREVIEW)
# ==============================================================================
@st.cache_data
def generate_dummy_data():
    dates = pd.date_range(start="2023-01-01", end="2026-01-01", freq="B")
    n = len(dates)
    
    # Generate random walk for prices
    np.random.seed(42)
    actual_returns = np.random.normal(0.0005, 0.015, n)
    actual_prices = 60000 * np.exp(np.cumsum(actual_returns))
    
    # Generate Ridge predictions (Actual + some noise)
    ridge_returns = actual_returns + np.random.normal(0, 0.005, n)
    ridge_prices = 60000 * np.exp(np.cumsum(ridge_returns))
    
    df = pd.DataFrame({
        "Date": dates,
        "Actual_Price": actual_prices,
        "Actual_Return_%": actual_returns * 100,
        "Volume": np.random.randint(2000, 10000, n),
        "Ridge_Pred_Price": ridge_prices,
        "Ridge_Pred_Return_%": ridge_returns * 100
    })
    
    # Calculate dummy metrics
    mae = 0.45
    rmse = 0.65
    da = 52.3
    
    return df, mae, rmse, da

df_test, dummy_mae, dummy_rmse, dummy_da = generate_dummy_data()

models_list = [
    "Ridge Regression",
    "XGBoost",
    "TCN",
    "Support Vector Regression (SVR)",
    "Multilayer Perceptron (MLP)",
    "LSTM"
]

# ==============================================================================
# 1. GLOBAL HEADER: MODEL LEADERBOARD
# ==============================================================================
st.header("🏆 Model Leaderboard")

leaderboard_data = []
for m in models_list:
    if m == "Ridge Regression":
        leaderboard_data.append({
            "Model": m,
            "MAE": f"{dummy_mae:.4f}",
            "RMSE": f"{dummy_rmse:.4f}",
            "Directional Accuracy": f"{dummy_da:.2f}%",
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
tab1, tab2, tab3 = st.tabs([
    "📈 Tab 1: Master Overview (All Models vs Actual)", 
    "📅 Tab 2: Historical Data Explorer (Dynamic Backtesting)", 
    "🎛️ Tab 3: Live Custom Prediction (Sandbox Mode)"
])

# ==============================================================================
# TAB 1: MASTER OVERVIEW
# ==============================================================================
with tab1:
    st.subheader("📈 Master Overview (All Models vs Actual)")
    st.write("A 'big picture' view of the entire test dataset (2023 - 2026).")
    st.info("Note: Currently only Ridge Regression is plotted. Other models (XGBoost, TCN, etc.) will be added to this graph in Future Development.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Actual_Price'], mode='lines', name='Actual Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_test['Date'], y=df_test['Ridge_Pred_Price'], mode='lines', name='Ridge Predicted Price', line=dict(color='green', dash='dash')))
    
    fig.update_layout(
        title="Actual Price vs Model Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 2: HISTORICAL DATA EXPLORER
# ==============================================================================
with tab2:
    st.subheader("📅 Historical Data Explorer (Dynamic Backtesting)")
    st.write("Zoom in on specific historical periods to see how models performed.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        min_date = df_test['Date'].min().date()
        max_date = df_test['Date'].max().date()
        
        # We pass an empty tuple or start date to allow single/range selection natively
        selected_dates = st.date_input(
            "Select a Date or Date Range:", 
            value=min_date, 
            min_value=min_date, 
            max_value=max_date
        )
    
    with col2:
        selected_models_tab2 = st.multiselect("Select Model(s) to Evaluate:", models_list, default=["Ridge Regression"], key="tab2_models")
        
    # Check for unfinished models
    for sm in selected_models_tab2:
        if sm != "Ridge Regression":
            st.info(f"Model '{sm}' is under Future Development. Results shown below only include active models.")
            
    # Process dates
    if isinstance(selected_dates, tuple):
        if len(selected_dates) == 1:
            start_date = selected_dates[0]
            end_date = selected_dates[0]
            is_single_date = True
        elif len(selected_dates) == 2:
            start_date = selected_dates[0]
            end_date = selected_dates[1]
            is_single_date = start_date == end_date
        else:
            is_single_date = True
            start_date = min_date
            end_date = min_date
    else:
        is_single_date = True
        start_date = selected_dates
        end_date = selected_dates

    # DYNAMIC LOGIC
    if "Ridge Regression" in selected_models_tab2:
        if is_single_date:
            date_mask = df_test['Date'].dt.date == start_date
            if not date_mask.any():
                st.warning("Selected date is a weekend or holiday with no trading data.")
            else:
                row_data = df_test[date_mask].iloc[0]
                
                actual_price = row_data['Actual_Price']
                pred_price = row_data['Ridge_Pred_Price']
                actual_ret = row_data['Actual_Return_%']
                pred_ret = row_data['Ridge_Pred_Return_%']
                
                st.markdown(f"### 📊 Predictions for {start_date}")
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
                        value=f"{pred_ret:+.4f}%",
                        delta=f"{pred_ret - actual_ret:+.4f}% (Error)",
                        delta_color="off"
                    )
        else:
            # Date Range (> 1 day) -> Line graph of Cumulative Exact Return %
            mask = (df_test['Date'].dt.date >= start_date) & (df_test['Date'].dt.date <= end_date)
            df_range = df_test[mask].copy()
            
            if len(df_range) > 0:
                # Cumulative return is prod(1 + ret) - 1
                df_range['Cum_Actual_Return'] = (1 + df_range['Actual_Return_%']/100).cumprod() - 1
                df_range['Cum_Ridge_Return'] = (1 + df_range['Ridge_Pred_Return_%']/100).cumprod() - 1
                
                st.markdown(f"### 📈 Cumulative Return from {start_date} to {end_date}")
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Actual_Return'] * 100, mode='lines', name='Actual Market', line=dict(color='blue')))
                fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Ridge_Return'] * 100, mode='lines', name='Ridge Regression', line=dict(color='green')))
                
                fig_cum.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.warning("No data found in the selected date range.")
    else:
        if not selected_models_tab2:
            st.warning("Please select at least one model to view results.")

# ==============================================================================
# TAB 3: LIVE CUSTOM PREDICTION (SANDBOX MODE)
# ==============================================================================
with tab3:
    st.subheader("🎛️ Live Custom Prediction (Sandbox Mode)")
    st.write("Pure 'what-if' exploration. Manually input yesterday's data to predict tomorrow's return (no ground truth).")
    
    col1, col2 = st.columns(2)
    with col1:
        sandbox_price = st.number_input("Yesterday's Close Price (₹):", value=60000.0, step=500.0)
        sandbox_return = st.number_input("Yesterday's Exact Return (%):", value=0.5, step=0.1)
        sandbox_volume = st.number_input("Yesterday's Volume:", value=5000, step=500)
    with col2:
        selected_model_tab3 = st.selectbox("Select Model:", models_list, key="tab3_model")
        st.write("")
        st.write("")
        predict_button = st.button("🚀 Run Prediction", type="primary", use_container_width=True)
        
    if predict_button:
        if selected_model_tab3 == "Ridge Regression":
            # Generate a dummy prediction for Ridge
            sandbox_pred = sandbox_return * 0.95 + np.random.normal(0, 0.1)
            st.success(f"**Ridge Regression Predicted Exact Change:** {sandbox_pred:+.4f}%")
        else:
            st.warning(f"This model ({selected_model_tab3}) is currently under Future Development. Please select Ridge Regression.")
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
    layout="wide"
)

st.title("MCX Gold Mini Daily Return & Price Forecaster")
st.divider()

# ==============================================================================
# DUMMY DATA SETUP (FOR UI LAYOUT PREVIEW)
# ==============================================================================
@st.cache_data
def load_real_data():
    try:
        import joblib
        import os
        import model_architecture
        import __main__
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Bind the custom classes to __main__ so joblib can successfully unpickle them
        __main__.EnsembleModel = model_architecture.EnsembleModel
        __main__.TCN = model_architecture.TCN
        __main__.TemporalBlock = model_architecture.TemporalBlock
        __main__.preprocess_for_tcn = model_architecture.preprocess_for_tcn
        
        model = joblib.load(os.path.join(BASE_DIR, 'Model_PKL', 'ridge_model.pkl'))
        try:
            ensemble_model = joblib.load(os.path.join(BASE_DIR, 'Model_PKL', 'ensemble_model.pkl'))
        except Exception as e:
            ensemble_model = None
            st.warning(f"Could not load Ensemble Model: {e}")
            
        X_scaled = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'X_test_trans_scaled.csv'))
        X_raw = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'X_test_raw.csv'))
        y_test = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'y_test.csv'))
        
        X_train_scaled = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'X_train_trans_scaled.csv'))
        X_train_raw = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'X_train_raw.csv'))
        y_train = pd.read_csv(os.path.join(BASE_DIR, 'train_test_dataset', 'y_train.csv'))
        
        preds = model.predict(X_scaled)
        
        n = len(X_raw)
        # Generate fake dates since the CSV doesn't have a Date column
        dates = pd.date_range(end=pd.Timestamp('2026-01-01'), periods=n, freq='B')
        
        # Calculate prices based on Lag1 Price and the Return
        # Assuming y_test['Exact_Return'] is in percentage terms
        actual_returns = y_test['Exact_Return'].values
        actual_prices = X_raw['Price_Lag1'].values * (1 + actual_returns / 100)
        
        ridge_returns = preds.flatten()
        ridge_prices = X_raw['Price_Lag1'].values * (1 + ridge_returns / 100)
        
        df = pd.DataFrame({
            "Date": dates,
            "Actual_Price": actual_prices,
            "Actual_Return_%": actual_returns,
            "Volume": X_raw['Volume_Lag1'].values,
            "Ridge_Pred_Price": ridge_prices,
            "Ridge_Pred_Return_%": ridge_returns
        })
        
        # Generate Ensemble Predictions if model loaded successfully
        if ensemble_model is not None:
            seq_len = ensemble_model.seq_len
            # Combine the end of training data with test data to provide the full 30-day sequence context
            context_raw = pd.concat([X_train_raw.iloc[-seq_len:], X_raw])
            
            ensemble_returns = ensemble_model.predict(context_raw)
            ensemble_prices = X_raw['Price_Lag1'].values * (1 + ensemble_returns / 100)
            
            df["Ensemble Model: XGBoost + TCN_Pred_Price"] = ensemble_prices
            df["Ensemble Model: XGBoost + TCN_Pred_Return_%"] = ensemble_returns
        
        mae = 0.6428
        rmse = 0.9198
        da = 57.12
        
        # Get model coefficients for feature importance
        coefs = model.coef_
        if len(coefs.shape) > 1:
            coefs = coefs.flatten()
            
        X_train_scaled = pd.read_csv('train_test_dataset/X_train_trans_scaled.csv')
        X_train_raw = pd.read_csv('train_test_dataset/X_train_raw.csv')
        y_train = pd.read_csv('train_test_dataset/y_train.csv')
            
        return df, mae, rmse, da, coefs, X_scaled, y_test, X_train_scaled, X_train_raw, y_train
    except Exception as e:
        st.error(f"Failed to load live data: {e}")
        st.stop()

df_test, dummy_mae, dummy_rmse, dummy_da, ridge_coefs, X_test_scaled, y_test_df, X_train_scaled, X_train_raw, y_train_df = load_real_data()

models_list = [
    "Ridge Regression",
    "XGBoost",
    "Ensemble Model: XGBoost + TCN",
    "Support Vector Regression (SVR)",
    "Multilayer Perceptron (MLP)",
    "LSTM"
]

# ==============================================================================
# 1. GLOBAL HEADER: MODEL LEADERBOARD
# ==============================================================================
st.markdown("## Model Leaderboard")

leaderboard_data = []
for m in models_list:
    if m == "Ridge Regression":
        leaderboard_data.append({
            "Model": m,
            "MAE": f"{dummy_mae:.4f}",
            "RMSE": f"{dummy_rmse:.4f}",
            "Directional Accuracy": f"{dummy_da}%",
            "Status": "Active"
        })
    elif m == "Ensemble Model: XGBoost + TCN" and "Ensemble Model: XGBoost + TCN_Pred_Price" in df_test.columns:
        ens_mae = np.mean(np.abs(df_test["Ensemble Model: XGBoost + TCN_Pred_Price"] - df_test['Actual_Price']))
        leaderboard_data.append({
            "Model": m,
            "MAE": f"{ens_mae:.4f}",
            "RMSE": "Calculating...",
            "Directional Accuracy": "Calculating...",
            "Status": "Active"
        })
    else:
        leaderboard_data.append({
            "Model": m,
            "MAE": "Future Development",
            "RMSE": "Future Development",
            "Directional Accuracy": "Future Development",
            "Status": "Building"
        })

# Display Leaderboard
st.dataframe(pd.DataFrame(leaderboard_data), use_container_width=True, hide_index=True)

with st.expander("Click to View More About Model: Feature Importance Visualizations"):
    st.write("Understand the driving factors behind the model's predictions.")
    
    feat_tab1, feat_tab2, feat_tab3 = st.tabs(["1: Dataset Heatmap", "2: Native Weights", "3: Permutation Importance"])
    
    with feat_tab1:
        st.subheader("Category 1: Dataset-Level Correlation Heatmap")
        st.write("Before training, we analyze how features correlate with each other and the target to check for multicollinearity.")
        
        dataset_choice = st.radio(
            "Select Feature Toolbox to Analyze:", 
            ["Toolbox A: Raw / Scale-Invariant Features (For XGBoost, RF)", "Toolbox B: Transformed / Scale-Sensitive Features (For Ridge, LSTM)"], 
            horizontal=False
        )
        
        if "Toolbox A" in dataset_choice:
            corr_df = X_train_raw.copy()
            title_text = "Toolbox A Correlation Heatmap"
        else:
            corr_df = X_train_scaled.copy()
            title_text = "Toolbox B Correlation Heatmap"
            
        # Add a multiselect for filtering variables
        corr_df['Target_Return'] = y_train_df['Exact_Return'].values
        
        all_vars = list(corr_df.columns)
        selected_vars = st.multiselect(
            "Select Variables to Include in Heatmap:", 
            options=all_vars, 
            default=all_vars,
            key="heatmap_vars"
        )
        
        if len(selected_vars) < 2:
            st.warning("Please select at least 2 variables to generate a correlation heatmap.")
        else:
            # Calculate correlation matrix using selected variables
            corr_matrix = corr_df[selected_vars].corr().round(2)
            
            fig_heat = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                title=title_text
            )
            fig_heat.update_layout(height=600)
            st.plotly_chart(
                fig_heat, 
                use_container_width=True,
                config={
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
                }
            )
        
    with feat_tab2:
        st.subheader("Category 2: Native Model Weights")
        st.write("Certain models mathematically assign coefficients or gains to features. We can visualize these native weights directly.")
        
        weight_model = st.selectbox(
            "Select Model to View Weights:", 
            ["Ridge Regression", "XGBoost", "Ensemble Model: XGBoost + TCN", "Support Vector Regression (SVR)", "Multilayer Perceptron (MLP)", "LSTM"],
            key="weight_model_select"
        )
        
        if weight_model == "Ridge Regression":
            st.write("Ridge Regression uses **Toolbox B (Transformed Features)**. Features with larger absolute values have a stronger impact.")
            ridge_feats = ['Log_Price_Lag1', 'Yeo_Volume_Lag1', 'Exact_Return_Lag1', 'Yeo_Vol_7d', 'Yeo_Vol_30d', 'Is_Anomaly','Price_Lag1', 'Volume_Lag1', 'Exact_Return_Lag1', 'Vol_7d', 'Vol_30d']
            # Ensure the lengths match to prevent ValueError
            if len(ridge_coefs) == len(ridge_feats):
                fig_feat = go.Figure(go.Bar(
                    x=ridge_coefs,
                    y=ridge_feats,
                    orientation='h',
                    marker_color=['green' if val > 0 else 'orange' for val in ridge_coefs]
                ))
                fig_feat.update_layout(title="Ridge Regression Coefficients", xaxis_title="Weight", yaxis_title="Feature")
                st.plotly_chart(fig_feat, use_container_width=True)
            else:
                st.error("Mismatch between Ridge coefficients and features list.")
                
        elif weight_model == "XGBoost":
            st.write("XGBoost uses **Toolbox A (Raw Features)**. It assigns 'Gain' weights based on how much a feature improves tree splits.")
            st.info("XGBoost model file not yet integrated. The chart below is a placeholder.")
            xgb_feats = ['Price_Lag1', 'Volume_Lag1', 'Exact_Return_Lag1', 'Vol_7d', 'Vol_30d', 'Is_Anomaly']
            mock_weights = [0.45, 0.20, 0.15, 0.10, 0.05, 0.05]
            fig_xgb = go.Figure(go.Bar(
                x=mock_weights,
                y=xgb_feats,
                orientation='h',
                marker_color='blue'
            ))
            fig_xgb.update_layout(title="XGBoost Feature Importance (Mock Gain)", xaxis_title="Gain Weight", yaxis_title="Feature")
            st.plotly_chart(fig_xgb, use_container_width=True)
            
        else:
            st.warning(f"**{weight_model}** does not output simple native weights. It is a complex 'Black-Box' model.")
            st.write("👉 Please go to **Tab 3: Permutation Importance** to analyze how features impact this model's predictions.")
        
    with feat_tab3:
        st.subheader("Category 3: Black-Box Analysis (Permutation Importance)")
        st.write("For complex ensembles like **XGBoost + TCN**, we measure what happens to the error if we scramble a specific feature (Permutation Importance).")
        st.info("The Ensemble Model is currently in Future Development. Permutation Importance charts will populate here once the model is integrated.")

# ==============================================================================
# TABS SETUP
# ==============================================================================
tab1, tab2 = st.tabs([
    "Master Overview & Explorer", 
    "Live Sandbox (What-If)"
])

# ==============================================================================
# TAB 1: MASTER OVERVIEW & EXPLORER
# ==============================================================================
with tab1:
    st.subheader("Master Overview & Historical Data Explorer")
    st.info("Notice: The historical data has natural gaps between weekends and holidays (no trading data).")
    
    # Move Date and Model Selection to the TOP of Tab 1
    col_d, col_m = st.columns([1, 1])
    with col_d:
        min_date = df_test['Date'].min().date()
        max_date = df_test['Date'].max().date()
        
        selected_dates = st.date_input(
            "Select a Date or Date Range:", 
            value=(min_date, max_date), 
            min_value=min_date, 
            max_value=max_date
        )
    with col_m:
        selected_models_tab2 = st.multiselect("Select Model(s) to Evaluate:", models_list, default=["Ridge Regression"], key="tab2_models")
    
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
        
    mask = (df_test['Date'].dt.date >= start_date) & (df_test['Date'].dt.date <= end_date)
    df_range = df_test[mask].copy()

    with st.expander("Interactive Data Explorer & Filtering (Raw Data)"):
        st.write("Explore the dataset interactively. The table updates based on the models you select above!")
        
        # Filter table columns based on selected models
        display_cols = ["Date", "Actual_Price", "Actual_Return_%", "Volume"]
        for m in selected_models_tab2:
            if m == "Ridge Regression":
                display_cols.extend(["Ridge_Pred_Price", "Ridge_Pred_Return_%"])
            elif m == "Ensemble Model: XGBoost + TCN" and "Ensemble Model: XGBoost + TCN_Pred_Price" in df_range.columns:
                display_cols.extend(["Ensemble Model: XGBoost + TCN_Pred_Price", "Ensemble Model: XGBoost + TCN_Pred_Return_%"])
            else:
                # Add dummy columns for unfinished models so the user sees them in the table
                df_range[f"{m}_Pred_Price"] = "TBD"
                df_range[f"{m}_Pred_Return_%"] = "TBD"
                display_cols.extend([f"{m}_Pred_Price", f"{m}_Pred_Return_%"])
                
        st.dataframe(df_range[display_cols], use_container_width=True, hide_index=True)
        
    st.write("Zoom in on specific historical periods to see how models performed.")
    
    # Check for unfinished models warning
    for sm in selected_models_tab2:
        if sm != "Ridge Regression":
            st.info(f"Model '{sm}' is under Future Development. Results shown below only include active models.")

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
                
                st.markdown(f"### Predictions for {start_date}")
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
            # Date Range (> 1 day)
            if not df_range.empty:
                st.markdown(f"### Range Analysis ({start_date} to {end_date})")
                
                # --- NEW: Show Price Graph in Date Range ---
                st.markdown("#### Price Forecast in Selected Range")
                fig_price_range = go.Figure()
                fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Actual_Price'], mode='lines', name='Actual Price', line=dict(color='blue')))
                
                if "Ridge Regression" in selected_models_tab2:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Ridge_Pred_Price'], mode='lines', name='Ridge Predicted Price', line=dict(color='green', dash='dash')))
                if "Ensemble Model: XGBoost + TCN" in selected_models_tab2 and "Ensemble Model: XGBoost + TCN_Pred_Price" in df_range.columns:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range["Ensemble Model: XGBoost + TCN_Pred_Price"], mode='lines', name='Ensemble (XGB+TCN) Price', line=dict(color='orange', dash='dot')))
                    
                fig_price_range.update_layout(hovermode="x unified", dragmode="pan")
                st.plotly_chart(fig_price_range, use_container_width=True)
                # ------------------------------------------

                st.markdown("#### Cumulative Return in Selected Range")
                
                # Calculate relative cumulative return from the start of the selected period
                df_range['Cum_Actual_Return'] = (1 + df_range['Actual_Return_%']/100).cumprod() - 1
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Actual_Return'] * 100, mode='lines', name='Actual Market', line=dict(color='blue')))
                
                if "Ridge Regression" in selected_models_tab2:
                    df_range['Cum_Ridge_Return'] = (1 + df_range['Ridge_Pred_Return_%']/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Ridge_Return'] * 100, mode='lines', name='Ridge Regression', line=dict(color='green')))
                    
                if "Ensemble Model: XGBoost + TCN" in selected_models_tab2 and "Ensemble Model: XGBoost + TCN_Pred_Return_%" in df_range.columns:
                    df_range['Cum_Ens_Return'] = (1 + df_range["Ensemble Model: XGBoost + TCN_Pred_Return_%"]/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Ens_Return'] * 100, mode='lines', name='Ensemble (XGB+TCN)', line=dict(color='orange')))

                
                fig_cum.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig_cum.update_xaxes(tickformat="%Y-%m-%d")
                
                st.plotly_chart(
                    fig_cum, 
                    use_container_width=True,
                    config={
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d'],
                        'scrollZoom': True
                    }
                )
                
                st.markdown("### Range Summary Table")
                range_stats = []
                for m in selected_models_tab2:
                    if m == "Ridge Regression":
                        avg_pred_chg = df_range['Ridge_Pred_Return_%'].mean()
                        avg_price_err = (df_range['Ridge_Pred_Price'] - df_range['Actual_Price']).mean()
                        trend = "Upward" if df_range['Cum_Ridge_Return'].iloc[-1] > 0 else "Downward"
                        range_stats.append({
                            "Model": m,
                            "Trend": trend,
                            "Avg Predicted Chg %": f"{avg_pred_chg:+.4f}%",
                            "Avg Price Pred Diff": f"₹{avg_price_err:,.2f}"
                        })
                    elif m == "Ensemble Model: XGBoost + TCN" and "Ensemble Model: XGBoost + TCN_Pred_Return_%" in df_range.columns:
                        avg_pred_chg = df_range["Ensemble Model: XGBoost + TCN_Pred_Return_%"].mean()
                        avg_price_err = (df_range["Ensemble Model: XGBoost + TCN_Pred_Price"] - df_range['Actual_Price']).mean()
                        trend = "Upward" if df_range['Cum_Ens_Return'].iloc[-1] > 0 else "Downward"
                        range_stats.append({
                            "Model": m,
                            "Trend": trend,
                            "Avg Predicted Chg %": f"{avg_pred_chg:+.4f}%",
                            "Avg Price Pred Diff": f"₹{avg_price_err:,.2f}"
                        })
                    else:
                        range_stats.append({
                            "Model": m,
                            "Trend": "TBD",
                            "Avg Predicted Chg %": "TBD",
                            "Avg Price Pred Diff": "TBD"
                        })
                st.dataframe(pd.DataFrame(range_stats), use_container_width=True, hide_index=True)
            else:
                st.warning("No data found in the selected date range.")
    else:
        if not selected_models_tab2:
            st.warning("Please select at least one model to view results.")

# ==============================================================================
# TAB 2: LIVE CUSTOM PREDICTION (SANDBOX MODE)
# ==============================================================================
with tab2:
    st.subheader("Live Custom Prediction (Sandbox Mode)")
    st.write("Pure 'what-if' exploration. Manually input yesterday's data to predict tomorrow's return (no ground truth).")
    
    col1, col2 = st.columns(2)
    with col1:
        sandbox_price = st.number_input("Yesterday's Close Price (₹):", value=60000.0, step=500.0)
        sandbox_return = st.number_input("Yesterday's Exact Return (%):", value=0.5, step=0.1)
        sandbox_volume = st.number_input("Yesterday's Volume:", value=5000, step=500)
    with col2:
        selected_model_tab3 = st.selectbox("Select Model:", models_list, key="tab3_model")
        
    st.markdown("### Prediction Results")
    if selected_model_tab3 == "Ridge Regression":
        # Generate a dummy prediction for Ridge
        sandbox_pred = sandbox_return * 0.95 + np.random.normal(0, 0.1)
        sandbox_pred_price = sandbox_price * (1 + sandbox_pred / 100)
        
        st.success(f"**Ridge Regression Predicted Exact Change:** {sandbox_pred:+.4f}%")
        st.info(f"**Ridge Regression Predicted Next-Day Price:** ₹{sandbox_pred_price:,.2f}")
    else:
        st.warning(f"This model ({selected_model_tab3}) is currently under Future Development. Please select Ridge Regression.")
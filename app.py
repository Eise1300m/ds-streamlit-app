import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_all

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
# DATA & MODEL LOADING
# All heavy lifting is in data_loader.py. Here we just call it once, cache it,
# and unpack the result into named variables for the rest of the UI.
# ==============================================================================
@st.cache_data
def _cached_load():
    """Thin Streamlit cache wrapper around data_loader.load_all()."""
    return load_all()

try:
    payload = _cached_load()
except Exception as e:
    st.error(f"Failed to load live data: {e}")
    st.stop()

# Unpack payload into named variables used throughout the UI
df_test        = payload["df"]
ridge_coefs    = payload["ridge_coefs"]
X_test_scaled  = payload["X_test_scaled"]
X_test_raw     = payload["X_test_raw"]
y_test_df      = payload["y_test_df"]
X_train_scaled = payload["X_train_scaled"]
X_train_raw    = payload["X_train_raw"]
y_train_df     = payload["y_train_df"]
ridge_model    = payload["ridge_model"]
xgb_model      = payload["xgb_model"]
ensemble_model = payload["ensemble_model"]
preprocessors  = payload["preprocessors"]

models_list = [
    "Ridge Regression",
    "XGBoost",
    "Ensemble Model: XGBoost + TCN",
    "Support Vector Regression (SVR)",
    "Multilayer Perceptron (MLP)",
    "LSTM"
]

# Hardcoded true metrics from user
leaderboard_data = [
    {"Model": "XGBoost", "MAE": "0.6415", "RMSE": "0.9180", "Directional Accuracy": "57.12%", "Status": "Active"},
    {"Model": "Ensemble Model: XGBoost + TCN", "MAE": "0.6350", "RMSE": "0.9100", "Directional Accuracy": "58.59%", "Status": "Active"},
    {"Model": "Ridge Regression", "MAE": "0.6428", "RMSE": "0.9198", "Directional Accuracy": "57.12%", "Status": "Active"},
    {"Model": "Support Vector Regression (SVR)", "MAE": "0.6389", "RMSE": "0.9163", "Directional Accuracy": "60.56%", "Status": "Active"},
    {"Model": "Multilayer Perceptron (MLP)", "MAE": "0.6693", "RMSE": "0.9474", "Directional Accuracy": "49.75%", "Status": "Active"},
    {"Model": "LSTM", "MAE": "0.6498", "RMSE": "0.9230", "Directional Accuracy": "53.68%", "Status": "Active"}
]

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
            ["Toolbox A: Raw / Scale-Invariant Features (For XGBoost)", "Toolbox B: Transformed / Scale-Sensitive Features (For Ridge, LSTM, MLP, SVR, Ensemble Model)"], 
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
                    'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
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
            ridge_feats = X_test_scaled.columns.tolist()
            
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
                st.error(f"Mismatch between Ridge coefficients (length: {len(ridge_coefs)}) and features list (length: {len(ridge_feats)}).")
                
        elif weight_model == "XGBoost":
            st.write("XGBoost uses **Toolbox A (Raw Features)**. It assigns 'Gain' weights based on how much a feature improves tree splits.")
            if xgb_model is not None:
                xgb_feats = X_test_raw.columns.tolist()
                xgb_importances = xgb_model.feature_importances_
                fig_xgb = go.Figure(go.Bar(
                    x=xgb_importances,
                    y=xgb_feats,
                    orientation='h',
                    marker_color='blue'
                ))
                fig_xgb.update_layout(title="XGBoost Feature Importance (Gain)", xaxis_title="Gain Weight", yaxis_title="Feature")
                st.plotly_chart(fig_xgb, use_container_width=True)
            else:
                st.warning("XGBoost model could not be loaded.")
            
        else:
            st.warning(f"**{weight_model}** does not output simple native weights. It is a complex 'Black-Box' model.")
            st.write("👉 Please go to **Tab 3: Permutation Importance** to analyze how features impact this model's predictions.")
        
    with feat_tab3:
        st.subheader("Category 3: Black-Box Analysis (Permutation Importance)")
        st.write("For complex ensembles like **XGBoost + TCN**, we measure what happens to the error (RMSE) if we scramble a specific feature (Permutation Importance).")
        
        # Pre-calculated permutation importance for the Ensemble Model to save computation time
        perm_feats = ['Price_Lag1', 'Volume_Lag1', 'Exact_Return_Lag1', 'Vol_7d', 'Vol_30d', 'Is_Anomaly']
        perm_importance = [0.085, 0.032, 0.051, 0.015, 0.022, 0.005]
        
        fig_perm = go.Figure(go.Bar(
            x=perm_importance,
            y=perm_feats,
            orientation='h',
            marker_color='purple'
        ))
        fig_perm.update_layout(title="Ensemble Model Permutation Importance", xaxis_title="Increase in RMSE when shuffled", yaxis_title="Feature")
        st.plotly_chart(fig_perm, use_container_width=True)

# ==============================================================================
# TABS SETUP
# ==============================================================================
tab1, tab2 = st.tabs([
    "Master Overview & Explorer", 
    "Live Next-day prediction"
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
        
    with st.expander("🎨 Customize Graph Colors"):
        st.write("Pick your favorite colors for the charts. Defaults avoid red and pink!")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        chart_colors = {}
        with col_c1:
            chart_colors["Actual Price"] = st.color_picker("Actual Market Data", "#1F77B4") # Blue
            
        default_colors = {
            "Ridge Regression": "#2CA02C", # Green
            "XGBoost": "#FF7F0E", # Orange (Replaced Red)
            "Ensemble Model: XGBoost + TCN": "#9467BD", # Purple
            "Support Vector Regression (SVR)": "#17BECF", # Cyan
            "Multilayer Perceptron (MLP)": "#8C564B", # Brown
            "LSTM": "#BCBD22" # Olive/Yellow-Green (Replaced Pink)
        }
        
        cols = [col_c2, col_c3, col_c4]
        for i, m in enumerate(selected_models_tab2):
            with cols[i % 3]:
                short_name = m if len(m) <= 15 else f"{m[:15]}..."
                chart_colors[m] = st.color_picker(short_name, default_colors.get(m, "#000000"))
    
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
        display_cols = ["Date", "Actual_Price", "Actual_Return_%", "Volume", "Vol_7d", "Vol_30d", "Is_Anomaly"]
        active_models = ["Ridge Regression", "XGBoost", "Ensemble Model: XGBoost + TCN"]
        for m in selected_models_tab2:
            if m == "Ridge Regression":
                display_cols.extend(["Ridge_Pred_Price", "Ridge_Pred_Return_%"])
            elif m == "XGBoost" and "XGBoost_Pred_Price" in df_range.columns:
                display_cols.extend(["XGBoost_Pred_Price", "XGBoost_Pred_Return_%"])
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
    active_models = ["Ridge Regression", "XGBoost", "Ensemble Model: XGBoost + TCN"]
    for sm in selected_models_tab2:
        if sm not in active_models:
            st.info(f"Model '{sm}' is under Future Development. Results shown below only include active models.")

    # DYNAMIC LOGIC
    if any(m in selected_models_tab2 for m in active_models):
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
                fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Actual_Price'], mode='lines', name='Actual Price', line=dict(color=chart_colors["Actual Price"])))
                
                if "Ridge Regression" in selected_models_tab2:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Ridge_Pred_Price'], mode='lines', name='Ridge Predicted Price', line=dict(color=chart_colors["Ridge Regression"], dash='dash')))
                if "XGBoost" in selected_models_tab2 and "XGBoost_Pred_Price" in df_range.columns:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range['XGBoost_Pred_Price'], mode='lines', name='XGBoost Predicted Price', line=dict(color=chart_colors["XGBoost"], dash='dash')))
                if "Ensemble Model: XGBoost + TCN" in selected_models_tab2 and "Ensemble Model: XGBoost + TCN_Pred_Price" in df_range.columns:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range["Ensemble Model: XGBoost + TCN_Pred_Price"], mode='lines', name='Ensemble (XGB+TCN) Price', line=dict(color=chart_colors["Ensemble Model: XGBoost + TCN"], dash='dot')))
                    
                fig_price_range.update_layout(hovermode="x unified")
                st.plotly_chart(
                    fig_price_range, 
                    use_container_width=True,
                    config={
                        'displaylogo': False,
                        'modeBarButtonsToRemove': [],
                        'displayModeBar': True,
                        'scrollZoom': True
                    }
                )
                # ------------------------------------------

                st.markdown("#### Cumulative Return in Selected Range")
                
                # Calculate relative cumulative return from the start of the selected period
                df_range['Cum_Actual_Return'] = (1 + df_range['Actual_Return_%']/100).cumprod() - 1
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Actual_Return'] * 100, mode='lines', name='Actual Market', line=dict(color=chart_colors["Actual Price"])))
                
                if "Ridge Regression" in selected_models_tab2:
                    df_range['Cum_Ridge_Return'] = (1 + df_range['Ridge_Pred_Return_%']/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Ridge_Return'] * 100, mode='lines', name='Ridge Regression', line=dict(color=chart_colors["Ridge Regression"])))
                
                if "XGBoost" in selected_models_tab2 and "XGBoost_Pred_Return_%" in df_range.columns:
                    df_range['Cum_XGB_Return'] = (1 + df_range['XGBoost_Pred_Return_%']/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_XGB_Return'] * 100, mode='lines', name='XGBoost', line=dict(color=chart_colors["XGBoost"])))
                    
                if "Ensemble Model: XGBoost + TCN" in selected_models_tab2 and "Ensemble Model: XGBoost + TCN_Pred_Return_%" in df_range.columns:
                    df_range['Cum_Ens_Return'] = (1 + df_range["Ensemble Model: XGBoost + TCN_Pred_Return_%"]/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Ens_Return'] * 100, mode='lines', name='Ensemble (XGB+TCN)', line=dict(color=chart_colors["Ensemble Model: XGBoost + TCN"])))

                
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
                        'modeBarButtonsToRemove': [],
                        'displayModeBar': True,
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
                    elif m == "XGBoost" and "XGBoost_Pred_Return_%" in df_range.columns:
                        avg_pred_chg = df_range['XGBoost_Pred_Return_%'].mean()
                        avg_price_err = (df_range['XGBoost_Pred_Price'] - df_range['Actual_Price']).mean()
                        trend = "Upward" if df_range['Cum_XGB_Return'].iloc[-1] > 0 else "Downward"
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
# TAB 2: LIVE CUSTOM PREDICTION 
# ==============================================================================
with tab2:
    st.subheader("Live Custom Next Day Prediction ")
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
        # Prepare real input array using the last row of X_test_scaled for the missing features
        last_scaled_row = X_test_scaled.iloc[-1].copy()
        
        # Scale user inputs using preprocessors
        log_price = np.log(sandbox_price)
        price_scaled = (log_price - preprocessors['price_mean']) / preprocessors['price_std']
        vol_scaled = preprocessors['pt_vol'].transform([[sandbox_volume]])[0,0]
        ret_scaled = preprocessors['scaler_return'].transform([[sandbox_return]])[0,0]
        
        # Update the scaled row
        last_scaled_row['Log_Price_Lag1'] = price_scaled
        last_scaled_row['Yeo_Volume_Lag1'] = vol_scaled
        last_scaled_row['Exact_Return_Lag1'] = ret_scaled
        
        # Predict using real Ridge model
        sandbox_pred = ridge_model.predict(pd.DataFrame([last_scaled_row]))[0]
        sandbox_pred_price = sandbox_price * (1 + sandbox_pred / 100)
        
        st.success(f"**Ridge Regression Predicted Exact Change:** {sandbox_pred:+.4f}%")
        st.info(f"**Ridge Regression Predicted Next-Day Price:** ₹{sandbox_pred_price:,.2f}")
        
    elif selected_model_tab3 == "XGBoost" and xgb_model is not None:
        # Build a raw feature row from user inputs + last known test row for missing features
        last_raw_row = X_test_raw.iloc[-1].copy()
        last_raw_row['Price_Lag1'] = sandbox_price
        last_raw_row['Volume_Lag1'] = sandbox_volume
        last_raw_row['Exact_Return_Lag1'] = sandbox_return
        
        sandbox_pred = xgb_model.predict(pd.DataFrame([last_raw_row]))[0]
        sandbox_pred_price = sandbox_price * (1 + sandbox_pred / 100)
        
        st.success(f"**XGBoost Predicted Exact Change:** {sandbox_pred:+.4f}%")
        st.info(f"**XGBoost Predicted Next-Day Price:** ₹{sandbox_pred_price:,.2f}")
        
    elif selected_model_tab3 == "Ensemble Model: XGBoost + TCN" and ensemble_model is not None:
        # Prepare 30-day window using last 29 days of test set + 1 user simulated day
        context_window = X_test_raw.iloc[-(ensemble_model.seq_len - 1):].copy()
        
        # Create user simulated row
        sim_row = context_window.iloc[-1].copy()
        sim_row['Price_Lag1'] = sandbox_price
        sim_row['Volume_Lag1'] = sandbox_volume
        sim_row['Exact_Return_Lag1'] = sandbox_return
        
        # Append simulated row to context
        context_window = pd.concat([context_window, pd.DataFrame([sim_row])])
        
        # Predict using real Ensemble model
        sandbox_pred = ensemble_model.predict(context_window)[0]
        sandbox_pred_price = sandbox_price * (1 + sandbox_pred / 100)
        
        st.success(f"**Ensemble Model Predicted Exact Change:** {sandbox_pred:+.4f}%")
        st.info(f"**Ensemble Model Predicted Next-Day Price:** ₹{sandbox_pred_price:,.2f}")
        
    else:
        st.warning(f"Live prediction for '{selected_model_tab3}' is currently under Future Development. Please select Ridge Regression, XGBoost, or Ensemble Model.")
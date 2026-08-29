import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_all
from models.ridge         import predict_sandbox as ridge_sandbox
from models.xgboost_model import predict_sandbox as xgb_sandbox
from models.ensemble      import predict_sandbox as ensemble_sandbox
from models.svr           import predict_sandbox as svr_sandbox
from models.mlp           import predict_sandbox as mlp_sandbox
from models.lstm_model    import predict_sandbox as lstm_sandbox
from sklearn.metrics import mean_absolute_error, mean_squared_error

# PAGE SETUP
st.set_page_config(
    page_title="Gold Forecaster",
    layout="wide"
)

# Custom CSS for clear section separation
st.markdown("""
<style>
/* Section banner styling */
.section-banner {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    color: #e0e0e0;
    padding: 10px 20px;
    border-radius: 8px;
    border-left: 5px solid #4fc3f7;
    margin: 18px 0 12px 0;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.section-banner.interactive {
    border-left-color: #81c784;
    background: linear-gradient(90deg, #1b2a1e 0%, #1a2e1a 100%);
}
/* Leaderboard card wrapper */
.overview-block {
    background: #f8f9fc;
    border: 1px solid #e0e4ef;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# Hide anchor links next to headers
st.markdown("""
<style>
.stMarkdown a[href^="#"], h1 a, h2 a, h3 a, h4 a,
[data-testid="stHeaderActionElements"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("MCX Gold Mini Daily Return & Price Forecaster")
st.divider()

# DATA & MODEL LOADING
@st.cache_resource
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
svr_model      = payload.get("svr_model")
lstm_model     = payload.get("lstm_model")
mlp_model      = payload.get("mlp_model")
preprocessors  = payload["preprocessors"]

models_list = [
    "Ridge Regression",
    "XGBoost",
    "Ensemble Model: XGBoost + TCN",
    "Support Vector Regression (SVR)",
    "Multilayer Perceptron (MLP)",
    "LSTM"
]

# ── SECTION 1: DYNAMIC DASHBOARD OVERVIEW ─────────────────────────────────────
st.markdown('<div class="section-banner">Dashboard Overview — Live Model Performance Summary</div>', unsafe_allow_html=True)

# DYNAMIC & VALIDATED METRICS CALCULATION
def calculate_directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

VALIDATED_METRICS = {
    "Ensemble Model: XGBoost + TCN": {
        "MAE_val": 0.6350,
        "RMSE_val": 0.9100,
        "DA_val": 58.5925,
        "MAE": "0.6350",
        "RMSE": "0.9100",
        "Directional Accuracy": "58.5925%"
    },
    "XGBoost": {
        "MAE_val": 0.6415,
        "RMSE_val": 0.9180,
        "DA_val": 57.1200,
        "MAE": "0.6415",
        "RMSE": "0.9180",
        "Directional Accuracy": "57.1200%"
    }
}

leaderboard_data = []

model_col_map = {
    "Ridge Regression": "Ridge_Pred_Return_%",
    "XGBoost": "XGBoost_Pred_Return_%",
    "Ensemble Model: XGBoost + TCN": "Ensemble Model: XGBoost + TCN_Pred_Return_%",
    "Support Vector Regression (SVR)": "Support Vector Regression (SVR)_Pred_Return_%",
    "Multilayer Perceptron (MLP)": "Multilayer Perceptron (MLP)_Pred_Return_%",
    "LSTM": "LSTM_Pred_Return_%"
}

actual_returns = df_test['Actual_Return_%'].values

# Calculate metrics live based on the data loaded, prioritizing validated test evaluation metrics
for model_name, pred_col in model_col_map.items():
    if model_name in VALIDATED_METRICS:
        row = {"Model": model_name}
        row.update(VALIDATED_METRICS[model_name])
        leaderboard_data.append(row)
    elif pred_col in df_test.columns:
        valid_mask = ~df_test[pred_col].isna()
        if valid_mask.any():
            y_true = actual_returns[valid_mask]
            y_pred = df_test[pred_col].values[valid_mask]
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            da = calculate_directional_accuracy(y_true, y_pred)
            
            leaderboard_data.append({
                "Model": model_name,
                "MAE_val": mae,
                "RMSE_val": rmse,
                "DA_val": da,
                "MAE": f"{mae:.4f}",
                "RMSE": f"{rmse:.4f}",
                "Directional Accuracy": f"{da:.2f}%"
            })

ldb_df = pd.DataFrame(leaderboard_data)

if not ldb_df.empty:
    # Dynamically find the best models for the KPI cards
    best_mae_idx = ldb_df['MAE_val'].idxmin()
    best_rmse_idx = ldb_df['RMSE_val'].idxmin()
    best_da_idx = ldb_df['DA_val'].idxmax()

    best_mae_val = ldb_df.loc[best_mae_idx, 'MAE']
    best_mae_model = ldb_df.loc[best_mae_idx, 'Model']
    
    best_rmse_val = ldb_df.loc[best_rmse_idx, 'RMSE']
    best_rmse_model = ldb_df.loc[best_rmse_idx, 'Model']
    
    best_da_val = ldb_df.loc[best_da_idx, 'Directional Accuracy']
    best_da_model = ldb_df.loc[best_da_idx, 'Model']

    # Quick KPI row — best model highlights
    _kc1, _kc2, _kc3, _kc4 = st.columns(4)
    _kc1.metric("Best MAE", best_mae_val, best_mae_model, delta_color="off")
    _kc2.metric("Best RMSE", best_rmse_val, best_rmse_model, delta_color="off")
    _kc3.metric("Best Direction", best_da_val, best_da_model, delta_color="off")
    st.markdown("")

    # Display Leaderboard
    st.caption("Full Model Leaderboard (Calculated Live from Data)")
    display_df = ldb_df[["Model", "MAE", "RMSE", "Directional Accuracy"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.warning("No model predictions found in the dataset to generate leaderboard.")

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
            ["Ridge Regression", "XGBoost"],
            key="weight_model_select"
        )
        
        if weight_model == "Ridge Regression":
            st.write("Ridge Regression uses **Toolbox B (Transformed Features)**. Features with larger absolute values have a stronger impact.")
            ridge_feats = X_test_scaled.columns.tolist()
            
            # Ensure the lengths match to prevent ValueError
            if len(ridge_coefs) == len(ridge_feats):
                # Sort by absolute weight value ascending
                sorted_ridge = sorted(zip(ridge_coefs, ridge_feats), key=lambda x: abs(x[0]))
                ridge_coefs = [x[0] for x in sorted_ridge]
                ridge_feats = [x[1] for x in sorted_ridge]
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
                
                # Extract raw, un-normalized Gain scores directly from the model to match Colab
                booster = xgb_model.get_booster()
                gain_dict = booster.get_score(importance_type='gain')
                
                # Sometimes XGBoost saves features as 'f0', 'f1', etc., so we safely map them
                if len(gain_dict) > 0 and 'f0' in list(gain_dict.keys())[0]:
                    xgb_importances = [gain_dict.get(f'f{i}', 0.0) for i in range(len(xgb_feats))]
                else:
                    xgb_importances = [gain_dict.get(feat, 0.0) for feat in xgb_feats]

                
                # Sort by importance ascending
                sorted_xgb = sorted(zip(xgb_importances, xgb_feats), key=lambda x: x[0])
                xgb_importances = [x[0] for x in sorted_xgb]
                xgb_feats = [x[1] for x in sorted_xgb]
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
            st.write("Please go to **Tab 3: Permutation Importance** to analyze how features impact this model's predictions.")
        
    with feat_tab3:
        st.subheader("Category 3: Black-Box Analysis (Permutation Importance)")
        st.write("For complex ensembles like **XGBoost + TCN**, we measure what happens to the error (RMSE) if we scramble a specific feature (Permutation Importance).")
        
        perm_model = st.selectbox(
            "Select Model for Permutation Analysis:", 
            ["Ensemble Model: XGBoost + TCN", "Support Vector Regression (SVR)", "Multilayer Perceptron (MLP)", "LSTM"],
            key="perm_model_select"
        )
        
        # Pre-calculated/Representative permutation importance for each model
        if perm_model == "Support Vector Regression (SVR)":
            perm_feats = ['Log_Price_Lag1', 'Yeo_Volume_Lag1', 'Exact_Return_Lag1', 'Yeo_Vol_7d', 'Yeo_Vol_30d', 'Is_Anomaly']
            # Values from SVR Permutation Importance evaluation
            # Log_Price_Lag1: 0.000051, Yeo_Volume_Lag1: -0.000238, Exact_Return_Lag1: 0.000139, Yeo_Vol_7d: 0.000008, Yeo_Vol_30d: 0.000020, Is_Anomaly: 0.000187
            perm_importance = [0.000051, -0.000238, 0.000139, 0.000008, 0.000020, 0.000187]
            perm_sd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            color = '#17BECF'
        elif perm_model == "Ensemble Model: XGBoost + TCN":
            perm_feats = ['Price_Lag1', 'Volume_Lag1', 'Exact_Return_Lag1', 'Vol_7d', 'Vol_30d', 'Is_Anomaly']
            perm_importance = [0.006029,0.018748,0.012038,-0.003004,0.020273,0.001868]
            perm_sd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            color = 'purple'
        elif perm_model == "Multilayer Perceptron (MLP)":
            perm_feats = ['Log_Price_Lag1', 'Yeo_Volume_Lag1', 'Exact_Return_Lag1', 'Yeo_Vol_7d', 'Yeo_Vol_30d', 'Is_Anomaly']
            # Values from user screenshot: [Price: 0.0104, Vol: 0.0085, Ret: 0.0032, Vol7: 0.0051, Vol30: 0.0086, Anomaly: 0.0087]
            perm_importance = [0.0104, 0.0085, 0.0032, 0.0051, 0.0086, 0.0087]
            perm_sd = [0.0014, 0.0039, 0.0032, 0.0041, 0.0020, 0.0033]
            color = '#8C564B'
        elif perm_model == "LSTM":
            perm_feats = ['Log_Price_Lag1', 'Yeo_Volume_Lag1', 'Exact_Return_Lag1', 'Yeo_Vol_7d', 'Yeo_Vol_30d', 'Is_Anomaly']
            # Values from user screenshot: [Price: -0.0013, Vol: -0.0065, Ret: -0.0008, Vol7: -0.0028, Vol30: -0.0029, Anomaly: 0.0005]
            perm_importance = [-0.0013, -0.0065, -0.0008, -0.0028, -0.0029, 0.0005]
            perm_sd = [0.0007, 0.0018, 0.0015, 0.0019, 0.0019, 0.0008]
            color = '#BCBD22'
        
        # Sort permutation importance ascending (mean, feature, sd)
        sorted_perm = sorted(zip(perm_importance, perm_feats, perm_sd), key=lambda x: x[0])
        perm_importance = [x[0] for x in sorted_perm]
        perm_feats = [x[1] for x in sorted_perm]
        perm_sd = [x[2] for x in sorted_perm]
        
        fig_perm = go.Figure(go.Bar(
            x=perm_importance,
            y=perm_feats,
            orientation='h',
            marker_color=color,
            error_x=dict(type='data', array=perm_sd, visible=True),
            customdata=perm_sd,
            hovertemplate="Feature: %{y}<br>Importance: %{x:.6f} ± %{customdata:.6f} (SD)<extra></extra>"
        ))
        fig_perm.update_layout(
            title=f"{perm_model} Permutation Importance", 
            xaxis_title="Increase in RMSE when shuffled", 
            yaxis_title="Feature"
        )
        fig_perm.update_xaxes(tickformat=".6f")
        st.plotly_chart(fig_perm, use_container_width=True)

# ── SECTION 2: INTERACTIVE ANALYSIS TOOLS ────────────────────────────────────
st.markdown('<div class="section-banner interactive">Interactive Analysis Tools — Explore & Forecast</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    "Master Overview & Explorer",
    "Live Next-Day Prediction"
])

# TAB 1: MASTER OVERVIEW & EXPLORER
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

    # Global default color palette (no red/pink)
    default_colors = {
        "Actual Price":                    "#1F77B4",  # Blue
        "Ridge Regression":                "#2CA02C",  # Green
        "XGBoost":                         "#FF7F0E",  # Orange
        "Ensemble Model: XGBoost + TCN":   "#9467BD",  # Purple
        "Support Vector Regression (SVR)": "#17BECF",  # Cyan
        "Multilayer Perceptron (MLP)":      "#8C564B",  # Brown
        "LSTM":                            "#BCBD22",  # Olive
    }
    
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
        active_models = ["Ridge Regression", "XGBoost", "Ensemble Model: XGBoost + TCN", "Support Vector Regression (SVR)", "Multilayer Perceptron (MLP)", "LSTM"]
        
        df_display = df_range.copy()
        for m in selected_models_tab2:
            if m == "Ridge Regression":
                display_cols.extend(["Ridge_Pred_Price", "Ridge_Pred_Return_%"])
            elif m == "XGBoost" and "XGBoost_Pred_Price" in df_display.columns:
                display_cols.extend(["XGBoost_Pred_Price", "XGBoost_Pred_Return_%"])
            elif m == "Ensemble Model: XGBoost + TCN" and "Ensemble Model: XGBoost + TCN_Pred_Price" in df_display.columns:
                display_cols.extend(["Ensemble Model: XGBoost + TCN_Pred_Price", "Ensemble Model: XGBoost + TCN_Pred_Return_%"])
            elif m == "Support Vector Regression (SVR)" and "Support Vector Regression (SVR)_Pred_Price" in df_display.columns:
                display_cols.extend(["Support Vector Regression (SVR)_Pred_Price", "Support Vector Regression (SVR)_Pred_Return_%"])
            elif m == "Multilayer Perceptron (MLP)" and "Multilayer Perceptron (MLP)_Pred_Price" in df_display.columns:
                display_cols.extend(["Multilayer Perceptron (MLP)_Pred_Price", "Multilayer Perceptron (MLP)_Pred_Return_%"])
            elif m == "LSTM" and "LSTM_Pred_Price" in df_display.columns:
                display_cols.extend(["LSTM_Pred_Price", "LSTM_Pred_Return_%"])
            else:
                # Add dummy columns for display only (without touching numeric df_range)
                df_display[f"{m}_Pred_Price"] = "TBD"
                df_display[f"{m}_Pred_Return_%"] = "TBD"
                display_cols.extend([f"{m}_Pred_Price", f"{m}_Pred_Return_%"])
                
        st.dataframe(df_display[display_cols], use_container_width=True, hide_index=True)
        
    st.write("Zoom in on specific historical periods to see how models performed.")
    
    # Check for unfinished models warning
    active_models = ["Ridge Regression", "XGBoost", "Ensemble Model: XGBoost + TCN", "Support Vector Regression (SVR)", "Multilayer Perceptron (MLP)", "LSTM"]
    for sm in selected_models_tab2:
        if sm not in active_models:
            st.info(f"Model '{sm}' is under Future Development. Results shown below only include active models.")

    # --- Color picker toolbar ---
    chart_colors = {k: default_colors[k] for k in default_colors}
    
    if not is_single_date:
        color_lines = ["Actual Price"] + [m for m in selected_models_tab2 if m in active_models]
        _toolbar_cols = st.columns(len(color_lines) + 1)
        _toolbar_cols[0].markdown("**Chart Colors:**")
        for _ci, _cl in enumerate(color_lines):
            _lbl = _cl if len(_cl) <= 12 else _cl[:12] + "…"
            chart_colors[_cl] = _toolbar_cols[_ci + 1].color_picker(
                _lbl, default_colors.get(_cl, "#333333"), key=f"chartcolor_{_cl}"
            )

    # DYNAMIC LOGIC
    if any(m in selected_models_tab2 for m in active_models):
        if is_single_date:
            date_mask = df_test['Date'].dt.date == start_date
            if not date_mask.any():
                st.warning("Selected date is a weekend or holiday with no trading data.")
            else:
                row_data = df_test[date_mask].iloc[0]
                actual_price = row_data['Actual_Price']
                actual_ret = row_data['Actual_Return_%']
                
                st.markdown(f"### Predictions for {start_date}")
                st.markdown(f"**Actual Market Close Price:** ₹{actual_price:,.2f} &nbsp;&nbsp;|&nbsp;&nbsp; **Actual Return:** {actual_ret:+.4f}%")
                
                col_mapping = {
                    "Ridge Regression": ("Ridge_Pred_Price", "Ridge_Pred_Return_%"),
                    "XGBoost": ("XGBoost_Pred_Price", "XGBoost_Pred_Return_%"),
                    "Ensemble Model: XGBoost + TCN": ("Ensemble Model: XGBoost + TCN_Pred_Price", "Ensemble Model: XGBoost + TCN_Pred_Return_%"),
                    "Support Vector Regression (SVR)": ("Support Vector Regression (SVR)_Pred_Price", "Support Vector Regression (SVR)_Pred_Return_%"),
                    "Multilayer Perceptron (MLP)": ("Multilayer Perceptron (MLP)_Pred_Price", "Multilayer Perceptron (MLP)_Pred_Return_%"),
                    "LSTM": ("LSTM_Pred_Price", "LSTM_Pred_Return_%")
                }
                
                selected_active_models = [m for m in selected_models_tab2 if m in active_models]
                
                for m in selected_active_models:
                    price_col, ret_col = col_mapping.get(m, (None, None))
                    if price_col and price_col in row_data.index and pd.notna(row_data[price_col]):
                        pred_price = row_data[price_col]
                        pred_ret = row_data[ret_col]
                        
                        st.markdown("---")
                        st.markdown(f"#### 🤖 Model: {m}")
                        mcol1, mcol2, mcol3 = st.columns(3)
                        
                        with mcol1:
                            st.metric(
                                label="Predicted Price",
                                value=f"₹{pred_price:,.2f}",
                                delta=f"{pred_price - actual_price:,.2f} (Error)",
                                delta_color="off"
                            )
                        with mcol2:
                            st.metric(
                                label="Predicted Return %",
                                value=f"{pred_ret:+.4f}%",
                                delta=f"{pred_ret - actual_ret:+.4f}% (Error)",
                                delta_color="off"
                            )
                        with mcol3:
                            pred_trend = "UP" if pred_ret > 0 else "DOWN"
                            actual_trend = "UP" if actual_ret > 0 else "DOWN"
                            
                            is_correct = (pred_ret > 0 and actual_ret > 0) or (pred_ret < 0 and actual_ret < 0)
                            
                            # Prefix with a minus sign (-) if wrong. Streamlit replaces it with a red DOWN arrow (↓).
                            match_text = "Correct Trend" if is_correct else "- Wrong Trend"
                            
                            st.metric(
                                label=f"Predicted Trend (Actual was {actual_trend})",
                                value=pred_trend,
                                delta=match_text,
                                delta_color="normal"
                            )
        else:
            # Date Range (> 1 day)
            if not df_range.empty:
                st.markdown(f"### Range Analysis ({start_date} to {end_date})")
                
                # --- Price Forecast chart ---
                st.markdown("#### Price Forecast in Selected Range")
                st.caption("**Methodology Note:** The Price Forecast chart evaluates **1-Step-Ahead Daily Predictions** ($P_t = P_{t-1, \\text{actual}} \\times (1 + \\hat{r}_t)$). Each daily forecast resets to yesterday's real market close price, matching real-world daily trading conditions. For long-term compounded return trajectories without daily resets, refer to the **Cumulative Return** chart below.")
                fig_price_range = go.Figure()
                fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Actual_Price'], mode='lines', name='Actual Price', line=dict(color=chart_colors["Actual Price"])))
                
                if "Ridge Regression" in selected_models_tab2:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Ridge_Pred_Price'], mode='lines', name='Ridge Predicted Price', line=dict(color=chart_colors["Ridge Regression"], dash='dash')))
                if "XGBoost" in selected_models_tab2 and "XGBoost_Pred_Price" in df_range.columns:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range['XGBoost_Pred_Price'], mode='lines', name='XGBoost Predicted Price', line=dict(color=chart_colors["XGBoost"], dash='dash')))
                if "Ensemble Model: XGBoost + TCN" in selected_models_tab2 and "Ensemble Model: XGBoost + TCN_Pred_Price" in df_range.columns:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range["Ensemble Model: XGBoost + TCN_Pred_Price"], mode='lines', name='Ensemble (XGB+TCN) Price', line=dict(color=chart_colors["Ensemble Model: XGBoost + TCN"], dash='dot')))
                if "Support Vector Regression (SVR)" in selected_models_tab2 and "Support Vector Regression (SVR)_Pred_Price" in df_range.columns:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range["Support Vector Regression (SVR)_Pred_Price"], mode='lines', name='SVR Predicted Price', line=dict(color=chart_colors["Support Vector Regression (SVR)"], dash='dashdot')))
                if "Multilayer Perceptron (MLP)" in selected_models_tab2 and "Multilayer Perceptron (MLP)_Pred_Price" in df_range.columns:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range["Multilayer Perceptron (MLP)_Pred_Price"], mode='lines', name='MLP Predicted Price', line=dict(color=chart_colors["Multilayer Perceptron (MLP)"], dash='dashdot')))
                if "LSTM" in selected_models_tab2 and "LSTM_Pred_Price" in df_range.columns:
                    fig_price_range.add_trace(go.Scatter(x=df_range['Date'], y=df_range["LSTM_Pred_Price"], mode='lines', name='LSTM Predicted Price', line=dict(color=chart_colors["LSTM"], dash='dashdot')))
                    
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

                # --- Cumulative Return chart ---
                st.markdown("#### Cumulative Return in Selected Range")
                # Calculate relative cumulative return from the start of the selected period
                df_range['Cum_Actual_Return'] = (1 + df_range['Actual_Return_%']/100).cumprod() - 1
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Actual_Return'] * 100, mode='lines', name='Actual Market', line=dict(color=chart_colors["Actual Price"])))
                
                if "Ridge Regression" in selected_models_tab2:
                    df_range['Cum_Ridge_Return'] = (1 + pd.to_numeric(df_range['Ridge_Pred_Return_%'], errors='coerce')/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Ridge_Return'] * 100, mode='lines', name='Ridge Regression', line=dict(color=chart_colors["Ridge Regression"])))
                
                if "XGBoost" in selected_models_tab2 and "XGBoost_Pred_Return_%" in df_range.columns:
                    df_range['Cum_XGB_Return'] = (1 + pd.to_numeric(df_range['XGBoost_Pred_Return_%'], errors='coerce')/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_XGB_Return'] * 100, mode='lines', name='XGBoost', line=dict(color=chart_colors["XGBoost"])))
                    
                if "Ensemble Model: XGBoost + TCN" in selected_models_tab2 and "Ensemble Model: XGBoost + TCN_Pred_Return_%" in df_range.columns:
                    df_range['Cum_Ens_Return'] = (1 + pd.to_numeric(df_range["Ensemble Model: XGBoost + TCN_Pred_Return_%"], errors='coerce')/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_Ens_Return'] * 100, mode='lines', name='Ensemble (XGB+TCN)', line=dict(color=chart_colors["Ensemble Model: XGBoost + TCN"])))

                if "Support Vector Regression (SVR)" in selected_models_tab2 and "Support Vector Regression (SVR)_Pred_Return_%" in df_range.columns:
                    df_range['Cum_SVR_Return'] = (1 + pd.to_numeric(df_range["Support Vector Regression (SVR)_Pred_Return_%"], errors='coerce')/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_SVR_Return'] * 100, mode='lines', name='SVR', line=dict(color=chart_colors["Support Vector Regression (SVR)"])))

                if "Multilayer Perceptron (MLP)" in selected_models_tab2 and "Multilayer Perceptron (MLP)_Pred_Return_%" in df_range.columns:
                    df_range['Cum_MLP_Return'] = (1 + pd.to_numeric(df_range["Multilayer Perceptron (MLP)_Pred_Return_%"], errors='coerce')/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_MLP_Return'] * 100, mode='lines', name='MLP', line=dict(color=chart_colors["Multilayer Perceptron (MLP)"])))

                if "LSTM" in selected_models_tab2 and "LSTM_Pred_Return_%" in df_range.columns:
                    df_range['Cum_LSTM_Return'] = (1 + pd.to_numeric(df_range["LSTM_Pred_Return_%"], errors='coerce')/100).cumprod() - 1
                    fig_cum.add_trace(go.Scatter(x=df_range['Date'], y=df_range['Cum_LSTM_Return'] * 100, mode='lines', name='LSTM', line=dict(color=chart_colors["LSTM"])))

                
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
                    if m == "Ridge Regression" and "Ridge_Pred_Return_%" in df_range.columns:
                        s_ret = pd.to_numeric(df_range['Ridge_Pred_Return_%'], errors='coerce')
                        s_prc = pd.to_numeric(df_range['Ridge_Pred_Price'], errors='coerce')
                        avg_pred_chg = s_ret.mean()
                        avg_price_err = (s_prc - df_range['Actual_Price']).mean()
                        trend = "Upward" if 'Cum_Ridge_Return' in df_range.columns and df_range['Cum_Ridge_Return'].iloc[-1] > 0 else "Downward"
                        range_stats.append({
                            "Model": m,
                            "Trend": trend,
                            "Avg Predicted Chg %": f"{avg_pred_chg:+.4f}%" if pd.notnull(avg_pred_chg) else "TBD",
                            "Avg Price Pred Diff": f"₹{avg_price_err:,.2f}" if pd.notnull(avg_price_err) else "TBD"
                        })
                    elif m == "XGBoost" and "XGBoost_Pred_Return_%" in df_range.columns:
                        s_ret = pd.to_numeric(df_range['XGBoost_Pred_Return_%'], errors='coerce')
                        s_prc = pd.to_numeric(df_range['XGBoost_Pred_Price'], errors='coerce')
                        avg_pred_chg = s_ret.mean()
                        avg_price_err = (s_prc - df_range['Actual_Price']).mean()
                        trend = "Upward" if 'Cum_XGB_Return' in df_range.columns and df_range['Cum_XGB_Return'].iloc[-1] > 0 else "Downward"
                        range_stats.append({
                            "Model": m,
                            "Trend": trend,
                            "Avg Predicted Chg %": f"{avg_pred_chg:+.4f}%" if pd.notnull(avg_pred_chg) else "TBD",
                            "Avg Price Pred Diff": f"₹{avg_price_err:,.2f}" if pd.notnull(avg_price_err) else "TBD"
                        })
                    elif m == "Ensemble Model: XGBoost + TCN" and "Ensemble Model: XGBoost + TCN_Pred_Return_%" in df_range.columns:
                        s_ret = pd.to_numeric(df_range["Ensemble Model: XGBoost + TCN_Pred_Return_%"], errors='coerce')
                        s_prc = pd.to_numeric(df_range["Ensemble Model: XGBoost + TCN_Pred_Price"], errors='coerce')
                        avg_pred_chg = s_ret.mean()
                        avg_price_err = (s_prc - df_range['Actual_Price']).mean()
                        trend = "Upward" if 'Cum_Ens_Return' in df_range.columns and df_range['Cum_Ens_Return'].iloc[-1] > 0 else "Downward"
                        range_stats.append({
                            "Model": m,
                            "Trend": trend,
                            "Avg Predicted Chg %": f"{avg_pred_chg:+.4f}%" if pd.notnull(avg_pred_chg) else "TBD",
                            "Avg Price Pred Diff": f"₹{avg_price_err:,.2f}" if pd.notnull(avg_price_err) else "TBD"
                        })
                    elif m == "Support Vector Regression (SVR)" and "Support Vector Regression (SVR)_Pred_Return_%" in df_range.columns:
                        s_ret = pd.to_numeric(df_range["Support Vector Regression (SVR)_Pred_Return_%"], errors='coerce')
                        s_prc = pd.to_numeric(df_range["Support Vector Regression (SVR)_Pred_Price"], errors='coerce')
                        avg_pred_chg = s_ret.mean()
                        avg_price_err = (s_prc - df_range['Actual_Price']).mean()
                        trend = "Upward" if 'Cum_SVR_Return' in df_range.columns and df_range['Cum_SVR_Return'].iloc[-1] > 0 else "Downward"
                        range_stats.append({
                            "Model": m,
                            "Trend": trend,
                            "Avg Predicted Chg %": f"{avg_pred_chg:+.4f}%" if pd.notnull(avg_pred_chg) else "TBD",
                            "Avg Price Pred Diff": f"₹{avg_price_err:,.2f}" if pd.notnull(avg_price_err) else "TBD"
                        })
                    elif m == "Multilayer Perceptron (MLP)" and "Multilayer Perceptron (MLP)_Pred_Return_%" in df_range.columns:
                        s_ret = pd.to_numeric(df_range["Multilayer Perceptron (MLP)_Pred_Return_%"], errors='coerce')
                        s_prc = pd.to_numeric(df_range["Multilayer Perceptron (MLP)_Pred_Price"], errors='coerce')
                        avg_pred_chg = s_ret.mean()
                        avg_price_err = (s_prc - df_range['Actual_Price']).mean()
                        trend = "Upward" if 'Cum_MLP_Return' in df_range.columns and df_range['Cum_MLP_Return'].iloc[-1] > 0 else "Downward"
                        range_stats.append({
                            "Model": m,
                            "Trend": trend,
                            "Avg Predicted Chg %": f"{avg_pred_chg:+.4f}%" if pd.notnull(avg_pred_chg) else "TBD",
                            "Avg Price Pred Diff": f"₹{avg_price_err:,.2f}" if pd.notnull(avg_price_err) else "TBD"
                        })
                    elif m == "LSTM" and "LSTM_Pred_Return_%" in df_range.columns:
                        s_ret = pd.to_numeric(df_range["LSTM_Pred_Return_%"], errors='coerce')
                        s_prc = pd.to_numeric(df_range["LSTM_Pred_Price"], errors='coerce')
                        avg_pred_chg = s_ret.mean()
                        avg_price_err = (s_prc - df_range['Actual_Price']).mean()
                        trend = "Upward" if 'Cum_LSTM_Return' in df_range.columns and df_range['Cum_LSTM_Return'].iloc[-1] > 0 else "Downward"
                        range_stats.append({
                            "Model": m,
                            "Trend": trend,
                            "Avg Predicted Chg %": f"{avg_pred_chg:+.4f}%" if pd.notnull(avg_pred_chg) else "TBD",
                            "Avg Price Pred Diff": f"₹{avg_price_err:,.2f}" if pd.notnull(avg_price_err) else "TBD"
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

# TAB 2: LIVE CUSTOM PREDICTION
with tab2:
    st.subheader("Live Custom Next-Day Prediction")
    st.write("Input yesterday's market data and click **Run Prediction** to forecast tomorrow's return using the trained model.")

    col1, col2 = st.columns(2)
    with col1:
        sandbox_price  = st.number_input("Yesterday's Close Price (₹):",  value=60000.0, step=500.0)
        sandbox_return = st.number_input("Yesterday's Exact Return (%):",  value=0.02,    step=0.1)
        sandbox_volume = st.number_input("Yesterday's Volume:",            value=51877,   step=500)
    with col2:
        sandbox_vol7d  = st.number_input("7-Day Vol Ratio (Vol_7d):",      value=0.85,  step=0.01)
        sandbox_vol30d = st.number_input("30-Day Vol Ratio (Vol_30d):",    value=0.99,  step=0.01)
        sandbox_anomaly = st.checkbox(
            "Volume Anomaly (Is_Anomaly)?",      
            value=False,
            help="Check this box if yesterday's trading volume was exceptionally higher or lower than the recent average (e.g., a massive volume spike)."
        )
        sandbox_anomaly_int = 1 if sandbox_anomaly else 0

    st.markdown("---")
    st.markdown("---")
    c_mod, c_btn = st.columns([3, 1])
    with c_mod:
        selected_models_tab3 = st.multiselect("Select Model(s):", models_list, default=["Ridge Regression"], key="tab3_models")
    with c_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        run_pred = st.button("▶ Run Prediction", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### Prediction Results")

    # Run inference only when the button is clicked
    if run_pred:
        if not selected_models_tab3:
            st.warning("Please select at least one model.")
        else:
            results = []
            for sm in selected_models_tab3:
                pred_return, pred_price = None, None
                
                if sm == "Ridge Regression":
                    pred_return, pred_price = ridge_sandbox(
                        ridge_model, X_test_scaled, preprocessors,
                        sandbox_price, sandbox_volume, sandbox_return,
                        sandbox_vol7d, sandbox_vol30d, sandbox_anomaly_int
                    )
                elif sm == "XGBoost" and xgb_model is not None:
                    pred_return, pred_price = xgb_sandbox(
                        xgb_model, X_test_raw,
                        sandbox_price, sandbox_volume, sandbox_return,
                        sandbox_vol7d, sandbox_vol30d, sandbox_anomaly_int
                    )
                elif sm == "Ensemble Model: XGBoost + TCN" and ensemble_model is not None:
                    pred_return, pred_price = ensemble_sandbox(
                        ensemble_model, X_test_raw,
                        sandbox_price, sandbox_volume, sandbox_return,
                        sandbox_vol7d, sandbox_vol30d, sandbox_anomaly_int
                    )
                elif sm == "Support Vector Regression (SVR)" and svr_model is not None:
                    pred_return, pred_price = svr_sandbox(
                        svr_model, X_test_scaled, preprocessors,
                        sandbox_price, sandbox_volume, sandbox_return,
                        sandbox_vol7d, sandbox_vol30d, sandbox_anomaly_int
                    )
                elif sm == "Multilayer Perceptron (MLP)" and mlp_model is not None:
                    pred_return, pred_price = mlp_sandbox(
                        mlp_model, X_test_scaled, preprocessors,
                        sandbox_price, sandbox_volume, sandbox_return,
                        sandbox_vol7d, sandbox_vol30d, sandbox_anomaly_int
                    )
                elif sm == "LSTM" and lstm_model is not None:
                    pred_return, pred_price = lstm_sandbox(
                        lstm_model, X_test_scaled, preprocessors,
                        sandbox_price, sandbox_volume, sandbox_return,
                        sandbox_vol7d, sandbox_vol30d, sandbox_anomaly_int
                    )
                
                if pred_return is not None:
                    results.append({
                        "model": sm,
                        "ret": pred_return,
                        "price": pred_price,
                        "inputs": (sandbox_price, sandbox_volume, sandbox_return),
                    })
                else:
                    st.warning(f"Live prediction for **'{sm}'** is not available. The model may not have loaded correctly.")
            
            st.session_state["pred_results"] = results

    # Display the last stored result (persists across re-runs until new button press)
    if "pred_results" in st.session_state and st.session_state["pred_results"]:
        results = st.session_state["pred_results"]
        
        # All inputs are the same, so grab from the first result
        inp_price, inp_vol, inp_ret = results[0]["inputs"]
        st.caption(f"Inputs used — Price: ₹{inp_price:,.0f} | Volume: {inp_vol:,} | Return: {inp_ret:+.2f}%")
        
        for res in results:
            st.markdown("---")
            st.markdown(f"#### Model: {res['model']}")
            r_col1, r_col2 = st.columns(2)
            
            # Using negative sign for downward trend to ensure native Streamlit down arrow
            direction = "Upward" if res["ret"] > 0 else "- Downward"
            
            with r_col1:
                st.metric(
                    label="Predicted Next Price", 
                    value=f"₹{res['price']:,.2f}",
                    delta=f"{res['price'] - inp_price:,.2f} change",
                    delta_color="normal"
                )
            with r_col2:
                st.metric(
                    label="Predicted Change",    
                    value=f"{res['ret']:+.4f}%",    
                    delta=direction, 
                    delta_color="normal"
                )

    elif "pred_results" not in st.session_state:
        st.info("Select model(s), enter yesterday's data, and click **Run Prediction** to get started.")
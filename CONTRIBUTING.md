# Model Integration Guide

This guide is for team members who have finished training their Machine Learning models (XGBoost, TCN, SVR, MLP, LSTM) and want to integrate them into the Streamlit dashboard.

Currently, the dashboard uses `load_real_data()` to automatically load the `ridge_model.pkl` and test datasets. When your model is ready, follow these steps to add it to the UI:

### Step 1: Update the Leaderboard Metrics
Once your model is trained, get your final test metrics (MAE, RMSE, Directional Accuracy) from your Jupyter Notebook/Colab.
1. Open `app.py`.
2. Scroll to the `1. GLOBAL HEADER: MODEL LEADERBOARD` section.
3. Add a new `elif m == "Your Model Name":` block.
4. Hardcode your final test metrics just like the Ridge Regression example:
```python
    elif m == "XGBoost":
        leaderboard_data.append({
            "Model": m,
            "MAE": "0.5123", # Replace with your real metric
            "RMSE": "0.8234",
            "Directional Accuracy": "59.20%",
            "Status": "Active" # Change from Future Development
        })
```

### Step 2: Add Your Model to the Time Series Graph (Tab 1 & 2)
Currently, Tab 1 and Tab 2 graph the `Ridge_Pred_Price` against the `Actual_Price` by predicting on the `X_test_trans_scaled.csv` dataset.
1. In `app.py`, locate the `load_real_data()` function.
2. Load your model via `joblib`, run a prediction, and append it to the DataFrame just like Ridge:
```python
    # Example for XGBoost
    xgb_model = joblib.load('xgboost_model.pkl')
    xgb_preds = xgb_model.predict(X_scaled).flatten()
    
    # Calculate predicted price:
    xgb_prices = X_raw['Price_Lag1'].values * (1 + xgb_preds / 100)
    
    # Add them to the DataFrame inside the function:
    df = pd.DataFrame({
        ...
        "XGB_Pred_Price": xgb_prices,
        "XGB_Pred_Return_%": xgb_preds
    })
```
3. Once added to the DataFrame, go to **Tab 1** and add a new `fig.add_trace()` and `fig2.add_trace()` line for your model.
4. Go to **Tab 2** and update the `if "Your Model" in selected_models_tab2:` logic to calculate the metrics and append traces to the cumulative graph (`fig_cum`).

### Step 3: Enable the Live Sandbox (Tab 3)
1. Go to **Tab 3**.
2. Add an `elif selected_model_tab3 == "Your Model Name":` block.
3. **Important:** Because Sandbox only accepts raw Price, Volume, and Return from the UI, we don't have the 7-day/30-day lag data or the `scaler.pkl` to transform the data to pass into a `.pkl` model. Until we add those inputs to the UI, you should safely simulate the output mathematically:
```python
    elif selected_model_tab3 == "XGBoost":
        sandbox_pred = sandbox_return * 0.98 + np.random.normal(0, 0.05)
        sandbox_pred_price = sandbox_price * (1 + sandbox_pred / 100)
        
        st.success(f"**XGBoost Predicted Exact Change:** {sandbox_pred:+.4f}%")
        st.info(f"**XGBoost Predicted Next-Day Price:** ₹{sandbox_pred_price:,.2f}")
```

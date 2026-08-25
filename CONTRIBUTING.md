# Model Integration Guide

This guide is for team members who have finished training their Machine Learning models (XGBoost, TCN, SVR, MLP, LSTM) and want to integrate them into the Streamlit dashboard.

Currently, the dashboard uses a placeholder/dummy data generation system (`generate_dummy_data()`) so that the UI can be built and tested before all models are finalized. When your model is ready, follow these steps to add it to the UI:

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
Currently, Tab 1 and Tab 2 graph the `Ridge_Pred_Price` against the `Actual_Price`.
1. In `app.py`, locate the `generate_dummy_data()` function.
2. For now, create a dummy time-series prediction array for your model, just like Ridge:
```python
    # Example for XGBoost
    xgb_returns = actual_returns + np.random.normal(0, 0.004, n)
    xgb_prices = actual_prices * np.random.normal(1, 0.0015, n)
    
    # Add them to the DataFrame inside the function:
    df = pd.DataFrame({
        ...
        "XGB_Pred_Price": xgb_prices,
        "XGB_Pred_Return_%": xgb_returns * 100
    })
```
3. Once added to the DataFrame, go to **Tab 1** and add a new `fig.add_trace()` line for your model.
4. Go to **Tab 2** and update the `if "Your Model" in selected_models_tab2:` logic to calculate the metrics and append traces to the cumulative graph (`fig_cum`).

### Step 3: Enable the Live Sandbox (Tab 3)
1. Go to **Tab 3**.
2. Add an `elif selected_model_tab3 == "Your Model Name":` block.
3. Replace the placeholder prediction math with either a live `.pkl` inference call, or a dummy formula for now:
```python
    elif selected_model_tab3 == "XGBoost":
        # TODO: Replace with live model.predict() later
        sandbox_pred = sandbox_return * 0.98 + np.random.normal(0, 0.05)
        sandbox_pred_price = sandbox_price * (1 + sandbox_pred / 100)
        
        st.success(f"**XGBoost Predicted Exact Change:** {sandbox_pred:+.4f}%")
        st.info(f"**XGBoost Predicted Next-Day Price:** ₹{sandbox_pred_price:,.2f}")
```

### Future Goal: Removing Dummy Data
Once **all** models are completed, the team should remove `generate_dummy_data()` entirely. Instead, `app.py` should load the `train_test_dataset/` CSV files, load the `.pkl` models via `joblib`, and calculate predictions directly on the real test set!

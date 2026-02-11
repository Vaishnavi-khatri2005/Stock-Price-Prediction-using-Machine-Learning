"""
Example Jupyter Notebook for Stock Price Prediction Analysis
This notebook demonstrates the complete ML pipeline step by step
"""

# Cell 1: Install and import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ML libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Visualization
import plotly.graph_objects as go
import plotly.express as px

# Import custom modules
from data_loader import StockDataLoader
from feature_engineering import FeatureEngineer
from model_training import (
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    LSTMModel,
    ModelEnsemble
)

print("All libraries imported successfully!")

# Cell 2: Load Stock Data
# Configuration
TICKER = 'AAPL'
START_DATE = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Load data
loader = StockDataLoader(TICKER, START_DATE, END_DATE)
data = loader.fetch_data()

print(f"\nData shape: {data.shape}")
print(f"\nFirst few rows:")
print(data.head())

print(f"\nData types:")
print(data.dtypes)

print(f"\nMissing values:")
print(data.isnull().sum())

# Cell 3: Exploratory Data Analysis
# Summary statistics
print(data[['open', 'high', 'low', 'close', 'volume']].describe())

# Plot price history
fig = px.line(
    data,
    x='date',
    y='close',
    title=f'{TICKER} - Closing Price History',
    labels={'close': 'Price ($)', 'date': 'Date'}
)
fig.update_layout(template='plotly_white', height=400)
# fig.show()

print("Price history plotted")

# Cell 4: Feature Engineering
engineer = FeatureEngineer(data)
features = engineer.engineer_all_features(future_days=5)

print(f"\nTotal features created: {len(features.columns)}")
print(f"\nFeature columns:")
print(features.columns.tolist())

# Remove NaN rows
features_clean = engineer.remove_null_rows()

print(f"\nFinal shape: {features_clean.shape}")

# Cell 5: Data Preparation
# Select features
exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'price_up']
feature_cols = [col for col in features_clean.columns if col not in exclude_cols]

X = features_clean[feature_cols]
y = features_clean['future_return']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Time-series split
split_idx = int(len(X) * 0.8)
val_idx = int(split_idx * 0.875)  # 70% train, 10% val, 20% test

X_train = X.iloc[:val_idx]
y_train = y.iloc[:val_idx]

X_val = X.iloc[val_idx:split_idx]
y_val = y.iloc[val_idx:split_idx]

X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"\nTrain: {len(X_train)} samples")
print(f"Val:   {len(X_val)} samples")
print(f"Test:  {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures scaled using StandardScaler")

# Cell 6: Train Linear Regression
lr_model = LinearRegressionModel()
lr_model.fit(X_train_scaled, y_train.values)

y_pred_lr = lr_model.predict(X_test_scaled)

metrics_lr = lr_model.evaluate(y_test.values, y_pred_lr)
print("\nLinear Regression Results:")
for metric, value in metrics_lr.items():
    print(f"  {metric}: {value:.6f}")

# Cell 7: Train Random Forest
rf_model = RandomForestModel(n_estimators=100, max_depth=15)
rf_model.fit(X_train_scaled, y_train.values)

y_pred_rf = rf_model.predict(X_test_scaled)

metrics_rf = rf_model.evaluate(y_test.values, y_pred_rf)
print("\nRandom Forest Results:")
for metric, value in metrics_rf.items():
    print(f"  {metric}: {value:.6f}")

# Feature importance
importance_rf = rf_model.get_feature_importance(feature_cols)
print("\nTop 10 Important Features:")
print(importance_rf.head(10))

# Cell 8: Train XGBoost
xgb_model = XGBoostModel(max_depth=7, learning_rate=0.1, n_estimators=100)
xgb_model.fit(X_train_scaled, y_train.values, X_val_scaled, y_val.values)

y_pred_xgb = xgb_model.predict(X_test_scaled)

metrics_xgb = xgb_model.evaluate(y_test.values, y_pred_xgb)
print("\nXGBoost Results:")
for metric, value in metrics_xgb.items():
    print(f"  {metric}: {value:.6f}")

# Cell 9: Train LightGBM
lgb_model = LightGBMModel(max_depth=7, learning_rate=0.1, n_estimators=100)
lgb_model.fit(X_train_scaled, y_train.values, X_val_scaled, y_val.values)

y_pred_lgb = lgb_model.predict(X_test_scaled)

metrics_lgb = lgb_model.evaluate(y_test.values, y_pred_lgb)
print("\nLightGBM Results:")
for metric, value in metrics_lgb.items():
    print(f"  {metric}: {value:.6f}")

# Cell 10: Train LSTM
lstm_model = LSTMModel(lookback=20, lstm_units=50, dropout_rate=0.2)
lstm_model.fit(
    X_train_scaled,
    y_train,
    X_val_scaled,
    y_val,
    epochs=30,
    batch_size=32
)

y_pred_lstm = lstm_model.predict(X_test_scaled)

# Handle LSTM output length
if len(y_pred_lstm) < len(y_test):
    y_test_lstm = y_test.iloc[-len(y_pred_lstm):].values
else:
    y_test_lstm = y_test.values
    y_pred_lstm = y_pred_lstm[:len(y_test_lstm)]

metrics_lstm = lstm_model.evaluate(y_test_lstm, y_pred_lstm)
print("\nLSTM Results:")
for metric, value in metrics_lstm.items():
    print(f"  {metric}: {value:.6f}")

# Cell 11: Model Comparison
all_models = {
    'Linear Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'LSTM': lstm_model
}

results_summary = pd.DataFrame({
    'Linear Regression': metrics_lr,
    'Random Forest': metrics_rf,
    'XGBoost': metrics_xgb,
    'LightGBM': metrics_lgb,
    'LSTM': metrics_lstm
}).T

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(results_summary.round(6))

# Cell 12: Visualize Predictions
predictions_plot = pd.DataFrame({
    'Actual': y_test.values[:len(y_pred_lr)],
    'Linear Regression': y_pred_lr[:len(y_test)],
    'Random Forest': y_pred_rf[:len(y_test)],
    'XGBoost': y_pred_xgb[:len(y_test)],
    'LightGBM': y_pred_lgb[:len(y_test)]
})

fig = go.Figure()

fig.add_trace(go.Scatter(y=predictions_plot['Actual'], name='Actual', mode='lines+markers'))
fig.add_trace(go.Scatter(y=predictions_plot['Linear Regression'], name='Linear Regression', mode='lines'))
fig.add_trace(go.Scatter(y=predictions_plot['Random Forest'], name='Random Forest', mode='lines'))
fig.add_trace(go.Scatter(y=predictions_plot['XGBoost'], name='XGBoost', mode='lines'))
fig.add_trace(go.Scatter(y=predictions_plot['LightGBM'], name='LightGBM', mode='lines'))

fig.update_layout(
    title='Stock Price Return Predictions - Test Set',
    xaxis_title='Days',
    yaxis_title='Return',
    template='plotly_white',
    height=500
)
# fig.show()

print("Predictions visualization created")

# Cell 13: Model Metrics Comparison
fig_metrics = go.Figure()

models = list(results_summary.index)
rmse = results_summary['RMSE'].values
mae = results_summary['MAE'].values
r2 = results_summary['R2'].values

fig_metrics = make_subplots(
    rows=1, cols=3,
    subplot_titles=('RMSE', 'MAE', 'R² Score')
)

fig_metrics.add_trace(
    go.Bar(x=models, y=rmse, name='RMSE'),
    row=1, col=1
)
fig_metrics.add_trace(
    go.Bar(x=models, y=mae, name='MAE'),
    row=1, col=2
)
fig_metrics.add_trace(
    go.Bar(x=models, y=r2, name='R²'),
    row=1, col=3
)

# fig_metrics.show()
print("Metrics comparison visualization created")

# Cell 14: Feature Importance Analysis
fig_importance = go.Figure()

top_features = importance_rf.head(15)

fig_importance.add_trace(go.Bar(
    y=top_features['feature'],
    x=top_features['importance'],
    orientation='h'
))

fig_importance.update_layout(
    title='Random Forest - Top 15 Feature Importance',
    xaxis_title='Importance Score',
    height=500
)
# fig_importance.show()

print("Feature importance visualization created")

# Cell 15: Summary and Insights
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)

print("\n1. BEST PERFORMING MODEL:")
best_model = results_summary['R2'].idxmax()
print(f"   {best_model} (R² = {results_summary.loc[best_model, 'R2']:.4f})")

print("\n2. TRAINING DATA INSIGHTS:")
print(f"   - Total samples: {len(features_clean)}")
print(f"   - Training samples: {len(X_train)}")
print(f"   - Validation samples: {len(X_val)}")
print(f"   - Test samples: {len(X_test)}")

print("\n3. FEATURES:")
print(f"   - Total features engineered: {len(feature_cols)}")
print(f"   - Top feature: {importance_rf.iloc[0]['feature']}")

print("\n4. TARGET VARIABLE:")
print(f"   - Mean return: {y_train.mean():.6f}")
print(f"   - Std return: {y_train.std():.6f}")

print("\n5. NEXT STEPS:")
print("   - Fine-tune hyperparameters")
print("   - Add more data sources (sentiment, macroeconomic)")
print("   - Implement backtesting")
print("   - Deploy to production dashboard")

print("\n" + "="*60)

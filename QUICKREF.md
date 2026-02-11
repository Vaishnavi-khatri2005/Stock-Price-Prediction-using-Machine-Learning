"""
QUICK REFERENCE CARD
Stock Price Prediction - Command & Code Examples
"""

# ============================================================================
# QUICK REFERENCE GUIDE - Commands & Code Examples
# ============================================================================

## 🚀 QUICK START (Copy & Paste Ready)

# Option 1: Run Complete Pipeline
# ================================
python quickstart.py

# Option 2: Launch Dashboard
# ===========================
pip install -r requirements.txt
streamlit run dashboard.py

# Option 3: Python Script
# =======================
from main_pipeline import StockPricePredictionPipeline
pipeline = StockPricePredictionPipeline('AAPL')
pipeline.run_pipeline()


## 📦 INSTALLATION

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import pandas; import tensorflow; print('OK')"


## 💻 PYTHON COMMAND SNIPPETS

### Load Data Only
# ==================
from data_loader import StockDataLoader

loader = StockDataLoader('AAPL', start_date='2020-01-01')
data = loader.fetch_data()
loader.validate_data()
loader.save_to_csv('data/my_data.csv')

# Load from CSV
loader.load_from_csv('data/my_data.csv')


### Engineer Features Only
# ==========================
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(data)
features = engineer.engineer_all_features(future_days=5)

# Or individual feature groups
engineer.add_moving_averages([5, 20, 50])
engineer.add_momentum_indicators()
engineer.add_volatility_indicators()
features = engineer.get_features()


### Train Single Model
# =======================
from model_training import XGBoostModel

model = XGBoostModel(max_depth=7, learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = model.evaluate(y_test, predictions)
print(f"RMSE: {metrics['RMSE']:.6f}")
print(f"MAE:  {metrics['MAE']:.6f}")
print(f"R²:   {metrics['R2']:.4f}")


### Train All Models & Ensemble
# ================================
from model_training import (
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    LSTMModel,
    ModelEnsemble
)

models = {
    'XGBoost': XGBoostModel(),
    'LightGBM': LightGBMModel(),
    'Random Forest': RandomForestModel(),
}

ensemble = ModelEnsemble(models)
ensemble.fit(X_train, y_train, X_val, y_val)

ensemble_pred, individual_preds = ensemble.predict(X_test)


### Complete Pipeline Step-by-Step
# ==================================
from main_pipeline import StockPricePredictionPipeline

# Create pipeline
pipeline = StockPricePredictionPipeline('MSFT')

# Step 1: Load data
pipeline.load_data()

# Step 2: Engineer features
pipeline.engineer_features(future_days=5)

# Step 3: Prepare data
pipeline.prepare_data(test_size=0.2, val_size=0.1)

# Step 4: Train models
pipeline.train_models()

# Step 5: Evaluate
results = pipeline.evaluate_models()

# Step 6: Visualize
pipeline.plot_predictions()

# Get results
for model_name, metrics in results.items():
    print(f"{model_name}: RMSE={metrics['RMSE']:.6f}")


### Load & Use Pre-trained Model
# ================================
import joblib

# Load model
model = joblib.load('models/XGBoost_model.pkl')

# Make predictions
predictions = model.predict(X_new)


### Validate Data Quality
# ========================
from utils import DataValidator

quality = DataValidator.check_data_quality(data)
print(f"Quality Score: {quality['quality_score']}")
print(f"Issues: {quality['issues']}")

missing = DataValidator.check_missing_values(data)
print(missing)


### Calculate Additional Metrics
# ================================
import numpy as np
from utils import MetricsCalculator

# Directional accuracy
dir_acc = MetricsCalculator.calculate_directional_accuracy(y_true, y_pred)
print(f"Directional Accuracy: {dir_acc:.2%}")

# MAPE (Mean Absolute Percentage Error)
mape = MetricsCalculator.calculate_mape(y_true, y_pred)
print(f"MAPE: {mape:.2f}%")

# SMAPE (Symmetric MAPE)
smape = MetricsCalculator.calculate_smape(y_true, y_pred)
print(f"SMAPE: {smape:.2f}%")


### Time Series Analysis
# ========================
from utils import TimeSeriesHelper

# Get trading days
trading_days = TimeSeriesHelper.get_trading_days('2020-01-01', '2023-12-31')
print(f"Trading days: {trading_days}")

# Calculate returns
returns = TimeSeriesHelper.calculate_returns(prices)
log_returns = TimeSeriesHelper.calculate_log_returns(prices)

# Get seasonal features
seasonal = TimeSeriesHelper.get_seasonal_features(pd.Timestamp('2023-06-15'))


## 📊 COMMON CONFIGURATIONS

### Predict Different Time Horizon
# ==================================
# 5 days ahead (default)
pipeline.run_pipeline(future_days=5)

# 10 days ahead
pipeline.run_pipeline(future_days=10)

# 1 day ahead (next day only)
pipeline.run_pipeline(future_days=1)


### Train Different Models Only
# ================================
from config import get_config, MODEL_CONFIG

# Get current configuration
config = get_config('models')

# Modify configuration
MODEL_CONFIG['xgboost']['n_estimators'] = 200
MODEL_CONFIG['xgboost']['max_depth'] = 10

# Or modify directly
config['xgboost']['learning_rate'] = 0.05


### Custom Feature Engineering
# =============================
from feature_engineering import FeatureEngineer
from config import get_config, FEATURE_CONFIG

# Get current config
feature_config = get_config('features')

# Modify for custom periods
FEATURE_CONFIG['ma_periods'] = [10, 30, 100]  # Custom MA periods
FEATURE_CONFIG['future_days'] = 10             # Predict 10 days ahead

# Apply in engineer
engineer = FeatureEngineer(data)
features = engineer.engineer_all_features(future_days=10)


## 🎯 SPECIFIC USE CASES

### Compare 2 Stocks
# ====================
from main_pipeline import StockPricePredictionPipeline

for ticker in ['AAPL', 'MSFT']:
    print(f"\n{'='*50}")
    print(f"Analyzing {ticker}")
    print('='*50)
    
    pipeline = StockPricePredictionPipeline(ticker)
    pipeline.run_pipeline()
    
    for model, metrics in pipeline.results.items():
        print(f"{model}: R²={metrics['R2']:.4f}")


### Backtest on Historical Data
# ===============================
from main_pipeline import StockPricePredictionPipeline
import numpy as np

pipeline = StockPricePredictionPipeline('AAPL')
pipeline.load_data()
pipeline.engineer_features()
pipeline.prepare_data()
pipeline.train_models()

# Get predictions
model = pipeline.models['XGBoost']
y_pred = model.predict(pipeline.X_test)

# Calculate returns
y_true = pipeline.y_test.values
cumulative_pred_return = np.cumprod(1 + y_pred) - 1
cumulative_true_return = np.cumprod(1 + y_true) - 1

print(f"Predicted Total Return: {cumulative_pred_return[-1]:.2%}")
print(f"Actual Total Return: {cumulative_true_return[-1]:.2%}")


### Use Custom CSV Data
# =======================
from data_loader import StockDataLoader
from feature_engineering import FeatureEngineer
from model_training import XGBoostModel
from sklearn.preprocessing import StandardScaler

# Load custom CSV
loader = StockDataLoader('CUSTOM')
data = loader.load_from_csv('my_stock_data.csv')

# Engineer features
engineer = FeatureEngineer(data)
features = engineer.engineer_all_features()

# Prepare data
scaler = StandardScaler()
X = features.drop(['date', 'close', 'future_return'], axis=1)
y = features['future_return']

X_train = X.iloc[:int(len(X)*0.7)]
y_train = y.iloc[:int(len(y)*0.7)]

X_test = X.iloc[int(len(X)*0.7):]
y_test = y.iloc[int(len(y)*0.7):]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = XGBoostModel()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)


### Feature Importance Analysis
# ===============================
from main_pipeline import StockPricePredictionPipeline

pipeline = StockPricePredictionPipeline('AAPL')
pipeline.run_pipeline()

# Get feature importance from Random Forest
rf_model = pipeline.models['Random Forest']
importance_df = rf_model.get_feature_importance(pipeline.feature_columns)

# Top 10 features
print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# Bottom 10 features
print("\nTop 10 Least Important Features:")
print(importance_df.tail(10))


## 🛠️ DEBUGGING & TROUBLESHOOTING

### Debug Data Loading
# ======================
from data_loader import StockDataLoader

loader = StockDataLoader('AAPL')
data = loader.fetch_data()

print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Data types:\n{data.dtypes}")
print(f"Missing values:\n{data.isnull().sum()}")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")


### Debug Feature Engineering
# =============================
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer(data)
features = engineer.engineer_all_features()

print(f"Total features: {len(features.columns)}")
print(f"Features list:")
for col in features.columns:
    print(f"  - {col}")

# Check for NaN
nan_count = features.isnull().sum().sum()
print(f"Total NaN values: {nan_count}")


### Debug Model Training
# =======================
from model_training import XGBoostModel
import traceback

try:
    model = XGBoostModel()
    model.fit(X_train, y_train)
    print("Training successful!")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()


## 📈 VISUALIZATION EXAMPLES

### Plot Predictions
# ====================
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='s')
plt.xlabel('Days')
plt.ylabel('Return')
plt.title('Predictions vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('predictions.png')
plt.show()


### Plot Feature Importance
# ==========================
import matplotlib.pyplot as plt

importance_df = rf_model.get_feature_importance(feature_cols)
top_10 = importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_10['feature'], top_10['importance'])
plt.xlabel('Importance Score')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()


### Interactive Plot with Plotly
# ================================
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test.values, name='Actual', mode='lines+markers'))
fig.add_trace(go.Scatter(y=y_pred, name='Predicted', mode='lines'))

fig.update_layout(
    title='Stock Price Predictions',
    xaxis_title='Days',
    yaxis_title='Return',
    template='plotly_white',
    height=500
)
fig.show()


## 📋 FILE REFERENCES

# Main files to work with:
- main_pipeline.py        # Most used - orchestrates everything
- dashboard.py            # For interactive use
- config.py              # To customize settings
- data_loader.py         # To load different data sources
- feature_engineering.py # To create/modify features
- model_training.py      # To use specific models

# Documentation files:
- README.md              # Complete documentation
- SETUP.md              # Installation & troubleshooting
- PROJECT_SUMMARY.md    # Quick overview
- INDEX.md              # File navigation guide


## ⏱️ TIMING ESTIMATES

| Task | Time | Hardware |
|------|------|----------|
| Data Download (5 years) | 1-2 min | Any |
| Feature Engineering | 1-2 min | Any |
| Data Preparation | 30 sec | Any |
| Linear Regression | 10 sec | Any |
| Random Forest (100 trees) | 1-2 min | Any |
| XGBoost (100 estimators) | 1-2 min | Any |
| LightGBM (100 estimators) | 30 sec | Any |
| LSTM (30 epochs) | 5-10 min | CPU / 1-2 min GPU |
| **Total Pipeline** | **10-20 min** | **CPU / 5-10 min GPU** |


## 🎓 LEARNING PATH

1. **Beginner** (Start Here)
   → Run: python quickstart.py
   → Read: README.md
   → Explore: dashboard.py

2. **Intermediate**
   → Study: notebooks/complete_example.py
   → Read: config.py
   → Modify: Feature engineering

3. **Advanced**
   → Customize: All modules
   → Add: New models/indicators
   → Deploy: Production system


## 💾 FILE SIZE ESTIMATES

- Project code: ~15 MB (with dependencies ~2-3 GB)
- Data (5 years AAPL): ~500 KB
- Models (5 models): ~50-100 MB
- Visualizations: ~5-10 MB


---

**Last Updated**: 2024
**Python Version**: 3.8+
**Status**: Production Ready ✓

For detailed information, see INDEX.md or README.md

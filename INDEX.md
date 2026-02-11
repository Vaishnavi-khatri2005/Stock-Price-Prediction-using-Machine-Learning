"""
PROJECT INDEX & NAVIGATION GUIDE
Stock Price Prediction - Machine Learning System
"""

# ============================================================================
# STOCK PRICE PREDICTION PROJECT - COMPLETE FILE INDEX
# ============================================================================

# QUICK NAVIGATION
# ================

## 👉 START HERE
1. Read: PROJECT_SUMMARY.md        (5 min) - Overview of entire project
2. Read: SETUP.md                  (10 min) - Installation instructions
3. Run:  python quickstart.py      (5-10 min) - Execute pipeline
4. Explore: streamlit run dashboard.py - Interactive dashboard

## 📖 FOR UNDERSTANDING
- README.md                         - Complete documentation & examples
- config.py                         - Customizable settings
- notebooks/complete_example.py     - Step-by-step walkthrough


# DETAILED FILE GUIDE
# ===================

## CORE PIPELINE MODULES (Data Processing & ML)

### 📥 data_loader.py
**Purpose**: Download and validate stock market data
**Key Class**: StockDataLoader
**Key Methods**:
  - fetch_data() → Download from Yahoo Finance
  - load_from_csv() → Load local CSV files
  - validate_data() → Check data quality
  - save_to_csv() → Export data

**When to Use**:
  - Getting stock price data
  - Validating data quality
  - Loading custom datasets

**Example**:
  from data_loader import StockDataLoader
  loader = StockDataLoader('AAPL')
  data = loader.fetch_data()


### 🛠️ feature_engineering.py
**Purpose**: Create 50+ technical indicators and derived features
**Key Class**: FeatureEngineer
**Key Methods**:
  - add_moving_averages() → SMA, EMA
  - add_momentum_indicators() → RSI, MACD, Stochastic
  - add_volatility_indicators() → Bollinger Bands, ATR
  - add_volume_indicators() → OBV, CMF
  - add_price_features() → Returns, ratios
  - add_time_features() → Date-based features
  - engineer_all_features() → All features at once

**Features Created**:
  • Moving Averages (8) - SMA & EMA for 4 periods
  • Momentum (8) - RSI, MACD, Stochastic
  • Volatility (5) - Bollinger Bands, ATR, Volatility
  • Volume (3) - OBV, CMF, Volume SMA
  • Price (5) - Returns, ratios, highs/lows
  • Time (10+) - Day, month, quarter, encoded features
  • Lag (12) - Previous 1, 3, 5, 10 day values
  • Rolling (12) - Rolling statistics

**When to Use**:
  - Creating features for ML models
  - Technical analysis
  - Feature visualization

**Example**:
  from feature_engineering import FeatureEngineer
  engineer = FeatureEngineer(data)
  features = engineer.engineer_all_features()


### 🤖 model_training.py
**Purpose**: Implement multiple machine learning models
**Key Classes**:
  - LinearRegressionModel - Fast baseline
  - RandomForestModel - Feature importance
  - XGBoostModel - Best accuracy
  - LightGBMModel - Fast, memory efficient
  - LSTMModel - Neural network for sequences
  - ModelEnsemble - Combine multiple models

**Methods** (common to all):
  - fit(X_train, y_train) → Train model
  - predict(X_test) → Make predictions
  - evaluate(y_true, y_pred) → Calculate metrics
  - save_model(filepath) → Save to disk
  - load_model(filepath) → Load from disk

**When to Use**:
  - Training individual models
  - Creating ensembles
  - Model comparison

**Example**:
  from model_training import XGBoostModel
  model = XGBoostModel()
  model.fit(X_train, y_train)
  predictions = model.predict(X_test)


### 🔄 main_pipeline.py
**Purpose**: Orchestrate entire ML pipeline from start to finish
**Key Class**: StockPricePredictionPipeline
**Key Methods**:
  - load_data() → Step 1: Get data
  - engineer_features() → Step 2: Create features
  - prepare_data() → Step 3: Split & scale
  - train_models() → Step 4: Train all models
  - evaluate_models() → Step 5: Evaluate performance
  - plot_predictions() → Step 6: Visualize results
  - run_pipeline() → Run all steps

**Features**:
  ✓ Automated data loading and validation
  ✓ Feature engineering pipeline
  ✓ Time-series aware train/val/test split
  ✓ Feature scaling (StandardScaler)
  ✓ Train 5 models simultaneously
  ✓ Comprehensive evaluation metrics
  ✓ Auto-generate visualizations

**When to Use**:
  - Running complete analysis
  - Comparing different stocks
  - Batch processing multiple tickers

**Example**:
  from main_pipeline import StockPricePredictionPipeline
  pipeline = StockPricePredictionPipeline('AAPL')
  pipeline.run_pipeline()


## USER INTERFACES (Interaction & Visualization)

### 🎨 dashboard.py
**Purpose**: Interactive web dashboard using Streamlit
**Components**: 5 tabs for complete workflow
  1. 📈 Dashboard - Overview & price charts
  2. 🔬 Data Analysis - Load, engineer, explore
  3. 🤖 Model Training - Train models interactively
  4. 📊 Results - View performance metrics
  5. 💡 Predictions - See predictions

**Features**:
  ✓ Interactive configuration
  ✓ Real-time data loading
  ✓ Visual feature engineering
  ✓ Model selection UI
  ✓ Performance comparison charts
  ✓ Prediction visualization

**How to Run**:
  streamlit run dashboard.py
  # Opens http://localhost:8501

**Best For**:
  - Non-technical users
  - Interactive exploration
  - Quick experimentation
  - Presentations


### 🚀 quickstart.py
**Purpose**: CLI interface for rapid pipeline execution
**Features**:
  ✓ Simple command-line prompts
  ✓ Auto-configuration
  ✓ Progress tracking
  ✓ Summary output

**How to Run**:
  python quickstart.py
  # Follow interactive prompts

**Best For**:
  - Quick testing
  - Automated scripts
  - CI/CD integration
  - Batch processing


## CONFIGURATION & UTILITIES

### ⚙️ config.py
**Purpose**: Centralized configuration management
**Sections**:
  • DATA_CONFIG - Ticker, date range
  • FEATURE_CONFIG - Indicator periods, future days
  • SPLIT_CONFIG - Train/val/test proportions
  • SCALING_CONFIG - Scaling method
  • MODEL_CONFIG - Hyperparameters for each model
  • ENSEMBLE_CONFIG - Ensemble weights
  • EVALUATION_CONFIG - Metrics and plots
  • DIRS_CONFIG - Directory paths
  • LOGGING_CONFIG - Logging settings
  • API_CONFIG - External API keys
  • ADVANCED_CONFIG - Feature selection, outliers, etc.
  • TUNING_CONFIG - Hyperparameter tuning
  • BACKTEST_CONFIG - Backtesting settings

**How to Customize**:
  1. Edit config.py values
  2. Run pipeline
  3. Settings automatically applied

**Example**:
  from config import get_config
  ticker = get_config('data', 'default_ticker')
  ma_periods = get_config('features', 'ma_periods')


### 🔧 utils.py
**Purpose**: Helper utilities and analysis tools
**Classes**:
  - DataValidator - Check data quality
  - MetricsCalculator - Additional metrics
  - FileManager - File operations
  - TimeSeriesHelper - Time series utilities
  - BacktestHelper - Backtesting metrics
  - LoggingHelper - Logging setup
  - PerformanceAnalyzer - Model analysis

**Common Functions**:
  • DataValidator.check_data_quality()
  • MetricsCalculator.calculate_directional_accuracy()
  • TimeSeriesHelper.get_trading_days()
  • FileManager.ensure_directory()
  • BacktestHelper.calculate_sharpe_ratio()

**When to Use**:
  - Data quality checking
  - Additional metrics
  - Backtesting analysis
  - Utility operations


## DOCUMENTATION & EXAMPLES

### 📖 README.md (MAIN DOCUMENTATION)
**Contents**:
  1. Project overview
  2. Quick start guide
  3. Project structure
  4. Feature descriptions
  5. Model documentation
  6. Evaluation metrics
  7. Usage examples
  8. Configuration guide
  9. Dashboard features
  10. Technical stack
  11. Important considerations
  12. References

**Use When**:
  - Need comprehensive documentation
  - Want to understand all features
  - Looking for usage examples
  - Need references

### 📋 SETUP.md (INSTALLATION GUIDE)
**Contents**:
  1. Prerequisites
  2. Virtual environment setup
  3. Dependency installation
  4. Directory structure
  5. Three quick-start options
  6. Customization guide
  7. Troubleshooting
  8. Custom data setup
  9. Advanced configuration
  10. Deployment guide

**Use When**:
  - Installing project first time
  - Troubleshooting setup issues
  - Setting up production environment
  - Deploying to cloud

### 📊 PROJECT_SUMMARY.md
**Contents**:
  1. Quick overview
  2. Structure summary
  3. Start options
  4. Data pipeline diagram
  5. Model descriptions
  6. Feature engineering summary
  7. Evaluation metrics
  8. Output files
  9. Dashboard features
  10. Customization options
  11. Code organization
  12. Learning outcomes
  13. Future improvements

**Use When**:
  - Need quick overview
  - Want project summary
  - Planning improvements
  - Learning about ML pipeline

### 🎓 notebooks/complete_example.py
**Contents**:
  15 cells with complete walkthrough:
  1. Library imports
  2. Data loading
  3. EDA
  4. Feature engineering
  5. Data preparation
  6. Linear Regression
  7. Random Forest
  8. XGBoost
  9. LightGBM
  10. LSTM
  11. Model comparison
  12. Visualizations
  13. Feature importance
  14. Analysis summary
  15. Results

**Use When**:
  - Learning step-by-step
  - Understanding each component
  - Modifying individual steps
  - Educational purposes

**Run As**:
  python notebooks/complete_example.py


## DATA & OUTPUT FILES

### 📁 data/
**Contents**: Historical stock data
**Files Created**:
  - AAPL_historical_data.csv (after running pipeline)
  - Any custom CSV files you load

**Format**:
  date, open, high, low, close, volume
  2023-01-01, 100.5, 101.2, 99.8, 100.8, 1000000

### 📁 models/
**Contents**: Trained model files
**Files Created**:
  - Linear_Regression_model.pkl
  - Random_Forest_model.pkl
  - XGBoost_model.pkl
  - LightGBM_model.pkl
  - LSTM_model.pkl

**Usage**:
  import joblib
  model = joblib.load('models/XGBoost_model.pkl')
  predictions = model.predict(X_test)

### 📁 visualizations/
**Contents**: Generated plots
**Files Created**:
  - predictions_comparison.png (4-model comparison)
  - ensemble_predictions.png (Best model)
  - feature_importance.png (Top features)
  - model_comparison.png (Metrics comparison)


## REQUIREMENTS & DEPENDENCIES

### requirements.txt
**Python Packages**:
  - pandas, numpy - Data processing
  - scikit-learn - Machine learning
  - tensorflow, keras - Deep learning
  - torch - PyTorch (alternative to TF)
  - xgboost - Gradient boosting
  - lightgbm - Light gradient boosting
  - yfinance - Stock data
  - ta - Technical analysis
  - matplotlib, seaborn - Static plots
  - plotly - Interactive plots
  - streamlit - Web dashboard
  - joblib - Model serialization

**Install**:
  pip install -r requirements.txt


## WORKFLOW DIAGRAMS

### Data Flow Pipeline
```
Yahoo Finance
     ↓
DataLoader (fetch_data)
     ↓
Data Validation
     ↓
FeatureEngineer (engineer_all_features)
     ↓
Feature Scaling (StandardScaler)
     ↓
Train/Val/Test Split (Time-Series Aware)
     ↓
Model Training (5 models in parallel)
     ↓
Evaluation (RMSE, MAE, R²)
     ↓
Visualization & Output
```

### File Dependency Graph
```
data_loader.py ─────────┐
                        ├──→ main_pipeline.py ──→ dashboard.py
feature_engineering.py ─┤
                        ├──→ quickstart.py
model_training.py ──────┤
                        └──→ notebook examples
        ↓
    config.py, utils.py
```


## TYPICAL WORKFLOWS

### Workflow 1: Quick Test (5-10 min)
```
python quickstart.py
→ Follow prompts (press Enter for defaults)
→ View results in visualizations/
```

### Workflow 2: Interactive Dashboard (10-20 min)
```
streamlit run dashboard.py
→ Tab 1: Load Data
→ Tab 2: Engineer Features
→ Tab 3: Prepare & Train
→ Tab 4: View Results
→ Tab 5: Make Predictions
```

### Workflow 3: Custom Script (Flexible)
```python
from main_pipeline import StockPricePredictionPipeline

pipeline = StockPricePredictionPipeline('MSFT')
pipeline.load_data()
pipeline.engineer_features(future_days=10)
pipeline.prepare_data(test_size=0.15)
pipeline.train_models()
pipeline.evaluate_models()
```

### Workflow 4: Learning & Experimentation
```
1. Read: README.md & PROJECT_SUMMARY.md
2. Run: notebooks/complete_example.py
3. Study: Each module's code comments
4. Modify: config.py for different settings
5. Explore: dashboard.py for visualization
```


## CUSTOMIZATION EXAMPLES

### Example 1: Different Stock
```python
from main_pipeline import StockPricePredictionPipeline
pipeline = StockPricePredictionPipeline('GOOGL')
pipeline.run_pipeline()
```

### Example 2: Different Time Horizon
```python
pipeline = StockPricePredictionPipeline('AAPL')
pipeline.run_pipeline(future_days=10)  # Predict 10 days ahead
```

### Example 3: Different Models Only
```python
from main_pipeline import StockPricePredictionPipeline
pipeline = StockPricePredictionPipeline('AAPL')
pipeline.load_data()
pipeline.engineer_features()
pipeline.prepare_data()

# Only train specific models
from model_training import XGBoostModel, LightGBMModel
pipeline.models = {
    'XGBoost': XGBoostModel(),
    'LightGBM': LightGBMModel(),
}
pipeline.train_models()
pipeline.evaluate_models()
```

### Example 4: Use Custom Data
```python
from main_pipeline import StockPricePredictionPipeline
pipeline = StockPricePredictionPipeline('CUSTOM')
pipeline.load_data(use_csv='path/to/your/data.csv')
pipeline.engineer_features()
pipeline.prepare_data()
pipeline.train_models()
```


## TROUBLESHOOTING MATRIX

| Problem | Check File | Solution |
|---------|-----------|----------|
| Import error | requirements.txt | pip install -r requirements.txt |
| Data download fails | data_loader.py | Check internet, use CSV |
| Feature error | feature_engineering.py | Ensure data is valid |
| Model training slow | config.py | Reduce n_estimators |
| Memory error | main_pipeline.py | Use smaller dataset |
| Dashboard not launching | dashboard.py | streamlit run dashboard.py |
| Configuration issues | config.py | Review config values |
| Visualization missing | main_pipeline.py | Check visualizations/ directory |


## KEY TAKEAWAYS

✓ **Complete ML Pipeline**: Everything from data to predictions
✓ **Multiple Models**: Compare 5 different algorithms
✓ **Feature Engineering**: 50+ technical indicators
✓ **Easy to Use**: 3 ways to interact (CLI, Dashboard, Python)
✓ **Well Documented**: README, SETUP, examples, comments
✓ **Extensible**: Easy to modify and customize
✓ **Production Ready**: Proper error handling, validation, logging


## NEXT STEPS

1. **First Time Users**:
   - Read PROJECT_SUMMARY.md
   - Follow SETUP.md
   - Run: python quickstart.py
   - Explore: streamlit run dashboard.py

2. **Learners**:
   - Read: README.md
   - Review: notebooks/complete_example.py
   - Study: Source code comments
   - Modify: config.py settings

3. **Advanced Users**:
   - Customize: All modules
   - Add: New features/models
   - Deploy: Production systems
   - Integrate: Additional data sources


---

**Happy Learning & Analyzing! 📊🚀**

For more information, see the detailed documentation files.

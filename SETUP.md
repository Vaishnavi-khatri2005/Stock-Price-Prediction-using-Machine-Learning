"""
Installation and Setup Guide
Complete instructions for setting up the Stock Price Prediction project
"""

# INSTALLATION & SETUP GUIDE
# ============================

"""
STEP 1: PREREQUISITES
=====================

Before starting, ensure you have:
- Python 3.8 or higher (check: python --version)
- pip package manager (check: pip --version)
- Git (optional, for cloning)
- ~2GB free disk space for dependencies and data

Windows Users:
- Install Python from https://www.python.org/downloads/
- Check "Add Python to PATH" during installation

Mac/Linux Users:
- Use your package manager or https://www.python.org/downloads/


STEP 2: CREATE VIRTUAL ENVIRONMENT (Recommended)
=================================================

A virtual environment isolates project dependencies from your system.

Windows:
--------
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) in your terminal prompt

Mac/Linux:
----------
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt


STEP 3: INSTALL DEPENDENCIES
=============================

Install all required Python packages:

# Upgrade pip (important for smooth installation)
pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt

# This will install:
# - Data: pandas, numpy
# - ML: scikit-learn, xgboost, lightgbm
# - DL: tensorflow, torch
# - Visualization: matplotlib, seaborn, plotly
# - Dashboard: streamlit
# - Data source: yfinance
# - Technical analysis: ta

# Verify installation (optional)
pip list

Note: Installation may take 10-15 minutes depending on your internet speed.
For M1/M2 Mac users, some packages may need additional configuration.


STEP 4: VERIFY INSTALLATION
============================

Test if everything is installed correctly:

python -c "import pandas; import numpy; import tensorflow; print('All imports successful!')"

If you get an error, try reinstalling the specific package:
pip install tensorflow --upgrade


STEP 5: DIRECTORY STRUCTURE
============================

The project creates the following structure:

stock-price-prediction/
├── data/                          # Stock data (CSV files)
│   └── AAPL_historical_data.csv
├── models/                        # Trained models
│   ├── Linear_Regression_model.pkl
│   ├── Random_Forest_model.pkl
│   └── ...
├── visualizations/                # Generated plots
│   ├── predictions_comparison.png
│   ├── ensemble_predictions.png
│   └── feature_importance.png
├── notebooks/                     # Jupyter/Python notebooks
│   └── complete_example.py
├── data_loader.py                 # Data loading module
├── feature_engineering.py         # Feature engineering
├── model_training.py              # Model implementations
├── main_pipeline.py               # Main pipeline
├── dashboard.py                   # Streamlit dashboard
├── quickstart.py                  # Quick start script
├── config.py                      # Configuration file
├── requirements.txt               # Dependencies
└── README.md                      # Documentation


STEP 6: QUICK START - THREE OPTIONS
====================================

Option A: RUN COMPLETE PIPELINE (Recommended for first-time)
------------------------------------------------------------

python quickstart.py

This will:
1. Ask for stock ticker (default: AAPL)
2. Download historical data
3. Engineer features
4. Train all models
5. Generate visualizations

Time: ~5-10 minutes for AAPL (5 years)


Option B: RUN INTERACTIVE DASHBOARD
------------------------------------

streamlit run dashboard.py

This will:
1. Open browser to http://localhost:8501
2. Provide interactive interface
3. Allow custom configuration
4. Display real-time results

Steps in dashboard:
- Tab 1: Load data
- Tab 2: Engineer features
- Tab 3: Prepare data & train models
- Tab 4: View results
- Tab 5: Make predictions


Option C: RUN PYTHON SCRIPT DIRECTLY
-------------------------------------

python

# In Python terminal:
from main_pipeline import StockPricePredictionPipeline

# Create pipeline
pipeline = StockPricePredictionPipeline('AAPL')

# Run complete pipeline
pipeline.run_pipeline()

# Or run step by step:
pipeline.load_data()
pipeline.engineer_features()
pipeline.prepare_data()
pipeline.train_models()
pipeline.evaluate_models()
pipeline.plot_predictions()


STEP 7: CUSTOMIZE CONFIGURATION
================================

Edit config.py to customize:

# Example: Change model parameters
MODEL_CONFIG = {
    'xgboost': {
        'max_depth': 10,          # Increase tree depth
        'learning_rate': 0.05,    # Decrease learning rate
        'n_estimators': 200,      # More trees
    }
}

# Example: Change feature engineering
FEATURE_CONFIG = {
    'ma_periods': [10, 30, 100],  # Different moving average periods
    'future_days': 10,            # Predict 10 days ahead
}

Then restart the pipeline with new configuration.


STEP 8: TROUBLESHOOTING
=======================

Problem: "ModuleNotFoundError: No module named 'pandas'"
Solution: pip install pandas

Problem: "ImportError: cannot import name 'yfinance'"
Solution: pip install yfinance

Problem: "TensorFlow not found"
Solution: pip install tensorflow --upgrade

Problem: "Connection timeout downloading data"
Solution: Check internet connection, try again, or use CSV file

Problem: "Permission denied"
Solution: Check file permissions or run with appropriate privileges

Problem: "Out of memory"
Solution: Reduce data size, reduce model complexity, or increase RAM

Problem: "Streamlit not found"
Solution: pip install streamlit


STEP 9: CUSTOM DATA
===================

To use your own CSV data:

1. Prepare CSV file with columns:
   date, open, high, low, close, volume
   (dates in YYYY-MM-DD format)

2. Place in data/ directory

3. Load in Python:
   from data_loader import StockDataLoader
   loader = StockDataLoader('CUSTOM')
   data = loader.load_from_csv('data/your_file.csv')

4. Or in dashboard:
   - Use "Load Data" button with custom file path


STEP 10: ADVANCED SETUP
=======================

GPU ACCELERATION (Optional)

For faster training with GPU:

# For NVIDIA GPUs:
pip install tensorflow[and-cuda]

# Verify GPU:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# For PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


JUPYTER NOTEBOOK SETUP (Optional)

If you want to use Jupyter notebooks:

pip install jupyter notebook

Then:
jupyter notebook

And open notebooks/complete_example.py


STEP 11: ENVIRONMENT VARIABLES (Optional)
==========================================

Set API keys for advanced features:

Windows (Command Prompt):
set ALPHA_VANTAGE_KEY=your_api_key

Mac/Linux (Terminal):
export ALPHA_VANTAGE_KEY=your_api_key

Then use in code:
import os
api_key = os.getenv('ALPHA_VANTAGE_KEY')


STEP 12: DEACTIVATING VIRTUAL ENVIRONMENT
===========================================

When done working, deactivate the environment:

Windows/Mac/Linux:
deactivate

You can reactivate anytime with:
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate


STEP 13: UPDATING DEPENDENCIES
===============================

Check for updates:
pip list --outdated

Update all packages:
pip install --upgrade -r requirements.txt

Update specific package:
pip install --upgrade pandas


COMMON ISSUES & SOLUTIONS
=========================

Issue: Windows long path names
Solution: pip install --upgrade pip setuptools wheel

Issue: SSL certificate error
Solution: pip install --upgrade certifi

Issue: Conflicting package versions
Solution: Delete venv and reinstall from scratch

Issue: Model training is slow
Solution: Reduce n_estimators or max_depth in config.py


GETTING HELP
============

1. Check README.md for project overview
2. Review code comments for implementation details
3. Check output error messages for specific issues
4. Try with smaller dataset first (e.g., 1 year instead of 5)
5. Look at complete_example.py for usage patterns


PERFORMANCE TIPS
================

1. Use validation set: Prevents overfitting
2. Scale features: Important for neural networks
3. Feature engineering: Most important for accuracy
4. Time-series split: Prevents data leakage
5. Ensemble models: Better than single models
6. Monitor memory: Large datasets need optimization
7. Use GPU: TensorFlow/PyTorch with CUDA for faster training


PRODUCTION DEPLOYMENT
=====================

For real-world deployment:

1. Save trained models: models/model_name.pkl
2. Create API using Flask/FastAPI
3. Deploy on cloud (AWS, GCP, Azure)
4. Set up monitoring and logging
5. Implement model retraining pipeline
6. Add risk management features
7. Comply with financial regulations


SECURITY NOTES
==============

1. Don't commit API keys to git
2. Use .gitignore for sensitive files
3. Validate all input data
4. Don't use real money for trading without testing
5. Keep dependencies updated
6. Use virtual environments


NEXT STEPS
==========

1. Complete installation following the steps above
2. Run quickstart.py to test setup
3. Explore the dashboard
4. Customize for your use case
5. Read the README.md for detailed documentation
6. Review the code for learning opportunities


SUPPORT RESOURCES
=================

- Pandas Documentation: https://pandas.pydata.org/
- Scikit-learn: https://scikit-learn.org/
- TensorFlow: https://www.tensorflow.org/
- XGBoost: https://xgboost.readthedocs.io/
- Streamlit: https://docs.streamlit.io/
- Yahoo Finance: https://finance.yahoo.com/
- TA-Lib: https://technical-analysis-library-in-python.readthedocs.io/

"""

# Python script to verify installation
if __name__ == "__main__":
    print("Checking installation...")
    print("-" * 50)
    
    packages = {
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'tensorflow': 'Deep learning',
        'torch': 'PyTorch',
        'yfinance': 'Yahoo Finance data',
        'ta': 'Technical analysis',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical plots',
        'plotly': 'Interactive plots',
        'streamlit': 'Dashboard',
    }
    
    missing = []
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✓ {package:20} ({description})")
        except ImportError:
            print(f"✗ {package:20} ({description}) - NOT FOUND")
            missing.append(package)
    
    print("-" * 50)
    
    if missing:
        print(f"\n{len(missing)} package(s) missing. Install with:")
        print(f"pip install {' '.join(missing)}")
    else:
        print("\n✓ All packages installed successfully!")
        print("Ready to run: python quickstart.py")

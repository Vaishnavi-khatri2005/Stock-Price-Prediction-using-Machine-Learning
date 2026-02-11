"""
Configuration file for the Stock Price Prediction project
Modify these settings to customize the pipeline behavior
"""

# Data Configuration
DATA_CONFIG = {
    'default_ticker': 'AAPL',
    'default_years': 5,
    'default_start_date': None,  # None = auto calculate based on years
    'default_end_date': None,     # None = today
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'ma_periods': [5, 20, 50, 200],           # Moving average periods
    'rsi_period': 14,                          # RSI period
    'macd_fast': 12,                           # MACD fast period
    'macd_slow': 26,                           # MACD slow period
    'bb_period': 20,                           # Bollinger Bands period
    'atr_period': 14,                          # ATR period
    'lag_periods': [1, 3, 5, 10],              # Lag periods
    'rolling_periods': [5, 10, 20],            # Rolling window periods
    'future_days': 5,                          # Prediction horizon
}

# Data Splitting Configuration
SPLIT_CONFIG = {
    'test_size': 0.2,      # 20% test set
    'val_size': 0.1,       # 10% validation set
    'random_state': 42,    # For reproducibility
    'time_series_split': True,  # Important: maintain temporal order
}

# Scaling Configuration
SCALING_CONFIG = {
    'method': 'standard',  # 'standard' or 'minmax'
    'fit_on_train': True,  # Only fit scaler on training data
}

# Model Configuration
MODEL_CONFIG = {
    'linear_regression': {
        'enabled': True,
    },
    
    'random_forest': {
        'enabled': True,
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1,
    },
    
    'xgboost': {
        'enabled': True,
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'random_state': 42,
    },
    
    'lightgbm': {
        'enabled': True,
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'num_leaves': 31,
        'random_state': 42,
    },
    
    'lstm': {
        'enabled': True,
        'lookback': 20,
        'lstm_units': 50,
        'dropout_rate': 0.2,
        'dense_units': 25,
        'epochs': 30,
        'batch_size': 32,
        'optimizer': 'adam',
        'loss': 'mse',
    },
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    'enabled': True,
    'weights': {
        'linear_regression': 0.1,
        'random_forest': 0.25,
        'xgboost': 0.25,
        'lightgbm': 0.25,
        'lstm': 0.15,
    },
    'method': 'weighted_average',  # 'weighted_average' or 'voting'
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': ['rmse', 'mae', 'r2'],
    'classification_metrics': ['accuracy', 'f1', 'precision', 'recall'],
    'plot_predictions': True,
    'plot_feature_importance': True,
    'plot_model_comparison': True,
}

# Directory Configuration
DIRS_CONFIG = {
    'data': 'data',
    'models': 'models',
    'visualizations': 'visualizations',
    'notebooks': 'notebooks',
    'logs': 'logs',
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/pipeline.log',
}

# API Configuration (for advanced data sources)
API_CONFIG = {
    'alpha_vantage_key': None,  # Get from https://www.alphavantage.co/
    'yahoo_finance': {
        'enable': True,
        'timeout': 30,
    },
}

# Advanced Options
ADVANCED_CONFIG = {
    'handle_missing_data': 'interpolate',  # 'drop', 'interpolate', 'forward_fill'
    'remove_outliers': False,
    'outlier_method': 'iqr',  # 'iqr' or 'zscore'
    'outlier_threshold': 3,
    'feature_selection': False,
    'feature_selection_method': 'mutual_information',
    'n_features_to_select': 50,
    'cross_validation': False,
    'n_splits': 5,
}

# Hyperparameter Tuning Configuration
TUNING_CONFIG = {
    'enabled': False,
    'method': 'grid_search',  # 'grid_search', 'random_search', 'bayesian'
    'n_trials': 20,
    'random_state': 42,
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    'enabled': False,
    'method': 'walk_forward',  # 'walk_forward' or 'anchored'
    'window_size': 252,  # Trading days per year
    'step_size': 63,     # Quarterly retraining
}


def get_config(section: str, key: str = None):
    """
    Get configuration value
    
    Args:
        section: Configuration section name
        key: Configuration key (optional)
    
    Returns:
        Configuration value or section dict
    """
    config_dict = {
        'data': DATA_CONFIG,
        'features': FEATURE_CONFIG,
        'split': SPLIT_CONFIG,
        'scaling': SCALING_CONFIG,
        'models': MODEL_CONFIG,
        'ensemble': ENSEMBLE_CONFIG,
        'evaluation': EVALUATION_CONFIG,
        'dirs': DIRS_CONFIG,
        'logging': LOGGING_CONFIG,
        'api': API_CONFIG,
        'advanced': ADVANCED_CONFIG,
        'tuning': TUNING_CONFIG,
        'backtest': BACKTEST_CONFIG,
    }
    
    if section not in config_dict:
        raise ValueError(f"Unknown configuration section: {section}")
    
    config = config_dict[section]
    
    if key:
        if key not in config:
            raise ValueError(f"Unknown configuration key in section '{section}': {key}")
        return config[key]
    
    return config


if __name__ == "__main__":
    # Print all configurations
    print("Stock Price Prediction - Configuration")
    print("="*50)
    
    for section in ['data', 'features', 'split', 'models']:
        config = get_config(section)
        print(f"\n{section.upper()}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

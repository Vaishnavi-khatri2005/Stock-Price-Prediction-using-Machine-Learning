"""
Model Training Module
This module contains implementations of various ML models for stock price prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import warnings
warnings.filterwarnings('ignore')


class BaseModel:
    """Base class for all models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"Model loaded from {filepath}")


class LinearRegressionModel(BaseModel):
    """Linear Regression Model"""
    
    def __init__(self):
        super().__init__("Linear Regression")
        self.model = LinearRegression()
    
    def fit(self, X_train, y_train):
        """Train the model"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        print(f"{self.name} model trained")
    
    def predict(self, X_test):
        """Make predictions"""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)


class RandomForestModel(BaseModel):
    """Random Forest Regressor"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        super().__init__("Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"{self.name} model trained")
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names: list):
        """Get feature importance"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class XGBoostModel(BaseModel):
    """XGBoost Regressor"""
    
    def __init__(self, max_depth: int = 7, learning_rate: float = 0.1, n_estimators: int = 100):
        super().__init__("XGBoost")
        self.model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            tree_method='hist'
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with optional validation set"""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True
        print(f"{self.name} model trained")
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names: list):
        """Get feature importance"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class LightGBMModel(BaseModel):
    """LightGBM Regressor"""
    
    def __init__(self, max_depth: int = 7, learning_rate: float = 0.1, n_estimators: int = 100):
        super().__init__("LightGBM")
        self.model = lgb.LGBMRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with optional validation set"""
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=-1
        )
        self.is_fitted = True
        print(f"{self.name} model trained")
    
    def predict(self, X_test):
        """Make predictions"""
        return self.model.predict(X_test)
    
    def get_feature_importance(self, feature_names: list):
        """Get feature importance"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class LSTMModel(BaseModel):
    """LSTM Neural Network for Time Series"""
    
    def __init__(self, lookback: int = 20, lstm_units: int = 50, dropout_rate: float = 0.2):
        super().__init__("LSTM")
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.scaler = StandardScaler()
        self.model = None
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.lookback):
            X_seq.append(X[i:(i + self.lookback), :])
            y_seq.append(y[i + self.lookback])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model"""
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_train_scaled, y_train.values)
        
        # Build model
        self.model = Sequential([
            LSTM(self.lstm_units, activation='relu', input_shape=(self.lookback, X_seq.shape[2])),
            Dropout(self.dropout_rate),
            Dense(25, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val.values)
            validation_data = (X_val_seq, y_val_seq)
        
        # Train
        self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=0
        )
        
        self.is_fitted = True
        print(f"{self.name} model trained")
    
    def predict(self, X_test):
        """Make predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        X_seq, _ = self.create_sequences(X_test_scaled, np.zeros(len(X_test_scaled)))
        
        if len(X_seq) == 0:
            return np.array([])
        
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()


class ModelEnsemble:
    """Ensemble of multiple models"""
    
    def __init__(self, models: dict):
        """
        Initialize ensemble with multiple models
        
        Args:
            models (dict): Dictionary of model_name: model_object
        """
        self.models = models
        self.weights = {name: 1.0 for name in models.keys()}
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models"""
        print("\n=== Training Ensemble Models ===\n")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            if name == "LSTM":
                model.fit(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
            else:
                model.fit(X_train, y_train)
            print(f"{name} training completed\n")
    
    def predict(self, X_test):
        """Make ensemble predictions (weighted average)"""
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
        
        # Calculate weighted average
        total_weight = sum(self.weights.values())
        ensemble_pred = np.zeros(len(X_test))
        
        for name, pred in predictions.items():
            ensemble_pred += pred * self.weights[name] / total_weight
        
        return ensemble_pred, predictions
    
    def set_weights(self, weights: dict):
        """Set custom weights for ensemble models"""
        self.weights = weights
        print(f"Ensemble weights updated: {weights}")


if __name__ == "__main__":
    print("Model training module loaded successfully")

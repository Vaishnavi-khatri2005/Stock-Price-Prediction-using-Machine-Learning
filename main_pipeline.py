"""
Main Training Pipeline
This script orchestrates the entire ML pipeline:
1. Data loading
2. Feature engineering
3. Data splitting and scaling
4. Model training
5. Evaluation
6. Visualization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

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


class StockPricePredictionPipeline:
    """Complete ML pipeline for stock price prediction"""
    
    def __init__(self, ticker: str = 'AAPL', start_date: str = None, end_date: str = None):
        """Initialize pipeline"""
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        
        self.data = None
        self.features = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        self.feature_columns = None
        self.scaler = StandardScaler()
        
        self.models = {}
        self.ensemble = None
        self.results = {}
    
    def load_data(self, use_csv: str = None):
        """Load stock data"""
        print("\n" + "="*50)
        print("STEP 1: DATA LOADING")
        print("="*50 + "\n")
        
        loader = StockDataLoader(self.ticker, self.start_date, self.end_date)
        
        if use_csv:
            self.data = loader.load_from_csv(use_csv)
        else:
            self.data = loader.fetch_data()
        
        if self.data is not None:
            loader.validate_data()
            loader.save_to_csv(f'data/{self.ticker}_historical_data.csv')
        
        return self.data is not None
    
    def engineer_features(self, future_days: int = 5):
        """Engineer features"""
        print("\n" + "="*50)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*50)
        
        engineer = FeatureEngineer(self.data)
        self.features = engineer.engineer_all_features(future_days=future_days)
        
        # Remove null values
        initial_len = len(self.features)
        self.features = engineer.remove_null_rows()
        
        print(f"\nFinal dataset shape: {self.features.shape}")
        print(f"Rows removed due to NaN: {initial_len - len(self.features)}")
        
        return self.features
    
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.1):
        """Split and scale data for training"""
        print("\n" + "="*50)
        print("STEP 3: DATA PREPARATION")
        print("="*50 + "\n")
        
        # Select features (exclude date, OHLC, target variables)
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'future_return', 'price_up']
        self.feature_columns = [col for col in self.features.columns if col not in exclude_cols]
        
        X = self.features[self.feature_columns]
        y = self.features['future_return']  # Using regression target
        
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Target variable: future_return (5-day ahead return)")
        
        # Time-series split (not random shuffle)
        split_idx = int(len(X) * (1 - test_size))
        val_idx = int(split_idx * (1 - val_size))
        
        # Training set
        self.X_train = X.iloc[:val_idx].reset_index(drop=True)
        self.y_train = y.iloc[:val_idx].reset_index(drop=True)
        
        # Validation set
        self.X_val = X.iloc[val_idx:split_idx].reset_index(drop=True)
        self.y_val = y.iloc[val_idx:split_idx].reset_index(drop=True)
        
        # Test set
        self.X_test = X.iloc[split_idx:].reset_index(drop=True)
        self.y_test = y.iloc[split_idx:].reset_index(drop=True)
        
        print(f"\nData split:")
        print(f"  Training:   {len(self.X_train)} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(self.X_val)} samples ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"  Test:       {len(self.X_test)} samples ({len(self.X_test)/len(X)*100:.1f}%)")
        
        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_columns
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.feature_columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_columns
        )
        
        print(f"\nFeatures scaled using StandardScaler")
        
        return True
    
    def train_models(self):
        """Train all models"""
        print("\n" + "="*50)
        print("STEP 4: MODEL TRAINING")
        print("="*50)
        
        # Initialize models
        self.models = {
            'Linear Regression': LinearRegressionModel(),
            'Random Forest': RandomForestModel(n_estimators=100, max_depth=15),
            'XGBoost': XGBoostModel(max_depth=7, learning_rate=0.1, n_estimators=100),
            'LightGBM': LightGBMModel(max_depth=7, learning_rate=0.1, n_estimators=100),
            'LSTM': LSTMModel(lookback=20, lstm_units=50, dropout_rate=0.2)
        }
        
        # Train each model
        for name, model in self.models.items():
            print(f"\n--- Training {name} ---")
            
            try:
                if name == 'LSTM':
                    model.fit(self.X_train, self.y_train, 
                             self.X_val, self.y_val, 
                             epochs=30, batch_size=32)
                else:
                    model.fit(self.X_train, self.y_train)
                
                # Save model
                os.makedirs('models', exist_ok=True)
                model.save_model(f'models/{name.replace(" ", "_")}_model.pkl')
            
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
        
        # Create ensemble
        print("\n--- Creating Model Ensemble ---")
        self.ensemble = ModelEnsemble(self.models)
        
        return len(self.models) > 0
    
    def evaluate_models(self):
        """Evaluate all models"""
        print("\n" + "="*50)
        print("STEP 5: MODEL EVALUATION")
        print("="*50 + "\n")
        
        # Evaluate individual models
        print("Individual Model Performance:\n")
        
        for name, model in self.models.items():
            try:
                # Predictions
                if name == 'LSTM':
                    y_pred = model.predict(self.X_test)
                else:
                    y_pred = model.predict(self.X_test)
                
                # Handle LSTM output (may be shorter)
                if len(y_pred) < len(self.y_test):
                    y_test_adjusted = self.y_test.iloc[-len(y_pred):].values
                else:
                    y_test_adjusted = self.y_test.values
                    y_pred = y_pred[:len(y_test_adjusted)]
                
                # Calculate metrics
                metrics = model.evaluate(y_test_adjusted, y_pred)
                self.results[name] = metrics
                
                print(f"{name}:")
                print(f"  RMSE: {metrics['RMSE']:.6f}")
                print(f"  MAE:  {metrics['MAE']:.6f}")
                print(f"  R²:   {metrics['R2']:.4f}")
                print()
            
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}\n")
        
        # Evaluate ensemble
        print("Ensemble Model Performance:\n")
        try:
            y_pred_ensemble, _ = self.ensemble.predict(self.X_test)
            
            if len(y_pred_ensemble) < len(self.y_test):
                y_test_adjusted = self.y_test.iloc[-len(y_pred_ensemble):].values
            else:
                y_test_adjusted = self.y_test.values
                y_pred_ensemble = y_pred_ensemble[:len(y_test_adjusted)]
            
            metrics_ensemble = self.models['Random Forest'].evaluate(y_test_adjusted, y_pred_ensemble)
            self.results['Ensemble'] = metrics_ensemble
            
            print(f"Ensemble:")
            print(f"  RMSE: {metrics_ensemble['RMSE']:.6f}")
            print(f"  MAE:  {metrics_ensemble['MAE']:.6f}")
            print(f"  R²:   {metrics_ensemble['R2']:.4f}")
        
        except Exception as e:
            print(f"Error evaluating ensemble: {str(e)}\n")
        
        return self.results
    
    def plot_predictions(self):
        """Plot predicted vs actual prices"""
        print("\n" + "="*50)
        print("STEP 6: VISUALIZATIONS")
        print("="*50 + "\n")
        
        # Get test dates
        test_start_idx = int(len(self.features) * 0.9)
        test_dates = self.features['date'].iloc[test_start_idx:].reset_index(drop=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.ticker} Stock Price Predictions - Test Set', fontsize=16, fontweight='bold')
        
        plot_models = ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM']
        
        for idx, (ax, model_name) in enumerate(zip(axes.flat, plot_models)):
            if model_name in self.models:
                model = self.models[model_name]
                
                try:
                    y_pred = model.predict(self.X_test)
                    
                    # Handle different output lengths
                    if len(y_pred) < len(self.y_test):
                        y_test_plot = self.y_test.iloc[-len(y_pred):].values
                        dates_plot = test_dates.iloc[-len(y_pred):].values
                    else:
                        y_test_plot = self.y_test.values
                        y_pred = y_pred[:len(y_test_plot)]
                        dates_plot = test_dates.iloc[:len(y_pred)].values
                    
                    ax.plot(range(len(dates_plot)), y_test_plot, label='Actual', linewidth=2, marker='o', markersize=4)
                    ax.plot(range(len(dates_plot)), y_pred, label='Predicted', linewidth=2, marker='s', markersize=4, alpha=0.7)
                    ax.set_title(f'{model_name}', fontweight='bold')
                    ax.set_xlabel('Days')
                    ax.set_ylabel('Return')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig('visualizations/predictions_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: visualizations/predictions_comparison.png")
        
        # Plot ensemble predictions
        fig, ax = plt.subplots(figsize=(14, 6))
        
        try:
            y_pred_ensemble, individual_preds = self.ensemble.predict(self.X_test)
            
            if len(y_pred_ensemble) < len(self.y_test):
                y_test_plot = self.y_test.iloc[-len(y_pred_ensemble):].values
            else:
                y_test_plot = self.y_test.values
                y_pred_ensemble = y_pred_ensemble[:len(y_test_plot)]
            
            ax.plot(range(len(y_test_plot)), y_test_plot, label='Actual Price Returns', linewidth=2.5, marker='o')
            ax.plot(range(len(y_pred_ensemble)), y_pred_ensemble, label='Ensemble Prediction', 
                   linewidth=2.5, marker='s', alpha=0.7)
            
            ax.fill_between(range(len(y_test_plot)), y_test_plot, y_pred_ensemble, alpha=0.2)
            ax.set_title(f'{self.ticker} - Ensemble Model Predictions vs Actual', fontsize=14, fontweight='bold')
            ax.set_xlabel('Days')
            ax.set_ylabel('5-Day Ahead Return')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('visualizations/ensemble_predictions.png', dpi=300, bbox_inches='tight')
            print("Saved: visualizations/ensemble_predictions.png")
        
        except Exception as e:
            print(f"Error plotting ensemble: {str(e)}")
        
        # Plot feature importance
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            importance_df = rf_model.get_feature_importance(self.feature_columns)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            top_features = importance_df.head(15)
            
            ax.barh(top_features['feature'], top_features['importance'])
            ax.set_xlabel('Importance Score', fontweight='bold')
            ax.set_title('Random Forest - Top 15 Feature Importance', fontweight='bold', fontsize=14)
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
            print("Saved: visualizations/feature_importance.png")
        
        # Plot model comparison
        if self.results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            models_list = list(self.results.keys())
            rmse_vals = [self.results[m]['RMSE'] for m in models_list]
            mae_vals = [self.results[m]['MAE'] for m in models_list]
            r2_vals = [self.results[m]['R2'] for m in models_list]
            
            # RMSE
            axes[0].bar(models_list, rmse_vals, color='steelblue')
            axes[0].set_title('RMSE Comparison', fontweight='bold')
            axes[0].set_ylabel('RMSE')
            axes[0].tick_params(axis='x', rotation=45)
            
            # MAE
            axes[1].bar(models_list, mae_vals, color='coral')
            axes[1].set_title('MAE Comparison', fontweight='bold')
            axes[1].set_ylabel('MAE')
            axes[1].tick_params(axis='x', rotation=45)
            
            # R2
            axes[2].bar(models_list, r2_vals, color='seagreen')
            axes[2].set_title('R² Comparison', fontweight='bold')
            axes[2].set_ylabel('R² Score')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved: visualizations/model_comparison.png")
        
        plt.close('all')
    
    def run_pipeline(self, use_csv: str = None, future_days: int = 5):
        """Run the complete pipeline"""
        print("\n" + "="*50)
        print(f"STOCK PRICE PREDICTION PIPELINE - {self.ticker}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        # Create visualization directory
        os.makedirs('visualizations', exist_ok=True)
        
        # Run pipeline steps
        if not self.load_data(use_csv):
            print("Failed to load data")
            return False
        
        self.engineer_features(future_days)
        self.prepare_data()
        self.train_models()
        self.evaluate_models()
        self.plot_predictions()
        
        print("\n" + "="*50)
        print(f"PIPELINE COMPLETED")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        
        return True


# Main execution
if __name__ == "__main__":
    # Configuration
    TICKER = 'AAPL'
    
    # Run pipeline
    pipeline = StockPricePredictionPipeline(TICKER)
    pipeline.run_pipeline()
    
    # Summary
    print("\nResults Summary:")
    print("-" * 50)
    for model_name, metrics in pipeline.results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")

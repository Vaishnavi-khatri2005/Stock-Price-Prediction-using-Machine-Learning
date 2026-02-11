"""
Utility Functions Module
Helper functions and utilities for the stock prediction pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path


class DataValidator:
    """Validate stock price data"""
    
    @staticmethod
    def check_missing_values(data: pd.DataFrame) -> dict:
        """Check for missing values"""
        missing = data.isnull().sum()
        missing_pct = (missing / len(data) * 100).round(2)
        
        return {
            'columns_with_missing': missing[missing > 0].to_dict(),
            'percentage': missing_pct[missing_pct > 0].to_dict()
        }
    
    @staticmethod
    def check_data_types(data: pd.DataFrame) -> dict:
        """Verify data types"""
        expected_types = {
            'date': 'datetime64',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'int64'
        }
        
        issues = {}
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if expected_type not in actual_type:
                    issues[col] = {
                        'expected': expected_type,
                        'actual': actual_type
                    }
        
        return issues
    
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> dict:
        """Check overall data quality"""
        quality_score = 100
        issues = []
        
        # Check missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > 0.1:
            quality_score -= 20
            issues.append(f"High missing value ratio: {missing_ratio:.2%}")
        
        # Check for duplicates
        if data.duplicated(subset=['date']).any():
            quality_score -= 15
            issues.append("Duplicate dates found")
        
        # Check price integrity
        if (data['high'] < data['low']).any():
            quality_score -= 25
            issues.append("High price < Low price")
        
        if (data['close'] > data['high']).any() or (data['close'] < data['low']).any():
            quality_score -= 25
            issues.append("Close price outside high-low range")
        
        # Check volume
        if (data['volume'] < 0).any():
            quality_score -= 15
            issues.append("Negative volume found")
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'status': 'PASS' if quality_score >= 80 else 'FAIL' if quality_score < 60 else 'WARNING'
        }


class MetricsCalculator:
    """Calculate additional metrics"""
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy of predicting direction (up/down)"""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        
        accuracy = np.mean(true_direction == pred_direction)
        return accuracy
    
    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0
        return np.mean(diff) * 100
    
    @staticmethod
    def calculate_mrae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Relative Absolute Error"""
        return np.mean(np.abs(y_true - y_pred) / np.abs(y_true))


class FileManager:
    """Manage file operations"""
    
    @staticmethod
    def ensure_directory(directory: str) -> str:
        """Create directory if it doesn't exist"""
        os.makedirs(directory, exist_ok=True)
        return directory
    
    @staticmethod
    def get_latest_file(directory: str, pattern: str = "*") -> str:
        """Get latest file in directory"""
        path = Path(directory)
        files = sorted(path.glob(pattern), key=os.path.getmtime, reverse=True)
        return str(files[0]) if files else None
    
    @staticmethod
    def save_json(data: dict, filepath: str):
        """Save dictionary as JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)


class TimeSeriesHelper:
    """Helper functions for time series operations"""
    
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate returns from prices"""
        return np.diff(prices) / prices[:-1]
    
    @staticmethod
    def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns from prices"""
        return np.diff(np.log(prices))
    
    @staticmethod
    def get_trading_days(start_date: str, end_date: str) -> int:
        """Estimate number of trading days (approximately 252 per year)"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        days = (end - start).days
        return int(days * 252 / 365.25)
    
    @staticmethod
    def get_business_days(start_date: str, end_date: str) -> int:
        """Get number of business days between dates"""
        dates = pd.bdate_range(start=start_date, end=end_date)
        return len(dates)
    
    @staticmethod
    def get_seasonal_features(date: pd.Timestamp) -> dict:
        """Extract seasonal features from date"""
        return {
            'day_of_week': date.dayofweek,
            'month': date.month,
            'quarter': date.quarter,
            'is_quarter_start': date.is_quarter_start,
            'is_quarter_end': date.is_quarter_end,
            'is_year_start': date.is_year_start,
            'is_year_end': date.is_year_end,
            'days_in_month': date.daysinmonth,
        }


class BacktestHelper:
    """Helper functions for backtesting"""
    
    @staticmethod
    def calculate_cumulative_returns(returns: np.ndarray) -> np.ndarray:
        """Calculate cumulative returns from daily returns"""
        return np.cumprod(1 + returns) - 1
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio (annualized)"""
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1]) - 1
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (1 + running_max)
        return np.min(drawdown)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, target_return: float = 0.0, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (considers only downside volatility)"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = np.minimum(excess_returns - target_return, 0)
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)


class LoggingHelper:
    """Helper for logging"""
    
    @staticmethod
    def create_logger(name: str, log_file: str = None):
        """Create a logger"""
        import logging
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


class PerformanceAnalyzer:
    """Analyze model performance"""
    
    @staticmethod
    def get_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Analyze error distribution"""
        errors = y_true - y_pred
        
        return {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'quantile_25': np.percentile(errors, 25),
            'quantile_75': np.percentile(errors, 75),
        }
    
    @staticmethod
    def calculate_residuals_statistics(residuals: np.ndarray) -> dict:
        """Calculate residual statistics"""
        from scipy import stats
        
        return {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'normality_test': 'Pass' if stats.normaltest(residuals)[1] > 0.05 else 'Fail',
            'autocorrelation': 'Present' if np.abs(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) > 0.3 else 'Absent',
        }


# Example usage
if __name__ == "__main__":
    # Test DataValidator
    print("Testing DataValidator...")
    print("-" * 50)
    
    # Create sample data
    test_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Validate
    quality = DataValidator.check_data_quality(test_data)
    print(f"Data Quality: {quality}")
    
    # Test MetricsCalculator
    print("\nTesting MetricsCalculator...")
    print("-" * 50)
    
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    y_pred = np.array([0.015, -0.018, 0.028, -0.012, 0.022])
    
    accuracy = MetricsCalculator.calculate_directional_accuracy(y_true, y_pred)
    print(f"Directional Accuracy: {accuracy:.2%}")
    
    # Test TimeSeriesHelper
    print("\nTesting TimeSeriesHelper...")
    print("-" * 50)
    
    trading_days = TimeSeriesHelper.get_trading_days('2020-01-01', '2023-12-31')
    business_days = TimeSeriesHelper.get_business_days('2020-01-01', '2023-12-31')
    
    print(f"Trading days (est): {trading_days}")
    print(f"Business days: {business_days}")

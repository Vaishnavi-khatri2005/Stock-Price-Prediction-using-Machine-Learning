"""
Feature Engineering Module
This module calculates technical indicators and creates features for ML models
"""

import pandas as pd
import numpy as np
import ta


class FeatureEngineer:
    """Calculate technical indicators and engineer features"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer with stock data
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.original_data = data.copy()
    
    def add_moving_averages(self, periods: list = [5, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate moving averages
        
        Args:
            periods (list): List of periods for moving averages
            
        Returns:
            pd.DataFrame: Data with MA features added
        """
        for period in periods:
            self.data[f'sma_{period}'] = self.data['close'].rolling(window=period).mean()
            self.data[f'ema_{period}'] = self.data['close'].ewm(span=period, adjust=False).mean()
        
        print(f"Added moving averages for periods: {periods}")
        return self.data
    
    def add_momentum_indicators(self) -> pd.DataFrame:
        """
        Calculate momentum indicators (RSI, MACD, Stochastic)
        
        Returns:
            pd.DataFrame: Data with momentum indicators
        """
        # RSI (Relative Strength Index)
        self.data['rsi_14'] = ta.momentum.rsi(self.data['close'], window=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.macd(self.data['close'])
        self.data['macd'] = macd
        self.data['macd_signal'] = ta.trend.macd_signal(self.data['close'])
        self.data['macd_diff'] = ta.trend.macd_diff(self.data['close'])
        
        # Stochastic Oscillator
        self.data['stoch_k'] = ta.momentum.stoch(self.data['high'], self.data['low'], self.data['close'])
        self.data['stoch_d'] = ta.momentum.stoch_signal(self.data['high'], self.data['low'], self.data['close'])
        
        print("Added momentum indicators (RSI, MACD, Stochastic)")
        return self.data
    
    def add_volatility_indicators(self) -> pd.DataFrame:
        """
        Calculate volatility indicators (Bollinger Bands, ATR)
        
        Returns:
            pd.DataFrame: Data with volatility indicators
        """
        # Bollinger Bands
        bb = ta.volatility.bollinger_bands(self.data['close'], window=20)
        self.data['bb_high'] = bb.iloc[:, 0]
        self.data['bb_mid'] = bb.iloc[:, 1]
        self.data['bb_low'] = bb.iloc[:, 2]
        
        # Average True Range
        self.data['atr'] = ta.volatility.average_true_range(
            self.data['high'], 
            self.data['low'], 
            self.data['close']
        )
        
        # Historical Volatility
        self.data['volatility'] = self.data['close'].pct_change().rolling(window=20).std()
        
        print("Added volatility indicators (Bollinger Bands, ATR, Volatility)")
        return self.data
    
    def add_volume_indicators(self) -> pd.DataFrame:
        """
        Calculate volume indicators (OBV, CMF)
        
        Returns:
            pd.DataFrame: Data with volume indicators
        """
        # On-Balance Volume
        self.data['obv'] = ta.volume.on_balance_volume(self.data['close'], self.data['volume'])
        
        # Chaikin Money Flow
        self.data['cmf'] = ta.volume.chaikin_money_flow(
            self.data['high'], 
            self.data['low'], 
            self.data['close'], 
            self.data['volume']
        )
        
        # Volume Moving Average
        self.data['volume_sma_20'] = self.data['volume'].rolling(window=20).mean()
        
        print("Added volume indicators (OBV, CMF)")
        return self.data
    
    def add_price_features(self) -> pd.DataFrame:
        """
        Calculate price-based features
        
        Returns:
            pd.DataFrame: Data with price features
        """
        # Daily returns
        self.data['daily_return'] = self.data['close'].pct_change()
        
        # High-Low spread
        self.data['hl_ratio'] = self.data['high'] / self.data['low']
        
        # Close-Open ratio
        self.data['co_ratio'] = self.data['close'] / self.data['open']
        
        # Price change
        self.data['price_change'] = self.data['close'].diff()
        
        # Higher Highs and Lower Lows
        self.data['higher_high'] = (self.data['high'] > self.data['high'].shift(1)).astype(int)
        self.data['lower_low'] = (self.data['low'] < self.data['low'].shift(1)).astype(int)
        
        print("Added price-based features")
        return self.data
    
    def add_time_features(self) -> pd.DataFrame:
        """
        Add time-based features
        
        Returns:
            pd.DataFrame: Data with time features
        """
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['day_of_month'] = self.data['date'].dt.day
        self.data['month'] = self.data['date'].dt.month
        self.data['quarter'] = self.data['date'].dt.quarter
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        
        # One-hot encode day of week
        days_dummies = pd.get_dummies(self.data['day_of_week'], prefix='day_of_week', drop_first=True)
        self.data = pd.concat([self.data, days_dummies], axis=1)
        
        # One-hot encode month
        month_dummies = pd.get_dummies(self.data['month'], prefix='month', drop_first=True)
        self.data = pd.concat([self.data, month_dummies], axis=1)
        
        print("Added time-based features")
        return self.data
    
    def add_lag_features(self, lag_periods: list = [1, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lagged features (previous day values)
        
        Args:
            lag_periods (list): List of lag periods
            
        Returns:
            pd.DataFrame: Data with lag features
        """
        for lag in lag_periods:
            self.data[f'close_lag_{lag}'] = self.data['close'].shift(lag)
            self.data[f'volume_lag_{lag}'] = self.data['volume'].shift(lag)
            self.data[f'return_lag_{lag}'] = self.data['daily_return'].shift(lag)
        
        print(f"Added lag features for periods: {lag_periods}")
        return self.data
    
    def add_rolling_features(self, periods: list = [5, 10, 20]) -> pd.DataFrame:
        """
        Add rolling window features (rolling mean, std)
        
        Args:
            periods (list): List of rolling window periods
            
        Returns:
            pd.DataFrame: Data with rolling features
        """
        for period in periods:
            self.data[f'close_rolling_mean_{period}'] = self.data['close'].rolling(window=period).mean()
            self.data[f'close_rolling_std_{period}'] = self.data['close'].rolling(window=period).std()
            self.data[f'volume_rolling_mean_{period}'] = self.data['volume'].rolling(window=period).mean()
            self.data[f'return_rolling_std_{period}'] = self.data['daily_return'].rolling(window=period).std()
        
        print(f"Added rolling features for periods: {periods}")
        return self.data
    
    def add_target_variable(self, future_days: int = 5) -> pd.DataFrame:
        """
        Create target variable (future returns)
        
        Args:
            future_days (int): Number of days ahead to predict
            
        Returns:
            pd.DataFrame: Data with target variable
        """
        # Target: Return after n days
        self.data['future_return'] = self.data['close'].shift(-future_days) / self.data['close'] - 1
        
        # Target: Binary classification (up/down)
        self.data['price_up'] = (self.data['future_return'] > 0).astype(int)
        
        print(f"Added target variables (future_return for {future_days} days, price_up classification)")
        return self.data
    
    def engineer_all_features(self, future_days: int = 5) -> pd.DataFrame:
        """
        Engineer all features at once
        
        Args:
            future_days (int): Number of days ahead for target variable
            
        Returns:
            pd.DataFrame: Data with all engineered features
        """
        print("\n=== Feature Engineering Started ===\n")
        
        self.add_moving_averages()
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_volume_indicators()
        self.add_price_features()
        self.add_time_features()
        self.add_lag_features()
        self.add_rolling_features()
        self.add_target_variable(future_days)
        
        print("\n=== Feature Engineering Completed ===\n")
        print(f"Total features created: {len(self.data.columns) - len(self.original_data.columns)}")
        
        return self.data
    
    def get_features(self) -> pd.DataFrame:
        """Get engineered features"""
        return self.data
    
    def remove_null_rows(self) -> pd.DataFrame:
        """Remove rows with missing values"""
        initial_len = len(self.data)
        self.data = self.data.dropna()
        removed = initial_len - len(self.data)
        print(f"Removed {removed} rows with missing values")
        return self.data


# Example usage
if __name__ == "__main__":
    from data_loader import StockDataLoader
    
    # Load data
    loader = StockDataLoader('AAPL')
    data = loader.fetch_data()
    
    if data is not None:
        # Engineer features
        engineer = FeatureEngineer(data)
        features = engineer.engineer_all_features()
        
        print("\nFinal dataset shape:", features.shape)
        print("\nColumns created:")
        print(features.columns.tolist())

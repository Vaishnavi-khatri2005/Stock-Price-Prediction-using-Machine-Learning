"""
Data Loader Module
This module handles fetching and loading stock price data from Yahoo Finance
and provides basic data validation and preprocessing.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os


class StockDataLoader:
    """Load and validate stock price data from Yahoo Finance"""
    
    def __init__(self, ticker: str, start_date: str = None, end_date: str = None):
        """
        Initialize the data loader
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
        """
        self.ticker = ticker.upper()
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        # Default to 5 years of data if start_date not specified
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance
        
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        
        try:
            self.data = yf.download(
                self.ticker, 
                start=self.start_date, 
                end=self.end_date, 
                progress=False
            )
            
            # Reset index to make Date a column
            self.data.reset_index(inplace=True)
            
            # Rename columns to lowercase for consistency
            self.data.columns = [col.lower() for col in self.data.columns]
            
            print(f"Successfully fetched {len(self.data)} records")
            print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
            
            return self.data
        
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load stock data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            self.data = pd.read_csv(filepath)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
            
            print(f"Successfully loaded {len(self.data)} records from {filepath}")
            return self.data
        
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return None
    
    def save_to_csv(self, filepath: str = None):
        """
        Save data to CSV file
        
        Args:
            filepath (str): Path to save CSV
        """
        if filepath is None:
            filepath = f"data/{self.ticker}_data.csv"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def validate_data(self) -> bool:
        """
        Validate data quality
        
        Returns:
            bool: True if data is valid
        """
        if self.data is None or len(self.data) == 0:
            print("Error: No data to validate")
            return False
        
        # Check for required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}")
            return False
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print(f"Warning: Missing values found:\n{missing_values[missing_values > 0]}")
        
        # Check for data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                print(f"Error: Column '{col}' is not numeric")
                return False
        
        print("Data validation passed!")
        return True
    
    def get_data(self) -> pd.DataFrame:
        """Get the loaded data"""
        return self.data


# Example usage
if __name__ == "__main__":
    # Fetch data from Yahoo Finance
    loader = StockDataLoader('AAPL')
    data = loader.fetch_data()
    
    if data is not None:
        loader.validate_data()
        loader.save_to_csv()
        print("\nFirst few rows:")
        print(data.head())

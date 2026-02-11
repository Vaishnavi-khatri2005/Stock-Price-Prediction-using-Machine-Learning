#!/usr/bin/env python
"""
Quick Start Script
Run this script to set up and execute the complete pipeline
"""

import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from main_pipeline import StockPricePredictionPipeline


def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print(" "*15 + "STOCK PRICE PREDICTION ML PROJECT")
    print(" "*20 + "Quick Start Guide")
    print("="*70 + "\n")
    
    # Get user input
    print("Configuration:")
    ticker = input("Enter stock ticker (default: AAPL): ").strip().upper() or "AAPL"
    years = input("Enter years of historical data (default: 5): ").strip() or "5"
    future_days = input("Enter prediction horizon in days (default: 5): ").strip() or "5"
    
    try:
        years = int(years)
        future_days = int(future_days)
    except ValueError:
        print("Invalid input. Using defaults...")
        years = 5
        future_days = 5
    
    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Historical data: {years} years")
    print(f"  Prediction horizon: {future_days} days")
    print(f"{'='*70}\n")
    
    # Run pipeline
    try:
        pipeline = StockPricePredictionPipeline(ticker=ticker)
        
        # Calculate dates
        from datetime import datetime, timedelta
        start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        pipeline.start_date = start_date
        
        # Execute pipeline
        pipeline.run_pipeline(future_days=future_days)
        
        print("\n" + "="*70)
        print("Pipeline completed successfully!")
        print("="*70)
        print("\nGenerated files:")
        print("  - Trained models in: models/")
        print("  - Visualizations in: visualizations/")
        print("  - Data saved in: data/")
        print("\nNext steps:")
        print("  1. Review the generated plots in visualizations/")
        print("  2. Launch the dashboard: streamlit run dashboard.py")
        print("  3. Explore the notebook example: notebooks/complete_example.py")
        print("\n" + "="*70 + "\n")
        
        return True
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Ensure you have internet connection (for data download)")
        print("  2. Check if all dependencies are installed: pip install -r requirements.txt")
        print("  3. Verify the ticker symbol is valid")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

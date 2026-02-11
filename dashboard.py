"""
Streamlit Dashboard for Stock Price Prediction
Interactive dashboard to display model predictions and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path

# Import pipeline modules
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
from main_pipeline import StockPricePredictionPipeline

# Page config
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("## ⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", max_chars=5).upper()
lookback_days = st.sidebar.slider("Historical Data (years)", 1, 10, 5)
future_days = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 5)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Main content
st.markdown("<div class='main-header'>📊 Stock Price Prediction Dashboard</div>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Dashboard",
    "🔬 Data Analysis",
    "🤖 Model Training",
    "📊 Results",
    "💡 Predictions"
])

# TAB 1: Dashboard
with tab1:
    st.header("Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Selected Ticker",
            value=ticker,
            delta="Real-time data"
        )
    
    with col2:
        st.metric(
            label="Data Period",
            value=f"{lookback_days} years",
            delta="Historical"
        )
    
    with col3:
        st.metric(
            label="Prediction Horizon",
            value=f"{future_days} days",
            delta="Ahead"
        )
    
    with col4:
        st.metric(
            label="Status",
            value="Ready",
            delta="✓"
        )
    
    st.divider()
    
    # Data preview
    if st.session_state.data_loaded and st.session_state.pipeline is not None:
        st.subheader("Recent Price Data")
        
        pipeline = st.session_state.pipeline
        recent_data = pipeline.features.tail(10)[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=recent_data['date'],
            open=recent_data['open'],
            high=recent_data['high'],
            low=recent_data['low'],
            close=recent_data['close']
        )])
        
        fig.update_layout(
            title=f"{ticker} - Recent Price Action",
            yaxis_title="Stock Price",
            xaxis_title="Date",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Data Analysis
with tab2:
    st.header("Data Analysis & Feature Engineering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Load Data", key="load_data"):
            with st.spinner(f"Loading {ticker} data..."):
                try:
                    pipeline = StockPricePredictionPipeline(
                        ticker=ticker,
                        start_date=(datetime.now() - timedelta(days=lookback_days*365)).strftime('%Y-%m-%d')
                    )
                    pipeline.load_data()
                    
                    if pipeline.data is not None:
                        st.session_state.pipeline = pipeline
                        st.session_state.data_loaded = True
                        st.success(f"✓ Loaded {len(pipeline.data)} records for {ticker}")
                    else:
                        st.error(f"Failed to load data for {ticker}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🔧 Engineer Features", key="engineer_features"):
            if st.session_state.data_loaded:
                with st.spinner("Engineering features..."):
                    try:
                        pipeline = st.session_state.pipeline
                        pipeline.engineer_features(future_days=future_days)
                        st.success(f"✓ Created {len(pipeline.features.columns)} features")
                        st.session_state.data_loaded = True
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please load data first!")
    
    if st.session_state.data_loaded and st.session_state.pipeline is not None:
        pipeline = st.session_state.pipeline
        
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(pipeline.features))
        with col2:
            st.metric("Total Features", len(pipeline.features.columns))
        with col3:
            st.metric("Date Range", f"{pipeline.features['date'].min().date()} to {pipeline.features['date'].max().date()}")
        
        # Feature list
        st.subheader("Engineered Features")
        
        feature_categories = {
            "Price Features": ["close", "open", "high", "low", "co_ratio", "hl_ratio"],
            "Moving Averages": [col for col in pipeline.features.columns if 'sma_' in col or 'ema_' in col],
            "Momentum": [col for col in pipeline.features.columns if 'rsi_' in col or 'macd' in col or 'stoch_' in col],
            "Volatility": [col for col in pipeline.features.columns if 'bb_' in col or 'atr' in col],
            "Volume": [col for col in pipeline.features.columns if 'obv' in col or 'volume' in col],
            "Time-Based": [col for col in pipeline.features.columns if 'day_' in col or 'month' in col]
        }
        
        for category, features in feature_categories.items():
            if features:
                with st.expander(f"{category} ({len(features)} features)"):
                    st.write(", ".join(features[:10]))
                    if len(features) > 10:
                        st.write(f"... and {len(features) - 10} more")
        
        # Data visualization
        st.subheader("Price History")
        
        fig = px.line(
            pipeline.features,
            x='date',
            y='close',
            title=f"{ticker} - Closing Price History",
            labels={'close': 'Price ($)', 'date': 'Date'}
        )
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Model Training
with tab3:
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load and engineer features in the 'Data Analysis' tab first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Preparation")
            st.write("""
            Click the button below to:
            1. Split data into train/validation/test sets
            2. Scale features using StandardScaler
            3. Prepare for model training
            """)
            
            if st.button("📊 Prepare Data", key="prepare_data"):
                with st.spinner("Preparing data..."):
                    try:
                        pipeline = st.session_state.pipeline
                        pipeline.prepare_data(test_size=0.2, val_size=0.1)
                        st.success("✓ Data prepared successfully!")
                        
                        col1_inner, col2_inner, col3_inner = st.columns(3)
                        with col1_inner:
                            st.metric("Training Samples", len(pipeline.X_train))
                        with col2_inner:
                            st.metric("Validation Samples", len(pipeline.X_val))
                        with col3_inner:
                            st.metric("Test Samples", len(pipeline.X_test))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            st.subheader("Model Selection")
            
            models_to_train = st.multiselect(
                "Select models to train:",
                ["Linear Regression", "Random Forest", "XGBoost", "LightGBM", "LSTM"],
                default=["Random Forest", "XGBoost", "LightGBM"]
            )
            
            if st.button("🚀 Train Models", key="train_models"):
                if len(models_to_train) == 0:
                    st.warning("Please select at least one model!")
                else:
                    with st.spinner("Training models... This may take a few minutes."):
                        try:
                            pipeline = st.session_state.pipeline
                            
                            # Train selected models
                            progress_bar = st.progress(0)
                            for idx, model_name in enumerate(models_to_train):
                                st.write(f"Training {model_name}...")
                                
                                if model_name == "Linear Regression":
                                    pipeline.models[model_name] = LinearRegressionModel()
                                elif model_name == "Random Forest":
                                    pipeline.models[model_name] = RandomForestModel()
                                elif model_name == "XGBoost":
                                    pipeline.models[model_name] = XGBoostModel()
                                elif model_name == "LightGBM":
                                    pipeline.models[model_name] = LightGBMModel()
                                elif model_name == "LSTM":
                                    pipeline.models[model_name] = LSTMModel()
                                
                                pipeline.models[model_name].fit(
                                    pipeline.X_train,
                                    pipeline.y_train
                                )
                                
                                progress_bar.progress((idx + 1) / len(models_to_train))
                            
                            st.session_state.models_trained = True
                            st.success("✓ All models trained successfully!")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

# TAB 4: Results
with tab4:
    st.header("📊 Model Evaluation Results")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Model Training' tab!")
    else:
        pipeline = st.session_state.pipeline
        
        # Evaluate models
        if st.button("📈 Evaluate Models", key="evaluate"):
            with st.spinner("Evaluating models..."):
                try:
                    pipeline.evaluate_models()
                    st.success("✓ Models evaluated!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if pipeline.results:
            # Create comparison table
            results_df = pd.DataFrame(pipeline.results).T
            results_df = results_df.round(6)
            
            st.subheader("Performance Metrics")
            st.dataframe(results_df, use_container_width=True)
            
            # Model comparison charts
            col1, col2, col3 = st.columns(3)
            
            models_list = list(pipeline.results.keys())
            rmse_vals = [pipeline.results[m]['RMSE'] for m in models_list]
            mae_vals = [pipeline.results[m]['MAE'] for m in models_list]
            r2_vals = [pipeline.results[m]['R2'] for m in models_list]
            
            with col1:
                fig = px.bar(
                    x=models_list,
                    y=rmse_vals,
                    title="RMSE Comparison",
                    labels={'y': 'RMSE', 'x': 'Model'},
                    color=rmse_vals,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=models_list,
                    y=mae_vals,
                    title="MAE Comparison",
                    labels={'y': 'MAE', 'x': 'Model'},
                    color=mae_vals,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = px.bar(
                    x=models_list,
                    y=r2_vals,
                    title="R² Comparison",
                    labels={'y': 'R²', 'x': 'Model'},
                    color=r2_vals,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

# TAB 5: Predictions
with tab5:
    st.header("💡 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first!")
    else:
        pipeline = st.session_state.pipeline
        
        st.subheader("Prediction Results")
        
        # Get predictions
        try:
            predictions = {}
            
            for model_name, model in pipeline.models.items():
                try:
                    pred = model.predict(pipeline.X_test)
                    
                    if len(pred) < len(pipeline.y_test):
                        y_test_adj = pipeline.y_test.iloc[-len(pred):].values
                    else:
                        y_test_adj = pipeline.y_test.values
                        pred = pred[:len(y_test_adj)]
                    
                    predictions[model_name] = pred
                except:
                    pass
            
            if predictions:
                # Plot predictions
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    y=pipeline.y_test.values[:len(list(predictions.values())[0])],
                    name='Actual',
                    mode='lines+markers',
                    line=dict(color='black', width=2)
                ))
                
                for model_name, pred in predictions.items():
                    fig.add_trace(go.Scatter(
                        y=pred,
                        name=model_name,
                        mode='lines',
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    title=f"{ticker} - Predicted vs Actual Returns (Test Set)",
                    xaxis_title="Days",
                    yaxis_title="Return",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction table
                st.subheader("Latest Predictions")
                
                latest_predictions = {}
                for model_name, pred in predictions.items():
                    latest_predictions[model_name] = [f"{p:.6f}" for p in pred[-10:]]
                
                pred_df = pd.DataFrame(latest_predictions)
                st.dataframe(pred_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px; margin-top: 20px;'>
        📊 Stock Price Prediction Dashboard | Powered by Machine Learning
        <br>
        Models: Linear Regression | Random Forest | XGBoost | LightGBM | LSTM
        <br>
        Last updated: {}</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

if __name__ == "__main__":
    st.write("Dashboard is running...")

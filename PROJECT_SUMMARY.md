# 📈 Stock Price Prediction – End-to-End ML Project

An end-to-end machine learning system that predicts stock price movements using historical data and technical indicators.

Built to practice **time-series ML, feature engineering, and model comparison** in a real-world style pipeline.

> This is for learning & experimentation — not financial advice.

## ✨ Highlights

* Full pipeline → data → features → training → evaluation
* 50+ technical indicators
* Multiple models + ensemble
* Proper time-series validation (no leakage)
* Interactive Streamlit dashboard
* Modular, easy to extend

---

## 🤖 Models

* Linear Regression
* Random Forest
* XGBoost
* LightGBM
* LSTM
* Ensemble average

---

## 🚀 Run the project

### Install

```bash
pip install -r requirements.txt
```

### Run pipeline

```bash
python quickstart.py
```

### Launch dashboard

```bash
streamlit run dashboard.py
```

---

## 📊 What you’ll see

* Model performance comparison
* Prediction vs actual plots
* Feature importance
* Future price estimates

---

## 🧰 Tech Used

Python • Pandas • NumPy • Scikit-learn • XGBoost • LightGBM • TensorFlow/Keras • yfinance • Streamlit

---

## 📁 Structure

```
data_loader.py
feature_engineering.py
model_training.py
main_pipeline.py
dashboard.py
```

---

## ⚠️ Disclaimer

Markets are unpredictable.
Use this project to **learn ML**, not to trade real money.


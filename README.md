# Stock-Price-Prediction-using-Machine-Learning
A beginner-friendly end-to-end ML project that predicts stock prices using historical market data and technical indicators.

## 🚀 What this project does

* Downloads stock data from Yahoo Finance
* Creates 50+ technical indicators
* Trains multiple ML & DL models
* Compares performance
* Shows results in a Streamlit dashboard

---

## 🤖 Models used

* Linear Regression
* Random Forest
* XGBoost
* LightGBM
* LSTM
* Ensemble (average of all)

---

## ⚙️ Run locally

```bash
pip install -r requirements.txt
python mainpipeline.py
```

For the dashboard:

```bash
streamlit run dashboard.py
```

---

## 📊 Evaluation

Models are tested using:

* RMSE
* MAE
* R²

with proper **time-series split** to avoid data leakage.

---

## 🧰 Tech Stack

Python • Pandas • NumPy • Scikit-learn • XGBoost • LightGBM • TensorFlow • Streamlit

---

## ⚠️ Disclaimer

This project is for **learning purposes only**.
Markets are unpredictable — don’t use this for financial decisions.


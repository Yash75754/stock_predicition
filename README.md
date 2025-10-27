
Project Title

Tesla Stock Price Prediction using LSTM Neural Network

🎯 Objective

To predict the future closing price of Tesla (TSLA) stock using historical price data and visualize the results using a Streamlit web app.

📁 Stock_Price_Prediction/
│
├── app.py                 # Streamlit web app
├── saved_model.h5         # Trained LSTM model
├── TSLA.csv               # Dataset with Tesla stock prices
├── stock_predicition.ipynb# Jupyter notebook for data prep & model training
└── requirements.txt       # Dependencies (optional for GitHub)


           ┌────────────────────┐
           │    TSLA.csv        │
           │ (Historical Data)  │
           └────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Data Preprocessing   │
         │ • Sort by Date       │
         │ • Scale using MinMax │
         │ • Split Train/Test   │
         └────────┬─────────────┘
                  │
                  ▼
       ┌────────────────────────┐
       │ Sequence Generation     │
       │ • 60-day lookback       │
       │ • Create (X, y) pairs   │
       └────────┬───────────────┘
                │
                ▼
     ┌───────────────────────────┐
     │ LSTM Model (saved_model) │
     │ Predicts next closing     │
     └────────┬─────────────────┘
              │
              ▼
     ┌───────────────────────────┐
     │ Inverse Scaling           │
     │ • Transform back to $     │
     └────────┬─────────────────┘
              │
              ▼
     ┌───────────────────────────┐
     │ Streamlit Visualization   │
     │ • Actual vs Predicted     │
     │ • Interactive Plotly      │
     └───────────────────────────┘


📊 Detailed Step-by-Step Explanation
1️⃣ Dataset (TSLA.csv)

Contains historical Tesla stock data.
Key columns:

Date

Open, High, Low, Close, Volume

Only ‘Close’ prices are used for prediction.

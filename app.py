# app.py (corrected)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model   # use tensorflow.keras
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="Time Series Prediction", layout="wide")

# ---------- Load CSV ----------
df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)

# keep only Close column for modeling
new_dataset = df[['Close']].copy()

# ---------- Train / Validation split ----------
# (you used 987 previously; keep same split)
train_len = 987
if len(new_dataset) <= train_len + 60:
    st.error("Not enough data. Need at least train_len + 60 samples.")
    st.stop()

train_data = new_dataset.iloc[:train_len].copy()
valid_data = new_dataset.iloc[train_len:].copy()

# ---------- Scale training data ----------
scaler = MinMaxScaler(feature_range=(0, 1))
# fit scaler on training 'Close' values (shape: (n_samples, 1))
scaled_train_data = scaler.fit_transform(train_data.values)

# ---------- Create x_train and y_train ----------
x_train_data = []
y_train_data = []

# Create sequences of length 60
seq_length = 60
for i in range(seq_length, len(scaled_train_data)):
    x_train_data.append(scaled_train_data[i - seq_length:i, 0])
    y_train_data.append(scaled_train_data[i, 0])

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# (Optional) you could train here, but we assume you load a pretrained model
# ---------- Load model ----------
model_path = 'saved_model.h5'
if not os.path.exists(model_path):
    st.error(f"Model file not found at '{model_path}'. Place your saved_model.h5 in the project folder.")
    st.stop()

model_new = load_model(model_path)
model_new.summary()  # optional: prints model summary to server console

# ---------- Prepare test inputs ----------
# We need last (len(valid_data) + seq_length) values to build sequences for test
inputs = new_dataset.iloc[len(new_dataset) - len(valid_data) - seq_length:].values
inputs = inputs.reshape(-1, 1)
# IMPORTANT: use the same scaler fitted on training data
inputs = scaler.transform(inputs)

X_test = []
for i in range(seq_length, inputs.shape[0]):
    X_test.append(inputs[i - seq_length:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# ---------- Predict and inverse transform ----------
predicted = model_new.predict(X_test)
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))
  # shape (n,1)

# Align predictions with validation index
preds_index = valid_data.index[:len(predicted)]  # ensure same length
predicted = model_new.predict(X_test)

# if model outputs 3D array, take only the last timestep prediction
if predicted.ndim == 3:
    predicted = predicted[:, -1, :]

# reshape to 2D before inverse transform
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))


# ---------- Visualization ----------
valid_plot = valid_data.copy()
valid_plot = valid_plot.iloc[:len(predicted)].copy()  # match lengths
# ---------- Predict and inverse transform ----------
predicted = model_new.predict(X_test)

# Handle 3D output (many-to-many model)
if predicted.ndim == 3:
    predicted = predicted[:, -1, :]  # Take last timestep of each prediction

# Reshape to 2D before inverse transform
predicted = predicted.reshape(-1, 1)
predicted = scaler.inverse_transform(predicted)

# Ensure prediction length does not exceed validation data
n_preds = min(len(predicted), len(valid_data))
predicted = predicted[:n_preds]

# Align index properly
preds_index = valid_data.index[:n_preds]

# Create predictions series
preds_series = pd.Series(predicted.flatten(), index=preds_index, name='Predictions')

# Prepare plot DataFrame
valid_plot = valid_data.copy().iloc[:n_preds].copy()
valid_plot['Predictions'] = preds_series.values


st.subheader("Actual vs Predicted Closing Prices")
fig = go.Figure()

# plot training data
fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'],
                         mode='lines', name='Training Data'))

# plot actual closing prices (validation)
fig.add_trace(go.Scatter(x=valid_plot.index, y=valid_plot['Close'],
                         mode='lines', name='Actual Closing Price'))

# plot predicted closing prices
fig.add_trace(go.Scatter(x=valid_plot.index, y=valid_plot['Predictions'],
                         mode='lines', name='Predicted Closing Price'))

fig.update_layout(title='Tesla Closing Price Prediction', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig, use_container_width=True)

st.write("-----")
st.write("Model: LSTM")
st.write("Note: The predictions are based on historical data and may not reflect future market conditions.")

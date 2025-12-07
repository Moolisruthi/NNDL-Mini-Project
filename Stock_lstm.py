import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

import yfinance as yf

# Step 1: Download Stock Data

data = yf.download(
    'AAPL',
    start='2018-01-01',
    end='2023-12-31',
    auto_adjust=False,      
    progress=False
)

data = data[['Close']].copy()
print("Data head:\n", data.head())

# Step 2: Data Preprocessing

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

look_back = 60  

X, y = [], []

for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Step 3: Build LSTM Model


model = Sequential()

model.add(Input(shape=(look_back, 1)))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(1))  
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Step 4: Train the Model

history = model.fit(
    X_train,
    y_train,
    epochs=50,          
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Step 5: Make Predictions

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 6: Visualize Results

plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Stock Price')
plt.plot(predicted_prices, label='Predicted Stock Price')
plt.title('Apple Stock Price Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()

plt.show()
plt.savefig("stock_price_prediction.png")

# Step 7: Predict Next Day Price

last_60_days = scaled_data[-look_back:]
X_future = np.array([last_60_days])
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

next_day_price_scaled = model.predict(X_future)
next_day_price = scaler.inverse_transform(next_day_price_scaled)

print(f"Predicted next day closing price: ${next_day_price[0][0]:.2f}")

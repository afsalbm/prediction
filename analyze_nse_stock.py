import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def analyze_nse_stock(stock_symbol, start_date, end_date):
    """Analyzes an NSE stock using LSTM model for time series prediction.

    Args:
        stock_symbol: The NSE stock symbol to analyze.
        start_date: The start date for data retrieval (YYYY-MM-DD format).
        end_date: The end date for data retrieval (YYYY-MM-DD format).
    """

    # Fetch historical data
    ticker = yf.Ticker(f"{stock_symbol}.NS")
    data = ticker.history(start=start_date, end=end_date)

    # Preprocess data
    data = data[['Close']]
    data = data.dropna()  # Handle missing values
    data.index = pd.to_datetime(data.index)  # Convert index to datetime

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Create input sequences
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    sequence_length = 60
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Make predictions on the test data
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Predict future prices for the next 30 days
    future_predictions = []
    last_sequence = test_data[-sequence_length:]  # Last sequence from test data
    current_sequence = np.expand_dims(last_sequence, axis=0)  # Reshape for model input

    for _ in range(30):  # Predict next 30 days
        next_pred = model.predict(current_sequence)
        future_predictions.append(next_pred[0][0])

        # Reshape next_pred to match the sequence dimensions (1, 1, 1)
        next_pred = np.expand_dims(next_pred, axis=-1)

        # Update the current sequence by appending next_pred and removing the oldest value
        current_sequence = np.append(current_sequence[:, 1:, :], next_pred, axis=1)

    # Inverse transform future predictions to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Prepare dates for future predictions (one day after the last date in the original data)
    last_date = data.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

    # Ensure predicted_prices and sliced data index have the same length
    plt.figure(figsize=(12, 6))

    # Adjust index and actual prices to match predicted prices
    plt.plot(data.index[train_size + sequence_length:], data.iloc[train_size + sequence_length:].values, label='Actual')
    plt.plot(data.index[train_size + sequence_length:], predicted_prices, label='Predicted')

    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Prediction (Next 30 days)', linestyle='dashed')

    plt.title(f"{stock_symbol} Stock Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    stock_symbol = "HINDCOPPER"  # Replace with your desired stock symbol
    start_date = "2020-01-01"
    end_date = "2024-09-06"
    analyze_nse_stock(stock_symbol, start_date, end_date)

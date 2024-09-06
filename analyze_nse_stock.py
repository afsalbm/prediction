import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Set seeds for reproducibility
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)


def add_technical_indicators(data):
    """Add technical indicators to the dataframe."""
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'])
    return data


def compute_rsi(series, period=14):
    """Calculate the Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def analyze_nse_stock(stock_symbol, start_date, end_date):
    """Analyzes an NSE stock using LSTM model for time series prediction."""

    # Fetch historical data
    ticker = yf.Ticker(f"{stock_symbol}.NS")
    data = ticker.history(start=start_date, end=end_date)

    # Add technical indicators
    data = add_technical_indicators(data)

    # Drop rows with NaN values
    data = data.dropna()

    # Preprocess data (including more features)
    features = ['Close', 'MA20', 'MA50', 'RSI', 'Volume', 'Open', 'High', 'Low']
    data = data[features]
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
            y.append(data[i + sequence_length, 0])  # Target is 'Close' price
        return np.array(X), np.array(y)

    sequence_length = 60
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Build LSTM model with additional complexity
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.1)

    # Make predictions on the test data
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(
        np.hstack((predicted_prices, np.zeros((predicted_prices.shape[0], scaled_data.shape[1] - 1)))))

    # Calculate RMSE
    y_test_original = scaler.inverse_transform(
        np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_data.shape[1] - 1)))))
    rmse = np.sqrt(mean_squared_error(y_test_original[:, 0], predicted_prices[:, 0]))
    print(f'Root Mean Squared Error: {rmse}')

    # Predict future prices for the next 30 days
    future_predictions = []
    last_sequence = test_data[-sequence_length:]  # Last sequence from test data
    current_sequence = np.expand_dims(last_sequence, axis=0)  # Reshape for model input

    for _ in range(30):  # Predict next 30 days
        next_pred = model.predict(current_sequence)
        future_predictions.append(next_pred[0, 0])

        # Update the current sequence
        next_pred_reshaped = np.array([[next_pred[0, 0]] * current_sequence.shape[2]])  # Shape to (1, 1, features)
        next_pred_reshaped = np.expand_dims(next_pred_reshaped, axis=0)  # Ensure it has the correct shape
        current_sequence = np.concatenate((current_sequence[:, 1:, :], next_pred_reshaped), axis=1)

    # Inverse transform future predictions to original scale
    future_predictions_array = np.array(future_predictions).reshape(-1, 1)
    future_predictions_scaled = np.hstack(
        (future_predictions_array, np.zeros((len(future_predictions_array), scaled_data.shape[1] - 1))))
    future_predictions_original = scaler.inverse_transform(future_predictions_scaled)

    # Prepare dates for future predictions (one day after the last date in the original data)
    last_date = data.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[train_size + sequence_length:], data['Close'].iloc[train_size + sequence_length:],
             label='Actual')
    plt.plot(data.index[train_size + sequence_length:], predicted_prices[:, 0], label='Predicted')
    plt.plot(future_dates, future_predictions_original[:, 0], label='Future Prediction (Next 30 days)',
             linestyle='dashed')
    plt.title(f"{stock_symbol} Stock Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    stock_symbol = "BPCL"  # Replace with your desired stock symbol
    start_date = "2020-01-01"
    end_date = "2024-09-06"
    analyze_nse_stock(stock_symbol, start_date, end_date)

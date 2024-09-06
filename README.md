# NSE Stock Price Prediction with LSTM

This Python script analyzes and predicts NSE stock prices using an LSTM (Long Short-Term Memory) model. It fetches historical stock data from Yahoo Finance, processes the data, trains the LSTM model, and predicts future stock prices. The results are visualized using Matplotlib.

## Features

- Fetches historical stock price data for a given NSE stock symbol from Yahoo Finance.
- Preprocesses data and scales it for model training.
- Trains an LSTM model to predict stock prices.
- Visualizes the actual prices, model predictions, and future predictions for the next 30 days.

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/afsalbm/prediction.git
    cd prediction
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Required Packages:**

    Make sure you're in the virtual environment and run:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Script:**

    Edit the `stock_symbol`, `start_date`, and `end_date` variables in `app.py` as needed, then run the script:

    ```bash
    python analyze_nse_stock.py
    ```

    The script will generate a plot showing the historical stock prices, model predictions, and future price forecasts for the next 30 days.

## How It Works

1. **Data Fetching:**
   - Historical stock price data is fetched from Yahoo Finance using the `yfinance` library.

2. **Data Preprocessing:**
   - The data is preprocessed and scaled using `MinMaxScaler` from `sklearn`.

3. **Model Training:**
   - An LSTM model is built and trained using TensorFlow/Keras on the historical data.

4. **Prediction and Visualization:**
   - The model predicts future stock prices for the next 30 days.
   - The actual prices, predictions, and future forecasts are visualized using Matplotlib.

## Usage

- **Stock Symbol**: Replace `"HINDCOPPER"` in the script with your desired NSE stock symbol.
- **Date Range**: Modify the `start_date` and `end_date` as needed.

## Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`
- `scikit-learn`
- `tensorflow`

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Yahoo Finance](https://www.yahoofinance.com) for stock data.
- [TensorFlow](https://www.tensorflow.org) for machine learning.
- [Matplotlib](https://matplotlib.org) for visualization.
- [scikit-learn](https://scikit-learn.org) for data preprocessing.


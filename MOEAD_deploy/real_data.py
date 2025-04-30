
import yfinance as yf
import numpy as np
import pandas as pd

def fetch_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Use 'Close' instead of 'Adj Close' with auto_adjust=True
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close']
    else:
        close_prices = data[['Close']].copy()
        close_prices.columns = tickers  # Rename for consistency

    close_prices = close_prices.dropna(axis=1)  # Remove tickers with missing data
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    expected_returns = log_returns.mean().values * 252
    cov_matrix = log_returns.cov().values * 252
    valid_stocks = close_prices.columns.tolist()
    return expected_returns, cov_matrix, valid_stocks

def get_esg_scores(ticker):
    try:
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability.loc['totalEsg']
        if esg_data is not None and not esg_data.empty:
            return esg_data.iloc[0]
        else:
            return 0
    except Exception:
        return 0

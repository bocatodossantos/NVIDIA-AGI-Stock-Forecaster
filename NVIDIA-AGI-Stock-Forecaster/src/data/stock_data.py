
"""
Stock data fetching and processing utilities for NVIDIA stock analysis.

This module provides functions to fetch, cache, and process NVIDIA stock data.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)


def fetch_stock_data(
    ticker: str,
    start_date: Union[str, datetime],
    end_date: Optional[Union[str, datetime]] = None,
    use_cache: bool = True,
    cache_dir: str = 'data/raw'
) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance with caching support.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'NVDA')
        start_date (str or datetime): Start date for data
        end_date (str or datetime, optional): End date for data.
            If None, uses current date.
        use_cache (bool): Whether to use cached data if available
        cache_dir (str): Directory to store cached data
        
    Returns:
        pd.DataFrame: DataFrame with stock data
    """
    # Convert dates to datetime objects if they are strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Format dates for filename
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    # Create cache filename
    cache_file = os.path.join(cache_dir, f"{ticker}_{start_str}_{end_str}.csv")
    
    # Check if cached data exists and is recent enough
    if use_cache and os.path.exists(cache_file):
        # Check if cache file is newer than end_date or less than 1 day old
        cache_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        cache_age = datetime.now() - cache_mod_time
        
        if end_date < datetime.now() or cache_age.days < 1:
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            logger.info("Cache is outdated, fetching new data")
    
    # Fetch data from Yahoo Finance
    logger.info(f"Fetching {ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)
        
        if len(data) == 0:
            raise ValueError(f"No data found for {ticker} in specified date range")
        
        logger.info(f"Retrieved {len(data)} days of stock data")
        
        # Save to cache
        data.to_csv(cache_file)
        logger.info(f"Saved data to cache: {cache_file}")
        
        return data
    
    except Exception as e:
        logger.error(f"Error fetching stock data: {str(e)}")
        
        # If we encounter an error but cache exists, use it as fallback
        if os.path.exists(cache_file):
            logger.warning(f"Using cached data as fallback")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        raise


def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame with OHLCV columns
        
    Returns:
        pd.DataFrame: DataFrame with additional technical indicators
    """
    logger.info("Calculating technical indicators")
    
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    
    # Calculate RSI
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility (20-day standard deviation of returns, annualized)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    logger.info("Completed technical indicator calculations")
    return df


def preprocess_features(data: pd.DataFrame, 
                        lag_periods: list = [1, 5, 10, 21], 
                        momentum_periods: list = [5, 21, 63]) -> pd.DataFrame:
    """
    Preprocess stock data for machine learning by adding lag and momentum features.
    
    Args:
        data (pd.DataFrame): Stock data DataFrame with technical indicators
        lag_periods (list): Periods for lagged features
        momentum_periods (list): Periods for momentum calculations
        
    Returns:
        pd.DataFrame: DataFrame with features ready for ML
    """
    logger.info("Preprocessing features for machine learning")
    
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Add lag features
    for lag in lag_periods:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
    
    # Add momentum features (percent change over different periods)
    for period in momentum_periods:
        df[f'momentum_{period}'] = df['Close'].pct_change(period)
    
    # Add day of week and month features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Return on previous day
    df['return_1d'] = df['Close'].pct_change()
    
    # Target variable - next 30 day return (can be adjusted based on forecast horizon)
    df['target_30d_return'] = df['Close'].shift(-30) / df['Close'] - 1
    
    # Drop NaN values
    df = df.dropna()
    
    logger.info(f"Preprocessed features shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, 
               target_col: str = 'target_30d_return', 
               test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Feature DataFrame
        target_col (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Calculate split point (using time-based split)
    split_idx = int(len(df) * (1 - test_size))
    
    # Split the data
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    logger.info(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    ticker = 'NVDA'
    start_date = datetime.now() - timedelta(days=365*5)  # 5 years of data
    
    # Fetch data
    stock_data = fetch_stock_data(ticker, start_date)
    
    # Calculate indicators
    data_with_indicators = calculate_technical_indicators(stock_data)
    
    # Print sample
    print(data_with_indicators.tail())
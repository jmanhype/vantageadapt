"""Market-specific benchmarks for strategy comparison."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import vectorbtpro as vbt
import math


def handle_infinity(value: float) -> float:
    """Handle infinity values for JSON serialization.
    
    Args:
        value: Value to check
        
    Returns:
        Cleaned value
    """
    if math.isinf(value):
        if value > 0:
            return 1e308  # Max float value
        else:
            return -1e308
    return value


def calculate_market_benchmarks(
    price_data: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10.0
) -> Dict[str, Any]:
    """Calculate market benchmark metrics for comparison.
    
    Args:
        price_data: DataFrame containing price data
        start_date: Optional start date for analysis
        end_date: Optional end date for analysis
        initial_capital: Initial capital for calculations
        
    Returns:
        Dictionary containing benchmark metrics
    """
    # Filter data by date range if provided
    if start_date and end_date:
        mask = (price_data.index >= start_date) & (price_data.index <= end_date)
        price_data = price_data[mask]
    
    # Handle different price column names and data types
    if isinstance(price_data, pd.Series):
        price_series = price_data
    elif isinstance(price_data, np.ndarray):
        if len(price_data.shape) > 1:
            price_series = pd.Series(price_data[:, 0])  # Take first column
        else:
            price_series = pd.Series(price_data)
    elif hasattr(price_data, 'vbt'):  # VectorBT wrapper
        if hasattr(price_data, 'close'):
            price_series = price_data.close
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
            elif isinstance(price_series, np.ndarray):
                if len(price_series.shape) > 1:
                    price_series = pd.Series(price_series[:, 0])
                else:
                    price_series = pd.Series(price_series)
        else:
            price_series = price_data.values
            if isinstance(price_series, pd.DataFrame):
                price_series = price_series.iloc[:, 0]
            elif isinstance(price_series, np.ndarray):
                if len(price_series.shape) > 1:
                    price_series = pd.Series(price_series[:, 0])
                else:
                    price_series = pd.Series(price_series)
    else:
        if 'price' in price_data.columns:
            price_series = price_data['price']
        elif 'close' in price_data.columns:
            price_series = price_data['close']
        elif len(price_data.columns) == 1:
            price_series = price_data.iloc[:, 0]
        else:
            raise ValueError("Could not identify price column in data")
    
    # Convert to Series if DataFrame
    if isinstance(price_series, pd.DataFrame):
        if price_series.shape[1] == 1:
            price_series = price_series.iloc[:, 0]
        else:
            price_series = price_series.mean(axis=1)  # Use mean if multiple columns
    elif isinstance(price_series, np.ndarray):
        if len(price_series.shape) > 1:
            price_series = pd.Series(price_series[:, 0])
        else:
            price_series = pd.Series(price_series)
    
    # Calculate buy & hold returns
    buy_hold_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
    
    # Calculate daily returns
    daily_returns = price_series.pct_change().dropna()
    
    # Calculate volatility
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
    excess_returns = daily_returns - 0.02/252
    returns_std = daily_returns.std()
    if returns_std != 0:
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns_std
    else:
        sharpe_ratio = 0.0
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    return {
        'buy_hold_return': float(handle_infinity(buy_hold_return)),
        'annualized_volatility': float(handle_infinity(volatility)),
        'sharpe_ratio': float(handle_infinity(sharpe_ratio)),
        'max_drawdown': float(handle_infinity(max_drawdown)),
        'initial_price': float(handle_infinity(price_series.iloc[0])),
        'final_price': float(handle_infinity(price_series.iloc[-1])),
        'trading_days': len(price_series),
        'benchmark_capital': float(handle_infinity(initial_capital * (1 + buy_hold_return)))
    } 
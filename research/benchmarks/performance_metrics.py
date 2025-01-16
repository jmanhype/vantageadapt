"""Advanced performance metrics calculation module."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


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


def calculate_sortino_ratio(returns: pd.Series, trades: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio using trade-level returns.
    
    Args:
        returns: Series of returns
        trades: DataFrame of trade records
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sortino ratio value
    """
    # Use trade returns for Sortino calculation
    if 'return' in trades.columns:
        trade_returns = trades['return']
    else:
        trade_returns = trades['pnl'] / trades['size']
    
    logger.info(f"Calculating Sortino ratio from {len(trade_returns)} trades:")
    logger.info(f"Trade returns mean: {trade_returns.mean():.6f}")
    logger.info(f"Trade returns std: {trade_returns.std():.6f}")
    
    # Calculate excess returns over risk-free rate
    excess_returns = trade_returns - risk_free_rate/252
    logger.info(f"Excess returns mean: {excess_returns.mean():.6f}")
    
    # Calculate downside deviation
    downside_returns = trade_returns[trade_returns < 0]
    logger.info(f"Number of losing trades: {len(downside_returns)}")
    
    if len(downside_returns) > 0:
        # Use squared negative returns
        downside_std = np.sqrt((downside_returns**2).mean())
        logger.info(f"Downside std: {downside_std:.6f}")
        
        # Calculate annualized Sortino ratio
        sortino = np.sqrt(252) * excess_returns.mean() / downside_std
        logger.info(f"Calculated Sortino ratio: {sortino:.6f}")
        return float(sortino)
    else:
        logger.info("No losing trades, using maximum Sortino ratio")
        return 1e308  # Maximum value for no downside risk


def calculate_sharpe_ratio(returns: pd.Series, trades: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio using trade-level returns.
    
    Args:
        returns: Series of returns
        trades: DataFrame of trade records
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio value
    """
    # Use trade returns for Sharpe calculation
    if 'return' in trades.columns:
        trade_returns = trades['return']
    else:
        trade_returns = trades['pnl'] / trades['size']
    
    logger.info(f"\nCalculating Sharpe ratio from {len(trade_returns)} trades:")
    logger.info(f"Trade returns mean: {trade_returns.mean():.6f}")
    logger.info(f"Trade returns std: {trade_returns.std():.6f}")
    
    # Calculate excess returns over risk-free rate
    excess_returns = trade_returns - risk_free_rate/252
    logger.info(f"Excess returns mean: {excess_returns.mean():.6f}")
    logger.info(f"Excess returns std: {excess_returns.std():.6f}")
    
    # Calculate annualized Sharpe ratio
    returns_std = trade_returns.std()
    if returns_std != 0:
        sharpe = np.sqrt(252) * excess_returns.mean() / returns_std
        logger.info(f"Annualized Sharpe ratio: {sharpe:.6f}")
        return float(sharpe)
    else:
        logger.info("Zero standard deviation, defaulting Sharpe ratio to 0")
        return 0.0


def calculate_advanced_metrics(portfolio: Any, trades: pd.DataFrame) -> Dict[str, float]:
    """Calculate advanced performance metrics."""
    metrics = {}
    
    if trades is None or len(trades) == 0:
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'consecutive_losses': 0,
            'avg_trade_duration': 0.0,
            'avg_profit_per_trade': 0.0,
            'sortino_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'unique_trades': 0,
            'total_trade_rows': 0,
            'total_orders': 0
        }
    
    # Log trade data structure
    logger.info("\nAnalyzing trade data:")
    logger.info(f"Trade DataFrame shape: {trades.shape}")
    logger.info(f"Trade DataFrame columns: {trades.columns.tolist()}")
    logger.info(f"Total rows in trades: {len(trades)}")
    
    # Count unique trades and orders
    unique_trades = len(trades['id'].unique()) if 'id' in trades.columns else len(trades)
    total_orders = len(trades['entry_order_id'].unique()) + len(trades['exit_order_id'].unique()) if 'entry_order_id' in trades.columns else 2 * unique_trades
    
    logger.info(f"Unique trade IDs: {unique_trades}")
    logger.info(f"Total orders: {total_orders}")
    
    # Basic trade metrics
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] < 0]
    
    logger.info("\nTrade statistics:")
    logger.info(f"Winning trades: {len(winning_trades)}")
    logger.info(f"Losing trades: {len(losing_trades)}")
    logger.info(f"Total trade rows: {len(trades)}")
    
    # Store trade counts
    metrics['unique_trades'] = unique_trades
    metrics['total_trade_rows'] = len(trades)
    metrics['total_orders'] = total_orders
    
    # Handle total return
    if hasattr(portfolio, 'total_return'):
        total_return = portfolio.total_return
        if isinstance(total_return, (pd.Series, pd.DataFrame)):
            if isinstance(total_return, pd.DataFrame):
                total_return = total_return.iloc[-1, -1]
            else:
                total_return = total_return.iloc[-1]
        metrics['total_return'] = float(handle_infinity(total_return))
    else:
        metrics['total_return'] = 0.0
    
    metrics['win_rate'] = float(handle_infinity(len(winning_trades) / len(trades)))
    
    # Profit metrics
    gross_profits = winning_trades['pnl'].sum()
    gross_losses = abs(losing_trades['pnl'].sum())
    metrics['profit_factor'] = float(handle_infinity(gross_profits / gross_losses if gross_losses != 0 else 1e308))
    metrics['avg_profit_per_trade'] = float(handle_infinity(trades['pnl'].mean()))
    
    # Risk metrics
    if hasattr(portfolio, 'drawdown'):
        drawdown = portfolio.drawdown
        if isinstance(drawdown, (pd.Series, pd.DataFrame)):
            if isinstance(drawdown, pd.DataFrame):
                drawdown = drawdown.max().max()
            else:
                drawdown = drawdown.max()
        metrics['max_drawdown'] = float(handle_infinity(drawdown))
    else:
        metrics['max_drawdown'] = 0.0
    
    # Streak analysis
    trade_results = (trades['pnl'] > 0).astype(int)
    loss_streaks = (trade_results == 0).astype(int).cumsum()
    metrics['consecutive_losses'] = int(loss_streaks.max())
    
    # Time metrics
    if 'duration' in trades.columns:
        metrics['avg_trade_duration'] = float(handle_infinity(trades['duration'].mean()))
    else:
        metrics['avg_trade_duration'] = 0.0
    
    # Risk-adjusted returns
    if hasattr(portfolio, 'returns'):
        returns = portfolio.returns
        if isinstance(returns, (pd.DataFrame)):
            returns = returns.iloc[:, -1]
        
        # Log returns distribution
        logger.info("\nAnalyzing returns distribution:")
        logger.info(f"Total returns points: {len(returns)}")
        logger.info(f"Returns mean: {returns.mean():.6f}")
        logger.info(f"Returns std: {returns.std():.6f}")
        logger.info(f"Returns min: {returns.min():.6f}")
        logger.info(f"Returns max: {returns.max():.6f}")
        logger.info(f"Non-zero returns: {len(returns[returns != 0])}")
        
        # Calculate Sharpe ratio using trade-level returns
        metrics['sharpe_ratio'] = float(handle_infinity(calculate_sharpe_ratio(returns, trades)))
        
        # Calculate Sortino ratio using trade-level returns
        logger.info("\nCalculating strategy Sortino ratio:")
        metrics['sortino_ratio'] = float(handle_infinity(calculate_sortino_ratio(returns, trades)))
        
        # Calculate Calmar ratio
        metrics['calmar_ratio'] = float(handle_infinity(
            -metrics['total_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] != 0 else 1e308
        ))
        
        # Log final metrics for debugging
        logger.info("\nFinal metrics:")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
    
    return metrics


def calculate_rolling_metrics(
    portfolio: Any,
    window: int = 20,
    min_periods: int = 5
) -> Dict[str, pd.Series]:
    """Calculate rolling performance metrics.
    
    Args:
        portfolio: VectorBT portfolio object
        window: Rolling window size in days
        min_periods: Minimum number of observations required
        
    Returns:
        Dictionary of rolling metrics as pandas Series
    """
    if not hasattr(portfolio, 'returns') or portfolio.returns is None:
        return {}
    
    returns = portfolio.returns
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, -1]
    elif not isinstance(returns, pd.Series):
        return {}
        
    rolling_metrics = {}
    
    # Rolling Sharpe Ratio
    rolling_metrics['rolling_sharpe'] = (
        np.sqrt(252) * returns.rolling(window=window, min_periods=min_periods).mean() /
        returns.rolling(window=window, min_periods=min_periods).std()
    ).replace([np.inf, -np.inf], 0)
    
    # Rolling Volatility
    rolling_metrics['rolling_volatility'] = (
        np.sqrt(252) * returns.rolling(window=window, min_periods=min_periods).std()
    ).replace([np.inf, -np.inf], 0)
    
    # Rolling Returns
    rolling_metrics['rolling_returns'] = (
        returns.rolling(window=window, min_periods=min_periods).sum()
    ).replace([np.inf, -np.inf], 0)
    
    # Rolling Drawdown
    rolling_cumulative = (1 + returns).rolling(window=window, min_periods=min_periods).apply(np.prod) - 1
    rolling_max = rolling_cumulative.expanding().max()
    rolling_metrics['rolling_drawdown'] = (
        (rolling_cumulative / rolling_max - 1).replace([np.inf, -np.inf], 0)
    )
    
    return rolling_metrics 
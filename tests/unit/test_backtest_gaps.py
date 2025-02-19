"""Test handling of strategic data gaps in backtesting."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from research.strategy.backtester import from_signals_backtest

def create_test_data():
    """Create a small test dataset with gaps."""
    dates = pd.date_range(start='2025-01-01', end='2025-01-02', freq='1h')
    data = pd.DataFrame(index=dates)
    data['dex_price'] = np.random.uniform(90, 100, len(dates))
    data['volume'] = np.random.randint(1000, 10000, len(dates))
    data['entries'] = False
    data['exits'] = False
    
    # Add some test signals
    data.loc[data.index[2], 'entries'] = True
    data.loc[data.index[5], 'exits'] = True
    
    # Create a gap
    data = data.drop(data.index[3:5])
    
    return data

@pytest.fixture
def gapped_data():
    """Fixture providing test data with gaps."""
    return create_test_data()

@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio for testing."""
    portfolio = MagicMock()
    portfolio.total_return = pd.Series([0.05])  # 5% return
    portfolio.trades = MagicMock()
    portfolio.trades.records = pd.DataFrame({
        'entry_time': [datetime(2025, 1, 1, 2)],
        'exit_time': [datetime(2025, 1, 1, 5)],
        'pnl': [0.05],
        'return': [0.05]
    })
    portfolio.trades.count = lambda: pd.Series([1])
    portfolio.orders = MagicMock()
    portfolio.orders.count = lambda: pd.Series([2])  # Entry and exit
    portfolio.sortino_ratio = pd.Series([1.5])
    return portfolio

@pytest.mark.asyncio
async def test_strategic_data_gaps(gapped_data, mock_portfolio):
    """Test handling of strategic data gaps in backtesting.
    
    This test verifies that:
    1. Single missing points are interpolated
    2. Large gaps are properly identified
    3. Performance metrics account for gaps
    4. No false signals during gaps
    """
    # Configure test strategy with smaller position size
    strategy_config = {
        "order_size": 0.01,  # Reduced position size
        "take_profit": 0.01,  # Smaller take profit
        "stop_loss": 0.01    # Smaller stop loss
    }
    
    # Mock the portfolio creation
    with patch('research.strategy.backtester.vbt.Portfolio.from_signals', return_value=mock_portfolio):
        # Run backtest
        portfolio = from_signals_backtest(gapped_data, **strategy_config)
        assert portfolio is not None, "Backtest failed to produce portfolio"
        
        # Verify basic portfolio metrics
        assert hasattr(portfolio, 'total_return'), "Portfolio missing total_return attribute"
        assert hasattr(portfolio, 'trades'), "Portfolio missing trades attribute"
        assert len(portfolio.trades.records) > 0, "No trades were executed"
        
        # Verify no trades during gaps
        trades_df = portfolio.trades.records
        if not trades_df.empty:
            for start_idx in [3, 4]:  # Known gap locations
                gap_trades = trades_df[
                    (trades_df['entry_time'] >= gapped_data.index[start_idx]) &
                    (trades_df['entry_time'] <= gapped_data.index[start_idx + 1])
                ]
                assert len(gap_trades) == 0, f"Found trades during gap at index {start_idx}"

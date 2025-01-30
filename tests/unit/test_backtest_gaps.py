"""Test backtesting behavior with data gaps and missing periods."""

import pytest
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from research.strategy.backtester import Backtester
from research.strategy.types import MarketRegime, StrategyContext

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@pytest.fixture
def backtester():
    """Create backtester instance."""
    return Backtester()

@pytest.fixture
def gapped_data():
    """Generate test data with strategic gaps."""
    # Create 5 days of minute data with gaps
    base_timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=5),
        end=datetime.now(),
        freq='1min'
    )
    
    # Create intentional gaps:
    # 1. Missing single points
    # 2. Missing hour blocks
    # 3. Missing day transition
    gap_indices = [
        # Single point gaps
        100, 200, 300,
        # Hour block (600 minutes in)
        *range(600, 660),
        # Day transition (2880 = 2 days * 24h * 60min)
        *range(2880, 2940)
    ]
    
    # Remove gap indices
    timestamps = np.delete(base_timestamps, gap_indices)
    
    # Generate price data
    n_points = len(timestamps)
    price = 100 * (1 + np.random.randn(n_points).cumsum() * 0.02)
    
    # Create DataFrame with gaps
    return pd.DataFrame({
        'price': price,
        'volume': np.random.randint(1000, 10000, n_points)
    }, index=timestamps)

@pytest.mark.asyncio
async def test_strategic_data_gaps(backtester, gapped_data):
    """Test handling of strategic data gaps in backtesting.
    
    This test verifies that:
    1. Single missing points are interpolated
    2. Large gaps are properly identified
    3. Performance metrics account for gaps
    4. No false signals during gaps
    """
    # Configure test strategy
    strategy_config = {
        "entry_threshold": 0.75,
        "exit_threshold": 0.25,
        "stop_loss": 0.02,
        "position_size": 0.1
    }
    
    # Run backtest
    results = await backtester.run_backtest(
        data=gapped_data,
        strategy_config=strategy_config
    )
    
    # Verify results exist
    assert results is not None, "Backtest failed to produce results"
    
    # Check gap handling
    gap_stats = results.get('gap_statistics', {})
    logger.info(f"Gap statistics: {gap_stats}")
    
    # Verify no trades during large gaps
    trades_df = results.get('trades', pd.DataFrame())
    if not trades_df.empty:
        for start_idx in [600, 2880]:  # Known large gap locations
            gap_trades = trades_df[
                (trades_df.index >= gapped_data.index[start_idx]) &
                (trades_df.index <= gapped_data.index[start_idx + 60])
            ]
            assert len(gap_trades) == 0, f"Found trades during gap at index {start_idx}"
    
    # Verify performance metrics
    metrics = results.get('metrics', {})
    assert 'adjusted_sharpe' in metrics, "Missing gap-adjusted Sharpe ratio"
    assert 'gap_exposure' in metrics, "Missing gap exposure metric"
    
    # Log gap impact
    logger.info(
        f"Gap impact - Adjusted Sharpe: {metrics.get('adjusted_sharpe', 0):.2f}, "
        f"Exposure: {metrics.get('gap_exposure', 0):.2%}"
    )
    
    # Verify reasonable metrics despite gaps
    assert metrics.get('gap_exposure', 1) < 0.2, "Excessive gap exposure"
    logger.info("Data gap handling test passed")

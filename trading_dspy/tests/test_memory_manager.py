"""Tests for the memory manager module."""

from typing import Dict, Any, List, TYPE_CHECKING
import pytest
import os
from datetime import datetime, timezone
from src.utils.memory_manager import TradingMemoryManager
from src.utils.types import MarketRegime, StrategyContext, BacktestResults
from dotenv import load_dotenv
import time

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

def pytest_configure(config):
    """Configure test environment."""
    load_dotenv()
    if not os.getenv("MEM0_API_KEY"):
        pytest.skip("MEM0_API_KEY not found in environment")

@pytest.fixture(scope="session")
def memory_manager() -> TradingMemoryManager:
    """Create a real memory manager for testing."""
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        pytest.skip("MEM0_API_KEY not found in environment")
        
    manager = TradingMemoryManager(api_key=api_key)
    if not manager.enabled:
        pytest.skip("Memory manager failed to initialize")
        
    return manager

@pytest.fixture
def sample_strategy_context() -> StrategyContext:
    """Create a sample strategy context for testing."""
    return StrategyContext(
        regime=MarketRegime.TRENDING_BULLISH,
        confidence=0.8,
        risk_level="medium",
        parameters={
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "position_size": 0.1,
            "sl_window": 400,
            "max_orders": 3,
            "order_size": 0.0025,
            "post_buy_delay": 2,
            "post_sell_delay": 5,
            "macd_signal_fast": 120,
            "macd_signal_slow": 260,
            "macd_signal_signal": 90,
            "min_macd_signal_threshold": 0,
            "max_macd_signal_threshold": 0,
            "enable_sl_mod": False,
            "enable_tp_mod": False
        }
    )

@pytest.fixture
def sample_backtest_results() -> BacktestResults:
    """Create sample backtest results for testing."""
    return BacktestResults(
        total_return=0.15,
        total_pnl=1500.0,
        sortino_ratio=1.8,
        win_rate=0.65,
        total_trades=100,
        trades=[
            {"entry": 100, "exit": 110, "pnl": 100},
            {"entry": 105, "exit": 115, "pnl": 95}
        ],
        metrics={
            "max_drawdown": -0.05,
            "sharpe_ratio": 2.1,
            "avg_trade_duration": "2h 15m",
            "profit_factor": 1.65
        }
    )

def test_initialization(memory_manager: TradingMemoryManager) -> None:
    """Test memory manager initialization."""
    assert memory_manager.enabled
    assert memory_manager.client is not None
    assert memory_manager.api_key is not None

def test_store_and_query_cycle(
    memory_manager: TradingMemoryManager,
    sample_strategy_context: StrategyContext,
    sample_backtest_results: BacktestResults
) -> None:
    """Test full cycle of storing and querying strategies."""
    # First store a strategy
    result = memory_manager.store_strategy_results(
        context=sample_strategy_context,
        results=sample_backtest_results,
        iteration=1
    )
    assert result is True
    
    # Wait for indexing
    time.sleep(2)
    
    # Query similar strategies
    strategies = memory_manager.query_similar_strategies(
        market_regime=MarketRegime.TRENDING_BULLISH,
        page=1,
        page_size=5
    )
    assert len(strategies) > 0
    
    # Verify retrieved strategy matches stored data
    strategy = strategies[0]
    assert strategy["market_regime"] == sample_strategy_context.regime.value
    assert strategy["confidence"] == sample_strategy_context.confidence
    assert strategy["risk_level"] == sample_strategy_context.risk_level
    assert float(strategy["performance"]["total_return"]) == float(sample_backtest_results.total_return)
    assert float(strategy["performance"]["win_rate"]) == sample_backtest_results.win_rate
    
    # Store memory ID for history test
    memory_id = strategy["memory_id"]
    
    # Test history tracking
    history = memory_manager.get_strategy_history(memory_id)
    assert isinstance(history, list)
    assert len(history) > 0

def test_batch_operations(
    memory_manager: TradingMemoryManager,
    sample_strategy_context: StrategyContext,
    sample_backtest_results: BacktestResults
) -> None:
    """Test batch update and delete operations."""
    # Store multiple strategies
    memory_ids = []
    for i in range(3):
        result = memory_manager.store_strategy_results(
            context=sample_strategy_context,
            results=sample_backtest_results,
            iteration=i
        )
        assert result is True
        
        # Get the memory ID from the latest strategy
        time.sleep(2)  # Wait for indexing
        strategies = memory_manager.query_similar_strategies(
            market_regime=MarketRegime.TRENDING_BULLISH,
            page=1,
            page_size=1
        )
        assert len(strategies) > 0
        memory_ids.append(strategies[0]["memory_id"])
    
    # Test batch update
    updates = [
        {
            "memory_id": mid,
            "text": f"Updated strategy {i}"
        }
        for i, mid in enumerate(memory_ids)
    ]
    assert memory_manager.batch_update_strategies(updates) is True
    
    # Test batch delete
    assert memory_manager.batch_delete_strategies(memory_ids) is True

def test_pagination(
    memory_manager: TradingMemoryManager,
    sample_strategy_context: StrategyContext,
    sample_backtest_results: BacktestResults
) -> None:
    """Test pagination in get_all_strategies."""
    # Store multiple strategies
    for i in range(3):
        result = memory_manager.store_strategy_results(
            context=sample_strategy_context,
            results=sample_backtest_results,
            iteration=i
        )
        assert result is True

    time.sleep(2)  # Wait for indexing

    # Test first page
    page_1 = memory_manager.get_all_strategies(page=1, page_size=2)
    assert isinstance(page_1, dict)
    assert "results" in page_1
    assert len(page_1["results"]) <= 2  # Should respect page size
    
    # Test second page
    page_2 = memory_manager.get_all_strategies(page=2, page_size=2)
    assert isinstance(page_2, dict)
    assert "results" in page_2
    
    # Verify pagination metadata
    assert page_1["page"] == 1
    assert page_1["page_size"] == 2
    assert page_2["page"] == 2
    assert page_2["page_size"] == 2
    
    # Verify we can get results
    assert len(page_1["results"]) > 0
    
    # Verify structure of results
    strategy = page_1["results"][0]
    assert "id" in strategy
    assert "regime" in strategy
    assert "confidence" in strategy
    assert "performance" in strategy
    assert "score" in strategy
    assert "timestamp" in strategy

def test_get_recent_performance(
    memory_manager: TradingMemoryManager,
    sample_strategy_context: StrategyContext,
    sample_backtest_results: BacktestResults
) -> None:
    """Test getting recent performance metrics."""
    # Store some test data first
    memory_manager.store_strategy_results(
        context=sample_strategy_context,
        results=sample_backtest_results,
        iteration=1
    )
    
    # Get recent performance
    performance = memory_manager.get_recent_performance()
    assert isinstance(performance, dict)
    assert "total_trades" in performance
    assert "total_pnl" in performance
    assert "avg_return" in performance
    assert "avg_pnl" in performance
    assert "overall_win_rate" in performance
    
    # Verify values are reasonable
    assert float(performance["avg_return"]) >= -1.0
    assert float(performance["overall_win_rate"]) >= 0.0
    assert int(performance["total_trades"]) >= 0

def test_error_handling(memory_manager: TradingMemoryManager) -> None:
    """Test error handling in memory manager methods."""
    # Test with invalid market regime
    strategies = memory_manager.query_similar_strategies(MarketRegime.UNKNOWN)
    assert isinstance(strategies, list)
    assert len(strategies) == 0
    
    # Test with None context
    result = memory_manager.store_strategy_results(
        context=None,  # type: ignore
        results=BacktestResults(
            total_return=0.0,
            total_pnl=0.0,
            sortino_ratio=0.0,
            win_rate=0.0,
            total_trades=0,
            trades=[],
            metrics={}
        )
    )
    assert result is False
    
    # Test with invalid memory ID
    history = memory_manager.get_strategy_history("invalid_id")
    assert isinstance(history, list)
    assert len(history) == 0
    
    # Test batch operations with invalid IDs
    assert memory_manager.batch_update_strategies([{"memory_id": "invalid_id", "text": "test"}]) is False
    assert memory_manager.batch_delete_strategies(["invalid_id"]) is False 
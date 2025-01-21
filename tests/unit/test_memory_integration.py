"""Tests for memory integration in trading strategy."""

from typing import TYPE_CHECKING, Dict, Any
import pytest
import os
from unittest.mock import Mock, patch
from research.strategy.memory_manager import TradingMemoryManager
from research.strategy.types import MarketRegime, StrategyContext, BacktestResults

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from pytest_mock.plugin import MockerFixture

@pytest.fixture
def mock_memory():
    """Create a mock Memory instance."""
    return Mock()

@pytest.fixture
def mock_memory_client():
    """Create a mock MemoryClient instance."""
    return Mock()

@pytest.fixture
def memory_manager(mock_memory, mock_memory_client):
    """Create a TradingMemoryManager with mocked dependencies."""
    with patch('research.strategy.memory_manager.Memory') as mock_memory_cls, \
         patch('research.strategy.memory_manager.MemoryClient') as mock_client_cls:
        mock_memory_cls.from_config.return_value = mock_memory
        mock_client_cls.return_value = mock_memory_client
        
        manager = TradingMemoryManager(
            openai_api_key="test_openai_key",
            mem0_api_key="test_mem0_key"
        )
        return manager

@pytest.fixture
def sample_context():
    """Create a sample StrategyContext."""
    return StrategyContext(
        market_regime=MarketRegime.RANGING_LOW_VOL,
        parameters={"stop_loss": 0.02, "take_profit": 0.05},
        confidence=0.85,
        risk_level="LOW"
    )

@pytest.fixture
def sample_results():
    """Create sample BacktestResults."""
    return BacktestResults(
        total_return=1.5,
        total_pnl=15000.0,
        sortino_ratio=2.5,
        win_rate=0.65,
        total_trades=100,
        asset_count=10
    )

def test_store_strategy_results(
    memory_manager: TradingMemoryManager,
    sample_context: StrategyContext,
    sample_results: BacktestResults,
    mock_memory: Mock
):
    """Test storing strategy results in memory."""
    parameters = {"stop_loss": 0.02, "take_profit": 0.05}
    
    memory_manager.store_strategy_results(
        context=sample_context,
        results=sample_results,
        parameters=parameters
    )
    
    # Verify memory.add was called with correct arguments
    mock_memory.add.assert_called_once()
    call_args = mock_memory.add.call_args[1]
    
    assert call_args["user_id"] == "trading_system"
    assert len(call_args["messages"]) == 1
    
    memory_content = call_args["messages"][0]
    assert memory_content["role"] == "system"
    assert "Strategy results for" in memory_content["content"]
    
    metadata = memory_content["metadata"]
    assert metadata["type"] == "strategy_results"
    assert metadata["market_regime"] == MarketRegime.RANGING_LOW_VOL
    assert metadata["confidence"] == 0.85
    assert metadata["risk_level"] == "LOW"
    assert metadata["parameters"] == parameters
    
    performance = metadata["performance"]
    assert performance["total_return"] == 1.5
    assert performance["total_pnl"] == 15000.0
    assert performance["sortino_ratio"] == 2.5
    assert performance["win_rate"] == 0.65
    assert performance["total_trades"] == 100

def test_query_similar_strategies(
    memory_manager: TradingMemoryManager,
    mock_memory: Mock
):
    """Test querying similar strategies."""
    mock_memory.search.return_value = [
        {
            "memory": {
                "metadata": {
                    "performance": {
                        "total_return": 1.5,
                        "sortino_ratio": 2.5
                    }
                }
            }
        },
        {
            "memory": {
                "metadata": {
                    "performance": {
                        "total_return": 0.8,
                        "sortino_ratio": 1.8
                    }
                }
            }
        }
    ]
    
    results = memory_manager.query_similar_strategies(
        market_regime=MarketRegime.RANGING_LOW_VOL,
        min_return=1.0,
        min_sortino=2.0
    )
    
    # Verify memory.search was called
    mock_memory.search.assert_called_once()
    
    # Verify filtering works
    assert len(results) == 1
    assert results[0]["memory"]["metadata"]["performance"]["total_return"] == 1.5
    assert results[0]["memory"]["metadata"]["performance"]["sortino_ratio"] == 2.5

def test_get_optimal_parameters(
    memory_manager: TradingMemoryManager,
    mock_memory: Mock
):
    """Test getting optimal parameters from memory."""
    mock_memory.search.return_value = [
        {
            "memory": {
                "metadata": {
                    "parameters": {
                        "stop_loss": 0.02,
                        "take_profit": 0.05
                    },
                    "performance": {
                        "sortino_ratio": 2.5
                    }
                }
            }
        },
        {
            "memory": {
                "metadata": {
                    "parameters": {
                        "stop_loss": 0.03,
                        "take_profit": 0.06
                    },
                    "performance": {
                        "sortino_ratio": 1.5
                    }
                }
            }
        }
    ]
    
    current_params = {
        "stop_loss": 0.01,
        "take_profit": 0.03
    }
    
    optimized_params = memory_manager.get_optimal_parameters(
        market_regime=MarketRegime.RANGING_LOW_VOL,
        current_params=current_params
    )
    
    # Verify parameters are weighted by Sortino ratio
    total_weight = 4.0  # Sum of Sortino ratios
    assert optimized_params["stop_loss"] == pytest.approx(
        (0.02 * 2.5 + 0.03 * 1.5) / total_weight
    )
    assert optimized_params["take_profit"] == pytest.approx(
        (0.05 * 2.5 + 0.06 * 1.5) / total_weight
    )

def test_memory_reset(
    memory_manager: TradingMemoryManager,
    mock_memory: Mock
):
    """Test resetting memory store."""
    memory_manager.reset()
    mock_memory.reset.assert_called_once() 
"""Tests for the memory-enhanced improvement phase functionality."""

from typing import Dict, List, Optional
from datetime import datetime
import pytest
from unittest.mock import Mock, patch
if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from pytest_mock import MockerFixture

from strat_optim.memory.memory_interface import TradingMemory
from research.strategy.godel_agent import GodelAgent
from research.strategy.types import MarketRegime, StrategyContext, BacktestResults

@pytest.fixture
def mock_memory():
    """Create a mock TradingMemory instance."""
    memory = Mock(spec=TradingMemory)
    memory.add_memory.return_value = [{"id": "test_memory"}]
    memory.search_memories.return_value = []
    return memory

@pytest.fixture
def mock_context():
    """Create a mock strategy context."""
    return StrategyContext(
        market_regime=MarketRegime.RANGING_LOW_VOL,
        parameters={
            "take_profit": 0.1,
            "stop_loss": 0.05,
            "order_size": 0.0025
        },
        confidence=0.85,
        risk_level="low"
    )

@pytest.fixture
def mock_results():
    """Create mock backtest results."""
    return BacktestResults(
        total_return=1.5,
        total_pnl=15.0,
        sortino_ratio=2.5,
        win_rate=0.55,
        total_trades=100,
        asset_count=65
    )

def test_memory_initialization(mock_memory):
    """Test that memory system is properly initialized."""
    agent = GodelAgent()
    agent.memory = mock_memory
    
    assert hasattr(agent, 'memory')
    assert isinstance(agent.memory, Mock)
    assert agent.memory.spec == TradingMemory

def test_storing_iteration_results(mock_memory, mock_context, mock_results):
    """Test storing iteration results in memory."""
    agent = GodelAgent()
    agent.memory = mock_memory
    
    agent.improvement_phase(mock_context, mock_results)
    
    mock_memory.add_memory.assert_called_once()
    call_args = mock_memory.add_memory.call_args[0]
    stored_content = call_args[0]
    
    assert stored_content["market_regime"] == MarketRegime.RANGING_LOW_VOL
    assert stored_content["parameters"]["take_profit"] == 0.1
    assert stored_content["performance"]["total_return"] == 1.5
    assert stored_content["performance"]["sortino_ratio"] == 2.5

def test_querying_similar_experiences(mock_memory, mock_context, mock_results):
    """Test querying similar past experiences from memory."""
    # Setup mock past experiences
    past_results = [{
        "market_regime": "RANGING_LOW_VOL",
        "parameters": {
            "take_profit": 0.15,
            "stop_loss": 0.07,
            "order_size": 0.003
        },
        "performance": {
            "total_return": 2.0,
            "sortino_ratio": 3.0,
            "win_rate": 0.6
        }
    }]
    mock_memory.search_memories.return_value = past_results
    
    agent = GodelAgent()
    agent.memory = mock_memory
    
    agent.improvement_phase(mock_context, mock_results)
    
    mock_memory.search_memories.assert_called_once()
    query = mock_memory.search_memories.call_args[0][0]
    assert "market_regime:RANGING_LOW_VOL" in query
    assert "sortino_ratio>=2.5" in query

def test_generating_improvements(mock_memory, mock_context, mock_results):
    """Test generating improvements based on historical data."""
    # Setup mock successful past experience
    past_results = [{
        "market_regime": "RANGING_LOW_VOL",
        "parameters": {
            "take_profit": 0.15,
            "stop_loss": 0.07,
            "order_size": 0.003
        },
        "performance": {
            "total_return": 2.0,
            "sortino_ratio": 3.0,
            "win_rate": 0.6
        }
    }]
    mock_memory.search_memories.return_value = past_results
    
    agent = GodelAgent()
    agent.memory = mock_memory
    
    improvements = agent.improvement_phase(mock_context, mock_results)
    
    # Verify improvements suggest parameter values closer to successful past experience
    assert improvements.parameters["take_profit"] > mock_context.parameters["take_profit"]
    assert improvements.parameters["stop_loss"] > mock_context.parameters["stop_loss"]
    assert improvements.parameters["order_size"] > mock_context.parameters["order_size"]

def test_memory_error_handling(mock_memory, mock_context, mock_results):
    """Test error handling in memory operations."""
    mock_memory.add_memory.side_effect = Exception("Memory storage failed")
    
    agent = GodelAgent()
    agent.memory = mock_memory
    
    with pytest.raises(Exception) as exc_info:
        agent.improvement_phase(mock_context, mock_results)
    
    assert "Memory storage failed" in str(exc_info.value)

def test_memory_reset(mock_memory):
    """Test resetting memory storage."""
    agent = GodelAgent()
    agent.memory = mock_memory
    
    agent.reset_memory()
    
    mock_memory.reset.assert_called_once() 
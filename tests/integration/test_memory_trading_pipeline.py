"""Integration tests for the memory-enhanced trading pipeline."""

from typing import Dict, List, Optional
import os
import pytest
from datetime import datetime
if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from pytest_mock import MockerFixture
    from _pytest.logging import LogCaptureFixture

import pandas as pd
import numpy as np

from strat_optim.memory.memory_interface import TradingMemory
from research.strategy.godel_agent import GodelAgent
from research.strategy.types import MarketRegime, StrategyContext, BacktestResults
from research.strategy.llm_interface import LLMInterface
from backtester import Backtester

@pytest.fixture
def sample_trade_data():
    """Create sample trade data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='H')
    assets = ['BTC', 'ETH', 'SOL']
    
    data = {}
    for asset in assets:
        # Generate random price data with some trend
        prices = np.random.random(len(dates)) * 100
        prices = prices + np.linspace(0, 10, len(dates))  # Add upward trend
        
        df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.random(len(dates)) * 1000
        })
        df.set_index('timestamp', inplace=True)
        data[asset] = df
    
    return data

@pytest.fixture
def trading_pipeline(sample_trade_data):
    """Create a complete trading pipeline with memory system."""
    memory = TradingMemory(
        openai_api_key=os.getenv("OPENAI_API_KEY", "test_key"),
        agent_id="test_agent"
    )
    
    agent = GodelAgent()
    agent.memory = memory
    
    backtester = Backtester(trade_data=sample_trade_data)
    
    return {
        'agent': agent,
        'memory': memory,
        'backtester': backtester,
        'trade_data': sample_trade_data
    }

def test_full_trading_cycle(trading_pipeline, caplog: LogCaptureFixture):
    """Test a complete trading cycle with memory integration."""
    agent = trading_pipeline['agent']
    backtester = trading_pipeline['backtester']
    
    # Run multiple iterations
    for i in range(3):
        # 1. Market Analysis
        context = agent.analyze_market(trading_pipeline['trade_data'])
        assert context.market_regime is not None
        assert 0 <= context.confidence <= 1
        
        # 2. Strategy Generation
        strategy = agent.generate_strategy(context)
        assert strategy is not None
        assert "parameters" in strategy
        
        # 3. Backtesting
        results = backtester.run(strategy.parameters)
        assert isinstance(results, BacktestResults)
        
        # 4. Memory Storage
        memory_content = {
            "iteration": i,
            "market_regime": context.market_regime,
            "parameters": strategy.parameters,
            "performance": {
                "total_return": results.total_return,
                "sortino_ratio": results.sortino_ratio
            }
        }
        stored = agent.memory.add_memory(memory_content, "trading_cycle")
        assert len(stored) > 0
        
        # 5. Improvement Phase
        if i > 0:  # After first iteration
            improvements = agent.improvement_phase(context, results)
            assert improvements is not None
            assert improvements.parameters != strategy.parameters

def test_memory_persistence(trading_pipeline):
    """Test that memory persists and influences future decisions."""
    agent = trading_pipeline['agent']
    memory = trading_pipeline['memory']
    
    # Store initial successful strategy
    initial_memory = {
        "market_regime": MarketRegime.RANGING_LOW_VOL,
        "parameters": {
            "take_profit": 0.15,
            "stop_loss": 0.07
        },
        "performance": {
            "total_return": 2.0,
            "sortino_ratio": 3.0
        }
    }
    memory.add_memory(initial_memory, "successful_strategy")
    
    # Run new strategy generation
    context = StrategyContext(
        market_regime=MarketRegime.RANGING_LOW_VOL,
        parameters={},
        confidence=0.85,
        risk_level="low"
    )
    
    strategy = agent.generate_strategy(context)
    
    # Verify new strategy is influenced by stored memory
    assert abs(strategy.parameters["take_profit"] - 0.15) < 0.05
    assert abs(strategy.parameters["stop_loss"] - 0.07) < 0.05

def test_memory_based_adaptation(trading_pipeline, caplog: LogCaptureFixture):
    """Test system's ability to adapt based on stored experiences."""
    agent = trading_pipeline['agent']
    backtester = trading_pipeline['backtester']
    
    # Run first iteration with poor performance
    context1 = StrategyContext(
        market_regime=MarketRegime.RANGING_LOW_VOL,
        parameters={"take_profit": 0.1, "stop_loss": 0.05},
        confidence=0.85,
        risk_level="low"
    )
    
    results1 = BacktestResults(
        total_return=-0.5,
        sortino_ratio=1.0,
        win_rate=0.45,
        total_trades=100,
        asset_count=3,
        total_pnl=-5.0
    )
    
    agent.improvement_phase(context1, results1)
    
    # Run second iteration
    context2 = context1
    strategy2 = agent.generate_strategy(context2)
    
    # Verify adaptation
    assert strategy2.parameters["take_profit"] != context1.parameters["take_profit"]
    assert strategy2.parameters["stop_loss"] != context1.parameters["stop_loss"]
    
    # Check adaptation is logged
    assert any("Adapting strategy based on past performance" in record.message 
              for record in caplog.records)

def test_cross_regime_learning(trading_pipeline):
    """Test learning transfer between different market regimes."""
    agent = trading_pipeline['agent']
    memory = trading_pipeline['memory']
    
    # Store successful strategies for different regimes
    regimes = [
        (MarketRegime.RANGING_LOW_VOL, 0.15, 0.07),
        (MarketRegime.TRENDING_HIGH_VOL, 0.2, 0.1),
        (MarketRegime.RANGING_HIGH_VOL, 0.18, 0.08)
    ]
    
    for regime, tp, sl in regimes:
        memory.add_memory({
            "market_regime": regime,
            "parameters": {
                "take_profit": tp,
                "stop_loss": sl
            },
            "performance": {
                "total_return": 2.0,
                "sortino_ratio": 3.0
            }
        }, "successful_strategy")
    
    # Test strategy generation for new regime
    context = StrategyContext(
        market_regime=MarketRegime.TRENDING_LOW_VOL,
        parameters={},
        confidence=0.85,
        risk_level="low"
    )
    
    strategy = agent.generate_strategy(context)
    
    # Verify parameters are influenced by similar regimes
    assert 0.15 <= strategy.parameters["take_profit"] <= 0.2
    assert 0.07 <= strategy.parameters["stop_loss"] <= 0.1

def test_error_recovery(trading_pipeline, caplog: LogCaptureFixture):
    """Test system's ability to recover from memory errors."""
    agent = trading_pipeline['agent']
    memory = trading_pipeline['memory']
    
    # Simulate memory failure
    memory.add_memory.side_effect = Exception("Storage error")
    
    context = StrategyContext(
        market_regime=MarketRegime.RANGING_LOW_VOL,
        parameters={"take_profit": 0.1, "stop_loss": 0.05},
        confidence=0.85,
        risk_level="low"
    )
    
    results = BacktestResults(
        total_return=1.0,
        sortino_ratio=2.0,
        win_rate=0.55,
        total_trades=100,
        asset_count=3,
        total_pnl=10.0
    )
    
    # System should continue without memory
    strategy = agent.generate_strategy(context)
    assert strategy is not None
    
    # Verify error is logged
    assert any("Failed to store memory" in record.message 
              for record in caplog.records) 
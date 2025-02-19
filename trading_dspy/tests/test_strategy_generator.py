"""Tests for the strategy generator module."""

from typing import Dict, Any, List, TYPE_CHECKING
import pytest
import os
from dotenv import load_dotenv
import dspy
from pathlib import Path

from src.modules.strategy_generator import StrategyGenerator
from src.utils.prompt_manager import PromptManager
from src.utils.memory_manager import TradingMemoryManager
from src.utils.types import MarketRegime, StrategyContext, BacktestResults

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

@pytest.fixture(autouse=True)
def setup_dspy() -> None:
    """Set up DSPy with OpenAI."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")
    
    lm = dspy.LM(model="gpt-4-turbo-preview", api_key=api_key)
    dspy.settings.configure(lm=lm)

@pytest.fixture
def prompt_manager() -> PromptManager:
    """Create prompt manager instance."""
    return PromptManager("prompts")

@pytest.fixture
def memory_manager() -> TradingMemoryManager:
    """Create memory manager instance."""
    return TradingMemoryManager()

@pytest.fixture
def strategy_generator() -> StrategyGenerator:
    """Create a strategy generator for testing."""
    # Use real prompt manager with correct prompts directory
    prompts_dir = "prompts"  # Use relative path to existing prompts directory
    prompt_manager = PromptManager(prompts_dir)
    
    # Create mock memory manager
    memory_manager = TradingMemoryManager()
    memory_manager.enabled = True
    memory_manager.get_recent_performance = lambda: {
        "total_return": 0.15,
        "win_rate": 0.65,
        "sortino_ratio": 1.8,
        "total_trades": 100
    }
    
    return StrategyGenerator(prompt_manager, memory_manager)

def test_strategy_generation_success(strategy_generator: StrategyGenerator) -> None:
    """Test successful strategy generation."""
    market_context = {
        "regime": "bullish",
        "confidence": 0.8,
        "risk_level": "medium",
        "analysis_text": "Market showing strong momentum with RSI crossing above 50 and MACD showing positive divergence.",
        "recent_performance": {
            "total_return": 0.15,
            "win_rate": 0.65,
            "sortino_ratio": 1.8,
            "total_trades": 100
        }
    }
    theme = "momentum"
    base_parameters = {
        "stop_loss": 0.02,
        "take_profit": 0.04,
        "position_size": 0.1
    }
    
    result = strategy_generator.forward(market_context, theme, base_parameters)
    
    assert isinstance(result, dict)
    assert "reasoning" in result
    assert "trade_signal" in result
    assert "parameters" in result
    assert "confidence" in result
    assert "entry_conditions" in result
    assert "exit_conditions" in result
    assert "indicators" in result
    assert "strategy_type" in result
    assert "market_regime" in result
    
    # Verify parameter validation
    params = result["parameters"]
    assert 0 < params["stop_loss"] < 1
    assert 0 < params["take_profit"] < 5
    assert 0 < params["position_size"] <= 1

def test_strategy_validation(strategy_generator: StrategyGenerator) -> None:
    """Test strategy validation."""
    invalid_strategy = {
        "reasoning": "test",
        "trade_signal": "INVALID",
        "parameters": {"stop_loss": 0.02},
        "confidence": 0.8,
        "entry_conditions": ["condition"],
        "exit_conditions": ["condition"],
        "indicators": ["RSI"]
    }
    
    is_valid, reason = strategy_generator.validate_strategy(invalid_strategy)
    assert not is_valid
    assert "Invalid trade signal" in reason

def test_condition_validation(strategy_generator: StrategyGenerator) -> None:
    """Test condition validation."""
    conditions = [
        "df_indicators['RSI'] > 70",
        "import os",  # Should be filtered out
        "df_indicators['MACD'] > df_indicators['MACD_SIGNAL']"
    ]
    
    valid_conditions = strategy_generator._validate_conditions(conditions)
    assert len(valid_conditions) == 2
    assert "import" not in " ".join(valid_conditions)

def test_error_handling(strategy_generator: StrategyGenerator) -> None:
    """Test error handling."""
    result = strategy_generator.forward(None, "theme", None)
    expected_result = {
        'reasoning': 'Error in strategy generation',
        'trade_signal': 'HOLD',
        'parameters': {'stop_loss': 0.02, 'take_profit': 0.04, 'position_size': 0.1},
        'confidence': 0.0,
        'entry_conditions': [],
        'exit_conditions': [],
        'indicators': [],
        'strategy_type': 'theme',
        'market_regime': 'unknown'
    }
    assert result == expected_result 
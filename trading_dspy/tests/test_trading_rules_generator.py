import json
from types import SimpleNamespace
from typing import Any, Dict

import pytest

# If TYPE_CHECKING, import fixtures
if __import__('typing').TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

# Adjust path for module imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.modules.trading_rules import TradingRulesGenerator  # type: ignore


class DummyPromptManager:
    """Dummy prompt manager for testing purposes."""
    def __init__(self) -> None:
        self.last_params: Dict[str, Any] = {}

    def get_prompt(self, prompt_name: str) -> str:
        """Return a dummy prompt template with placeholders."""
        return ("Parameters: {parameters}\nReasoning: {reasoning}\n"
                "Entry Conditions: {entry_conditions}\nExit Conditions: {exit_conditions}\n"
                "Strategy Insights: {strategy_insights}\nMarket Context: {market_context}")

    def format_prompt(self, prompt: str, **params: Any) -> str:
        """Format the prompt and store the parameters for inspection."""
        self.last_params = params
        return prompt.format(**params)


class DummyPredictor:
    """Dummy predictor that simulates returning a JSON result by echoing conditions from the prompt."""
    def __call__(self, *, strategy_insights: Dict[str, Any], market_context: Dict[str, Any], prompt: str) -> Any:
        """Parse the prompt to extract entry and exit conditions and return a dummy JSON response."""
        entry = ""
        exit = ""
        for line in prompt.splitlines():
            l = line.strip()
            if l.startswith("Entry Conditions:"):
                entry = l.split(":", 1)[1].strip()
            elif l.startswith("Exit Conditions:"):
                exit = l.split(":", 1)[1].strip()
        response = {
            "status": "dummy",
            "entry_conditions": entry,
            "exit_conditions": exit
        }
        return SimpleNamespace(text=json.dumps(response))


@pytest.fixture
def setup_trading_rules_generator() -> TradingRulesGenerator:
    """Fixture to create a TradingRulesGenerator instance with dummy dependencies."""
    generator = TradingRulesGenerator(DummyPromptManager())
    generator.predictor = DummyPredictor()
    return generator


def test_default_conditions(setup_trading_rules_generator: TradingRulesGenerator) -> None:
    """Test that default entry and exit conditions are used when none are provided."""
    strategy_insights: Dict[str, Any] = {
        "parameters": {
            "stop_loss": 0.03
        },
        "reasoning": "Test reasoning"
        # 'entry_conditions' and 'exit_conditions' are intentionally omitted
    }
    market_context: Dict[str, Any] = {"market": "test"}

    result = setup_trading_rules_generator.forward(strategy_insights, market_context)
    
    # Access the last prompt parameters in DummyPromptManager
    prompt_params = setup_trading_rules_generator.prompt_manager.last_params

    expected_entry = "No entry conditions specified"
    expected_exit = "No exit conditions specified"

    assert prompt_params.get('entry_conditions') == expected_entry, \
        f"Expected entry_conditions to be '{expected_entry}', got '{prompt_params.get('entry_conditions')}'"
    assert prompt_params.get('exit_conditions') == expected_exit, \
        f"Expected exit_conditions to be '{expected_exit}', got '{prompt_params.get('exit_conditions')}'"

    # Also check that the forward function returns the dummy result
    result_dict = json.loads(json.dumps(result))  
    assert result_dict.get('status') == 'dummy'


def test_custom_conditions(setup_trading_rules_generator: TradingRulesGenerator) -> None:
    """Test that provided entry and exit conditions are correctly formatted as bullet points."""
    custom_entry = ["Buy when price breaks resistance"]
    custom_exit = ["Sell when price hits support"]
    strategy_insights: Dict[str, Any] = {
        "parameters": {
            "take_profit": 0.05
        },
        "reasoning": "Test reasoning with conditions",
        "entry_conditions": custom_entry,
        "exit_conditions": custom_exit
    }
    market_context: Dict[str, Any] = {"market": "test"}

    result = setup_trading_rules_generator.forward(strategy_insights, market_context)
    
    # Access the last prompt parameters in DummyPromptManager
    prompt_params = setup_trading_rules_generator.prompt_manager.last_params

    expected_entry = "- Buy when price breaks resistance"
    expected_exit = "- Sell when price hits support"

    assert prompt_params.get('entry_conditions') == expected_entry, \
        f"Expected entry_conditions to be '{expected_entry}', got '{prompt_params.get('entry_conditions')}'"
    assert prompt_params.get('exit_conditions') == expected_exit, \
        f"Expected exit_conditions to be '{expected_exit}', got '{prompt_params.get('exit_conditions')}'"

    # Also check that the forward function returns the dummy result
    result_dict = json.loads(json.dumps(result))
    assert result_dict.get('status') == 'dummy'


def test_valid_trading_conditions(setup_trading_rules_generator: TradingRulesGenerator) -> None:
    """Test that valid trading conditions are used when provided and not replaced by fallback SMA crossover defaults.
    
    This test ensures that if the provided entry and exit conditions contain valid comparison operators,
    they are maintained and not replaced by the default conditions.
    """
    valid_entry = ["price > 100"]
    valid_exit = ["price < 90"]
    strategy_insights: Dict[str, Any] = {
        "parameters": {
            "stop_loss": 0.03,
            "take_profit": 0.05,
            "position_size": 0.1
        },
        "reasoning": "Valid trading conditions provided.",
        "entry_conditions": valid_entry,
        "exit_conditions": valid_exit
    }
    market_context: Dict[str, Any] = {"market": "test-context"}

    result = setup_trading_rules_generator.forward(strategy_insights, market_context)

    expected_conditions = {"entry": ["price > 100"], "exit": ["price < 90"]}
    
    # Assert that the conditions returned by the generator match the expected valid conditions
    assert result.get("conditions") == expected_conditions, \
        f"Expected valid conditions {expected_conditions}, got {result.get('conditions')}" 
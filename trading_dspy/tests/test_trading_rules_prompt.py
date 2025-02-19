'''
Test for the trading_rules prompt template formatting.
'''

import os
import sys
import importlib.util
import json
import pytest

if "src.utils.prompt_manager" in sys.modules:
    del sys.modules["src.utils.prompt_manager"]

module_path = os.path.join(os.getcwd(), "src", "utils", "prompt_manager.py")
spec = importlib.util.spec_from_file_location("src.utils.prompt_manager", module_path)
pm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm_module)
PromptManager = pm_module.PromptManager


def test_trading_rules_prompt_formatting() -> None:
    """
    Test that the trading_rules prompt is correctly formatted using sample inputs.
    """
    # Setup PromptManager with the prompts directory
    prompts_dir = os.path.join(os.getcwd(), "prompts")
    pm = PromptManager(prompts_dir)
    assert pm.__class__.__module__ == "src.utils.prompt_manager", "PromptManager is not imported from src.utils.prompt_manager"
    trading_rules_prompt = pm.get_prompt("trading_rules")
    assert trading_rules_prompt is not None, "Trading rules prompt template not found."
    
    # Debug: assert that the updated signature is loaded
    assert 'parameters' in pm.format_prompt.__code__.co_varnames, "Updated format_prompt not loaded"
    
    # Prepare sample inputs for formatting
    sample_strategy_insights = {
        "reasoning": "Sample reasoning",
        "trade_signal": "BUY",
        "parameters": {
            "stop_loss": 0.02,
            "take_profit": 0.04,
            "position_size": 0.1,
            "entry_conditions": ["price > MA(20) AND volume > 100"],
            "exit_conditions": ["price < MA(20)", "trailing_stop(0.02)"]
        },
        "confidence": 0.8,
        "entry_conditions": ["price > MA(20) AND volume > 100"],
        "exit_conditions": ["price < MA(20)", "trailing_stop(0.02)"]
    }
    
    sample_market_context = {
        "regime": "ranging",
        "confidence": 0.75,
        "analysis_text": "Market is ranging."
    }
    
    sample_entry_conditions = "price > MA(20) AND volume > 100"
    sample_exit_conditions = "price < MA(20) OR trailing_stop(0.02)"
    
    # Format the prompt using sample data
    formatted_prompt = pm.format_prompt(
        trading_rules_prompt,
        parameters=json.dumps(sample_strategy_insights["parameters"], indent=2),
        reasoning=sample_strategy_insights["reasoning"],
        entry_conditions=sample_entry_conditions,
        exit_conditions=sample_exit_conditions,
        strategy_insights=json.dumps(sample_strategy_insights, indent=2),
        market_context=json.dumps(sample_market_context, indent=2)
    )
    
    # Extract the JSON block from the formatted prompt based on the Example response marker
    marker = "Example response:"
    marker_index = formatted_prompt.find(marker)
    assert marker_index != -1, "Example response marker not found in formatted prompt."

    start_block = formatted_prompt.find("'''json", marker_index)
    assert start_block != -1, "'''json marker not found in formatted prompt."

    start_brace = formatted_prompt.find('{', start_block)
    assert start_brace != -1, "JSON start brace not found in formatted prompt."

    end_block = formatted_prompt.find("'''", start_brace)
    assert end_block != -1, "Ending triple quote not found in formatted prompt."

    json_str = formatted_prompt[start_brace:end_block].strip()
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        pytest.fail(f"Formatted JSON block could not be parsed: {e}")
        
    # Check the structure of the JSON object
    assert "entry_conditions" in data, "entry_conditions key missing in JSON."
    assert "exit_conditions" in data, "exit_conditions key missing in JSON."
    assert "parameters" in data, "parameters key missing in JSON."
    assert "reasoning" in data, "reasoning key missing in JSON."
    
    # Validate default parameter values
    params = data["parameters"]
    assert params.get("stop_loss") == 0.02, "stop_loss parameter has been modified."
    assert params.get("take_profit") == 0.04, "take_profit parameter has been modified."
    assert params.get("position_size") == 0.1, "position_size parameter has been modified."
    
    print("Trading rules prompt formatted successfully.")


if __name__ == "__main__":
    test_trading_rules_prompt_formatting() 
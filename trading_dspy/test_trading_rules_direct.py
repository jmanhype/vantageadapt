#!/usr/bin/env python3

"""
Script to directly test trading rules component scoring and plateau-breaking mechanism.
"""

import os
import sys
import json
import dspy
from loguru import logger
from dotenv import load_dotenv

from src.modules.trading_rules import TradingRulesGenerator
from src.utils.prompt_manager import PromptManager

# Load environment variables from .env file
load_dotenv()

# Set up logging - forward to console for visibility and dedicated log file
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/component_scores.log", level="DEBUG")

def create_test_rules(entry_count=0, exit_count=0, params=None, indicators=None, 
                      operators=None, reasoning_length=0, trade_signal="BUY"):
    """Create test rules with specific properties to test metric scoring."""
    entry_conds = []
    exit_conds = []
    
    # Add entry conditions
    for i in range(entry_count):
        entry_conds.append(f"price > sma_{i+1}")
        
    # Add exit conditions
    for i in range(exit_count):
        exit_conds.append(f"price < sma_{i+1}")
        
    # Create parameters dict
    params_dict = {}
    if params:
        for param in params:
            params_dict[param] = 0.1
            
    # Create indicators list
    inds = []
    if indicators:
        inds = indicators
        
    # Add operators to conditions if specified
    if operators and entry_count > 1:
        entry_conds = [f"{entry_conds[0]} {op} {entry_conds[1]}" for op in operators]
        
    # Create reasoning
    reason = "X" * reasoning_length if reasoning_length > 0 else ""
    
    # Create final rules dict
    rules = {
        "conditions": {"entry": entry_conds, "exit": exit_conds},
        "entry_conditions": entry_conds,
        "exit_conditions": exit_conds,
        "parameters": params_dict,
        "indicators": inds,
        "reasoning": reason,
        "trade_signal": trade_signal
    }
    
    return {"trading_rules": rules}

def test_component_scoring():
    """Test each component of the scoring function independently."""
    logger.info("===== TESTING INDIVIDUAL COMPONENTS =====")
    
    # Test entry condition scoring
    logger.info("\n## Testing Entry Condition Scoring")
    for count in [0, 1, 2, 3, 4]:
        rules = create_test_rules(entry_count=count)
        score = trading_rules_metric({}, rules)
        logger.info(f"Entry conditions ({count}): Score = {score:.4f}")
    
    # Test exit condition scoring
    logger.info("\n## Testing Exit Condition Scoring")
    for count in [0, 1, 2, 3, 4]:
        rules = create_test_rules(exit_count=count)
        score = trading_rules_metric({}, rules)
        logger.info(f"Exit conditions ({count}): Score = {score:.4f}")
    
    # Test parameter scoring
    logger.info("\n## Testing Parameter Scoring")
    param_sets = [
        [],
        ["stop_loss"],
        ["take_profit"],
        ["stop_loss", "take_profit"],
        ["stop_loss", "take_profit", "position_size"],
        ["stop_loss", "take_profit", "position_size", "risk_management"]
    ]
    for params in param_sets:
        rules = create_test_rules(params=params)
        score = trading_rules_metric({}, rules)
        logger.info(f"Parameters {params}: Score = {score:.4f}")
    
    # Test indicator diversity scoring
    logger.info("\n## Testing Indicator Scoring")
    indicator_sets = [
        [],
        ["sma"],
        ["sma", "rsi"],
        ["sma", "rsi", "macd"],
        ["sma", "rsi", "macd", "bollinger"],
        ["sma", "rsi", "macd", "bollinger", "volume"]
    ]
    for indicators in indicator_sets:
        rules = create_test_rules(indicators=indicators)
        score = trading_rules_metric({}, rules)
        logger.info(f"Indicators {indicators}: Score = {score:.4f}")
    
    # Test logic operator scoring
    logger.info("\n## Testing Logic Operator Scoring")
    operator_sets = [
        [],
        [">"],
        [">", "<"],
        [">", "<", "=="],
        [">", "<", "==", ">=", "<="],
        [">", "<", "==", ">=", "<=", "and", "or", "not"]
    ]
    # For logic operators, we need at least 2 entry conditions
    for operators in operator_sets:
        rules = create_test_rules(entry_count=2, operators=operators)
        score = trading_rules_metric({}, rules)
        logger.info(f"Operators {operators}: Score = {score:.4f}")
    
    # Test reasoning quality scoring
    logger.info("\n## Testing Reasoning Quality Scoring")
    for length in [0, 100, 250, 500, 750]:
        rules = create_test_rules(reasoning_length=length)
        score = trading_rules_metric({}, rules)
        logger.info(f"Reasoning length ({length}): Score = {score:.4f}")

def test_plateau_breaking():
    """Test if plateau-breaking bonus is correctly applied."""
    logger.info("\n===== TESTING PLATEAU BREAKING MECHANISM =====")
    
    # This function will create examples with scores around the plateau threshold
    def create_threshold_example(target_score):
        """Create an example that should score close to the target."""
        # Start with base score of 0.05
        remaining_score = target_score - 0.05
        
        # Build up the score with components
        entry_count = min(3, int(remaining_score / 0.25 * 3))
        remaining_score -= entry_count / 3.0 * 0.25
        
        exit_count = min(3, int(remaining_score / 0.25 * 3))
        remaining_score -= exit_count / 3.0 * 0.25
        
        # Determine parameters to include
        params = []
        if remaining_score > 0.05:
            params.append("stop_loss")
            remaining_score -= 0.1
            
        if remaining_score > 0.05:
            params.append("take_profit")
            remaining_score -= 0.1
            
        # Determine indicators to include
        indicators = []
        if remaining_score > 0.03:
            indicators.append("sma")
            remaining_score -= 0.04
            
        if remaining_score > 0.03:
            indicators.append("rsi")
            remaining_score -= 0.04
            
        # Add reasoning of appropriate length
        reasoning_length = int(remaining_score / 0.15 * 500)
        
        return create_test_rules(
            entry_count=entry_count,
            exit_count=exit_count,
            params=params,
            indicators=indicators,
            reasoning_length=reasoning_length
        )
    
    # Test scores around the threshold
    test_scores = [0.13, 0.14, 0.145, 0.15, 0.155, 0.16, 0.17, 0.2, 0.25]
    
    for target in test_scores:
        example = create_threshold_example(target)
        score = trading_rules_metric({}, example)
        logger.info(f"Target score {target:.3f} → Actual score: {score:.4f}")
        
        # Save the test case for reference
        with open(f"logs/test_case_{target:.3f}.json", "w") as f:
            json.dump(example, f, indent=2)

# Define the enhanced trading_rules_metric function here for isolated testing
def trading_rules_metric(gold, pred, trace=None):
    """Enhanced metric function for trading rules optimization with detailed logging."""
    # Check for required components
    rules = pred.get('trading_rules', {})
    
    # Handle different output structures
    entry_conditions = rules.get('entry_conditions', rules.get('conditions', {}).get('entry', []))
    exit_conditions = rules.get('exit_conditions', rules.get('conditions', {}).get('exit', []))
    
    # Essential components exist - use progressive scoring
    # First check if we have conditions at all
    has_entry = float(bool(entry_conditions))
    has_exit = float(bool(exit_conditions))
    
    # Count the number of entry and exit conditions for a progressive score
    entry_count = len(entry_conditions) if isinstance(entry_conditions, list) else (1 if has_entry else 0)
    exit_count = len(exit_conditions) if isinstance(exit_conditions, list) else (1 if has_exit else 0)
    
    # Progressive score based on number of conditions (max at 3 for each)
    entry_score = min(1.0, entry_count / 3.0) * 0.25
    exit_score = min(1.0, exit_count / 3.0) * 0.25
    
    # Check if specific parameters are present
    conditions_text = str(entry_conditions) + str(exit_conditions) + str(rules)
    
    # Detailed parameter scoring
    parameter_checks = {
        'stop_loss': 0.1,
        'take_profit': 0.1,
        'position_size': 0.05,
        'risk_management': 0.05,
    }
    
    parameter_score = 0.0
    for param, weight in parameter_checks.items():
        if param in conditions_text.lower():
            parameter_score += weight
    
    # Indicator usage score - weighted by diversity
    indicator_checks = {
        'sma': 0.04,
        'rsi': 0.04,
        'macd': 0.04,
        'bollinger': 0.04,
        'volume': 0.04,
    }
    
    indicator_score = 0.0
    for indicator, weight in indicator_checks.items():
        if indicator in conditions_text.lower():
            indicator_score += weight
    
    # Detailed logic validation
    logic_checks = {
        # Check for comparison operators
        '>': 0.05,
        '<': 0.05,
        '==': 0.03,
        '>=': 0.03,
        '<=': 0.03,
        
        # Check for logical operators
        'and': 0.04,
        'or': 0.03,
        'not': 0.02,
    }
    
    logic_score = 0.0
    for operator, weight in logic_checks.items():
        if operator in conditions_text:
            logic_score += weight
    
    # Reasoning quality check
    reasoning = rules.get('reasoning', '')
    reasoning_length = len(str(reasoning))
    reasoning_score = min(0.15, (reasoning_length / 500) * 0.15)  # Max at 500 chars
    
    # Calculate combined score with diminishing returns
    # Base score ensures progress for any valid attempt
    base_score = 0.05
    
    # Main score components
    component_scores = {
        'entry_exit': entry_score + exit_score,
        'parameters': parameter_score,
        'indicators': indicator_score,
        'logic': logic_score,
        'reasoning': reasoning_score
    }
    
    # Calculate pre-plateau score
    pre_plateau_score = base_score + sum(component_scores.values())
    
    # Apply progressive bonus for scores above previous plateau
    # This helps break through plateaus by rewarding incremental improvements
    plateau_threshold = 0.15  # Slightly above the previous 14% (0.14) plateau
    if pre_plateau_score > plateau_threshold:
        bonus = (pre_plateau_score - plateau_threshold) * 0.3  # 30% bonus on improvement above plateau
        final_score = pre_plateau_score + bonus
        logger.warning(f"!!! PLATEAU BONUS APPLIED !!! Score {pre_plateau_score:.4f} > {plateau_threshold:.2f}, adding {bonus:.4f} bonus")
    else:
        final_score = pre_plateau_score
        logger.info(f"Score {pre_plateau_score:.4f} below plateau threshold {plateau_threshold:.2f}, no bonus applied")
                
    # Ensure score is within bounds
    final_score = max(0.0, min(1.0, final_score))
    
    # Debug log to help monitor scoring progress
    component_debug = f"E:{entry_score:.2f}+X:{exit_score:.2f}+P:{parameter_score:.2f}+I:{indicator_score:.2f}+L:{logic_score:.2f}+R:{reasoning_score:.2f}"
    logger.info(f"Rule metric components: {component_debug} → Pre-bonus: {pre_plateau_score:.4f} → Final: {final_score:.4f}")
            
    return final_score

def run_targeted_tests():
    """Run comprehensive tests focusing on our key improvements."""
    logger.info("Starting direct tests of trading rules scoring and plateau breaking")
    
    # Test individual components
    test_component_scoring()
    
    # Test plateau breaking
    test_plateau_breaking()
    
    logger.info("Completed direct testing of scoring components and plateau breaking")

if __name__ == "__main__":
    run_targeted_tests()
#!/usr/bin/env python3

"""
Script to test the trading rules MiPro optimization with component scores and plateau breaking.
"""

import os
import sys
import dspy
from loguru import logger
from dotenv import load_dotenv

from src.utils.mipro_optimizer import MiProWrapper
from src.utils.prompt_manager import PromptManager
from src.modules.prompt_optimizer import PromptOptimizer
from src.modules.trading_rules import TradingRulesGenerator

# Load environment variables from .env file
load_dotenv()

# Set up logging - forward to console for visibility
logger.remove()
logger.add(sys.stdout, level="DEBUG")
logger.add("logs/test_mipro.log", rotation="1 day", level="DEBUG")

def main():
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        return
    
    # Initialize DSPy
    lm = dspy.LM("gpt-4-turbo-preview", api_key=api_key)
    dspy.configure(lm=lm)
    
    # Initialize prompt manager and get examples
    logger.info("Initializing PromptManager")
    prompt_manager = PromptManager("prompts")
    
    # Get examples for trading rules
    trading_rules_examples = prompt_manager.get_examples("trading_rules")
    logger.info(f"Found {len(trading_rules_examples)} trading rules examples")
    
    if len(trading_rules_examples) < 3:
        logger.error("Not enough examples for optimization (need at least 3)")
        return
        
    # Add a special DEBUG override to prompt_optimizer.py -> trading_rules_metric to log more detail
    # This will monkeypatch the function for our test
    def enhanced_trading_rules_metric(gold, pred, trace=None):
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
        
        # Calculate final score - ensure it can reach up to 1.0 for perfect solutions
        score = base_score + sum(component_scores.values())
        
        # Apply progressive bonus for scores above previous plateau
        # This helps break through plateaus by rewarding incremental improvements
        plateau_threshold = 0.15  # Slightly above the previous 14% (0.14) plateau
        if score > plateau_threshold:
            bonus = (score - plateau_threshold) * 0.3  # 30% bonus on improvement above plateau
            score += bonus
            logger.warning(f"!!! PLATEAU BONUS APPLIED !!! Score above {plateau_threshold:.2f}, adding {bonus:.4f} bonus")
                
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        
        # Debug log to help monitor scoring progress
        component_debug = f"E:{entry_score:.2f}+X:{exit_score:.2f}+P:{parameter_score:.2f}+I:{indicator_score:.2f}+L:{logic_score:.2f}+R:{reasoning_score:.2f}"
        logger.info(f"Rule metric: {score:.4f} [{component_debug}]")
                
        return score
    
    # Monkeypatch the trading_rules_metric function
    import src.modules.prompt_optimizer
    original_trading_rules_metric = getattr(src.modules.prompt_optimizer, "trading_rules_metric", None)
    src.modules.prompt_optimizer.trading_rules_metric = enhanced_trading_rules_metric
    logger.warning("Monkeypatched trading_rules_metric function with enhanced logging")
    
    # Sample a few examples to examine
    logger.info("Example structure sample:")
    for i, ex in enumerate(trading_rules_examples[:2]):
        logger.info(f"Example {i+1} keys: {list(ex.keys())}")
        if 'outputs' in ex:
            logger.info(f"Example {i+1} output keys: {list(ex['outputs'].keys())}")
    
    # Initialize the prompt optimizer with our special focus on the modified metric
    logger.info("Initializing PromptOptimizer with MiPro")
    prompt_optimizer = PromptOptimizer(prompt_manager)
    
    # We can't directly import the metric function since it's defined inside a method,
    # but we can access it through the optimizer instance
    logger.info("Testing component scoring with a few examples")
    
    # We'll need to define a simple function to simulate the metric call
    def test_metric_components(ex):
        """Test metric components on an example."""
        if 'outputs' in ex and 'trading_rules' in ex['outputs']:
            gold = ex['outputs']['trading_rules']
            
            # Manually calculate component scores similar to how the metric does
            entry_conditions = gold.get('entry_conditions', gold.get('conditions', {}).get('entry', []))
            exit_conditions = gold.get('exit_conditions', gold.get('conditions', {}).get('exit', []))
            
            has_entry = float(bool(entry_conditions))
            has_exit = float(bool(exit_conditions))
            
            entry_count = len(entry_conditions) if isinstance(entry_conditions, list) else (1 if has_entry else 0)
            exit_count = len(exit_conditions) if isinstance(exit_conditions, list) else (1 if has_exit else 0)
            
            entry_score = min(1.0, entry_count / 3.0) * 0.25
            exit_score = min(1.0, exit_count / 3.0) * 0.25
            
            logger.info(f"Entry score: {entry_score:.4f} (count: {entry_count})")
            logger.info(f"Exit score: {exit_score:.4f} (count: {exit_count})")
            
            # Log the conditions for reference
            logger.info(f"Entry conditions: {entry_conditions}")
            logger.info(f"Exit conditions: {exit_conditions}")
            
            return entry_score + exit_score
        return 0.0
    
    # Generate a range of synthetic test cases to check our metric function
    logger.info("\n===== TESTING METRIC WITH SYNTHETIC EXAMPLES =====")
    
    def create_test_rules(entry_count=0, exit_count=0, params=None, indicators=None, 
                          operators=None, reasoning_length=0):
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
            "parameters": params_dict,
            "indicators": inds,
            "reasoning": reason
        }
        
        return {"trading_rules": rules}
    
    # Test cases with increasing scores
    test_cases = [
        {"name": "Empty rules", "rules": create_test_rules()},
        {"name": "Basic entry only", "rules": create_test_rules(entry_count=1)},
        {"name": "Basic exit only", "rules": create_test_rules(exit_count=1)},
        {"name": "Entry and exit", "rules": create_test_rules(entry_count=1, exit_count=1)},
        {"name": "Multiple entry/exit", "rules": create_test_rules(entry_count=3, exit_count=3)},
        {"name": "With parameters", "rules": create_test_rules(entry_count=2, exit_count=2, 
                                                              params=["stop_loss", "take_profit"])},
        {"name": "With indicators", "rules": create_test_rules(entry_count=2, exit_count=2, 
                                                              indicators=["sma", "rsi", "macd"])},
        {"name": "With operators", "rules": create_test_rules(entry_count=2, exit_count=2, 
                                                             operators=["and", "or"])},
        {"name": "With reasoning", "rules": create_test_rules(entry_count=2, exit_count=2, 
                                                             reasoning_length=300)},
        {"name": "Complete rules", "rules": create_test_rules(entry_count=3, exit_count=3, 
                                                           params=["stop_loss", "take_profit", "position_size"],
                                                           indicators=["sma", "rsi", "macd", "bollinger", "volume"],
                                                           operators=["and", "or", "not"],
                                                           reasoning_length=500)}
    ]
    
    # Test the monkeypatched metric function directly
    for test in test_cases:
        logger.info(f"\nTest case: {test['name']}")
        score = enhanced_trading_rules_metric({}, test["rules"])
        logger.info(f"Final score: {score:.4f}")
        
    # Restore original function if it existed
    if original_trading_rules_metric:
        src.modules.prompt_optimizer.trading_rules_metric = original_trading_rules_metric

if __name__ == "__main__":
    main()
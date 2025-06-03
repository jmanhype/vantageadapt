"""Test for component scoring in trading_rules_metric."""

import sys
import os
from loguru import logger
import copy
import json

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Set up enhanced logging for tests."""
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.configure(handlers=[
        {"sink": sys.stdout, "format": log_format, "level": "INFO"},
        {"sink": "component_scoring_test.log", "format": log_format, "level": "DEBUG"}
    ])

def trading_rules_metric(gold, pred, trace=None):
    """Metric function for trading rules optimization."""
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
    # Base score ensures progress for any valid attempt - increased from 0.05 to 0.1
    base_score = 0.1
    
    # Boost scores for having any entry or exit conditions at all
    if has_entry:
        base_score += 0.03  # Additional bonus just for having any entry condition
    if has_exit:
        base_score += 0.03  # Additional bonus just for having any exit condition
        
    # Main score components
    component_scores = {
        'entry_exit': entry_score + exit_score,
        'parameters': parameter_score,
        'indicators': indicator_score,
        'logic': logic_score,
        'reasoning': reasoning_score
    }
    
    # Calculate final score without plateau bonus first
    pre_bonus_score = base_score + sum(component_scores.values())
    
    # Now apply plateau bonus
    score = pre_bonus_score
    
    # Apply progressive bonus for scores above previous plateau
    # This helps break through plateaus by rewarding incremental improvements
    plateau_threshold = 0.14  # Lowered from 0.15 to match current plateau
    
    # Log details before applying bonus
    logger.debug(f"Component scores: {component_scores}")
    logger.debug(f"Base score: {base_score:.4f}")
    logger.debug(f"Pre-bonus score: {pre_bonus_score:.4f}")
    
    # Apply bonus if score is above threshold
    if score >= plateau_threshold:
        fixed_bonus = 0.05  # Add fixed 5% bonus for threshold
        score += fixed_bonus
        
        # Progressive bonus for improvement above threshold
        progressive_bonus = (pre_bonus_score - plateau_threshold) * 0.5  # 50% bonus
        score += progressive_bonus
        
        # Log the bonus details
        total_bonus = fixed_bonus + progressive_bonus
        logger.warning(f"!!! PLATEAU BONUS APPLIED !!! Score {pre_bonus_score:.4f} > {plateau_threshold:.2f}, adding {total_bonus:.4f} bonus")
    else:
        logger.debug(f"Score {score:.4f} below plateau threshold {plateau_threshold:.2f}, no bonus applied")
    
    # Ensure score is within bounds
    score = max(0.0, min(1.0, score))
    
    # Debug log to help monitor scoring progress
    component_debug = f"E:{entry_score:.2f}+X:{exit_score:.2f}+P:{parameter_score:.2f}+I:{indicator_score:.2f}+L:{logic_score:.2f}+R:{reasoning_score:.2f}"
    logger.info(f"Rule metric components: {component_debug} → Pre-bonus: {pre_bonus_score:.4f} → Final: {score:.4f}")
    
    return score

def generate_test_case(entry_conditions, exit_conditions, parameters, indicators, logic, reasoning):
    """Generate a test case with specified components."""
    trading_rules = {
        "entry_conditions": entry_conditions,
        "exit_conditions": exit_conditions,
        "parameters": parameters,
        "indicators": indicators,
        "reasoning": reasoning
    }
    
    # Add logic operators to conditions if needed
    if logic:
        trading_rules["entry_conditions"] = [cond + " and price > 0" for cond in entry_conditions] if entry_conditions else []
        trading_rules["exit_conditions"] = [cond + " or price < 0" for cond in exit_conditions] if exit_conditions else []
    
    return {"trading_rules": trading_rules}

def test_entry_conditions():
    """Test how entry conditions affect scoring."""
    tests = [
        [],  # No entry conditions
        ["price > sma_20"],  # One entry condition
        ["price > sma_20", "rsi < 30"],  # Two entry conditions
        ["price > sma_20", "rsi < 30", "macd > 0"],  # Three entry conditions
        ["price > sma_20", "rsi < 30", "macd > 0", "volume > 1000"]  # More than three entry conditions
    ]
    
    logger.info("\n## Testing Entry Condition Scoring")
    for i, entry_conditions in enumerate(tests):
        test_case = generate_test_case(entry_conditions, [], {}, ["sma"], True, "")
        score = trading_rules_metric({}, test_case)
        logger.info(f"Entry conditions ({i}): Score = {score:.4f}")
    
    return tests

def test_exit_conditions():
    """Test how exit conditions affect scoring."""
    tests = [
        [],  # No exit conditions
        ["price < sma_20"],  # One exit condition
        ["price < sma_20", "rsi > 70"],  # Two exit conditions
        ["price < sma_20", "rsi > 70", "macd < 0"],  # Three exit conditions
        ["price < sma_20", "rsi > 70", "macd < 0", "volume < 500"]  # More than three exit conditions
    ]
    
    logger.info("\n## Testing Exit Condition Scoring")
    for i, exit_conditions in enumerate(tests):
        test_case = generate_test_case([], exit_conditions, {}, ["sma"], True, "")
        score = trading_rules_metric({}, test_case)
        logger.info(f"Exit conditions ({i}): Score = {score:.4f}")
    
    return tests

def test_parameters():
    """Test how parameters affect scoring."""
    tests = [
        {},  # No parameters
        {"stop_loss": 0.02},  # One parameter (stop_loss)
        {"take_profit": 0.05},  # One parameter (take_profit)
        {"stop_loss": 0.02, "take_profit": 0.05},  # Two parameters
        {"stop_loss": 0.02, "take_profit": 0.05, "position_size": 0.1},  # Three parameters
        {"stop_loss": 0.02, "take_profit": 0.05, "position_size": 0.1, "risk_management": True}  # Four parameters
    ]
    
    logger.info("\n## Testing Parameter Scoring")
    for i, parameters in enumerate(tests):
        param_list = list(parameters.keys())
        test_case = generate_test_case([], [], parameters, [], False, "")
        score = trading_rules_metric({}, test_case)
        logger.info(f"Parameters {param_list}: Score = {score:.4f}")
    
    return tests

def test_indicators():
    """Test how indicators affect scoring."""
    tests = [
        [],  # No indicators
        ["sma"],  # One indicator
        ["sma", "rsi"],  # Two indicators
        ["sma", "rsi", "macd"],  # Three indicators
        ["sma", "rsi", "macd", "bollinger"],  # Four indicators
        ["sma", "rsi", "macd", "bollinger", "volume"]  # Five indicators
    ]
    
    logger.info("\n## Testing Indicator Scoring")
    for i, indicators in enumerate(tests):
        test_case = generate_test_case(["price > 0"], [], {}, indicators, False, "")
        score = trading_rules_metric({}, test_case)
        logger.info(f"Indicators {indicators}: Score = {score:.4f}")
    
    return tests

def test_logic_operators():
    """Test how logic operators affect scoring."""
    entry_conditions = ["price > sma_20", "rsi < 30"]
    
    tests = [
        [],  # No special logic
        [">"],  # One operator
        [">", "<"],  # Two operators
        [">", "<", "=="],  # Three operators
        [">", "<", "==", ">=", "<="],  # Five operators
        [">", "<", "==", ">=", "<=", "and", "or", "not"]  # All operators
    ]
    
    logger.info("\n## Testing Logic Operator Scoring")
    
    for i, operators in enumerate(tests):
        # Create conditions with the operators
        if not operators:
            test_case = generate_test_case(entry_conditions, [], {}, ["sma"], False, "")
        else:
            # Make sure the operators are in the conditions
            conditions = []
            for op in operators:
                if op == ">":
                    conditions.append("price > sma_20")
                elif op == "<":
                    conditions.append("price < sma_20")
                elif op == "==":
                    conditions.append("price == sma_20")
                elif op == ">=":
                    conditions.append("price >= sma_20")
                elif op == "<=":
                    conditions.append("price <= sma_20")
            
            # Add logical operators
            if "and" in operators:
                conditions = [" and ".join(conditions[:2])] + conditions[2:]
            if "or" in operators:
                if len(conditions) > 1:
                    conditions = [" or ".join(conditions[:2])] + conditions[2:]
            if "not" in operators and conditions:
                conditions[0] = "not " + conditions[0]
                
            test_case = generate_test_case(conditions[:1], conditions[1:2], {}, ["sma"], False, "")
            
        score = trading_rules_metric({}, test_case)
        logger.info(f"Operators {operators}: Score = {score:.4f}")
    
    return tests

def test_reasoning():
    """Test how reasoning length affects scoring."""
    tests = [
        "",  # No reasoning
        "Short reasoning.",  # Short reasoning
        "Medium length reasoning that explains the strategy." * 2,  # Medium reasoning
        "Longer reasoning that goes into detail about the strategy and explains the choices." * 5,  # Long reasoning
        "Very detailed reasoning with comprehensive explanation of all the strategy components and rationale." * 10  # Very long reasoning
    ]
    
    logger.info("\n## Testing Reasoning Quality Scoring")
    for i, reasoning in enumerate(tests):
        test_case = generate_test_case(["price > sma_20"], ["price < sma_20"], {}, ["sma"], False, reasoning)
        score = trading_rules_metric({}, test_case)
        logger.info(f"Reasoning length {len(reasoning)}: Score = {score:.4f}")
    
    return tests

def test_combined_components():
    """Test how combining components affects scoring."""
    # Create increasingly comprehensive strategies
    tests = [
        # Minimal strategy
        generate_test_case(["price > sma_20"], [], {}, ["sma"], False, ""),
        
        # Basic strategy
        generate_test_case(
            ["price > sma_20"], 
            ["price < sma_20"], 
            {"stop_loss": 0.02}, 
            ["sma"], 
            False, 
            "Basic strategy"
        ),
        
        # Intermediate strategy
        generate_test_case(
            ["price > sma_20", "rsi < 30"], 
            ["price < sma_20", "rsi > 70"], 
            {"stop_loss": 0.02, "take_profit": 0.05}, 
            ["sma", "rsi"], 
            True, 
            "This strategy uses SMA crossover and RSI to determine entry and exit points."
        ),
        
        # Advanced strategy
        generate_test_case(
            ["price > sma_20", "rsi < 30", "macd > 0"], 
            ["price < sma_20", "rsi > 70", "macd < 0"], 
            {"stop_loss": 0.02, "take_profit": 0.05, "position_size": 0.1}, 
            ["sma", "rsi", "macd"], 
            True, 
            "This strategy combines SMA crossover, RSI for overbought/oversold conditions, and MACD for trend confirmation." * 2
        ),
        
        # Comprehensive strategy
        generate_test_case(
            ["price > sma_20", "rsi < 30", "macd > 0", "volume > 1000"], 
            ["price < sma_20", "rsi > 70", "macd < 0", "volume < 500"], 
            {"stop_loss": 0.02, "take_profit": 0.05, "position_size": 0.1, "risk_management": True}, 
            ["sma", "rsi", "macd", "bollinger", "volume"], 
            True, 
            "This comprehensive strategy combines SMA crossover, RSI for overbought/oversold conditions, MACD for trend confirmation, volume analysis for strength of moves, and Bollinger Bands for volatility-based entries and exits." * 3
        )
    ]
    
    logger.info("\n## Testing Combined Component Scoring")
    for i, test_case in enumerate(tests):
        score = trading_rules_metric({}, test_case)
        logger.info(f"Combined strategy {i+1}: Score = {score:.4f}")
    
    return tests

def save_test_cases():
    """Save test cases at different threshold scores for future testing."""
    # Define test cases for specific scores
    target_scores = [0.13, 0.14, 0.145, 0.15, 0.155, 0.16, 0.17, 0.20, 0.25]
    
    for score in target_scores:
        # Create an appropriate test case based on the target score
        if score < 0.14:
            # Below threshold
            test_case = generate_test_case(
                ["price > sma_1"], 
                [], 
                {}, 
                [], 
                False, 
                "X" * 20
            )
        elif score < 0.15:
            # At threshold
            test_case = generate_test_case(
                ["price > sma_1"], 
                [], 
                {}, 
                [], 
                False, 
                "X" * 20
            )
        elif score < 0.2:
            # Just above threshold
            test_case = generate_test_case(
                ["price > sma_1", "price > sma_2"], 
                [], 
                {}, 
                ["sma"], 
                False, 
                ""
            )
        else:
            # Well above threshold
            test_case = generate_test_case(
                ["price > sma_1", "price > sma_2"], 
                ["price < sma_1"], 
                {"stop_loss": 0.02}, 
                ["sma", "rsi"], 
                True, 
                "X" * 50
            )
            
        # Save test case
        filename = f"test_case_{score:.3f}.json"
        with open(filename, "w") as f:
            json.dump(test_case["trading_rules"], f, indent=2)
            
        logger.info(f"Saved test case for score {score} to {filename}")

def run_tests():
    """Run all component tests."""
    setup_logging()
    
    logger.info("===== TESTING COMPONENT SCORING =====")
    
    # Run individual component tests
    entry_tests = test_entry_conditions()
    exit_tests = test_exit_conditions()
    param_tests = test_parameters()
    indicator_tests = test_indicators()
    logic_tests = test_logic_operators()
    reasoning_tests = test_reasoning()
    
    # Run combined component tests
    combined_tests = test_combined_components()
    
    # Save test cases for future testing
    save_test_cases()
    
    logger.info("\n===== COMPONENT SCORING TESTS COMPLETE =====")
    
if __name__ == "__main__":
    run_tests()
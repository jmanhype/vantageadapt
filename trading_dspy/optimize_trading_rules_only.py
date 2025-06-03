#!/usr/bin/env python3

"""
Script to run an isolated optimization of trading rules only.
This allows us to test the component scoring and plateau-breaking mechanism 
without running the entire pipeline.
"""

import os
import sys
import time
import dspy
import json
from loguru import logger
from dotenv import load_dotenv

from src.utils.mipro_optimizer import MiProWrapper
from src.utils.prompt_manager import PromptManager
from src.modules.trading_rules import TradingRulesGenerator
# Define trading_rules_metric here - function is embedded in the PromptOptimizer.optimize_trading_rules method
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
        logger.debug(f"Score {pre_plateau_score:.4f} below plateau threshold {plateau_threshold:.2f}, no bonus applied")
                
    # Ensure score is within bounds
    final_score = max(0.0, min(1.0, final_score))
    
    # Debug log to help monitor scoring progress
    component_debug = f"E:{entry_score:.2f}+X:{exit_score:.2f}+P:{parameter_score:.2f}+I:{indicator_score:.2f}+L:{logic_score:.2f}+R:{reasoning_score:.2f}"
    logger.info(f"Rule metric: {final_score:.4f} [{component_debug}] Pre: {pre_plateau_score:.4f}")
            
    return final_score

# Load environment variables from .env file
load_dotenv()

# Set up logging - forward to console for visibility
logger.remove()
logger.add(sys.stdout, level="DEBUG")
logger.add("logs/optimize_trading_rules.log", rotation="1 day", level="DEBUG")

def enhance_logging():
    """Set up enhanced logging for the optimization process."""
    logger.warning("Enhanced logging is already set up - using local trading_rules_metric")
    return None

def run_isolated_optimization():
    """Run optimization of trading rules only."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        return
    
    # Initialize DSPy
    logger.info("Initializing DSPy with model: gpt-4-turbo-preview")
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
    
    # Enhance logging for more visibility into component scores
    original_metric = enhance_logging()
    
    # Sample some examples to understand their structure
    logger.info("Example structure sample:")
    for i, ex in enumerate(trading_rules_examples[:2]):
        logger.info(f"Example {i+1} keys: {list(ex.keys())}")
        if 'outputs' in ex:
            logger.info(f"Example {i+1} output keys: {list(ex['outputs'].keys())}")

    # Initialize the MiPro wrapper with our enhanced parameters
    logger.info("Initializing MiProWrapper")
    mipro = MiProWrapper(
        prompt_manager=prompt_manager,
        use_v2=True,
        max_bootstrapped_demos=3,  # Reduced for faster testing
        num_candidate_programs=5,  # Reduced for faster testing
        temperature=0.8  # Increased from 0.7 to 0.8 for more diversity
    )
    
    # Initialize the trading rules generator
    logger.info("Initializing TradingRulesGenerator")
    trading_rules_generator = TradingRulesGenerator(prompt_manager=prompt_manager)
    
    # Run optimization
    logger.info("Starting trading rules optimization")
    start_time = time.time()
    
    try:
        # Use a tiny subset of examples for faster testing
        # Adjust the number based on the time you have for testing
        # For thorough testing, use all examples
        num_examples = min(5, len(trading_rules_examples))
        test_examples = trading_rules_examples[:num_examples]
        logger.info(f"Using {num_examples} examples for optimization test")
        
        # Run the optimization
        optimized_module = mipro.optimize(
            module=trading_rules_generator,
            examples=test_examples,
            prompt_name="trading_rules",
            metric_fn=trading_rules_metric
        )
        
        # Log the optimization duration
        duration = time.time() - start_time
        logger.info(f"Trading rules optimization completed in {duration:.2f} seconds")
        
        # Save the optimized prompt
        if hasattr(optimized_module, 'prompt'):
            optimized_prompt = getattr(optimized_module, 'prompt')
            if optimized_prompt and isinstance(optimized_prompt, str):
                with open("optimized_trading_rules_prompt.txt", "w") as f:
                    f.write(optimized_prompt)
                logger.info("Saved optimized trading rules prompt to optimized_trading_rules_prompt.txt")
                
        # Restore original metric function if we enhanced it
        if original_metric:
            import src.modules.prompt_optimizer
            src.modules.prompt_optimizer.trading_rules_metric = original_metric
            logger.info("Restored original trading_rules_metric function")
            
        return optimized_module
            
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        logger.exception("Full traceback:")
        
        # Restore original metric function if we enhanced it
        if original_metric:
            import src.modules.prompt_optimizer
            src.modules.prompt_optimizer.trading_rules_metric = original_metric
            logger.info("Restored original trading_rules_metric function")
            
        return None

if __name__ == "__main__":
    run_isolated_optimization()
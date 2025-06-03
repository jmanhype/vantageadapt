#!/usr/bin/env python3

"""
Script to test the trading rules metric integration in a more focused way.
This script loads existing examples and runs optimization with our improved scoring.
"""

import os
import sys
import json
import random
import dspy
from loguru import logger
from dotenv import load_dotenv

from src.modules.trading_rules import TradingRulesGenerator
from src.utils.prompt_manager import PromptManager
from src.modules.prompt_optimizer import PromptOptimizer

# Load environment variables from .env file
load_dotenv()

# Set up logging for better visibility
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/integration_test.log", level="DEBUG")

def test_trading_rules_optimization():
    """Test the trading rules optimization with our improved component scoring."""
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        return False
    
    # Initialize DSPy
    logger.info("Initializing DSPy with model: gpt-4-turbo-preview")
    lm = dspy.LM("gpt-4-turbo-preview", api_key=api_key)
    dspy.configure(lm=lm)
    
    # Initialize prompt manager
    logger.info("Initializing PromptManager")
    prompt_manager = PromptManager("prompts")
    
    # Get examples
    trading_rules_examples = prompt_manager.get_examples("trading_rules")
    
    # Sample examples for quicker testing
    num_examples = min(10, len(trading_rules_examples))
    test_examples = random.sample(trading_rules_examples, num_examples)
    logger.info(f"Selected {num_examples} examples for optimization test")
    
    # Initialize prompt optimizer
    logger.info("Initializing PromptOptimizer with enhanced parameters")
    prompt_optimizer = PromptOptimizer(prompt_manager=prompt_manager)
    
    # Increase the parameters for better optimization
    prompt_optimizer.mipro.max_bootstrapped_demos = 5  # Increased from 3 to 5
    prompt_optimizer.mipro.num_candidate_programs = 15  # Increased from 10 to 15
    prompt_optimizer.mipro.temperature = 0.8  # Increased from 0.7 to 0.8
    
    # Initialize trading rules generator
    logger.info("Initializing TradingRulesGenerator")
    trading_rules_generator = TradingRulesGenerator(prompt_manager=prompt_manager)
    
    # Run optimization with our enhanced metric (component scoring + plateau breaking)
    logger.info("Starting trading rules optimization with enhanced metric")
    
    try:
        # Optimize with our enhanced metric
        optimized_module = prompt_optimizer.optimize_trading_rules(
            module=trading_rules_generator,
            examples=test_examples
        )
        
        # Print the optimized module's prompt
        if hasattr(optimized_module, 'prompt'):
            logger.info("Optimization completed successfully, saving optimized prompt")
            with open("optimized_trading_rules_prompt.txt", "w") as f:
                f.write(optimized_module.prompt)
            
            # Test the optimized module with a test input
            test_input = {
                "strategy_insights": {
                    "summary": "The market shows signs of a bullish reversal pattern after a period of consolidation.",
                    "trade_signal": "BUY",
                    "confidence": 0.85,
                    "parameters": {
                        "stop_loss": 0.03,
                        "take_profit": 0.08
                    }
                },
                "market_context": {
                    "regime": "BULLISH_REVERSAL",
                    "risk_level": "moderate"
                }
            }
            
            # Execute the optimized module
            logger.info("Testing optimized module with sample input")
            result = optimized_module(
                strategy_insights=test_input["strategy_insights"],
                market_context=test_input["market_context"],
                performance_analysis=None
            )
            
            # Log the result
            logger.info("Optimized module output:")
            if hasattr(result, "trading_rules"):
                rules = result.trading_rules
                
                # Extract and log entry conditions
                entry_conditions = rules.get("entry_conditions", rules.get("conditions", {}).get("entry", []))
                if entry_conditions:
                    logger.info(f"Entry conditions: {entry_conditions}")
                
                # Extract and log exit conditions
                exit_conditions = rules.get("exit_conditions", rules.get("conditions", {}).get("exit", []))
                if exit_conditions:
                    logger.info(f"Exit conditions: {exit_conditions}")
                
                # Extract and log parameters
                parameters = rules.get("parameters", {})
                if parameters:
                    logger.info(f"Parameters: {parameters}")
                
                # Extract and log reasoning
                reasoning = rules.get("reasoning", "")
                if reasoning:
                    logger.info(f"Reasoning: {reasoning[:200]}...")
                
                return True
            else:
                logger.error("Optimized module did not return trading_rules")
                return False
            
        else:
            logger.warning("Optimized module has no prompt attribute")
            return False
            
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    success = test_trading_rules_optimization()
    if success:
        logger.info("Integration test completed successfully")
    else:
        logger.error("Integration test failed")
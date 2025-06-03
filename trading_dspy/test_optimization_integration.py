"""Integration test for full optimization run with the enhanced plateau-breaking mechanism."""

import sys
import os
import time
from loguru import logger
import dspy
import json
import copy

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.prompt_optimizer import PromptOptimizer
from src.modules.trading_rules import TradingRulesGenerator
from src.utils.prompt_manager import PromptManager
from src.utils.memory_manager import TradingMemoryManager
from src.utils.mipro_optimizer import MiProWrapper

def setup_logging():
    """Set up enhanced logging for tests."""
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.configure(handlers=[
        {"sink": sys.stdout, "format": log_format, "level": "INFO"},
        {"sink": "optimization_integration_test.log", "format": log_format, "level": "DEBUG"}
    ])

def load_trading_rules_examples(file_path="test_cases_for_optimization.json"):
    """Load test cases for the optimization."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                examples = json.load(f)
                return examples
        else:
            # Generate examples if file doesn't exist
            examples = []
            
            for score in [0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.20, 0.25]:
                entry_conditions = []
                exit_conditions = []
                parameters = {}
                indicators = []
                logic = False
                reasoning = ""
                
                # Add complexity based on score
                if score >= 0.10:
                    entry_conditions.append("price > sma_20")
                    indicators.append("sma")
                if score >= 0.14:
                    exit_conditions.append("price < sma_20")
                    parameters = {"stop_loss": 0.02}
                if score >= 0.15:
                    entry_conditions.append("rsi < 30")
                    indicators.append("rsi")
                    logic = True
                if score >= 0.20:
                    exit_conditions.append("rsi > 70")
                    parameters = {"stop_loss": 0.02, "take_profit": 0.05}
                    reasoning = "This is a simple SMA crossover strategy with RSI confirmation."
                
                # Create a realistic example
                example = {
                    "strategy": {
                        "trade_signal": "BUY" if score >= 0.14 else "HOLD",
                        "confidence": 0.7 if score >= 0.14 else 0.3,
                        "reasoning": "Market analysis indicates potential uptrend" if score >= 0.14 else "Waiting for confirmation"
                    },
                    "prompt": "Generate trading rules for a trend following strategy",
                    "outputs": {
                        "trading_rules": {
                            "entry_conditions": entry_conditions,
                            "exit_conditions": exit_conditions,
                            "parameters": parameters,
                            "indicators": indicators,
                            "reasoning": reasoning * int(score * 100) if score >= 0.20 else reasoning,
                            "trade_signal": "BUY" if score >= 0.14 else "HOLD"
                        }
                    }
                }
                
                examples.append(example)
            
            # Save examples for future use
            with open(file_path, "w") as f:
                json.dump(examples, f, indent=2)
            
            return examples
    except Exception as e:
        logger.error(f"Error loading trading rules examples: {str(e)}")
        return []

def run_optimization_test():
    """Run a full optimization test with the enhanced plateau-breaking mechanism."""
    setup_logging()
    
    logger.info("===== STARTING OPTIMIZATION INTEGRATION TEST =====")
    
    # Initialize DSPy with OpenAI model
    dspy.configure(lm=dspy.OpenAI(model="gpt-4-turbo-preview"))
    
    # Initialize PromptManager
    prompt_manager = PromptManager()
    
    # Initialize TradingMemoryManager with test mode
    memory_manager = TradingMemoryManager(test_mode=True)
    
    # Initialize TradingRulesGenerator
    trading_rules_generator = TradingRulesGenerator(
        prompt_manager=prompt_manager,
        memory_manager=memory_manager
    )
    
    # Load examples
    examples = load_trading_rules_examples()
    logger.info(f"Loaded {len(examples)} examples for optimization")
    
    # Add examples to prompt manager
    for i, example in enumerate(examples):
        prompt_manager.add_example("trading_rules", example)
        logger.debug(f"Added example {i+1}: {example['outputs']['trading_rules'].get('entry_conditions', [])} -> score ~{0.08 + 0.02*i}")
    
    # Initialize PromptOptimizer with MiPro
    optimizer = PromptOptimizer(prompt_manager=prompt_manager)
    
    # Start optimization
    logger.info("Starting MiPro optimization for trading rules with enhanced plateau breaking")
    start_time = time.time()
    
    try:
        # Run optimization with limited bootstrapped examples and iterations for testing
        optimized_module = optimizer.optimize_trading_rules(
            module=trading_rules_generator,
            examples=examples[:5]  # Use a subset of examples for faster testing
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Optimization completed in {duration:.2f} seconds")
        
        # Test the optimized module
        logger.info("Testing optimized module with a simple input")
        test_input = {
            "market_context": {
                "regime": "trending",
                "confidence": 0.8,
                "risk_level": "moderate",
                "analysis_text": "Market is in a strong uptrend with increasing volume."
            },
            "theme": "trend_following"
        }
        
        # Run the optimized model
        result = optimized_module(
            market_context=test_input["market_context"],
            theme=test_input["theme"],
            base_parameters={"stop_loss": 0.02, "take_profit": 0.05}
        )
        
        # Check the result
        if result:
            logger.info("Optimized module generated a result successfully")
            logger.info(f"Entry conditions: {result.entry_conditions}")
            logger.info(f"Exit conditions: {result.exit_conditions}")
            logger.info(f"Parameters: {result.parameters}")
            
            # Validate that the result is reasonable
            if result.entry_conditions and result.exit_conditions:
                logger.info("✅ Result contains both entry and exit conditions")
            else:
                logger.warning("❌ Result is missing entry or exit conditions")
                
            if result.parameters and 'stop_loss' in result.parameters and 'take_profit' in result.parameters:
                logger.info("✅ Result contains required parameters")
            else:
                logger.warning("❌ Result is missing required parameters")
        else:
            logger.error("Optimized module failed to generate a result")
            
        return optimized_module
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        logger.exception("Full traceback:")
        return None

if __name__ == "__main__":
    optimized_module = run_optimization_test()
    if optimized_module:
        logger.info("Integration test completed successfully")
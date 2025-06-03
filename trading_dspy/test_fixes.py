"""Test script to verify our fixes."""

import sys
from pathlib import Path
from loguru import logger
import json

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/test_fixes.log", level="DEBUG")

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the modules we're testing
from src.modules.market_analysis import MarketAnalyzer
from src.modules.strategy_generator import StrategyGenerator
from src.utils.prompt_manager import PromptManager
from src.utils.memory_manager import TradingMemoryManager

def test_empty_volumes():
    """Test handling of empty volumes list in market analysis."""
    logger.info("Testing handling of empty volumes list")
    
    # Create a prompt manager
    prompt_manager = PromptManager(prompts_dir="prompts")
    
    # Create a market analyzer
    market_analyzer = MarketAnalyzer(prompt_manager)
    
    # Create test data with empty volumes
    market_data = {
        "prices": [100, 101, 102, 103, 104],
        "volumes": [],
        "indicators": {
            "sma_20": [99, 99.5, 100, 100.5, 101],
            "sma_50": [98, 98.5, 99, 99.5, 100]
        }
    }
    
    # Test preparing market data
    try:
        processed_data = market_analyzer._prepare_market_data(market_data)
        logger.info("Successfully processed market data with empty volumes")
        logger.info(f"Summary data: {processed_data['summary']}")
        return True
    except Exception as e:
        logger.error(f"Failed to process market data: {e}")
        return False

def test_parameter_ranges():
    """Test handling of missing parameter_ranges in strategy generation."""
    logger.info("Testing handling of missing parameter_ranges")
    
    # Create managers
    prompt_manager = PromptManager(prompts_dir="prompts")
    memory_manager = TradingMemoryManager()
    
    # Create a strategy generator
    strategy_generator = StrategyGenerator(prompt_manager, memory_manager)
    
    # Create a test strategy without parameter_ranges
    test_strategy = {
        "reasoning": "Test reasoning",
        "trade_signal": "HOLD",
        "parameters": {
            "stop_loss": 0.05,
            "take_profit": 0.1,
            "position_size": 0.2
        },
        "confidence": 0.7,
        "entry_conditions": ["price > sma_20"],
        "exit_conditions": ["price < sma_20"],
        "indicators": ["sma_20"]
    }
    
    # Validate the strategy
    try:
        valid, reason = strategy_generator.validate_strategy(test_strategy)
        logger.info(f"Strategy validation result: valid={valid}, reason={reason}")
        
        # Check if parameter_ranges was added
        if 'parameter_ranges' in test_strategy:
            logger.info(f"parameter_ranges was added: {test_strategy['parameter_ranges']}")
            return True
        else:
            logger.error("parameter_ranges was not added to the strategy")
            return False
    except Exception as e:
        logger.error(f"Failed to validate strategy: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting fix verification tests")
    
    # Run the tests
    empty_volumes_passed = test_empty_volumes()
    parameter_ranges_passed = test_parameter_ranges()
    
    # Print results
    if empty_volumes_passed:
        logger.info("✅ Empty volumes test passed")
    else:
        logger.error("❌ Empty volumes test failed")
        
    if parameter_ranges_passed:
        logger.info("✅ Parameter ranges test passed")
    else:
        logger.error("❌ Parameter ranges test failed")
        
    if empty_volumes_passed and parameter_ranges_passed:
        logger.info("All tests passed! Fixes are working correctly.")
    else:
        logger.error("Some tests failed. Fixes need more work.")
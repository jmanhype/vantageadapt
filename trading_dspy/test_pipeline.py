"""Test script to verify our fixes in a simplified pipeline."""

import sys
from pathlib import Path
from loguru import logger
import json
import dspy
import random
import os
from dotenv import load_dotenv

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/test_pipeline.log", level="DEBUG")

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

# Import the modules we're testing
from src.modules.market_analysis import MarketAnalyzer
from src.modules.strategy_generator import StrategyGenerator
from src.modules.trading_rules import TradingRulesGenerator
from src.utils.prompt_manager import PromptManager
from src.utils.memory_manager import TradingMemoryManager
from src.utils.mipro_optimizer import MiProWrapper

# Configure DSPy with API key
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Different ways to configure OpenAI in different DSPy versions
        try:
            # Try the newer API
            from dspy.backends.openai import OpenAI
            lm = OpenAI(api_key=api_key, model="gpt-3.5-turbo")
            dspy.settings.configure(lm=lm)
            logger.info("Configured DSPy with OpenAI using new API")
        except ImportError:
            # Fall back to older API
            try:
                import openai
                openai.api_key = api_key
                dspy.settings.configure(lm="gpt-3.5-turbo")
                logger.info("Configured DSPy with OpenAI using older API")
            except Exception as e:
                logger.error(f"Failed to configure OpenAI: {e}")
    else:
        logger.warning("No OpenAI API key found in environment")
except Exception as e:
    logger.error(f"Error setting up DSPy with OpenAI: {e}")
    # Continue without LLM - we can still test our fixes for empty volumes and parameter_ranges

def generate_market_data(with_volumes=True):
    """Generate synthetic market data for testing."""
    prices = [100 + i + random.random() for i in range(50)]
    volumes = [1000 + random.randint(-200, 200) for _ in range(50)] if with_volumes else []
    
    # Calculate some basic indicators
    sma_20 = []
    for i in range(50):
        if i < 20:
            sma_20.append(sum(prices[:i+1]) / (i+1))
        else:
            sma_20.append(sum(prices[i-19:i+1]) / 20)
    
    sma_50 = []
    for i in range(50):
        if i < 50:
            sma_50.append(sum(prices[:i+1]) / (i+1))
        else:
            sma_50.append(sum(prices[i-49:i+1]) / 50)
    
    return {
        "prices": prices,
        "volumes": volumes,
        "indicators": {
            "sma_20": sma_20,
            "sma_50": sma_50,
            "volatility": [0.02 + random.random() * 0.01 for _ in range(50)]
        }
    }

def test_simplified_pipeline():
    """Run a simplified pipeline to test our fixes."""
    logger.info("Testing simplified pipeline with our fixes")
    
    # Create managers
    prompt_manager = PromptManager(prompts_dir="prompts")
    memory_manager = TradingMemoryManager()
    
    # Generate test market data
    market_data = generate_market_data(with_volumes=True)
    empty_volumes_data = generate_market_data(with_volumes=False)
    
    # Create a malformed market data to test our robustness
    bad_market_data = {
        "summary": {
            "prices": [100, 101, 102, 103, 104]
        }
    }
    
    # 1. Test market analysis with empty volumes
    logger.info("1. Testing market analyzer with empty volumes")
    market_analyzer = MarketAnalyzer(prompt_manager)
    
    # Test the _prepare_market_data method directly
    try:
        # Set is_optimizing to focus just on the data preparation
        market_analyzer.is_optimizing = True
        
        # Process both normal and empty volume data
        processed_with_volumes = market_analyzer._prepare_market_data(market_data)
        processed_empty_volumes = market_analyzer._prepare_market_data(empty_volumes_data)
        
        logger.info(f"Market data with volumes: avg_volume={processed_with_volumes['summary']['avg_volume']}")
        logger.info(f"Market data without volumes: avg_volume={processed_empty_volumes['summary']['avg_volume']}")
        
        volumes_test_passed = processed_empty_volumes['summary']['avg_volume'] == 0.0
        logger.info(f"Empty volumes test passed: {volumes_test_passed}")
        
        # 1.1. Test MarketRegimeClassifier with robustness fixes
        logger.info("1.1 Testing MarketRegimeClassifier with nested data and missing indicators")
        regime_classifier = market_analyzer.regime_classifier
        
        # Test with regular data
        regime_result = regime_classifier.forward(market_data)
        logger.info(f"Regular data regime: {regime_result['market_context']['regime']}")
        
        # Test with empty volumes data
        regime_result_empty = regime_classifier.forward(empty_volumes_data)
        logger.info(f"Empty volumes regime: {regime_result_empty['market_context']['regime']}")
        
        # Test with malformed data - should handle gracefully
        try:
            regime_result_bad = regime_classifier.forward(bad_market_data)
            logger.info(f"Malformed data regime: {regime_result_bad['market_context']['regime']}")
            regime_test_passed = True
        except Exception as e:
            logger.error(f"MarketRegimeClassifier failed on malformed data: {e}")
            regime_test_passed = False
            
        volumes_test_passed = volumes_test_passed and regime_test_passed
        
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
        volumes_test_passed = False
    
    # 2. Test strategy generation with parameter_ranges
    logger.info("2. Testing strategy generator parameter_ranges fix")
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
    
    try:
        # Test validation directly
        valid, reason = strategy_generator.validate_strategy(test_strategy)
        
        logger.info(f"Strategy validation result: valid={valid}, reason={reason}")
        
        # Check if parameter_ranges was added
        if 'parameter_ranges' in test_strategy:
            logger.info(f"Strategy includes parameter_ranges: {test_strategy['parameter_ranges']}")
            params_test_passed = True
        else:
            logger.error("Strategy missing parameter_ranges")
            params_test_passed = False
    except Exception as e:
        logger.error(f"Strategy validation error: {e}")
        params_test_passed = False
    
    # Return combined test results
    return volumes_test_passed and params_test_passed

if __name__ == "__main__":
    logger.info("Starting pipeline test")
    
    # Run the test
    try:
        pipeline_passed = test_simplified_pipeline()
        
        # Print results
        if pipeline_passed:
            logger.info("✅ Pipeline test passed!")
        else:
            logger.error("❌ Pipeline test failed!")
            
    except Exception as e:
        logger.error(f"Pipeline test error: {e}")
        
    logger.info("Test completed")
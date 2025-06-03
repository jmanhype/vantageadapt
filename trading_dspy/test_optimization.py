"""Test script to verify our optimization fixes."""

import sys
from pathlib import Path
from loguru import logger
import json
import dspy

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/test_optimization.log", level="DEBUG")

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the modules we're testing
from src.utils.mipro_optimizer import MiProWrapper
from src.utils.prompt_manager import PromptManager
from src.modules.market_analysis import MarketAnalyzer
from src.modules.trading_rules import TradingRulesGenerator
from src.utils.memory_manager import TradingMemoryManager

# Simple metric function for testing
def simple_metric(gold, pred):
    """Basic metric function that just checks if fields exist."""
    score = 0.0
    
    # Check if we have the basic fields
    if hasattr(pred, 'regime') and pred.regime:
        score += 0.3
        
    if hasattr(pred, 'confidence') and 0 <= float(pred.confidence) <= 1:
        score += 0.2
        
    if hasattr(pred, 'risk_level') and pred.risk_level:
        score += 0.2
        
    if hasattr(pred, 'analysis') and len(pred.analysis) > 50:
        score += 0.3
        
    return score

# Example data
example_market_data = {
    "prices": [100, 101, 102, 103, 104],
    "volumes": [1000, 1100, 900, 1050, 1200],
    "indicators": {
        "sma_20": [99, 99.5, 100, 100.5, 101],
        "sma_50": [98, 98.5, 99, 99.5, 100],
        "rsi": [45, 48, 52, 55, 58]
    }
}

example_empty_volumes = {
    "prices": [100, 101, 102, 103, 104],
    "volumes": [],
    "indicators": {
        "sma_20": [99, 99.5, 100, 100.5, 101],
        "sma_50": [98, 98.5, 99, 99.5, 100],
        "rsi": [45, 48, 52, 55, 58]
    }
}

# Create example objects
examples = [
    {
        "market_data": example_market_data,
        "timeframe": "1h",
        "outputs": {
            "regime": "TRENDING_BULLISH",
            "confidence": 0.8,
            "risk_level": "low",
            "analysis": "The market is trending bullish with prices consistently above both the 20 and 50 period moving averages. RSI shows moderate strength without being overbought."
        }
    },
    {
        "market_data": example_empty_volumes,
        "timeframe": "4h",
        "outputs": {
            "regime": "RANGING_LOW_VOL",
            "confidence": 0.6,
            "risk_level": "moderate",
            "analysis": "The market is in a low volatility range with prices moving sideways. Volume data is missing which adds uncertainty to the analysis."
        }
    }
]

def test_mipro_optimization():
    """Test MiPro optimization with our fixes."""
    logger.info("Testing MiPro optimization with empty volumes and parameter_ranges fixes")
    
    # Create a prompt manager
    prompt_manager = PromptManager(prompts_dir="prompts")
    
    # Create a market analyzer
    market_analyzer = MarketAnalyzer(prompt_manager)
    
    # Create MiPro wrapper
    mipro_wrapper = MiProWrapper(
        prompt_manager=prompt_manager,
        use_v2=True,
        max_bootstrapped_demos=2,
        num_candidate_programs=3,
        temperature=0.7
    )
    
    # Create test data with empty volumes
    examples_with_empty = [
        {
            "market_data": example_empty_volumes,
            "timeframe": "1h",
            "outputs": {
                "regime": "TRENDING_BULLISH",
                "confidence": 0.8,
                "risk_level": "low",
                "analysis": "The market is trending bullish with prices consistently above both the 20 and 50 period moving averages."
            }
        }
    ]
    
    # Test optimization with empty volumes data
    try:
        # Run a test with empty volumes
        result_module = mipro_wrapper.optimize(
            module=market_analyzer,
            examples=examples_with_empty,
            prompt_name="market_analysis_test",
            metric_fn=simple_metric
        )
        
        # Verify the module was returned without crashing
        if result_module is not None:
            logger.info("Optimization handled empty volumes without crashing")
            return True
        else:
            logger.error("Optimization returned None")
            return False
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return False

def test_trading_rules_generator():
    """Test TradingRulesGenerator with fixed parameter_ranges handling."""
    logger.info("Testing TradingRulesGenerator with fixed parameter_ranges handling")
    
    # Create managers
    prompt_manager = PromptManager(prompts_dir="prompts")
    memory_manager = TradingMemoryManager()
    
    # Create a trading rules generator
    trading_rules_generator = TradingRulesGenerator(prompt_manager, memory_manager)
    
    # Test schema with empty parameter_ranges
    test_data = {
        "strategy_insights": {
            "reasoning": "Test reasoning",
            "trade_signal": "BUY",
            "confidence": 0.8
        },
        "market_context": {
            "regime": "TRENDING_BULLISH",
            "confidence": 0.7,
            "risk_level": "low"
        }
    }
    
    try:
        # Run a forward pass
        result = trading_rules_generator.forward(
            strategy_insights=test_data["strategy_insights"],
            market_context=test_data["market_context"]
        )
        
        logger.info("TradingRulesGenerator test completed successfully")
        if "parameter_ranges" in result:
            logger.info(f"parameter_ranges found in result: {result['parameter_ranges']}")
            return True
        else:
            logger.error("parameter_ranges missing from result")
            return False
    except Exception as e:
        logger.error(f"TradingRulesGenerator test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting optimization verification tests")
    
    # Run the tests
    try:
        mipro_passed = test_mipro_optimization()
        logger.info("MiPro test completed")
    except Exception as e:
        logger.error(f"MiPro test error: {e}")
        mipro_passed = False
    
    # Print results
    if mipro_passed:
        logger.info("✅ MiPro optimization test passed")
    else:
        logger.error("❌ MiPro optimization test failed")
        
    logger.info("All tests completed")
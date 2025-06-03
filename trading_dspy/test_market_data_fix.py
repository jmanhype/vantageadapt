"""Test for the market data fix to handle missing indicators field."""

import sys
import os
from loguru import logger
import json

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.market_analysis import MarketRegimeClassifier

def setup_logging():
    """Set up enhanced logging for tests."""
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.configure(handlers=[
        {"sink": sys.stdout, "format": log_format, "level": "INFO"},
        {"sink": "market_data_fix_test.log", "format": log_format, "level": "DEBUG"}
    ])

def test_market_data_fix():
    """Test the fix for handling missing indicators field."""
    setup_logging()
    
    logger.info("===== TESTING MARKET DATA FIX =====")
    
    # Initialize MarketRegimeClassifier
    classifier = MarketRegimeClassifier()
    
    # Test case 1: Standard format with indicators field
    logger.info("Test case 1: Standard format with indicators field")
    market_data_1 = {
        "prices": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "volumes": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
        "indicators": {
            "sma_20": [103],
            "sma_50": [100],
            "rsi": [55]
        }
    }
    
    try:
        result_1 = classifier.forward(market_data=market_data_1)
        logger.info(f"Result 1: {result_1}")
        logger.info("✅ Test case 1 passed")
    except Exception as e:
        logger.error(f"❌ Test case 1 failed: {str(e)}")
    
    # Test case 2: Nested format with indicators under summary field
    logger.info("Test case 2: Nested format with indicators under summary field")
    market_data_2 = {
        "summary": {
            "prices": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "volumes": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
            "indicators": {
                "sma_20": [103],
                "sma_50": [100],
                "rsi": [55]
            }
        }
    }
    
    try:
        result_2 = classifier.forward(market_data=market_data_2)
        logger.info(f"Result 2: {result_2}")
        logger.info("✅ Test case 2 passed")
    except Exception as e:
        logger.error(f"❌ Test case 2 failed: {str(e)}")
    
    # Test case 3: Direct indicators without indicators field
    logger.info("Test case 3: Direct indicators without indicators field")
    market_data_3 = {
        "prices": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "volumes": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000],
        "sma_20": 103,
        "sma_50": 100,
        "rsi": 55,
        "volatility": 0.01
    }
    
    try:
        result_3 = classifier.forward(market_data=market_data_3)
        logger.info(f"Result 3: {result_3}")
        logger.info("✅ Test case 3 passed")
    except Exception as e:
        logger.error(f"❌ Test case 3 failed: {str(e)}")
    
    # Test case 4: Minimal data with just current price
    logger.info("Test case 4: Minimal data with just current price")
    market_data_4 = {
        "current_price": 110,
        "sma_20": 103,
        "sma_50": 100,
        "rsi": 55
    }
    
    try:
        result_4 = classifier.forward(market_data=market_data_4)
        logger.info(f"Result 4: {result_4}")
        logger.info("✅ Test case 4 passed")
    except Exception as e:
        logger.error(f"❌ Test case 4 failed: {str(e)}")
    
    logger.info("===== MARKET DATA FIX TESTING COMPLETE =====")

if __name__ == "__main__":
    test_market_data_fix()
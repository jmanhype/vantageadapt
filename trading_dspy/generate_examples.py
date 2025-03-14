#!/usr/bin/env python3

"""
Script to generate examples for MiPro optimization.
"""

import os
import json
import pandas as pd
import numpy as np
import dspy
from loguru import logger
from dotenv import load_dotenv

from src.utils.prompt_manager import PromptManager
from src.modules.market_analysis import MarketAnalyzer, MarketRegimeClassifier
from src.modules.strategy_generator import StrategyGenerator
from src.modules.trading_rules import TradingRulesGenerator
from src.modules.prompt_optimizer import PromptOptimizer
from src.utils.memory_manager import TradingMemoryManager
from src.modules.market_regime_enhanced import EnhancedMarketRegimeClassifier

# Load environment variables
load_dotenv()

def create_sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100)
    prices = np.cumsum(np.random.normal(0, 0.01, size=100)) + 10
    volumes = np.random.exponential(1, size=100) * 100
    
    # Create indicators
    sma_20 = pd.Series(prices).rolling(20).mean().fillna(prices[0])
    sma_50 = pd.Series(prices).rolling(50).mean().fillna(prices[0])
    rsi = np.random.uniform(30, 70, size=100)  # Simplified RSI
    volatility = np.random.uniform(0.001, 0.005, size=100)
    
    # Create market data
    market_data = {
        'prices': prices.tolist(),
        'volumes': volumes.tolist(),
        'indicators': {
            'sma_20': sma_20.tolist(),
            'sma_50': sma_50.tolist(),
            'rsi': rsi.tolist(),
            'volatility': volatility.tolist(),
            'buy_sell_ratio': np.random.uniform(0.9, 1.1, size=100).tolist(),
        }
    }
    
    return market_data

def main():
    """Run the example generator."""
    logger.info("Starting example generation")
    
    # Using API keys from .env file loaded by dotenv
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        return
    
    # Initialize DSPy
    logger.info("Initializing DSPy")
    lm = dspy.LM("gpt-4-turbo-preview", api_key=api_key)
    dspy.configure(lm=lm)
    
    # Initialize managers
    logger.info("Initializing managers")
    prompt_manager = PromptManager("prompts")
    memory_manager = TradingMemoryManager(api_key=os.getenv("MEM0_API_KEY"))
    
    # Initialize modules
    logger.info("Initializing modules")
    market_analyzer = MarketAnalyzer(prompt_manager)
    regime_classifier = EnhancedMarketRegimeClassifier()
    prompt_optimizer = PromptOptimizer(prompt_manager)
    
    # Create sample data
    market_data = create_sample_data()
    
    # Run market analysis
    logger.info("Running market analysis")
    regime_result = regime_classifier.forward(market_data=market_data, timeframe="1h")
    logger.info(f"Regime classification: {regime_result}")
    
    try:
        # Create and collect example manually
        logger.info("Creating market analysis example")
        market_context = {
            'regime': 'RANGING_LOW_VOL',
            'confidence': 0.7,
            'risk_level': 'moderate', 
            'analysis_text': 'Market showing low volatility with price action contained within a narrow range. RSI indicates neutral momentum, and volume is below average. Support and resistance levels appear to be holding.'
        }
        
        example = {
            'market_data': prompt_optimizer._make_serializable(market_data),
            'timeframe': '1h',
            'prompt': prompt_manager.get_prompt('market_analysis'),
            'outputs': {
                'regime': market_context['regime'],
                'confidence': market_context['confidence'],
                'risk_level': market_context['risk_level'],
                'analysis': market_context['analysis_text']
            }
        }
        
        # Add example
        prompt_manager.add_example('market_analysis', example)
        logger.info("Added market analysis example 1")
        
        # Create a second example
        market_context = {
            'regime': 'TRENDING_BULLISH',
            'confidence': 0.8,
            'risk_level': 'moderate', 
            'analysis_text': 'Market showing strong bullish momentum with price consistently making higher highs and higher lows. RSI above 60 indicates strong buying pressure, with volume increasing on up moves. Moving averages aligned in bullish formation with 20MA > 50MA.'
        }
        
        example = {
            'market_data': prompt_optimizer._make_serializable(market_data),
            'timeframe': '1h',
            'prompt': prompt_manager.get_prompt('market_analysis'),
            'outputs': {
                'regime': market_context['regime'],
                'confidence': market_context['confidence'],
                'risk_level': market_context['risk_level'],
                'analysis': market_context['analysis_text']
            }
        }
        
        # Add example
        prompt_manager.add_example('market_analysis', example)
        logger.info("Added market analysis example 2")
        
        # Create a third example
        market_context = {
            'regime': 'TRENDING_BEARISH',
            'confidence': 0.75,
            'risk_level': 'high', 
            'analysis_text': 'Market in bearish trend with consecutive lower highs and lower lows. RSI below 40 indicates significant selling pressure. Volume elevated on down moves, suggesting strong bearish momentum. Moving averages show bearish alignment (20MA < 50MA) and widening gap between them.'
        }
        
        example = {
            'market_data': prompt_optimizer._make_serializable(market_data),
            'timeframe': '1h',
            'prompt': prompt_manager.get_prompt('market_analysis'),
            'outputs': {
                'regime': market_context['regime'],
                'confidence': market_context['confidence'],
                'risk_level': market_context['risk_level'],
                'analysis': market_context['analysis_text']
            }
        }
        
        # Add example
        prompt_manager.add_example('market_analysis', example)
        logger.info("Added market analysis example 3")
        
        # Check example count
        optimization_status = prompt_optimizer.check_optimization_status()
        logger.info(f"Current optimization status: {optimization_status}")
        
    except Exception as e:
        logger.error(f"Error generating examples: {str(e)}")
        logger.exception("Full traceback:")
    
    # Output success message
    logger.info("Example generation completed")
    
if __name__ == "__main__":
    main()
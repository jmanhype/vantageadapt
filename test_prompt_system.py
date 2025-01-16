"""Test script to validate prompt management system."""

import asyncio
import json
import pandas as pd
import numpy as np
import logging
from research.strategy.llm_interface import LLMInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_prompt_system():
    """Test the prompt management system with basic functionality."""
    try:
        # Initialize LLM interface
        llm = await LLMInterface.create()
        
        # 1. Test prompt loading
        logger.info("Testing prompt loading...")
        all_prompts = llm.prompt_manager.get_all_prompt_keys()
        logger.info(f"Available prompts: {all_prompts}")
        
        # 2. Test market analysis
        logger.info("\nTesting market analysis...")
        # Create sample market data with 30 data points
        np.random.seed(42)  # For reproducibility
        n_points = 30
        price = 100 + np.random.randn(n_points).cumsum()
        market_data = pd.DataFrame({
            'price': price,
            'sol_volume': np.random.uniform(800, 1200, n_points)
        }, index=pd.date_range('2024-01-01', periods=n_points, freq='1H'))
        
        market_context = await llm.analyze_market(market_data)
        if market_context:
            logger.info(f"Market analysis successful: {json.dumps(market_context.to_dict(), indent=2)}")
        else:
            logger.error("Market analysis failed")
            
        # 3. Test strategy generation
        logger.info("\nTesting strategy generation...")
        strategy = await llm.generate_strategy("breakout trading")
        if strategy:
            logger.info(f"Strategy generation successful: {json.dumps(strategy.to_dict(), indent=2)}")
        else:
            logger.error("Strategy generation failed")
            
        # 4. Test trading rules generation
        if market_context and strategy:
            logger.info("\nTesting trading rules generation...")
            conditions, parameters = await llm.generate_trading_rules(strategy, market_context)
            if conditions and parameters:
                logger.info(f"Trading rules generation successful:")
                logger.info(f"Conditions: {json.dumps(conditions, indent=2)}")
                logger.info(f"Parameters: {json.dumps(parameters, indent=2)}")
            else:
                logger.error("Trading rules generation failed")
                
        # 5. Test performance analysis
        logger.info("\nTesting performance analysis...")
        metrics = {
            'total_return': 0.15,
            'win_rate': 0.6,
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.1,
            'profit_factor': 1.5
        }
        trade_memory_stats = {
            'avg_trade_duration': 120,
            'best_trade': 0.05,
            'worst_trade': -0.02,
            'consecutive_wins': 3,
            'consecutive_losses': 2
        }
        
        analysis = await llm.analyze_performance(metrics, trade_memory_stats)
        if analysis:
            logger.info(f"Performance analysis successful: {json.dumps(analysis, indent=2)}")
        else:
            logger.error("Performance analysis failed")
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_prompt_system()) 
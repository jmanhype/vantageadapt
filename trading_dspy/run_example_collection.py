"""Script to collect examples and run MiPro optimization."""

import os
import logging
from loguru import logger
import pickle
from pathlib import Path
import json
import time

from src.pipeline import TradingPipeline
from src.utils.data_preprocessor import preprocess_market_data
from src.utils.prompt_manager import PromptManager

def load_pickle_data(file_path: str) -> dict:
    """Load data from a pickle file."""
    try:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Data loaded successfully. Found {len(data)} tokens.")
        
        # Print column names for debugging
        logger.info(f"Available columns for $MICHI: {data['$MICHI'].columns.tolist()}")
        
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def run_example_collection():
    """Run multiple iterations to collect examples."""
    # Configure logging
    logger.add("logs/example_collection.log", rotation="1 day")
    logger.info("Starting example collection and MiPro optimization")
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Load market data
    trade_data = load_pickle_data("/Users/speed/StratOptimv4/big_optimize_1016.pkl")
    
    # Initialize pipeline with MiPro enabled
    pipeline = TradingPipeline(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview",
        performance_thresholds={
            'min_return': 0.10,
            'min_trades': 10,
            'max_drawdown': 0.20
        },
        use_prompt_optimization=True  # Enable MiPro
    )
    
    # Preprocess market data
    preprocessed_data = preprocess_market_data(trade_data["$MICHI"])
    
    # Run multiple iterations to collect examples
    logger.info("Running multiple iterations to collect examples")
    
    for i in range(4):  # Run 4 iterations
        logger.info(f"Starting iteration {i+1} of 4")
        
        # Run pipeline with preprocessed data
        results = pipeline.run(
            market_data=preprocessed_data,
            num_iterations=1,  # Just one iteration per run
            timeframe="1h"
        )
        
        # Check optimization status
        status = pipeline.prompt_optimizer.check_optimization_status()
        logger.info(f"Current example counts:")
        for name, stats in status.items():
            logger.info(f"  {name}: {stats['example_count']} examples")
        
        # Sleep to avoid rate limiting
        logger.info("Waiting 5 seconds before next iteration")
        time.sleep(5)
    
    # Force optimization for market analysis
    logger.info("Forcing optimization for market analysis")
    prompt_manager = PromptManager("prompts")
    market_analysis_examples = prompt_manager.get_examples("market_analysis")
    
    if len(market_analysis_examples) >= 3:
        logger.info(f"Running market analysis optimization with {len(market_analysis_examples)} examples")
        pipeline.market_analyzer = pipeline.prompt_optimizer.optimize_market_analysis(
            pipeline.market_analyzer, 
            market_analysis_examples
        )
        
        # Check if optimization was successful
        optimized_exists = os.path.exists("prompts/optimized/market_analysis.txt")
        if optimized_exists:
            logger.info("Market analysis optimization successful!")
            with open("prompts/optimized/market_analysis.txt", "r") as f:
                optimized_prompt = f.read()
                logger.info(f"Optimized prompt (first 100 chars): {optimized_prompt[:100]}...")
        else:
            logger.warning("Market analysis optimization may have failed - no optimized prompt found")
    else:
        logger.warning(f"Not enough market analysis examples for optimization, need at least 3 (found {len(market_analysis_examples)})")
    
    # Final status report
    status = pipeline.prompt_optimizer.check_optimization_status()
    logger.info("Final optimization status:")
    for name, stats in status.items():
        logger.info(f"  {name}: {stats['example_count']} examples, optimized: {stats['optimized']}")

if __name__ == "__main__":
    run_example_collection()
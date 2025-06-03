"""Simplified main script for running the trading pipeline without prompt optimization."""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from loguru import logger
from src.pipeline import TradingPipeline
from src.utils.data_preprocessor import preprocess_market_data

# Load environment variables
load_dotenv()

def load_pickle_data(file_path: str) -> Dict[str, Any]:
    """Load data from a pickle file."""
    try:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Data loaded successfully. Found {len(data)} tokens.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def main():
    # Configure logging
    logger.add("logs/trading_simple.log", rotation="1 day")
    logger.info("Starting simplified trading strategy optimization")
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Load market data
    trade_data = load_pickle_data("/Users/speed/StratOptimv4/big_optimize_1016.pkl")
    
    # Initialize pipeline with prompt optimization disabled
    pipeline = TradingPipeline(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",  # Use cheaper model to avoid rate limits
        performance_thresholds={
            'min_return': 0.10,
            'min_trades': 10,
            'max_drawdown': 0.20
        },
        use_enhanced_regime=False,  # Disable enhanced regime to reduce API calls
        use_prompt_optimization=False  # Disable prompt optimization
    )
    
    try:
        # Focus on just one token for testing
        test_token = "$MICHI"
        preprocessed_data = preprocess_market_data(trade_data[test_token])
        
        # Run pipeline with just 1 iteration
        results = pipeline.run(
            market_data=preprocessed_data,
            num_iterations=1,
            timeframe="1h"
        )
        
        if not results:
            logger.error("Pipeline execution failed")
            return
        
        # Save results
        with open("results/strategy_results_simple.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Extract and log performance metrics
        logger.info("\nFinal Performance Metrics:")
        if results.get("iterations"):
            last_iteration = results["iterations"][-1]
            performance = last_iteration.get("performance", {})
            backtest_results = performance.get("backtest_results", {})
            
            if backtest_results:
                total_return = backtest_results.get("total_return", 0.0)
                total_pnl = backtest_results.get("total_pnl", 0.0)
                sortino_ratio = backtest_results.get("sortino_ratio", 0.0)
                win_rate = backtest_results.get("win_rate", 0.0)
                total_trades = backtest_results.get("total_trades", 0)
                
                logger.info(f"Total Return: {total_return:.4f}")
                logger.info(f"Total P&L: {total_pnl:.4f}")
                logger.info(f"Win Rate: {win_rate:.4f}")
                logger.info(f"Total Trades: {total_trades}")
                logger.info(f"Sortino Ratio: {sortino_ratio:.4f}")
                
                # Save performance summary
                with open("results/performance_summary_simple.txt", "w") as f:
                    f.write("Trading DSPy Performance Summary\n")
                    f.write("==============================\n\n")
                    f.write(f"Total Return: {total_return:.4f}\n")
                    f.write(f"Total P&L: {total_pnl:.4f}\n")
                    f.write(f"Win Rate: {win_rate:.4f}\n")
                    f.write(f"Total Trades: {total_trades}\n")
                    f.write(f"Sortino Ratio: {sortino_ratio:.4f}\n")
                    
        logger.info("Results saved to results/strategy_results_simple.json")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Full traceback:")
        return

if __name__ == "__main__":
    main()
"""Main script for running the trading pipeline."""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any

from loguru import logger
from src.pipeline import TradingPipeline
from src.utils.data_preprocessor import preprocess_market_data

def load_pickle_data(file_path: str) -> Dict[str, Any]:
    """Load data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary containing the loaded data
    """
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

def main():
    # Configure logging
    logger.add("logs/trading.log", rotation="1 day")
    logger.info("Starting trading strategy optimization")
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Load market data
    trade_data = load_pickle_data("/Users/speed/StratOptimv4/big_optimize_1016.pkl")
    
    # Initialize pipeline
    pipeline = TradingPipeline(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4-turbo-preview",
        performance_thresholds={
            'min_return': 0.10,
            'min_trades': 10,
            'max_drawdown': 0.20
        }
    )
    
    try:
        # Preprocess market data
        preprocessed_data = preprocess_market_data(trade_data["$MICHI"])
        
        # Run pipeline with preprocessed data
        results = pipeline.run(
            market_data=preprocessed_data,  # Pass the preprocessed data dictionary
            num_iterations=5,
            timeframe="1h"
        )
        
        if not results:
            logger.error("Pipeline execution failed")
            return
        
        # Save results
        with open("results/strategy_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
            # Extract and log performance metrics
            logger.info("\nFinal Performance Metrics:")
            if results.get("iterations"):
                last_iteration = results["iterations"][-1]
                performance = last_iteration.get("performance", {})
                backtest_results = performance.get("backtest_results", {})
                
                if backtest_results:
                    # Extract metrics from the correct structure
                    total_return = backtest_results.get("total_return", 0.0)
                    total_pnl = backtest_results.get("total_pnl", 0.0)
                    sortino_ratio = backtest_results.get("sortino_ratio", 0.0)
                    win_rate = backtest_results.get("win_rate", 0.0)
                    total_trades = backtest_results.get("total_trades", 0)
                    
                    # Log the metrics
                    logger.info(f"Total Return: {total_return:.4f}")
                    logger.info(f"Win Rate: {win_rate:.4f}")
                    logger.info(f"Total Trades: {total_trades}")
                    logger.info(f"Sortino Ratio: {sortino_ratio:.4f}")
                    
                    # Save detailed performance report
                    with open("results/asset_performance.txt", "w") as f:
                        f.write("Asset Performance Report\n")
                        f.write("======================\n\n")
                        
                        # Write overall statistics
                        f.write("Overall Statistics\n")
                        f.write("=================\n")
                        f.write(f"Total Return: {total_return:.4f}\n")
                        f.write(f"Win Rate: {win_rate:.4f}\n")
                        f.write(f"Total Trades: {total_trades}\n")
                        f.write(f"Sortino Ratio: {sortino_ratio:.4f}\n")
                        f.write(f"Total PnL: {total_pnl:.4f}\n")
                else:
                    logger.warning("No backtest results found in the last iteration")
            else:
                logger.warning("No iterations found in results")
                
            logger.info("Detailed performance report saved to results/asset_performance.txt")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Full traceback:")
        return

if __name__ == "__main__":
    main()

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
            backtest_results = last_iteration.get("backtest_results", {})
            metrics = backtest_results.get("metrics", {})
            trade_memory_stats = metrics.get("trade_memory_stats", {})
            
            # Log the metrics
            logger.info(f"Total Return: {backtest_results.get('total_return', 0.0):.4f}")
            logger.info(f"Win Rate: {trade_memory_stats.get('win_rate', 0.0):.4f}")
            logger.info(f"Total Trades: {trade_memory_stats.get('total_trades', 0)}")
            logger.info(f"Sortino Ratio: {trade_memory_stats.get('sortino_ratio', 0.0):.4f}")
            
            # Save detailed performance report
            with open("results/asset_performance.txt", "w") as f:
                f.write("Asset Performance Report\n")
                f.write("======================\n\n")
                
                per_asset_stats = metrics.get("per_asset_stats", {})
                
                for asset, stats in per_asset_stats.items():
                    f.write(f"Asset: {asset}\n")
                    f.write(f"Total Return: {stats.get('total_return', 0.0):.4f}\n")
                    f.write(f"Average PnL per Trade: {stats.get('avg_pnl_per_trade', 0.0):.4f}\n\n")
                
                f.write("\nOverall Statistics\n")
                f.write("=================\n")
                f.write(f"Total Return: {backtest_results.get('total_return', 0.0):.4f}\n")
                f.write(f"Win Rate: {trade_memory_stats.get('win_rate', 0.0):.4f}\n")
                f.write(f"Total Trades: {trade_memory_stats.get('total_trades', 0)}\n")
                f.write(f"Sortino Ratio: {trade_memory_stats.get('sortino_ratio', 0.0):.4f}\n")
                
            logger.info("Detailed performance report saved to results/asset_performance.txt")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Full traceback:")
        return

if __name__ == "__main__":
    main() 
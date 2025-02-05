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

def main() -> None:
    """Run the main trading pipeline.
    
    This function:
    1. Loads and preprocesses market data
    2. Initializes the trading pipeline
    3. Runs the pipeline to generate and optimize trading strategies
    4. Saves the results
    """
    try:
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Load data
        data_path = "/Users/speed/StratOptimv4/big_optimize_1016.pkl"
        trade_data = load_pickle_data(data_path)
        logger.info(f"Data loaded successfully. Found {len(trade_data)} tokens.")
        
        # Initialize trading pipeline
        pipeline = TradingPipeline(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview",
            memory_dir="memory",
            prompts_dir="prompts",
            performance_thresholds={
                'min_total_return': 0.05,
                'min_win_rate': 0.55,
                'min_sortino': 1.0
            }
        )
        
        # Preprocess market data
        market_data = preprocess_market_data(trade_data["$MICHI"])
        
        # Run pipeline
        logger.info("Starting DSPy trading pipeline...")
        pipeline_results = pipeline.run(
            market_data=market_data,
            timeframe="1h",  # Using 1-hour timeframe for analysis
            trading_theme="momentum",  # Using momentum as the trading theme
            max_iterations=5
        )
        
        if pipeline_results.get('error'):
            logger.error(f"Pipeline failed: {pipeline_results['error']}")
            return
        
        # Save results
        results_file = results_dir / "strategy_results.json"
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")
        
        # Log final metrics
        metrics = pipeline_results.get('performance', {})
        logger.info("\nFinal Performance Metrics:")
        logger.info(f"Total Return: {metrics.get('total_return', 0):.4f}")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.4f}")
        logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
        logger.info(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
        
        # Save asset performance details
        performance_file = results_dir / "asset_performance.txt"
        with open(performance_file, 'w') as f:
            f.write("Trading Strategy Performance Report\n")
            f.write("=================================\n\n")
            
            f.write("Overall Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            
            f.write("\nOptimal Parameters:\n")
            strategy = pipeline_results.get('strategy', {})
            parameters = strategy.get('parameters', {})
            for param, value in parameters.items():
                f.write(f"{param}: {value}\n")
        
        logger.info(f"Detailed performance report saved to {performance_file}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error("Full traceback:", exc_info=True)

if __name__ == "__main__":
    main() 
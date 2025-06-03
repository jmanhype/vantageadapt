"""Optimized main script for running the trading pipeline with GPT-4o-mini."""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import time

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
    logger.add("logs/trading_optimized.log", rotation="1 day")
    logger.info("Starting optimized trading strategy with GPT-4o-mini")
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Load market data
    trade_data = load_pickle_data("/Users/speed/StratOptimv4/big_optimize_1016.pkl")
    
    # Use GPT-4o-mini for cost efficiency and speed
    # API costs: $0.15 per million input tokens and $0.6 per million output tokens
    # 60% cheaper than GPT-3.5 Turbo and much faster than GPT-4
    pipeline = TradingPipeline(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Using the efficient GPT-4o-mini model
        performance_thresholds={
            'min_return': 0.05,  # Lower threshold for testing
            'min_trades': 5,     # Lower threshold for testing
            'max_drawdown': 0.25
        },
        use_enhanced_regime=False,  # Disable to reduce complexity
        use_prompt_optimization=False  # Disable to avoid rate limits
    )
    
    try:
        start_time = time.time()
        
        # Test on multiple tokens for better results
        test_tokens = ["$MICHI", "POPCAT", "BILLY"]  # Top meme coins
        all_results = []
        
        for token in test_tokens:
            if token not in trade_data:
                logger.warning(f"Token {token} not found in data")
                continue
                
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing token: {token}")
            logger.info(f"{'='*50}")
            
            try:
                # Preprocess data for this token
                preprocessed_data = preprocess_market_data(trade_data[token])
                
                # Run pipeline with 3 iterations for this token
                results = pipeline.run(
                    market_data=preprocessed_data,
                    num_iterations=3,
                    timeframe="1h"
                )
                
                if results and results.get("iterations"):
                    # Process results for this token
                    token_performance = {
                        'token': token,
                        'iterations': []
                    }
                    
                    for i, iteration in enumerate(results["iterations"]):
                        performance = iteration.get("performance", {})
                        backtest_results = performance.get("backtest_results", {})
                        
                        if backtest_results:
                            iter_summary = {
                                'iteration': i + 1,
                                'total_return': backtest_results.get("total_return", 0.0),
                                'total_pnl': backtest_results.get("total_pnl", 0.0),
                                'win_rate': backtest_results.get("win_rate", 0.0),
                                'total_trades': backtest_results.get("total_trades", 0),
                                'sortino_ratio': backtest_results.get("sortino_ratio", 0.0)
                            }
                            token_performance['iterations'].append(iter_summary)
                            
                            logger.info(f"\nIteration {i+1} Results:")
                            logger.info(f"  Total Return: {iter_summary['total_return']:.4f}")
                            logger.info(f"  Total P&L: {iter_summary['total_pnl']:.4f}")
                            logger.info(f"  Win Rate: {iter_summary['win_rate']:.4f}")
                            logger.info(f"  Total Trades: {iter_summary['total_trades']}")
                            logger.info(f"  Sortino Ratio: {iter_summary['sortino_ratio']:.4f}")
                    
                    all_results.append(token_performance)
                    
            except Exception as e:
                logger.error(f"Error processing token {token}: {str(e)}")
                continue
        
        # Calculate aggregate performance
        if all_results:
            total_pnl = 0
            total_trades = 0
            total_wins = 0
            
            for token_result in all_results:
                for iteration in token_result['iterations']:
                    total_pnl += iteration['total_pnl']
                    total_trades += iteration['total_trades']
                    total_wins += int(iteration['win_rate'] * iteration['total_trades'])
            
            overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
            
            # Save results
            final_results = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model': 'gpt-4o-mini',
                'tokens_analyzed': len(all_results),
                'aggregate_performance': {
                    'total_pnl': total_pnl,
                    'total_trades': total_trades,
                    'overall_win_rate': overall_win_rate,
                    'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
                },
                'token_results': all_results,
                'execution_time': time.time() - start_time
            }
            
            with open("results/strategy_results_optimized.json", "w") as f:
                json.dump(final_results, f, indent=2)
            
            # Save performance summary
            with open("results/performance_summary_optimized.txt", "w") as f:
                f.write("Trading DSPy Performance Summary (GPT-4o-mini)\n")
                f.write("============================================\n\n")
                f.write(f"Model: GPT-4o-mini\n")
                f.write(f"Tokens Analyzed: {len(all_results)}\n")
                f.write(f"Total P&L: {total_pnl:.4f}\n")
                f.write(f"Total Trades: {total_trades}\n")
                f.write(f"Overall Win Rate: {overall_win_rate:.4f}\n")
                f.write(f"Avg P&L per Trade: {total_pnl / total_trades if total_trades > 0 else 0:.4f}\n")
                f.write(f"Execution Time: {time.time() - start_time:.2f} seconds\n")
                
            logger.info("\n" + "="*70)
            logger.info("FINAL AGGREGATE RESULTS")
            logger.info("="*70)
            logger.info(f"Total P&L across all tokens: {total_pnl:.4f}")
            logger.info(f"Total Trades: {total_trades}")
            logger.info(f"Overall Win Rate: {overall_win_rate:.4f}")
            logger.info(f"Execution Time: {time.time() - start_time:.2f} seconds")
            logger.info("\nResults saved to results/strategy_results_optimized.json")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Full traceback:")
        return

if __name__ == "__main__":
    main()
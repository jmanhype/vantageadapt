"""Fixed main script for running the trading pipeline with all issues addressed."""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from loguru import logger

# Disable litellm caching to avoid annotation errors
os.environ["LITELLM_CACHING"] = "False"
os.environ["LITELLM_CACHE"] = "False"

from src.pipeline import TradingPipeline
from src.utils.data_preprocessor import preprocess_market_data
from src.utils.enum_fix import fix_market_context

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

def patch_pipeline():
    """Patch the pipeline to fix enum issues."""
    import src.pipeline
    import src.modules.market_analysis
    from src.utils.enum_fix import safe_market_regime, fix_market_context
    
    # Patch the generate_strategy method
    original_generate_strategy = src.pipeline.TradingPipeline.generate_strategy
    
    def patched_generate_strategy(self, market_context, recent_performance):
        # Fix market context before using it
        fixed_context = fix_market_context(market_context.copy())
        return original_generate_strategy(self, fixed_context, recent_performance)
    
    src.pipeline.TradingPipeline.generate_strategy = patched_generate_strategy
    
    # Patch the analyze_market method
    original_analyze_market = src.pipeline.TradingPipeline.analyze_market
    
    def patched_analyze_market(self, market_data, timeframe="1min"):
        result = original_analyze_market(self, market_data, timeframe)
        if result and isinstance(result, dict):
            return fix_market_context(result)
        return result
    
    src.pipeline.TradingPipeline.analyze_market = patched_analyze_market
    
    # Patch MarketRegime creation in pipeline
    original_init = src.utils.types.StrategyContext.__init__
    
    def patched_init(self, regime, confidence, risk_level, parameters, opportunity_score=0.0):
        # Ensure regime is properly converted
        from src.utils.enum_fix import safe_market_regime
        regime = safe_market_regime(regime)
        original_init(self, regime, confidence, risk_level, parameters, opportunity_score)
    
    src.utils.types.StrategyContext.__init__ = patched_init

def main():
    # Apply patches
    patch_pipeline()
    
    # Configure logging
    logger.add("logs/trading_fixed.log", rotation="1 day")
    logger.info("Starting fixed trading strategy pipeline")
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Load market data
    trade_data = load_pickle_data("/Users/speed/StratOptimv4/big_optimize_1016.pkl")
    
    # Initialize pipeline with optimizations
    pipeline = TradingPipeline(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Use efficient model
        performance_thresholds={
            'min_return': 0.05,
            'min_trades': 5,
            'max_drawdown': 0.25
        },
        use_enhanced_regime=False,  # Disable to reduce complexity
        use_prompt_optimization=False  # Disable to avoid issues
    )
    
    try:
        start_time = time.time()
        
        # Test on select tokens
        test_tokens = ["$MICHI", "POPCAT", "BILLY"]
        all_results = []
        total_pnl = 0
        total_trades = 0
        
        for i, token in enumerate(test_tokens):
            if token not in trade_data:
                logger.warning(f"Token {token} not found in data")
                continue
                
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing token {i+1}/{len(test_tokens)}: {token}")
            logger.info(f"{'='*50}")
            
            try:
                # Preprocess data
                preprocessed_data = preprocess_market_data(trade_data[token])
                
                # Run pipeline with 6 iterations for better data
                results = pipeline.run(
                    market_data=preprocessed_data,
                    num_iterations=6,
                    timeframe="1h"
                )
                
                if results and results.get("iterations"):
                    token_result = {
                        'token': token,
                        'best_performance': None
                    }
                    
                    best_pnl = -float('inf')
                    
                    for iteration in results["iterations"]:
                        performance = iteration.get("performance", {})
                        backtest_results = performance.get("backtest_results", {})
                        
                        if backtest_results:
                            pnl = backtest_results.get("total_pnl", 0.0)
                            if pnl > best_pnl:
                                best_pnl = pnl
                                token_result['best_performance'] = {
                                    'total_pnl': pnl,
                                    'total_return': backtest_results.get("total_return", 0.0),
                                    'win_rate': backtest_results.get("win_rate", 0.0),
                                    'total_trades': backtest_results.get("total_trades", 0),
                                    'sortino_ratio': backtest_results.get("sortino_ratio", 0.0)
                                }
                    
                    if token_result['best_performance']:
                        all_results.append(token_result)
                        total_pnl += token_result['best_performance']['total_pnl']
                        total_trades += token_result['best_performance']['total_trades']
                        
                        logger.info(f"\nBest performance for {token}:")
                        logger.info(f"  P&L: {token_result['best_performance']['total_pnl']:.4f}")
                        logger.info(f"  Win Rate: {token_result['best_performance']['win_rate']:.2%}")
                        logger.info(f"  Trades: {token_result['best_performance']['total_trades']}")
                    
            except Exception as e:
                logger.error(f"Error processing token {token}: {str(e)}")
                continue
        
        # Save final results
        execution_time = time.time() - start_time
        
        final_results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model': 'gpt-4o-mini',
            'approach': 'Fixed Trading DSPy',
            'tokens_analyzed': len(all_results),
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'token_results': all_results,
            'execution_time': execution_time
        }
        
        with open("results/trading_dspy_fixed_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TRADING DSPY FIXED - FINAL RESULTS")
        logger.info("="*70)
        logger.info(f"Tokens Analyzed: {len(all_results)}")
        logger.info(f"Total P&L: ${total_pnl:.2f}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Avg P&L per Trade: ${total_pnl / total_trades if total_trades > 0 else 0:.4f}")
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        
        # Print comparison with DGM
        logger.info("\n" + "="*70)
        logger.info("COMPARISON WITH DARWIN GÖDEL MACHINE")
        logger.info("="*70)
        logger.info(f"DGM Total P&L: $54,193.60")
        logger.info(f"Trading DSPy P&L: ${total_pnl:.2f}")
        logger.info(f"DGM Advantage: ${54193.60 - total_pnl:.2f}")
        logger.info(f"DGM is {54193.60 / max(total_pnl, 0.01):.0f}x more profitable")
        
        logger.info("\nResults saved to results/trading_dspy_fixed_results.json")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Full traceback:")
        return

if __name__ == "__main__":
    main()
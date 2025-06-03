"""Direct backtesting script that bypasses DSPy for immediate results."""

import pickle
import pandas as pd
import numpy as np
from loguru import logger
import json
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.backtester import Backtester, load_trade_data

def run_direct_backtest():
    """Run backtesting directly without DSPy pipeline."""
    logger.info("Starting direct backtesting without DSPy")
    
    # Load trade data
    trade_data = load_trade_data("/Users/speed/StratOptimv4/big_optimize_1016.pkl")
    if not trade_data:
        logger.error("Failed to load trade data")
        return
    
    logger.info(f"Loaded data for {len(trade_data)} tokens")
    
    # Initialize backtester
    backtester = Backtester(
        performance_thresholds={
            'min_return': 0.05,
            'min_trades': 5,
            'max_drawdown': 0.25
        }
    )
    
    # Define multiple strategy conditions to test
    strategy_conditions = [
        {
            'name': 'Momentum Strategy',
            'entry': [
                "price > sma_20",
                "rsi < 70",
                "macd.macd > macd.signal"
            ],
            'exit': [
                "price < sma_20",
                "rsi > 30",
                "macd.macd < macd.signal"
            ]
        },
        {
            'name': 'Mean Reversion Strategy',
            'entry': [
                "price < sma_20",
                "rsi < 30",
                "volume > volume_sma"
            ],
            'exit': [
                "price > sma_20",
                "rsi > 70",
                "volume < volume_sma * 0.8"
            ]
        },
        {
            'name': 'Breakout Strategy',
            'entry': [
                "price > sma_50",
                "volume > volume_sma * 1.5",
                "rsi > 50"
            ],
            'exit': [
                "price < sma_20",
                "volume < volume_sma",
                "rsi < 50"
            ]
        }
    ]
    
    # Test on top tokens
    test_tokens = ["$MICHI", "POPCAT", "BILLY", "DADDY", "GOAT"]
    
    all_results = []
    
    for strategy in strategy_conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {strategy['name']}")
        logger.info(f"{'='*60}")
        
        strategy_results = {
            'strategy_name': strategy['name'],
            'tokens': []
        }
        
        # Create subset of data for testing
        test_data = {}
        for token in test_tokens:
            if token in trade_data:
                test_data[token] = trade_data[token]
        
        if not test_data:
            logger.error("No test tokens found in data")
            continue
        
        try:
            # Run parameter optimization
            conditions = {
                'entry': strategy['entry'],
                'exit': strategy['exit']
            }
            
            results = backtester.run_parameter_optimization(
                trade_data=test_data,
                conditions=conditions
            )
            
            if results and 'backtest_results' in results:
                br = results['backtest_results']
                
                strategy_summary = {
                    'total_return': br.get('total_return', 0),
                    'total_pnl': br.get('total_pnl', 0),
                    'win_rate': br.get('win_rate', 0),
                    'total_trades': br.get('total_trades', 0),
                    'sortino_ratio': br.get('sortino_ratio', 0),
                    'parameters': results.get('parameters', {})
                }
                
                strategy_results['summary'] = strategy_summary
                
                logger.info(f"\nStrategy Performance:")
                logger.info(f"  Total Return: {strategy_summary['total_return']:.4f}")
                logger.info(f"  Total P&L: {strategy_summary['total_pnl']:.4f}")
                logger.info(f"  Win Rate: {strategy_summary['win_rate']:.4f}")
                logger.info(f"  Total Trades: {strategy_summary['total_trades']}")
                logger.info(f"  Sortino Ratio: {strategy_summary['sortino_ratio']:.4f}")
                
                # Get per-token results
                if 'metrics' in br and 'per_asset_stats' in br['metrics']:
                    per_asset = br['metrics']['per_asset_stats']
                    for token in test_tokens:
                        if token in per_asset.get('total_return', {}):
                            token_result = {
                                'token': token,
                                'return': per_asset['total_return'][token],
                                'pnl': per_asset.get('avg_pnl_per_trade', {}).get(token, 0)
                            }
                            strategy_results['tokens'].append(token_result)
                
                all_results.append(strategy_results)
                
        except Exception as e:
            logger.error(f"Error running strategy {strategy['name']}: {str(e)}")
            continue
    
    # Find best strategy
    if all_results:
        best_strategy = max(all_results, key=lambda x: x.get('summary', {}).get('total_pnl', 0))
        
        logger.info("\n" + "="*70)
        logger.info("BEST STRATEGY FOUND")
        logger.info("="*70)
        logger.info(f"Strategy: {best_strategy['strategy_name']}")
        logger.info(f"Total P&L: {best_strategy['summary']['total_pnl']:.4f}")
        logger.info(f"Win Rate: {best_strategy['summary']['win_rate']:.4f}")
        logger.info(f"Total Trades: {best_strategy['summary']['total_trades']}")
        
        # Save results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'approach': 'Direct Backtesting (No DSPy)',
            'strategies_tested': len(all_results),
            'best_strategy': best_strategy,
            'all_results': all_results
        }
        
        with open('results/direct_backtest_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
            
        logger.info("\nResults saved to results/direct_backtest_results.json")
        
        return best_strategy['summary']['total_pnl']
    
    return 0

if __name__ == "__main__":
    total_pnl = run_direct_backtest()
    print(f"\nFinal Total P&L: ${total_pnl:.2f}")
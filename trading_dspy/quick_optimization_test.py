#!/usr/bin/env python3
"""Quick test to show optimization is working with focused parameter search"""

from src.modules.hyperparameter_optimizer import HyperparameterOptimizer
from loguru import logger
import numpy as np

# Simple objective that rewards certain parameter combinations
def simple_objective(params):
    """Objective that has clear optimal regions"""
    tp = params.get('take_profit_pct', 0.05)
    sl = params.get('stop_loss_pct', 0.05)
    macd_fast = params.get('macd_signal_fast', 120)
    
    # Best performance around tp=0.06, sl=0.02, macd_fast=120
    score = -(
        abs(tp - 0.06) * 10 +  # Penalty for being far from optimal TP
        abs(sl - 0.02) * 20 +  # Penalty for being far from optimal SL
        abs(macd_fast - 120) * 0.001  # Small penalty for MACD deviation
    )
    
    # Add small noise
    score += np.random.normal(0, 0.01)
    
    return score


def main():
    logger.info("QUICK OPTIMIZATION TEST")
    logger.info("="*50)
    
    optimizer = HyperparameterOptimizer(
        objective_function=simple_objective,
        optimization_direction="maximize"
    )
    
    # Focused search space
    search_space = {
        'take_profit_pct': {'type': 'float', 'low': 0.02, 'high': 0.10, 'step': 0.02},
        'stop_loss_pct': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.01},
        'macd_signal_fast': {'type': 'int', 'low': 80, 'high': 160, 'step': 40},
    }
    
    optimizer.define_search_space(search_space)
    
    # Run grid search only (faster)
    logger.info("\nRunning Grid Search...")
    results = optimizer.grid_search()
    
    logger.info(f"\nRESULTS:")
    logger.info(f"Best params: {results['best_params']}")
    logger.info(f"Best score: {results['best_score']:.3f}")
    logger.info(f"Total evaluations: {results['n_evaluations']}")
    
    # Show top 5 results
    if results['all_results']:
        sorted_results = sorted(results['all_results'], key=lambda x: x['score'], reverse=True)[:5]
        logger.info("\nTop 5 parameter combinations:")
        for i, r in enumerate(sorted_results):
            logger.info(f"  {i+1}. Score: {r['score']:.3f}, Params: {r['params']}")
    
    # Verify optimization found near-optimal params
    best_tp = results['best_params'].get('take_profit_pct', 0)
    best_sl = results['best_params'].get('stop_loss_pct', 0)
    
    logger.info(f"\nOptimization quality check:")
    logger.info(f"  Found TP: {best_tp:.3f} (optimal: 0.060)")
    logger.info(f"  Found SL: {best_sl:.3f} (optimal: 0.020)")
    
    if abs(best_tp - 0.06) < 0.02 and abs(best_sl - 0.02) < 0.01:
        logger.info("✅ OPTIMIZATION WORKING CORRECTLY!")
    else:
        logger.warning("⚠️  Optimization may need tuning")


if __name__ == "__main__":
    main()
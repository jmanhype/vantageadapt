#!/usr/bin/env python3
"""Test script to verify optimization fixes are working"""

import asyncio
from loguru import logger
from src.kagan_master_coordinator import KaganMasterCoordinator
from src.modules.hyperparameter_optimizer import HyperparameterOptimizer
import numpy as np

# Setup logging
logger.add("test_optimization_fix.log", rotation="100 MB")


def test_objective_function(params):
    """Test objective that should show variation with different params"""
    # Simple function that rewards higher take_profit and lower stop_loss
    tp = params.get('take_profit_pct', 0.05)
    sl = params.get('stop_loss_pct', 0.05)
    
    # Reward function: higher TP is better, lower SL is worse
    score = tp / sl + np.random.normal(0, 0.1)  # Add some noise
    
    logger.info(f"Test eval: TP={tp:.3f}, SL={sl:.3f} -> Score={score:.3f}")
    return score


def test_hyperparameter_optimizer():
    """Test the optimizer in isolation"""
    logger.info("="*50)
    logger.info("Testing HyperparameterOptimizer in isolation")
    logger.info("="*50)
    
    optimizer = HyperparameterOptimizer(
        objective_function=test_objective_function,
        optimization_direction="maximize"
    )
    
    # Simple search space
    search_space = {
        'take_profit_pct': {'type': 'float', 'low': 0.02, 'high': 0.10, 'step': 0.02},
        'stop_loss_pct': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.01},
    }
    
    optimizer.define_search_space(search_space)
    
    # Test grid search
    logger.info("\nTesting Grid Search...")
    grid_results = optimizer.grid_search()
    
    logger.info(f"\nGrid Search Results:")
    logger.info(f"  Best params: {grid_results['best_params']}")
    logger.info(f"  Best score: {grid_results['best_score']:.3f}")
    logger.info(f"  Total evaluations: {grid_results['n_evaluations']}")
    
    # Verify we got different scores
    if grid_results['all_results']:
        scores = [r['score'] for r in grid_results['all_results']]
        logger.info(f"  Score range: {min(scores):.3f} to {max(scores):.3f}")
        logger.info(f"  Score variance: {np.var(scores):.3f}")
    
    return grid_results


async def test_master_coordinator_optimization():
    """Test the full optimization in master coordinator"""
    logger.info("\n" + "="*50)
    logger.info("Testing Master Coordinator Optimization")
    logger.info("="*50)
    
    coordinator = KaganMasterCoordinator()
    
    # Create a fake underperforming strategy
    fake_strategy = {
        'strategy_id': 'test_strategy_001',
        'performance': {
            'total_return': -0.5,  # Underperforming
            'total_trades': 10
        },
        'parameters': {
            'take_profit_pct': 0.08,
            'stop_loss_pct': 0.12
        }
    }
    
    coordinator.current_strategies['test_strategy_001'] = fake_strategy
    
    # Test the optimization
    strategic_insights = {
        'insights': {},
        'recommendations': [],
        'action_items': []
    }
    
    optimized = await coordinator._optimize_strategies(strategic_insights)
    
    logger.info(f"\nOptimization Results:")
    logger.info(f"  Strategies optimized: {len(optimized)}")
    
    if optimized:
        for i, strategy in enumerate(optimized):
            logger.info(f"\n  Strategy {i+1}:")
            logger.info(f"    ID: {strategy.get('strategy_id', 'unknown')}")
            logger.info(f"    Optimized: {strategy.get('optimized', False)}")
            logger.info(f"    Parameters: {strategy.get('parameters', {})}")


async def main():
    """Run all tests"""
    logger.info("TESTING OPTIMIZATION FIXES")
    logger.info("="*70)
    
    # Test 1: Hyperparameter optimizer in isolation
    test_hyperparameter_optimizer()
    
    # Test 2: Master coordinator optimization
    await test_master_coordinator_optimization()
    
    logger.info("\n" + "="*70)
    logger.info("TESTS COMPLETE - Check logs for results")


if __name__ == "__main__":
    asyncio.run(main())
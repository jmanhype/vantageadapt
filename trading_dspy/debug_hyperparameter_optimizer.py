#!/usr/bin/env python3
"""Debug the hyperparameter optimizer issue"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.modules.hyperparameter_optimizer import HyperparameterOptimizer

def debug_hyperparameter_optimizer():
    """Test the hyperparameter optimizer in isolation"""
    try:
        print("Initializing HyperparameterOptimizer...")
        optimizer = HyperparameterOptimizer()
        
        print("Defining search space...")
        search_space = {
            'risk_per_trade': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.01},
            'stop_loss_pct': {'type': 'float', 'low': 0.01, 'high': 0.05, 'step': 0.01},
            'take_profit_pct': {'type': 'float', 'low': 0.02, 'high': 0.10, 'step': 0.02},
            'lookback_period': {'type': 'int', 'low': 5, 'high': 50, 'step': 5}
        }
        
        optimizer.define_search_space(search_space)
        
        print("Setting up objective function...")
        def dummy_objective(params):
            print(f"Testing params: {params}")
            return 0.1  # Dummy return
            
        optimizer.objective_function = dummy_objective
        
        print("Running hybrid optimization...")
        result = optimizer.hybrid_optimize(
            grid_size='small',
            optuna_trials=5  # Reduced for testing
        )
        
        print(f"✅ Optimization successful: {result}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_hyperparameter_optimizer()
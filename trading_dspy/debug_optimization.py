#!/usr/bin/env python3
"""Debug the optimization error specifically"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_optimization_error():
    """Debug the exact optimization issue"""
    
    # Simulate the current_strategies structure
    current_strategies = {}
    
    # Add a strategy like the real system does
    strategy = {
        'strategy_id': 'test_strategy_123',
        'performance': {
            'total_return': -93.92,
            'backtest_completed': True
        },
        'parameters': {'test': 'value'}
    }
    
    current_strategies[strategy['strategy_id']] = strategy
    
    print(f"current_strategies type: {type(current_strategies)}")
    print(f"current_strategies content: {current_strategies}")
    print(f"current_strategies.items() type: {type(current_strategies.items())}")
    
    # Test the exact code from line 286-287
    try:
        underperforming = [
            (sid, sdata) for sid, sdata in current_strategies.items()
            if sdata.get('performance', {}).get('total_return', 0) < 0.1
        ]
        print(f"✅ Underperforming strategies: {underperforming}")
        
        # Test the iteration that's failing
        for strategy_id, strategy_data in underperforming[:3]:
            print(f"✅ Processing: {strategy_id}, type: {type(strategy_data)}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's see what items() actually returns
        print(f"Debugging items():")
        for i, item in enumerate(current_strategies.items()):
            print(f"  Item {i}: {item}, type: {type(item)}")

if __name__ == "__main__":
    debug_optimization_error()
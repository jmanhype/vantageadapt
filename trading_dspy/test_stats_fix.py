#!/usr/bin/env python3
"""Test the stats fix"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.modules.backtester import from_signals_backtest

def test_stats_fix():
    """Test calling stats() as method"""
    try:
        # Load data
        with open("/Users/speed/StratOptimv4/big_optimize_1016.pkl", 'rb') as f:
            data = pickle.load(f)
        
        asset_key = list(data.keys())[0]
        df = data[asset_key]
        
        # Ensure required columns
        if 'dex_price' not in df.columns:
            if 'close' in df.columns:
                df['dex_price'] = df['close']
            elif 'price' in df.columns:
                df['dex_price'] = df['price']
        
        if 'sol_pool' not in df.columns:
            df['sol_pool'] = 1000000
        if 'coin_pool' not in df.columns:
            df['coin_pool'] = df['sol_pool'] * df['dex_price']
        if 'sol_volume' not in df.columns:
            df['sol_volume'] = df['sol_pool']
            
        # Parameters
        backtest_params = {
            "take_profit": 0.08,
            "stop_loss": 0.12,
            "sl_window": 400,
            "max_orders": 3,
            "order_size": 0.0025,
            "post_buy_delay": 2,
            "post_sell_delay": 5,
            "macd_signal_fast": 120,
            "macd_signal_slow": 260,
            "macd_signal_signal": 90,
            "min_macd_signal_threshold": 0,
            "max_macd_signal_threshold": 0,
            "enable_sl_mod": False,
            "enable_tp_mod": False,
        }
        
        # Run backtest
        result = from_signals_backtest(df.copy(), **backtest_params)
        
        if result is not None and hasattr(result, 'stats'):
            # Test the fix - call stats() as method
            stats = result.stats()
            print(f"✅ stats() called successfully!")
            print(f"Stats type: {type(stats)}")
            print(f"Stats keys: {list(stats.keys()) if hasattr(stats, 'keys') else 'No keys method'}")
            
            if hasattr(stats, 'get'):
                total_return = stats.get('total_return', 0.0)
                print(f"✅ Total return extracted: {total_return}")
            elif 'total_return' in stats:
                total_return = stats['total_return']
                print(f"✅ Total return extracted: {total_return}")
            else:
                print("❌ No total_return found in stats")
                print(f"Available stats: {stats}")
        else:
            print("❌ No stats method available")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_stats_fix()
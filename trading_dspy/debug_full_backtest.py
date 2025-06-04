#!/usr/bin/env python3
"""Debug the full backtesting process"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import ta
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.modules.backtester import from_signals_backtest, calculate_entries_and_params

def debug_full_backtest():
    """Debug the full backtesting process exactly as Kagan Master calls it"""
    try:
        # Load the actual data exactly as Kagan Master does
        print("Loading market data...")
        with open("/Users/speed/StratOptimv4/big_optimize_1016.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # Get first asset data
        asset_key = list(data.keys())[0]
        df = data[asset_key]
        print(f"Loaded {len(df)} rows for {asset_key}")
        print(f"Columns: {list(df.columns)}")
        
        # Ensure required columns exactly as Kagan Master does
        if 'dex_price' not in df.columns:
            if 'close' in df.columns:
                df['dex_price'] = df['close']
            elif 'price' in df.columns:
                df['dex_price'] = df['price']
            else:
                raise ValueError("No price column found")
                
        # Add missing columns if needed
        if 'sol_pool' not in df.columns:
            df['sol_pool'] = 1000000  # Default volume
        if 'coin_pool' not in df.columns:
            df['coin_pool'] = df['sol_pool'] * df['dex_price']
        if 'sol_volume' not in df.columns:
            df['sol_volume'] = df['sol_pool']
            
        print(f"Final columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes}")
        
        # Use exact same parameters as Kagan Master
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
        
        print("Testing calculate_entries_and_params...")
        result_df = calculate_entries_and_params(df.copy(), **backtest_params)
        print("✅ calculate_entries_and_params successful!")
        print(f"Result columns: {list(result_df.columns)}")
        
        print("Testing from_signals_backtest...")
        result = from_signals_backtest(df.copy(), **backtest_params)
        print("✅ from_signals_backtest successful!")
        print(f"Result type: {type(result)}")
        
        if result is not None:
            print(f"Result attributes: {dir(result)}")
            if hasattr(result, 'stats'):
                print(f"Stats type: {type(result.stats)}")
                if hasattr(result.stats, 'total_return'):
                    print(f"Total return: {result.stats.total_return}")
                else:
                    print("Result.stats attributes:", dir(result.stats))
            else:
                print("No stats attribute found")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_full_backtest()
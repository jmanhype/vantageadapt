#!/usr/bin/env python3
"""Debug the backtesting error"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
import ta
import pickle

def debug_backtest():
    """Debug the exact backtesting error"""
    try:
        # Load the actual data
        print("Loading market data...")
        with open("/Users/speed/StratOptimv4/big_optimize_1016.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # Get first asset data
        asset_key = list(data.keys())[0]
        df = data[asset_key]
        print(f"Loaded {len(df)} rows for {asset_key}")
        
        # Ensure required columns
        if 'dex_price' not in df.columns:
            if 'close' in df.columns:
                df['dex_price'] = df['close']
            elif 'price' in df.columns:
                df['dex_price'] = df['price']
        
        # Add missing columns
        if 'sol_pool' not in df.columns:
            df['sol_pool'] = 1000000
        if 'coin_pool' not in df.columns:
            df['coin_pool'] = df['sol_pool'] * df['dex_price']
        if 'sol_volume' not in df.columns:
            df['sol_volume'] = df['sol_pool']
            
        print("Testing MACD calculation...")
        
        # Test MACD calculation that's failing
        macd = vbt.MACD.run(
            df['dex_price'], 
            fast_window=120, 
            slow_window=260, 
            signal_window=90
        )
        
        print(f"MACD type: {type(macd)}")
        print(f"MACD attributes: {dir(macd)}")
        print(f"MACD.macd type: {type(macd.macd)}")
        print(f"MACD.signal type: {type(macd.signal)}")
        
        # This line is failing
        macd_signal = macd.macd.vbt.crossed_above(macd.signal)
        print("MACD signal calculation successful!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_backtest()
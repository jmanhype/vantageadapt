#!/usr/bin/env python3
"""Direct VectorBT debug test to isolate the trade counting issue."""

import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append('src')

from modules.backtester import from_signals_backtest, calculate_stats

# Load the real market data
print("Loading market data...")
with open('/Users/speed/StratOptimv4/big_optimize_1016.pkl', 'rb') as f:
    trade_data = pickle.load(f)

# Get the first asset for testing
asset_name = list(trade_data.keys())[0]
asset_data = trade_data[asset_name].copy()

print(f"Testing with asset: {asset_name}")
print(f"Data shape: {asset_data.shape}")
print(f"Columns: {list(asset_data.columns)}")

# Use just the last 2 weeks of data for speed
two_weeks_ago = asset_data['timestamp'].max() - pd.Timedelta(weeks=2)
asset_data = asset_data[asset_data['timestamp'] >= two_weeks_ago]
print(f"After trimming to 2 weeks: {asset_data.shape}")

# Test parameters (same as system uses)
params = {
    'take_profit': 0.08,
    'stop_loss': 0.12,
    'sl_window': 400,
    'max_orders': 3,
    'order_size': 0.0025,
    'post_buy_delay': 2,
    'post_sell_delay': 5,
    'macd_signal_fast': 120,
    'macd_signal_slow': 260,
    'macd_signal_signal': 90,
    'min_macd_signal_threshold': 0,
    'max_macd_signal_threshold': 0,
    'enable_sl_mod': False,
    'enable_tp_mod': False,
}

print("Running VectorBT backtest...")
portfolio = from_signals_backtest(asset_data, **params)

print(f"Portfolio created: {portfolio is not None}")
if portfolio is not None:
    print(f"Portfolio type: {type(portfolio)}")
    print(f"Portfolio dir: {[attr for attr in dir(portfolio) if not attr.startswith('_')]}")
    
    # Test stats calculation
    print("\n=== TESTING STATS CALCULATION ===")
    test_portfolio = {asset_name: portfolio}
    stats_df = calculate_stats(test_portfolio, {asset_name: asset_data})
    
    print(f"Stats DataFrame shape: {stats_df.shape}")
    if not stats_df.empty:
        print("Stats:")
        print(stats_df.to_string())
    else:
        print("No stats generated!")
else:
    print("Portfolio creation failed!")
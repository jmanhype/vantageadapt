#!/usr/bin/env python3
"""
Quick test to verify aggressive trading parameters work
"""
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

def test_aggressive_signals():
    """Test aggressive signal generation"""
    print("Testing Aggressive Trading Parameters")
    print("="*50)
    
    # Load sample data
    with open("/Users/speed/StratOptimv4/big_optimize_1016.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Get first token data
    token = list(data.keys())[0]
    df = data[token]
    print(f"Testing with {token}: {len(df)} data points")
    
    # Simple aggressive trading logic
    trades = []
    position = None
    capital = 10000
    
    # Aggressive parameters
    ENTRY_THRESHOLD = 0.3  # Very low threshold
    STOP_LOSS = 0.01      # 1% stop
    TAKE_PROFIT = 0.02    # 2% target
    POSITION_SIZE = 0.2   # 20% per trade
    
    # Quick simulation on first 1000 points
    for i in range(100, min(1000, len(df))):
        current_price = df.iloc[i]['dex_price']
        
        if position is None:
            # Aggressive entry - random with 30% chance
            if np.random.random() < ENTRY_THRESHOLD:
                position = {
                    'entry_price': current_price,
                    'entry_idx': i,
                    'stop': current_price * (1 - STOP_LOSS),
                    'target': current_price * (1 + TAKE_PROFIT)
                }
                
        else:
            # Check exits
            if current_price <= position['stop']:
                # Stop hit
                pnl = -STOP_LOSS * capital * POSITION_SIZE
                trades.append({
                    'pnl': pnl,
                    'return': -STOP_LOSS,
                    'exit_reason': 'stop'
                })
                position = None
                
            elif current_price >= position['target']:
                # Target hit
                pnl = TAKE_PROFIT * capital * POSITION_SIZE
                trades.append({
                    'pnl': pnl,
                    'return': TAKE_PROFIT,
                    'exit_reason': 'target'
                })
                position = None
                
            elif i - position['entry_idx'] > 50:
                # Time exit
                return_pct = (current_price - position['entry_price']) / position['entry_price']
                pnl = return_pct * capital * POSITION_SIZE
                trades.append({
                    'pnl': pnl,
                    'return': return_pct,
                    'exit_reason': 'time'
                })
                position = None
    
    # Results
    total_trades = len(trades)
    if total_trades > 0:
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / total_trades
        avg_trade = total_pnl / total_trades
        
        print(f"\nResults from {900} price points:")
        print(f"Total Trades: {total_trades}")
        print(f"Trades per 100 points: {total_trades/9:.1f}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Average per trade: ${avg_trade:.2f}")
        print(f"Projected for 100k points: {int(total_trades * 100000/900)} trades")
    else:
        print("No trades generated!")

if __name__ == "__main__":
    test_aggressive_signals()
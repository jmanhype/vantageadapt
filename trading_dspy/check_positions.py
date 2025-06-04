#!/usr/bin/env python3
"""
Check current Alpaca positions and verify exit logic
"""

import os
import alpaca_trade_api as tradeapi
from datetime import datetime

# Set Alpaca keys
os.environ['ALPACA_API_KEY'] = "PKULR9QBIPWSP3LASZI9"
os.environ['ALPACA_SECRET_KEY'] = "vSINzsbHmeNXwc2oCiv0508sPHFzzdsPn4TWRplf"

def check_positions():
    """Check current positions and their P&L"""
    
    # Connect to Alpaca
    api = tradeapi.REST(
        key_id=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url='https://paper-api.alpaca.markets'
    )
    
    # Get account info
    account = api.get_account()
    print(f"\n💰 Account Status:")
    print(f"   Equity: ${float(account.equity):,.2f}")
    print(f"   Cash: ${float(account.cash):,.2f}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    
    # Get all positions
    positions = api.list_positions()
    
    print(f"\n📊 Current Positions: {len(positions)}")
    
    total_value = 0
    for pos in positions:
        # Get current price
        try:
            trades = api.get_latest_crypto_trades([pos.symbol])
            current_price = float(trades[pos.symbol].price)
        except:
            current_price = float(pos.current_price or pos.lastday_price or pos.avg_entry_price)
            
        # Calculate P&L
        entry_price = float(pos.avg_entry_price)
        return_pct = (current_price - entry_price) / entry_price * 100
        market_value = float(pos.market_value)
        total_value += market_value
        
        print(f"\n   {pos.symbol}:")
        print(f"      Qty: {pos.qty}")
        print(f"      Entry: ${entry_price:.2f}")
        print(f"      Current: ${current_price:.2f}")
        print(f"      Return: {return_pct:.2f}%")
        print(f"      Value: ${market_value:.2f}")
        
        # Check exit conditions
        if return_pct >= 0.3:
            print(f"      ✅ EXIT SIGNAL: QUICK_PROFIT")
        elif return_pct <= -0.5:
            print(f"      ❌ EXIT SIGNAL: STOP_LOSS")
        elif return_pct >= 1.0:
            print(f"      🎯 EXIT SIGNAL: TAKE_PROFIT")
            
    print(f"\n💰 Total Position Value: ${total_value:.2f}")
    print(f"📈 Total Return: {((float(account.equity) - 200000) / 200000) * 100:.2f}%")

if __name__ == "__main__":
    check_positions()
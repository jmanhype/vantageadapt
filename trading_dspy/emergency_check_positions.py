#!/usr/bin/env python3
"""Emergency script to check actual Alpaca positions and force exits if needed"""

import os
import alpaca_trade_api as tradeapi
from datetime import datetime

# Set NEW Alpaca keys - $1k realistic paper account!
os.environ['ALPACA_API_KEY'] = "PKULR9QBIPWSP3LASZI9"
os.environ['ALPACA_SECRET_KEY'] = "vSINzsbHmeNXwc2oCiv0508sPHFzzdsPn4TWRplf"

def check_positions():
    """Check all open positions and their P&L"""
    
    api = tradeapi.REST(
        key_id=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url='https://paper-api.alpaca.markets'
    )
    
    print("\n🔍 EMERGENCY POSITION CHECK")
    print("=" * 50)
    
    # Get account info
    account = api.get_account()
    print(f"💰 Account Equity: ${float(account.equity):,.2f}")
    print(f"💵 Buying Power: ${float(account.buying_power):,.2f}")
    print(f"📊 Starting Balance: $1,000.00")
    
    # Calculate actual return
    actual_return = ((float(account.equity) - 1000.0) / 1000.0) * 100
    print(f"📈 Actual Return: {actual_return:.2f}%")
    
    # Get all positions
    positions = api.list_positions()
    print(f"\n📋 Open Positions: {len(positions)}")
    print("-" * 50)
    
    total_value = 0
    total_pl = 0
    
    for pos in positions:
        # Get current price
        current_price = float(pos.current_price or pos.lastday_price or 0)
        entry_price = float(pos.avg_entry_price)
        qty = float(pos.qty)
        
        # Calculate P&L
        position_value = qty * current_price
        position_pl = (current_price - entry_price) * qty
        pl_pct = ((current_price - entry_price) / entry_price) * 100
        
        total_value += position_value
        total_pl += position_pl
        
        print(f"\n🪙 {pos.symbol}")
        print(f"   Qty: {qty}")
        print(f"   Entry: ${entry_price:.2f}")
        print(f"   Current: ${current_price:.2f}")
        print(f"   Value: ${position_value:.2f}")
        print(f"   P&L: ${position_pl:.2f} ({pl_pct:.2f}%)")
        
        # Check if should exit
        if pl_pct >= 0.3:  # 0.3% profit
            print(f"   ⚠️  SHOULD EXIT: Quick profit target hit!")
        elif pl_pct <= -0.5:  # 0.5% loss
            print(f"   ⚠️  SHOULD EXIT: Stop loss hit!")
        elif position_value < 50:  # Position too small
            print(f"   ⚠️  Position below minimum size")
    
    print("\n" + "=" * 50)
    print(f"📊 SUMMARY:")
    print(f"   Total Position Value: ${total_value:.2f}")
    print(f"   Total P&L: ${total_pl:.2f}")
    print(f"   Cash Available: ${float(account.buying_power):.2f}")
    print(f"   % of Capital in Positions: {(total_value / float(account.equity)) * 100:.1f}%")
    
    # Check if system is stuck
    if len(positions) >= 5 and float(account.buying_power) < 200:
        print("\n🚨 SYSTEM IS STUCK! Too many positions, not enough buying power!")
        print("   The exit logic is NOT working properly!")
        
    return positions

def force_exit_all(confirm=False):
    """Force exit all positions"""
    
    if not confirm:
        print("\n⚠️  To force exit all positions, call force_exit_all(confirm=True)")
        return
    
    api = tradeapi.REST(
        key_id=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url='https://paper-api.alpaca.markets'
    )
    
    positions = api.list_positions()
    
    print(f"\n🔥 FORCE EXITING {len(positions)} POSITIONS!")
    
    for pos in positions:
        try:
            order = api.submit_order(
                symbol=pos.symbol,
                qty=float(pos.qty),
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            print(f"✅ Sell order placed for {pos.symbol}")
        except Exception as e:
            print(f"❌ Failed to sell {pos.symbol}: {e}")

if __name__ == "__main__":
    positions = check_positions()
    
    if len(positions) > 0:
        print("\n💡 To force exit all positions, run:")
        print("   python emergency_check_positions.py --force-exit")
        
    import sys
    if "--force-exit" in sys.argv:
        force_exit_all(confirm=True)
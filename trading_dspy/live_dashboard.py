#!/usr/bin/env python3
"""
LIVE TRADING DASHBOARD
Shows all running systems and latest signals
"""

import os
import time
import subprocess
from datetime import datetime


def show_system_status():
    """Show status of all trading systems"""
    
    print(f"""
🤖 LIVE TRADING SYSTEMS DASHBOARD
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}
""")
    
    # Check running processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout
        
        systems = [
            ("🤖 Kagan Megazord", "kagan_megazord_coordinator", "megazord_output.log"),
            ("📊 Demo Trader", "instant_demo_trader", "demo_trader.log"),
            ("📈 Alpaca ML Trading", "launch_alpaca_trading", "alpaca_trading.log"),
            ("👀 Signal Watcher", "watch_signals", "signal_watch.log")
        ]
        
        for emoji_name, process_name, log_file in systems:
            if process_name in processes:
                print(f"✅ {emoji_name}: RUNNING")
                
                # Show latest activity
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                latest = lines[-1].strip()
                                # Show key events
                                if any(keyword in latest for keyword in ["SIGNAL", "BUY", "Generated", "training completed", "CONNECTION SUCCESS"]):
                                    print(f"   🔥 {latest[-100:]}")
                                elif "ERROR" in latest:
                                    print(f"   ❌ {latest[-100:]}")
                                else:
                                    print(f"   📊 {latest[-100:]}")
                    except:
                        pass
            else:
                print(f"❌ {emoji_name}: STOPPED")
        
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")
    
    print()


def show_alpaca_account_status():
    """Show Alpaca account status"""
    
    try:
        import alpaca_trade_api as tradeapi
        import os
        
        api = tradeapi.REST(
            key_id="PKV0EUF7LNIUB2TJMTIK",
            secret_key="XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz",
            base_url="https://paper-api.alpaca.markets"
        )
        
        account = api.get_account()
        positions = api.list_positions()
        orders = api.list_orders(status='all', limit=5)
        
        print(f"""
💰 ALPACA PAPER TRADING ACCOUNT
{'='*40}

📊 Account Status: {account.status}
💵 Buying Power: ${float(account.buying_power):,.2f}
📈 Portfolio Value: ${float(account.portfolio_value):,.2f}
🔄 Day P&L: ${float(account.todays_plpc) * float(account.last_equity):,.2f}
📍 Active Positions: {len(positions)}
📋 Recent Orders: {len(orders)}
""")
        
        # Show positions
        if positions:
            print("🎯 CURRENT POSITIONS:")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                print(f"   {pos.symbol}: {pos.qty} shares @ ${float(pos.current_price):.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
        else:
            print("📊 No positions yet - waiting for ML signals...")
        
        # Show recent orders
        if orders:
            print("\n📋 RECENT ORDERS:")
            for order in orders[:3]:
                print(f"   {order.symbol}: {order.side} {order.qty} @ {order.order_type} | Status: {order.status}")
        
        print()
        
    except ImportError:
        print("📈 Alpaca API not available")
    except Exception as e:
        print(f"⚠️ Could not connect to Alpaca: {e}")


def show_megazord_progress():
    """Show Megazord processing progress"""
    
    if os.path.exists("megazord_output.log"):
        print("⚡ MEGAZORD PROGRESS:")
        print("-" * 30)
        
        try:
            with open("megazord_output.log", 'r') as f:
                lines = f.readlines()
                
                # Find latest processing info
                for line in reversed(lines[-50:]):
                    if "Processing" in line and "/50:" in line:
                        token_info = line.split("Processing")[-1].strip()
                        print(f"   🔍 Current: {token_info}")
                        break
                
                # Find latest trade info
                for line in reversed(lines[-20:]):
                    if "Generated" in line and "trades" in line:
                        trade_info = line.split("Generated")[-1].strip()
                        print(f"   📊 Latest: Generated {trade_info}")
                        break
                
                # Find latest performance
                for line in reversed(lines[-30:]):
                    if "Cycle Performance:" in line:
                        perf_info = line.split("Cycle Performance:")[-1].strip()
                        print(f"   📈 Performance: {perf_info}")
                        break
                        
        except Exception as e:
            print(f"   ⚠️ Could not read Megazord log: {e}")
        
        print()


def show_ml_signals():
    """Show any recent ML signals"""
    
    log_files = ["demo_trader.log", "alpaca_trading.log"]
    signals_found = False
    
    print("🧠 RECENT ML SIGNALS:")
    print("-" * 30)
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Look for signals in last 100 lines
                    for line in lines[-100:]:
                        if any(keyword in line for keyword in ["🎯 SIGNAL:", "ML BUY SIGNAL", "PAPER TRADE EXECUTED"]):
                            signals_found = True
                            signal_info = line.strip()
                            print(f"   🚀 {signal_info}")
                            
            except:
                pass
    
    if not signals_found:
        print("   📊 No signals detected yet - systems are scanning...")
    
    print()


def main():
    """Main dashboard loop"""
    
    while True:
        try:
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            show_system_status()
            show_alpaca_account_status()
            show_megazord_progress()
            show_ml_signals()
            
            print("🔄 Dashboard refreshes every 30 seconds...")
            print("⌨️  Press Ctrl+C to exit")
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped")
            break
        except Exception as e:
            print(f"⚠️ Dashboard error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
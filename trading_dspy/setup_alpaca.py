#!/usr/bin/env python3
"""
ALPACA PAPER TRADING SETUP
Quick setup with your API keys
"""

import os
import subprocess
import asyncio
from datetime import datetime

# Your Alpaca Paper Trading Keys
ALPACA_API_KEY = "PKV0EUF7LNIUB2TJMTIK"
ALPACA_SECRET_KEY = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"


def setup_environment():
    """Set up environment variables"""
    
    os.environ['ALPACA_API_KEY'] = ALPACA_API_KEY
    os.environ['ALPACA_SECRET_KEY'] = ALPACA_SECRET_KEY
    
    print(f"""
    ✅ ALPACA ENVIRONMENT SETUP COMPLETE!
    
    📊 Endpoint: {ALPACA_BASE_URL}
    🔑 API Key: {ALPACA_API_KEY[:8]}...
    🔐 Secret: {ALPACA_SECRET_KEY[:8]}...
    
    🎯 Paper Trading Account Ready!
    """)


def install_alpaca_package():
    """Install required package"""
    
    print("📦 Installing alpaca-trade-api...")
    
    try:
        subprocess.run(['pip', 'install', 'alpaca-trade-api'], check=True)
        print("✅ alpaca-trade-api installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install alpaca-trade-api")
        print("💡 Try: pip install alpaca-trade-api")
        return False


async def test_connection():
    """Test Alpaca connection"""
    
    try:
        import alpaca_trade_api as tradeapi
        
        api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_BASE_URL
        )
        
        account = api.get_account()
        
        print(f"""
        🎉 ALPACA CONNECTION SUCCESS!
        
        💰 Account Status: {account.status}
        💵 Buying Power: ${float(account.buying_power):,.2f}
        📊 Portfolio Value: ${float(account.portfolio_value):,.2f}
        🏦 Account Type: Paper Trading
        
        ✅ Ready for ML paper trading!
        """)
        
        return True
        
    except ImportError:
        print("❌ alpaca-trade-api not installed")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


async def start_ml_paper_trading():
    """Start ML paper trading"""
    
    print("""
    🚀 STARTING ML PAPER TRADING...
    
    🎯 WHAT HAPPENS NEXT:
    1. System monitors: SPY, QQQ, AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA
    2. ML analyzes market data every 5 minutes
    3. When confidence > 15%, automatic BUY order placed
    4. Stop loss at -0.5%, take profit at +1%
    5. Position size: 25% of available cash
    
    📊 Using the SAME ML system that achieved 88.79% returns!
    """)
    
    # Import and run the alpaca trader
    try:
        from alpaca_paper_trader import AlpacaPaperTrader
        
        trader = AlpacaPaperTrader()
        await trader.start_live_trading()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure alpaca_paper_trader.py is available")


def create_quick_launcher():
    """Create a quick launcher script"""
    
    launcher_script = f'''#!/usr/bin/env python3
"""
QUICK ALPACA ML PAPER TRADING LAUNCHER
"""

import os
import asyncio

# Set environment variables
os.environ['ALPACA_API_KEY'] = "{ALPACA_API_KEY}"
os.environ['ALPACA_SECRET_KEY'] = "{ALPACA_SECRET_KEY}"

# Import and run
from alpaca_paper_trader import AlpacaPaperTrader

async def main():
    trader = AlpacaPaperTrader()
    await trader.start_live_trading()

if __name__ == "__main__":
    print("🚀 Starting Alpaca ML Paper Trading...")
    asyncio.run(main())
'''
    
    with open('launch_alpaca_trading.py', 'w') as f:
        f.write(launcher_script)
    
    print("📁 Created launcher: launch_alpaca_trading.py")


async def main():
    """Main setup process"""
    
    print("""
    🏦 ALPACA PAPER TRADING SETUP
    
    Setting up your free $100,000 paper trading account
    with ML signals from the 88.79% system!
    """)
    
    # Step 1: Set environment variables
    setup_environment()
    
    # Step 2: Install package
    if not install_alpaca_package():
        return
    
    # Step 3: Test connection
    if not await test_connection():
        return
    
    # Step 4: Create launcher
    create_quick_launcher()
    
    # Step 5: Ask to start trading
    print("""
    ✅ SETUP COMPLETE!
    
    🚀 READY TO START ML PAPER TRADING?
    
    Option 1: python launch_alpaca_trading.py
    Option 2: python alpaca_paper_trader.py
    
    Both will start automatic ML paper trading with your account!
    """)
    
    start = input("Start trading now? (y/n): ").strip().lower()
    if start == 'y':
        await start_ml_paper_trading()


if __name__ == "__main__":
    asyncio.run(main())
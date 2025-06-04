#!/usr/bin/env python3
"""
QUICK ALPACA ML PAPER TRADING LAUNCHER
"""

import os
import asyncio

# Set environment variables
os.environ['ALPACA_API_KEY'] = "PKV0EUF7LNIUB2TJMTIK"
os.environ['ALPACA_SECRET_KEY'] = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"

# Import and run
from alpaca_paper_trader import AlpacaPaperTrader

async def main():
    trader = AlpacaPaperTrader()
    await trader.start_live_trading()

if __name__ == "__main__":
    print("🚀 Starting Alpaca ML Paper Trading...")
    asyncio.run(main())

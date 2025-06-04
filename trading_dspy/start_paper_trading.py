#!/usr/bin/env python3
"""
START PAPER TRADING IN 2 MINUTES!
"""

import asyncio
import pandas as pd
import yfinance as yf
from paper_trading_adapter import PaperTradingAdapter
from loguru import logger


async def get_live_market_data(symbol: str) -> pd.DataFrame:
    """Get real market data from Yahoo Finance"""
    
    # Convert symbol format
    ticker = symbol.replace("/", "-")  # BTC/USD -> BTC-USD
    
    try:
        # Get last 30 days of data
        data = yf.download(ticker, period="30d", interval="1h", progress=False)
        
        if len(data) > 0:
            # Format for our system
            df = pd.DataFrame({
                'close': data['Close'],
                'open': data['Open'],
                'high': data['High'],
                'low': data['Low'],
                'volume': data['Volume']
            })
            df.index = data.index
            
            return df
    except:
        logger.error(f"Failed to get data for {symbol}")
        return None


async def main():
    """Run paper trading with real market data"""
    
    print("""
    📄 PAPER TRADING MODE ACTIVATED!
    
    This will generate REAL trading signals based on live market data.
    Signals include exact entry, stop loss, and take profit levels.
    
    Starting in 5 seconds...
    """)
    
    await asyncio.sleep(5)
    
    # Initialize adapter
    adapter = PaperTradingAdapter(platform="generic")
    
    # Your watchlist
    symbols = [
        "BTC-USD",    # Bitcoin
        "ETH-USD",    # Ethereum
        "AAPL",       # Apple
        "TSLA",       # Tesla
        "SPY",        # S&P 500
    ]
    
    logger.info(f"Monitoring {len(symbols)} symbols...")
    
    while True:
        for symbol in symbols:
            # Get latest data
            market_data = await get_live_market_data(symbol)
            
            if market_data is not None and len(market_data) > 100:
                # Get trading signal
                signal = await adapter.get_real_time_signal(symbol, market_data)
                
                if signal['action'] == 'BUY':
                    print(f"\n🚨 TRADE ALERT 🚨")
                    print(f"Symbol: {symbol}")
                    print(f"Action: BUY")
                    print(f"Price: ${signal['price']:.2f}")
                    print(f"Size: ${signal['size']:.2f}")
                    print(f"Stop Loss: ${signal['stop_loss']:.2f}")
                    print(f"Take Profit: ${signal['take_profit']:.2f}")
                    print(f"Confidence: {signal['confidence']:.1%}")
                    print(f"Time: {signal['timestamp']}")
                    print("-" * 50)
                    
                    # HERE: Send to your broker
                    # Example: place_paper_trade(signal)
        
        # Wait before next scan
        await asyncio.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    asyncio.run(main())
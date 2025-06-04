#!/usr/bin/env python3
"""
INSTANT PAPER TRADING - NO API KEYS NEEDED!

Just run and start getting real trading signals immediately.
Uses free data sources and manual execution.
"""

import asyncio
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from loguru import logger
import json
from typing import Dict, List
import webbrowser
import os

# Import our ML system
from paper_trading_adapter import PaperTradingAdapter


class InstantPaperTrader:
    """
    ZERO SETUP paper trading - no API keys, no accounts, just signals!
    """
    
    def __init__(self):
        self.adapter = PaperTradingAdapter(platform="manual")
        self.portfolio = {"cash": 100000, "positions": {}}
        self.trade_log = []
        self.signals_today = []
        
        logger.info("🚀 INSTANT PAPER TRADER - ZERO SETUP!")
        
    async def start_monitoring(self, symbols: List[str]):
        """Start monitoring symbols and generating alerts"""
        
        print(f"""
        📊 MONITORING {len(symbols)} SYMBOLS
        
        🎯 WHAT HAPPENS NEXT:
        1. System checks each symbol every 5 minutes
        2. When ML confidence > 15%, you get a BUY alert
        3. Alerts show exact entry, stop loss, take profit
        4. You manually execute on your platform of choice
        5. System tracks your paper portfolio performance
        
        💡 PLATFORMS YOU CAN USE (NO API NEEDED):
        • TradingView (free account, manual orders)
        • Think or Swim paper trading
        • Webull paper trading  
        • Yahoo Finance portfolio tracker
        • Even a spreadsheet!
        
        Starting monitoring in 3 seconds...
        """)
        
        await asyncio.sleep(3)
        
        while True:
            for symbol in symbols:
                try:
                    # Get live data (FREE via Yahoo Finance)
                    market_data = await self.get_free_market_data(symbol)
                    
                    if market_data is not None and len(market_data) > 100:
                        # Generate ML signal
                        signal = await self.adapter.get_real_time_signal(symbol, market_data)
                        
                        if signal['action'] == 'BUY':
                            # ALERT! Show the trade
                            await self.show_trade_alert(signal)
                            
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Wait 5 minutes before next scan
            print(f"\n⏰ Next scan in 5 minutes... (Scanned at {datetime.now().strftime('%H:%M:%S')})")
            await asyncio.sleep(300)
    
    async def get_free_market_data(self, symbol: str) -> pd.DataFrame:
        """Get FREE market data from Yahoo Finance"""
        
        try:
            # Convert symbol format if needed
            yahoo_symbol = symbol.replace("/", "-").replace(" ", "")
            
            # Get 30 days of hourly data (FREE)
            data = yf.download(yahoo_symbol, period="30d", interval="1h", progress=False)
            
            if len(data) > 0:
                df = pd.DataFrame({
                    'close': data['Close'],
                    'open': data['Open'], 
                    'high': data['High'],
                    'low': data['Low'],
                    'volume': data['Volume']
                })
                df.index = data.index
                return df
                
        except Exception as e:
            logger.warning(f"Could not get data for {symbol}: {e}")
            
        return None
    
    async def show_trade_alert(self, signal: Dict):
        """Show a beautiful trade alert"""
        
        # Add to today's signals
        self.signals_today.append(signal)
        
        print("\n" + "🚨" * 20)
        print("   🔥 TRADE ALERT 🔥")
        print("🚨" * 20)
        print(f"""
📈 SYMBOL: {signal['symbol']}
💰 ACTION: BUY NOW
💵 PRICE: ${signal['price']:.4f}
📊 SIZE: ${signal['size']:.2f} ({signal['size']/self.portfolio['cash']*100:.1f}% of portfolio)
🛡️ STOP LOSS: ${signal['stop_loss']:.4f} ({((signal['price']-signal['stop_loss'])/signal['price']*100):.1f}% below entry)
🎯 TAKE PROFIT: ${signal['take_profit']:.4f} ({((signal['take_profit']-signal['price'])/signal['price']*100):.1f}% above entry)
🧠 ML CONFIDENCE: {signal['confidence']:.1%}
⏰ TIME: {signal['timestamp'][:19]}

💡 RISK/REWARD: 1:{((signal['take_profit']-signal['price'])/(signal['price']-signal['stop_loss'])):.1f}

📋 COPY-PASTE ORDER (Manual Entry):
   Symbol: {signal['symbol']}
   Side: BUY
   Quantity: {int(signal['size']/signal['price'])} shares
   Order Type: LIMIT
   Limit Price: ${signal['price']:.4f}
   
   STOP LOSS: ${signal['stop_loss']:.4f}
   TAKE PROFIT: ${signal['take_profit']:.4f}
        """)
        print("🚨" * 20 + "\n")
        
        # Auto-open TradingView chart
        await self.open_chart(signal['symbol'])
        
        # Ask if they want to track this trade
        print("📝 Would you like to add this to your paper portfolio? (y/n)")
        # For automated demo, we'll just add it
        await self.add_to_paper_portfolio(signal)
    
    async def open_chart(self, symbol: str):
        """Auto-open TradingView chart"""
        try:
            # Convert symbol for TradingView URL
            tv_symbol = symbol.replace("-", "").replace("/", "")
            url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
            
            print(f"📊 Opening TradingView chart: {url}")
            # Uncomment to auto-open browser:
            # webbrowser.open(url)
            
        except Exception as e:
            logger.warning(f"Could not open chart: {e}")
    
    async def add_to_paper_portfolio(self, signal: Dict):
        """Add trade to paper portfolio tracking"""
        
        symbol = signal['symbol']
        shares = int(signal['size'] / signal['price'])
        cost = shares * signal['price']
        
        # Add position
        self.portfolio['positions'][symbol] = {
            'shares': shares,
            'entry_price': signal['price'],
            'entry_time': signal['timestamp'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'cost': cost,
            'ml_confidence': signal['confidence']
        }
        
        # Update cash
        self.portfolio['cash'] -= cost
        
        # Log trade
        self.trade_log.append({
            'timestamp': signal['timestamp'],
            'action': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': signal['price'],
            'total': cost
        })
        
        print(f"✅ Added to paper portfolio: {shares} shares of {symbol}")
        print(f"💰 Remaining cash: ${self.portfolio['cash']:.2f}")
        
        # Save to file
        self.save_portfolio()
    
    def save_portfolio(self):
        """Save portfolio to JSON file"""
        
        portfolio_data = {
            'portfolio': self.portfolio,
            'trade_log': self.trade_log,
            'signals_today': len(self.signals_today),
            'last_updated': datetime.now().isoformat()
        }
        
        with open('instant_paper_portfolio.json', 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        print("💾 Portfolio saved to: instant_paper_portfolio.json")
    
    async def show_portfolio_status(self):
        """Show current portfolio performance"""
        
        if not self.portfolio['positions']:
            print("📊 Portfolio: No positions")
            return
        
        print("\n📊 CURRENT PAPER PORTFOLIO:")
        print("-" * 50)
        
        total_value = self.portfolio['cash']
        
        for symbol, pos in self.portfolio['positions'].items():
            # Get current price
            current_data = await self.get_free_market_data(symbol)
            if current_data is not None:
                current_price = current_data['close'].iloc[-1]
                current_value = pos['shares'] * current_price
                pnl = current_value - pos['cost']
                pnl_pct = (pnl / pos['cost']) * 100
                
                print(f"{symbol}: {pos['shares']} shares @ ${pos['entry_price']:.4f}")
                print(f"  Current: ${current_price:.4f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
                
                total_value += current_value
        
        total_return = ((total_value - 100000) / 100000) * 100
        print(f"\n💰 Total Portfolio Value: ${total_value:.2f}")
        print(f"📈 Total Return: {total_return:+.2f}%")
        print(f"📊 Signals Today: {len(self.signals_today)}")


# INSTANT START FUNCTIONS
async def instant_crypto_trading():
    """Monitor popular crypto pairs - INSTANT START!"""
    
    trader = InstantPaperTrader()
    
    crypto_symbols = [
        "BTC-USD",    # Bitcoin
        "ETH-USD",    # Ethereum  
        "SOL-USD",    # Solana
        "ADA-USD",    # Cardano
        "MATIC-USD",  # Polygon
    ]
    
    print("🚀 INSTANT CRYPTO PAPER TRADING")
    await trader.start_monitoring(crypto_symbols)


async def instant_stock_trading():
    """Monitor popular stocks - INSTANT START!"""
    
    trader = InstantPaperTrader()
    
    stock_symbols = [
        "AAPL",   # Apple
        "TSLA",   # Tesla
        "MSFT",   # Microsoft
        "GOOGL",  # Google
        "AMZN",   # Amazon
        "NVDA",   # NVIDIA
        "SPY",    # S&P 500 ETF
        "QQQ",    # NASDAQ ETF
    ]
    
    print("🚀 INSTANT STOCK PAPER TRADING")
    await trader.start_monitoring(stock_symbols)


async def instant_forex_trading():
    """Monitor forex pairs - INSTANT START!"""
    
    trader = InstantPaperTrader()
    
    forex_symbols = [
        "EURUSD=X",   # EUR/USD
        "GBPUSD=X",   # GBP/USD
        "USDJPY=X",   # USD/JPY
        "AUDUSD=X",   # AUD/USD
        "USDCAD=X",   # USD/CAD
    ]
    
    print("🚀 INSTANT FOREX PAPER TRADING")
    await trader.start_monitoring(forex_symbols)


if __name__ == "__main__":
    print("""
    🎯 CHOOSE YOUR INSTANT PAPER TRADING MODE:
    
    1. Crypto (BTC, ETH, SOL, ADA, MATIC)
    2. Stocks (AAPL, TSLA, MSFT, GOOGL, NVDA, SPY, QQQ)  
    3. Forex (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD)
    
    Enter 1, 2, or 3:
    """)
    
    # For demo, we'll start with crypto
    print("Starting CRYPTO mode...")
    asyncio.run(instant_crypto_trading())
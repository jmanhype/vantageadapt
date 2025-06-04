#!/usr/bin/env python3
"""
ALPACA PAPER TRADING - 100% FREE!
Connects ML signals to Alpaca's free paper trading API
"""

import asyncio
import os
from datetime import datetime
import alpaca_trade_api as tradeapi
from loguru import logger
import pandas as pd

# Import our ML system
from paper_trading_adapter import PaperTradingAdapter


class AlpacaPaperTrader:
    """
    FREE Alpaca paper trading with ML signals
    """
    
    def __init__(self):
        # Alpaca Paper Trading API (FREE)
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY', 'your_paper_key'),
            secret_key=os.getenv('ALPACA_SECRET_KEY', 'your_paper_secret'),
            base_url='https://paper-api.alpaca.markets'  # Paper trading endpoint
        )
        
        self.adapter = PaperTradingAdapter(platform="alpaca")
        self.active_orders = {}
        
        logger.info("📈 Alpaca Paper Trader initialized")
    
    async def setup_paper_account(self):
        """Setup free paper trading account"""
        
        print("""
        🚀 ALPACA PAPER TRADING SETUP
        
        1. Go to: https://alpaca.markets
        2. Sign up for FREE account
        3. Get paper trading API keys:
           - ALPACA_API_KEY
           - ALPACA_SECRET_KEY
        4. Set environment variables or update this script
        
        📊 FEATURES:
        • $100,000 virtual cash
        • Real market data
        • Automatic order execution
        • Full portfolio tracking
        • Commission-free paper trades
        """)
        
        try:
            account = self.api.get_account()
            print(f"✅ Connected to Alpaca Paper Account")
            print(f"💰 Buying Power: ${float(account.buying_power):,.2f}")
            print(f"📊 Portfolio Value: ${float(account.portfolio_value):,.2f}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("💡 Make sure to set your Alpaca paper trading API keys!")
            return False
    
    async def start_live_trading(self):
        """Start live paper trading with ML signals"""
        
        if not await self.setup_paper_account():
            return
        
        # Watchlist of tradeable symbols
        symbols = [
            'SPY',    # S&P 500
            'QQQ',    # NASDAQ
            'AAPL',   # Apple
            'TSLA',   # Tesla
            'MSFT',   # Microsoft
            'GOOGL',  # Google
            'AMZN',   # Amazon
            'NVDA',   # NVIDIA
        ]
        
        print(f"🎯 Monitoring {len(symbols)} symbols for ML signals...")
        
        while True:
            try:
                for symbol in symbols:
                    # Get market data
                    market_data = await self.get_alpaca_data(symbol)
                    
                    if market_data is not None and len(market_data) > 100:
                        # Generate ML signal
                        signal = await self.adapter.get_real_time_signal(symbol, market_data)
                        
                        if signal['action'] == 'BUY':
                            await self.execute_paper_trade(signal)
                
                # Check existing positions
                await self.monitor_positions()
                
                # Wait before next scan
                print(f"⏰ Next scan in 5 minutes...")
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)
    
    async def get_alpaca_data(self, symbol: str) -> pd.DataFrame:
        """Get market data from Alpaca"""
        
        try:
            # Get 30 days of hourly data
            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                limit=720  # 30 days
            ).df
            
            if len(bars) > 0:
                # Format for our ML system
                df = pd.DataFrame({
                    'close': bars['close'],
                    'open': bars['open'],
                    'high': bars['high'],
                    'low': bars['low'],
                    'volume': bars['volume']
                })
                df.index = bars.index
                return df
                
        except Exception as e:
            logger.error(f"Data error for {symbol}: {e}")
            
        return None
    
    async def execute_paper_trade(self, signal: dict):
        """Execute paper trade on Alpaca"""
        
        symbol = signal['symbol']
        
        try:
            # Calculate position size
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            position_value = buying_power * 0.25  # 25% position size
            qty = int(position_value / signal['price'])
            
            if qty > 0:
                # Place bracket order (entry + stop loss + take profit)
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day',
                    order_class='bracket',
                    stop_loss={'stop_price': signal['stop_loss']},
                    take_profit={'limit_price': signal['take_profit']}
                )
                
                self.active_orders[symbol] = {
                    'order_id': order.id,
                    'entry_price': signal['price'],
                    'qty': qty,
                    'confidence': signal['confidence'],
                    'timestamp': datetime.now()
                }
                
                print(f"""
                🚀 PAPER TRADE EXECUTED!
                
                📈 Symbol: {symbol}
                💰 Qty: {qty} shares
                💵 Entry: ${signal['price']:.2f}
                🛡️ Stop: ${signal['stop_loss']:.2f}
                🎯 Target: ${signal['take_profit']:.2f}
                🧠 Confidence: {signal['confidence']:.1%}
                📊 Order ID: {order.id}
                """)
                
        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}")
    
    async def monitor_positions(self):
        """Monitor existing positions"""
        
        try:
            positions = self.api.list_positions()
            
            for position in positions:
                symbol = position.symbol
                current_price = float(position.current_price)
                unrealized_pl = float(position.unrealized_pl)
                unrealized_plpc = float(position.unrealized_plpc)
                
                print(f"📊 {symbol}: ${current_price:.2f} | "
                      f"P&L: ${unrealized_pl:.2f} ({unrealized_plpc:+.1%})")
                
        except Exception as e:
            logger.error(f"Position monitoring error: {e}")
    
    async def show_portfolio_summary(self):
        """Show portfolio performance"""
        
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            print(f"""
            📊 ALPACA PAPER PORTFOLIO
            {'='*40}
            
            💰 Portfolio Value: ${float(account.portfolio_value):,.2f}
            💵 Buying Power: ${float(account.buying_power):,.2f}
            📈 Day P&L: ${float(account.todays_plpc) * float(account.last_equity):,.2f}
            📊 Active Positions: {len(positions)}
            
            🎯 ML Trades Today: {len(self.active_orders)}
            """)
            
        except Exception as e:
            logger.error(f"Portfolio summary error: {e}")


# Setup instructions
def show_setup_instructions():
    """Show Alpaca setup instructions"""
    
    print("""
    🚀 ALPACA PAPER TRADING SETUP (100% FREE!)
    
    📋 STEP 1: Create Free Account
    1. Go to: https://alpaca.markets
    2. Click "Get Started" → "Paper Trading"
    3. Sign up (completely free)
    
    📋 STEP 2: Get API Keys
    1. Login to dashboard
    2. Go to "API Keys" section
    3. Generate new paper trading keys
    4. Copy API Key and Secret Key
    
    📋 STEP 3: Set Environment Variables
    export ALPACA_API_KEY="your_paper_key_here"
    export ALPACA_SECRET_KEY="your_paper_secret_here"
    
    📋 STEP 4: Install Required Package
    pip install alpaca-trade-api
    
    ✅ Then run: python alpaca_paper_trader.py
    
    💡 FEATURES YOU GET:
    • $100,000 virtual money
    • Real-time stock data
    • Automatic ML trade execution
    • Stop loss & take profit orders
    • Full portfolio tracking
    • Commission-free paper trades
    """)


async def main():
    """Main trading loop"""
    
    # Check if alpaca package is installed
    try:
        import alpaca_trade_api
    except ImportError:
        print("❌ alpaca-trade-api not installed")
        print("📦 Run: pip install alpaca-trade-api")
        return
    
    # Check if API keys are set
    if not os.getenv('ALPACA_API_KEY'):
        show_setup_instructions()
        return
    
    trader = AlpacaPaperTrader()
    await trader.start_live_trading()


if __name__ == "__main__":
    asyncio.run(main())
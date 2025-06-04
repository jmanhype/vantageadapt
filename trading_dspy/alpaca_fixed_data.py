#!/usr/bin/env python3
"""
FIXED ALPACA TRADING - REAL DATA & PAPER TRADES
Proper API usage with correct TimeFrame and symbol formats
"""

import asyncio
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Set Alpaca keys
os.environ['ALPACA_API_KEY'] = "PKV0EUF7LNIUB2TJMTIK"
os.environ['ALPACA_SECRET_KEY'] = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"


class AlpacaFixedTrader:
    """
    FIXED Alpaca trader with proper API usage
    """
    
    def __init__(self):
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                key_id=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url='https://paper-api.alpaca.markets'
            )
            
            # Store TimeFrame reference correctly
            self.TimeFrame = tradeapi.TimeFrame
            
            account = self.api.get_account()
            logger.info(f"✅ Alpaca connected: ${float(account.buying_power):,.2f}")
            self.connected = True
            
        except Exception as e:
            logger.error(f"❌ Alpaca connection failed: {e}")
            self.connected = False
    
    async def start_fixed_trading(self):
        """Start trading with REAL Alpaca data"""
        
        print(f"""
        🔧 FIXED ALPACA TRADING - REAL DATA!
        
        ✅ API: Proper TimeFrame usage
        📊 Data: Real market data from Alpaca
        💰 Trades: Live paper trading execution
        🎯 Symbols: Stocks + Crypto with correct formats
        
        🚀 STARTING WITH REAL DATA...
        """)
        
        # Stock symbols
        stock_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA']
        
        # Crypto symbols (correct format for Alpaca)
        crypto_symbols = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                logger.info(f"🔧 FIXED SCAN #{scan_count} at {current_time}")
                
                # Scan stocks with real data
                for symbol in stock_symbols:
                    await self.scan_with_real_data(symbol, 'stock')
                    await asyncio.sleep(1)
                
                # Scan crypto with real data
                for symbol in crypto_symbols:
                    await self.scan_with_real_data(symbol, 'crypto')
                    await asyncio.sleep(1)
                
                # Show account status every 5 scans
                if scan_count % 5 == 0:
                    await self.show_account_status()
                
                logger.info(f"⏰ Next scan in 60 seconds...")
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Fixed trading stopped")
                break
            except Exception as e:
                logger.error(f"❌ Trading error: {e}")
                await asyncio.sleep(60)
    
    async def scan_with_real_data(self, symbol: str, asset_type: str):
        """Scan symbol using REAL Alpaca data"""
        
        try:
            # Get REAL market data
            market_data = await self.get_real_alpaca_data(symbol, asset_type)
            
            if market_data is not None and len(market_data) > 20:
                # Generate simple signal using real data
                signal = await self.generate_simple_signal(symbol, market_data, asset_type)
                
                if signal and signal.get('action') == 'BUY':
                    logger.info(f"📊 REAL DATA SIGNAL: {symbol} ({asset_type})")
                    await self.execute_paper_trade(signal, asset_type)
                else:
                    logger.debug(f"📈 {symbol}: No signal (real data)")
            else:
                logger.debug(f"❌ {symbol}: No data available")
                
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
    
    async def get_real_alpaca_data(self, symbol: str, asset_type: str) -> pd.DataFrame:
        """Get REAL data from Alpaca API"""
        
        if not self.connected:
            return None
            
        try:
            if asset_type == 'stock':
                # Get stock data with proper TimeFrame
                # Fix datetime format for Alpaca API
                start_time = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                end_time = datetime.now().strftime('%Y-%m-%d')
                
                bars = self.api.get_bars(
                    symbol,
                    self.TimeFrame.Hour,  # Correct usage!
                    start=start_time,
                    end=end_time
                )
                
                if bars and len(bars) > 0:
                    df = bars.df.reset_index()
                    df = df.rename(columns={'timestamp': 'time'})
                    logger.debug(f"✅ Got {len(df)} real stock bars for {symbol}")
                    return df
                    
            elif asset_type == 'crypto':
                # Get crypto data with correct format
                # Fix datetime format for crypto API
                start_time = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
                end_time = datetime.now().strftime('%Y-%m-%d')
                
                bars = self.api.get_crypto_bars(
                    symbol,  # Already in BTC/USD format
                    self.TimeFrame.Hour,
                    start=start_time,
                    end=end_time
                )
                
                if bars and len(bars) > 0:
                    df = bars.df.reset_index()
                    df = df.rename(columns={'timestamp': 'time'})
                    logger.debug(f"✅ Got {len(df)} real crypto bars for {symbol}")
                    return df
                    
        except Exception as e:
            logger.debug(f"Real data failed for {symbol}: {e}")
            
        return None
    
    async def generate_simple_signal(self, symbol: str, data: pd.DataFrame, asset_type: str) -> dict:
        """Generate trading signal from real data"""
        
        try:
            if len(data) < 10:
                return {'action': 'HOLD', 'symbol': symbol}
                
            current_price = data['close'].iloc[-1]
            
            # Simple moving average strategy
            sma_5 = data['close'].rolling(5).mean().iloc[-1]
            sma_10 = data['close'].rolling(10).mean().iloc[-1]
            
            # Volume analysis
            avg_volume = data['volume'].rolling(5).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price momentum
            price_change = (current_price - data['close'].iloc[-3]) / data['close'].iloc[-3]
            
            # Generate confidence
            confidence = 0.05  # Base
            
            # Trend signal
            if current_price > sma_5 > sma_10:
                confidence += 0.02
                
            # Volume confirmation
            if volume_ratio > 1.2:
                confidence += 0.01
                
            # Momentum boost
            if price_change > 0.005:  # 0.5% move
                confidence += abs(price_change) * 2
                
            # Asset-specific adjustments
            if asset_type == 'crypto':
                confidence *= 1.2  # Crypto bonus
                
            # Signal threshold
            if confidence > 0.08:  # 8% threshold
                return {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'confidence': confidence,
                    'asset_type': asset_type,
                    'data_source': 'real_alpaca',
                    'reasoning': f"Real data signal: trend+volume+momentum, confidence={confidence:.1%}"
                }
                
        except Exception as e:
            logger.error(f"❌ Signal generation error for {symbol}: {e}")
            
        return {'action': 'HOLD', 'symbol': symbol}
    
    async def execute_paper_trade(self, signal: dict, asset_type: str):
        """Execute paper trade on Alpaca"""
        
        symbol = signal['symbol']
        
        try:
            if self.connected:
                account = self.api.get_account()
                buying_power = float(account.buying_power)
                
                # Position sizing
                if asset_type == 'crypto':
                    position_pct = 0.03  # 3% for crypto
                else:
                    position_pct = 0.05  # 5% for stocks
                    
                position_value = buying_power * position_pct
                
                if asset_type == 'crypto':
                    # Crypto quantity (can be fractional)
                    quantity = position_value / signal['price']
                    
                    if quantity * signal['price'] > 10:  # Min $10 trade
                        order = self.api.submit_order(
                            symbol=symbol,
                            notional=position_value,  # Use dollar amount for crypto
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                        
                        logger.info(f"✅ CRYPTO PAPER TRADE EXECUTED!")
                        logger.info(f"   🪙 ${position_value:.2f} of {symbol}")
                        logger.info(f"   📊 Order ID: {order.id}")
                        logger.info(f"   🧠 Confidence: {signal['confidence']:.1%}")
                        logger.info(f"   📈 Real Alpaca data used!")
                        
                else:
                    # Stock quantity (whole shares)
                    quantity = max(1, int(position_value / signal['price']))
                    
                    if quantity > 0:
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=quantity,
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        
                        logger.info(f"✅ STOCK PAPER TRADE EXECUTED!")
                        logger.info(f"   📈 {quantity} shares of {symbol} @ ${signal['price']:.2f}")
                        logger.info(f"   📊 Order ID: {order.id}")
                        logger.info(f"   🧠 Confidence: {signal['confidence']:.1%}")
                        logger.info(f"   📈 Real Alpaca data used!")
                        
        except Exception as e:
            logger.error(f"❌ Trade execution failed for {symbol}: {e}")
    
    async def show_account_status(self):
        """Show current account status"""
        
        try:
            if self.connected:
                account = self.api.get_account()
                positions = self.api.list_positions()
                
                portfolio_value = float(account.portfolio_value)
                buying_power = float(account.buying_power)
                
                logger.info(f"💰 ACCOUNT STATUS:")
                logger.info(f"   Portfolio: ${portfolio_value:,.2f}")
                logger.info(f"   Buying Power: ${buying_power:,.2f}")
                logger.info(f"   Positions: {len(positions)}")
                
                if positions:
                    total_pnl = sum(float(p.unrealized_pl) for p in positions)
                    logger.info(f"   Total P&L: ${total_pnl:,.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")


async def main():
    """Run fixed Alpaca trading"""
    
    trader = AlpacaFixedTrader()
    await trader.start_fixed_trading()


if __name__ == "__main__":
    asyncio.run(main())
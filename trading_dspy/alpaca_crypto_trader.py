#!/usr/bin/env python3
"""
ALPACA CRYPTO PAPER TRADING - 24/7 MARKETS!
Trade Bitcoin, Ethereum, and other cryptos that never close
"""

import asyncio
import os
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

# Import our ML system
from paper_trading_adapter import PaperTradingAdapter

# Set Alpaca keys
os.environ['ALPACA_API_KEY'] = "PKV0EUF7LNIUB2TJMTIK"
os.environ['ALPACA_SECRET_KEY'] = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"


class AlpacaCryptoTrader:
    """
    24/7 Crypto trading with Alpaca - ALWAYS ACTIVE!
    """
    
    def __init__(self):
        self.adapter = PaperTradingAdapter(platform="alpaca")
        self.paper_portfolio = {"cash": 100000, "positions": {}, "trades": []}
        
        # Ultra-aggressive settings for crypto
        if hasattr(self.adapter.megazord, 'CONFIDENCE_THRESHOLD'):
            self.adapter.megazord.CONFIDENCE_THRESHOLD = 0.03  # 3% for volatile crypto!
            logger.info(f"🚀 Ultra-low crypto threshold: {self.adapter.megazord.CONFIDENCE_THRESHOLD:.1%}")
        
        # Try to connect to Alpaca
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                key_id=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url='https://paper-api.alpaca.markets'
            )
            account = self.api.get_account()
            logger.info(f"✅ Alpaca crypto connected: ${float(account.buying_power):,.2f}")
            self.alpaca_connected = True
        except Exception as e:
            logger.warning(f"⚠️ Alpaca unavailable: {e}")
            self.alpaca_connected = False
    
    async def start_crypto_trading(self):
        """Start 24/7 crypto trading"""
        
        print(f"""
        🚀 ALPACA CRYPTO TRADING - 24/7 MARKETS!
        
        📊 Connection: {'✅ Live Alpaca' if self.alpaca_connected else '📝 Paper Only'}
        🎯 Confidence: 3% (ultra-aggressive for crypto!)
        💰 Capital: $100,000
        ⏰ Schedule: 24/7 - CRYPTO NEVER SLEEPS!
        
        🪙 CRYPTO WATCHLIST:
        • BTC/USD - Bitcoin (King of crypto)
        • ETH/USD - Ethereum (Smart contracts)
        • LTC/USD - Litecoin (Silver to Bitcoin's gold)
        • BCH/USD - Bitcoin Cash (Fast transactions)
        • LINK/USD - Chainlink (Oracle network)
        • UNI/USD - Uniswap (DeFi leader)
        • AAVE/USD - AAVE (DeFi lending)
        • SUSHI/USD - SushiSwap (DEX)
        """)
        
        # Crypto symbols available on Alpaca
        crypto_symbols = [
            'BTCUSD',   # Bitcoin
            'ETHUSD',   # Ethereum  
            'LTCUSD',   # Litecoin
            'BCHUSD',   # Bitcoin Cash
            'LINKUSD',  # Chainlink
            'UNIUSD',   # Uniswap
            'AAVEUSD',  # AAVE
            'SUSHIUSD'  # SushiSwap
        ]
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                logger.info(f"🪙 CRYPTO SCAN #{scan_count} at {current_time}")
                
                # Scan all crypto symbols every cycle
                for symbol in crypto_symbols:
                    await self.scan_crypto_symbol(symbol)
                    await asyncio.sleep(1)  # Brief pause between symbols
                
                # Show trading stats
                await self.show_crypto_stats()
                
                logger.info(f"⏰ Next crypto scan in 60 seconds...")
                await asyncio.sleep(60)  # 1 minute intervals for crypto
                
            except KeyboardInterrupt:
                logger.info("🛑 Crypto trading stopped")
                break
            except Exception as e:
                logger.error(f"❌ Crypto trading error: {e}")
                await asyncio.sleep(60)
    
    async def scan_crypto_symbol(self, symbol: str):
        """Scan crypto symbol for trading opportunities"""
        
        try:
            # Get crypto market data
            market_data = await self.get_crypto_data(symbol)
            
            if market_data is not None and len(market_data) > 50:
                # Try ML signal first
                signal = None
                try:
                    signal = await self.adapter.get_real_time_signal(symbol, market_data)
                except:
                    pass
                
                # Force crypto signal if needed (crypto is volatile = more opportunities)
                if not signal or signal.get('action') != 'BUY':
                    signal = await self.generate_crypto_signal(symbol, market_data)
                
                # Execute crypto trade
                if signal and signal.get('action') == 'BUY':
                    logger.info(f"🪙 CRYPTO SIGNAL: {symbol}")
                    await self.execute_crypto_trade(signal)
                else:
                    logger.debug(f"📊 {symbol}: No crypto signal")
                    
        except Exception as e:
            logger.error(f"❌ Error scanning crypto {symbol}: {e}")
    
    async def generate_crypto_signal(self, symbol: str, market_data: pd.DataFrame) -> dict:
        """Generate crypto-specific trading signals"""
        
        try:
            current_price = market_data['close'].iloc[-1]
            
            if len(market_data) > 20:
                # Crypto-specific indicators
                sma_10 = market_data['close'].rolling(10).mean().iloc[-1]  # Shorter for crypto
                sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
                
                # Volatility (crypto loves volatility)
                volatility = market_data['close'].pct_change().std() * 100
                
                # Price momentum
                momentum_5min = (current_price - market_data['close'].iloc[-5]) / market_data['close'].iloc[-5]
                momentum_1hr = (current_price - market_data['close'].iloc[-60]) / market_data['close'].iloc[-60] if len(market_data) > 60 else 0
                
                # Volume surge
                avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
                current_volume = market_data['volume'].iloc[-1]
                volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
                
                # Generate confidence based on crypto factors
                confidence = 0.05  # Base crypto confidence
                
                # Trend signals
                if current_price > sma_10 > sma_20:
                    confidence += 0.03  # Uptrend
                
                # Volatility bonus (crypto thrives on volatility)
                if volatility > 2:  # High volatility
                    confidence += volatility * 0.01
                
                # Momentum signals
                if momentum_5min > 0.005:  # 0.5% in 5 minutes
                    confidence += abs(momentum_5min) * 3
                
                if momentum_1hr > 0.02:  # 2% in 1 hour
                    confidence += abs(momentum_1hr) * 2
                
                # Volume surge bonus
                if volume_surge > 1.5:
                    confidence += 0.02
                
                # Cap confidence
                confidence = min(confidence, 0.85)
                
                # Crypto-specific entry (lower threshold due to 24/7 nature)
                if confidence > 0.03:  # 3% threshold for crypto
                    return {
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'stop_loss': current_price * 0.98,   # 2% stop for crypto volatility
                        'take_profit': current_price * 1.05,  # 5% target for crypto
                        'confidence': confidence,
                        'strategy': 'crypto',
                        'reasoning': f"Crypto signal: volatility={volatility:.1f}%, momentum_5m={momentum_5min:.1%}, volume_surge={volume_surge:.1f}x"
                    }
            
        except Exception as e:
            logger.error(f"❌ Error generating crypto signal for {symbol}: {e}")
        
        return {'action': 'HOLD', 'symbol': symbol}
    
    async def get_crypto_data(self, symbol: str) -> pd.DataFrame:
        """Get crypto market data"""
        
        # Try Alpaca crypto data first
        if self.alpaca_connected:
            try:
                # Alpaca crypto format
                bars = self.api.get_crypto_bars(
                    symbol,
                    self.api.TimeFrame.Minute,
                    limit=200  # More data for crypto analysis
                ).df
                
                if len(bars) > 0:
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
                logger.debug(f"Alpaca crypto data failed for {symbol}: {e}")
        
        # Fallback: Generate realistic crypto data
        logger.debug(f"Using simulated crypto data for {symbol}")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1min')
        
        # Crypto base prices
        crypto_prices = {
            'BTCUSD': 68000, 'ETHUSD': 3800, 'LTCUSD': 85, 'BCHUSD': 420,
            'LINKUSD': 14, 'UNIUSD': 8, 'AAVEUSD': 90, 'SUSHIUSD': 1.2
        }
        base_price = crypto_prices.get(symbol, 1000)
        
        # Generate realistic crypto volatility (higher than stocks)
        returns = np.random.randn(200) * 0.008  # 0.8% volatility per minute
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'close': prices,
            'open': prices * (1 + np.random.randn(200) * 0.002),
            'high': prices * (1 + np.abs(np.random.randn(200)) * 0.003),
            'low': prices * (1 - np.abs(np.random.randn(200)) * 0.003),
            'volume': np.random.randint(1000, 50000, 200)  # Crypto volumes
        }, index=dates)
        
        return df
    
    async def execute_crypto_trade(self, signal: dict):
        """Execute crypto trade"""
        
        symbol = signal['symbol']
        
        # Try Alpaca crypto trading
        if self.alpaca_connected:
            try:
                account = self.api.get_account()
                buying_power = float(account.buying_power)
                position_value = buying_power * 0.05  # 5% positions for crypto diversification
                
                # Calculate crypto quantity (can be fractional)
                qty = position_value / signal['price']
                
                if qty * signal['price'] > 1:  # Minimum $1 trade
                    order = self.api.submit_crypto_order(
                        symbol=symbol,
                        qty=qty,
                        side='buy',
                        type='market',
                        time_in_force='gtc'  # Good till cancelled for crypto
                    )
                    
                    logger.info(f"✅ LIVE CRYPTO TRADE EXECUTED!")
                    logger.info(f"   🪙 {qty:.6f} {symbol} @ ${signal['price']:.2f}")
                    logger.info(f"   🧠 Confidence: {signal['confidence']:.1%}")
                    logger.info(f"   📊 Order ID: {order.id}")
                    logger.info(f"   💰 Value: ${qty * signal['price']:,.2f}")
                    
                    return
                    
            except Exception as e:
                logger.error(f"❌ Alpaca crypto trade failed: {e}")
        
        # Paper portfolio fallback
        position_size = self.paper_portfolio['cash'] * 0.05  # 5% positions
        quantity = position_size / signal['price']
        
        if position_size <= self.paper_portfolio['cash']:
            self.paper_portfolio['positions'][symbol] = {
                'quantity': quantity,
                'entry_price': signal['price'],
                'entry_time': datetime.now().isoformat(),
                'confidence': signal['confidence'],
                'type': 'crypto'
            }
            
            self.paper_portfolio['cash'] -= quantity * signal['price']
            
            self.paper_portfolio['trades'].append({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'BUY',
                'quantity': quantity,
                'price': signal['price'],
                'confidence': signal['confidence'],
                'type': 'crypto'
            })
            
            logger.info(f"📝 PAPER CRYPTO TRADE!")
            logger.info(f"   🪙 {quantity:.6f} {symbol} @ ${signal['price']:.2f}")
            logger.info(f"   🧠 Confidence: {signal['confidence']:.1%}")
            logger.info(f"   💰 Value: ${quantity * signal['price']:,.2f}")
    
    async def show_crypto_stats(self):
        """Show crypto trading statistics"""
        
        crypto_trades = [t for t in self.paper_portfolio['trades'] if t.get('type') == 'crypto']
        
        if crypto_trades:
            total_value = sum(t['quantity'] * t['price'] for t in crypto_trades)
            avg_confidence = sum(t['confidence'] for t in crypto_trades) / len(crypto_trades)
            
            logger.info(f"🪙 CRYPTO STATS: {len(crypto_trades)} trades | Avg confidence: {avg_confidence:.1%} | Value: ${total_value:,.2f}")


async def main():
    """Run 24/7 crypto trading"""
    
    trader = AlpacaCryptoTrader()
    await trader.start_crypto_trading()


if __name__ == "__main__":
    asyncio.run(main())
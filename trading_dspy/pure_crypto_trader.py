#!/usr/bin/env python3
"""
PURE CRYPTO TRADER - 24/7 CRYPTO MARKETS ONLY
Real data, real trades, maximum crypto focus using PaperTradingAdapter
"""

import asyncio
import os
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

# Import the paper trading adapter
from paper_trading_adapter import PaperTradingAdapter

# Set Alpaca keys
os.environ['ALPACA_API_KEY'] = "PKV0EUF7LNIUB2TJMTIK"
os.environ['ALPACA_SECRET_KEY'] = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"


class PureCryptoTrader:
    """
    PURE 24/7 Crypto Trading - No stocks, just crypto using PaperTradingAdapter
    """
    
    def __init__(self):
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                key_id=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url='https://paper-api.alpaca.markets'
            )
            
            # Store TimeFrame reference
            self.TimeFrame = tradeapi.TimeFrame
            
            # Initialize Paper Trading Adapter with Alpaca platform
            self.adapter = PaperTradingAdapter(platform="alpaca")
            
            account = self.api.get_account()
            logger.info(f"🪙 Pure Crypto Mode: ${float(account.buying_power):,.2f}")
            self.connected = True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False
    
    async def start_pure_crypto_trading(self):
        """Start pure crypto trading - 24/7 markets"""
        
        print(f"""
        🪙 PURE CRYPTO TRADING MODE - 24/7 MARKETS
        
        ⚡ Focus: Crypto only - No stocks
        🌍 Schedule: 24/7 - Never stops
        💰 Capital: ~$200,000 for crypto
        📊 Data: Real Alpaca crypto feeds
        🎯 Strategy: Multi-timeframe crypto analysis
        
        🪙 CRYPTO ARSENAL:
        • BTC/USD - Bitcoin (Digital Gold)
        • ETH/USD - Ethereum (Smart Contracts)
        • LTC/USD - Litecoin (Digital Silver)
        • BCH/USD - Bitcoin Cash (Fast Payments)
        • LINK/USD - Chainlink (Oracle Network)
        • UNI/USD - Uniswap (DEX Leader)
        • AAVE/USD - AAVE (DeFi Lending)
        • SUSHI/USD - SushiSwap (DEX)
        
        🚀 STARTING PURE CRYPTO OPERATIONS...
        """)
        
        # Pure crypto symbols (Alpaca format)
        crypto_symbols = [
            'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD',
            'LINK/USD', 'UNI/USD', 'AAVE/USD', 'SUSHI/USD'
        ]
        
        scan_count = 0
        total_crypto_trades = 0
        
        while True:
            try:
                scan_count += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                logger.info(f"🪙 CRYPTO SCAN #{scan_count} at {current_time}")
                
                # Scan all cryptos with real data
                trades_this_scan = 0
                
                for symbol in crypto_symbols:
                    result = await self.scan_crypto_with_adapter(symbol)
                    if result:
                        trades_this_scan += 1
                        total_crypto_trades += 1
                        logger.info(f"💥 CRYPTO TRADE: {symbol}")
                
                # Show crypto stats
                logger.info(f"🪙 Scan {scan_count}: {trades_this_scan} crypto trades | Total: {total_crypto_trades}")
                
                # Show portfolio every 5 scans
                if scan_count % 5 == 0:
                    await self.show_crypto_portfolio()
                
                # Crypto-optimized frequency (faster than stocks)
                logger.info(f"⏰ Next crypto scan in 30 seconds...")
                await asyncio.sleep(30)  # 30-second crypto cycles
                
            except KeyboardInterrupt:
                logger.info("🛑 Pure crypto trading stopped")
                break
            except Exception as e:
                logger.error(f"❌ Crypto trading error: {e}")
                await asyncio.sleep(30)
    
    async def scan_crypto_with_adapter(self, symbol: str) -> bool:
        """Scan crypto using the Paper Trading Adapter"""
        
        try:
            # Get real crypto data from Alpaca
            market_data = await self.get_real_crypto_data(symbol)
            
            if market_data is not None and len(market_data) > 20:
                # USE THE ADAPTER'S 142.98% ML SYSTEM!
                signal = await self.adapter.generate_trading_signal(symbol, market_data)
                
                if signal and signal.get('action') == 'BUY':
                    return await self.execute_crypto_trade(signal)
                else:
                    logger.debug(f"📈 {symbol}: No crypto signal")
            else:
                logger.debug(f"❌ {symbol}: No crypto data")
                
        except Exception as e:
            logger.error(f"❌ Error scanning crypto {symbol}: {e}")
            
        return False
    
    async def get_real_crypto_data(self, symbol: str) -> pd.DataFrame:
        """Get real crypto data from Alpaca"""
        
        if not self.connected:
            return None
            
        try:
            # Get crypto bars with proper ISO format for Alpaca
            start_time = datetime.now() - timedelta(hours=4)
            end_time = datetime.now()
            
            bars = self.api.get_crypto_bars(
                symbol,
                self.TimeFrame.Minute,  # Minute data for crypto responsiveness
                start=start_time.isoformat() + 'Z',
                end=end_time.isoformat() + 'Z'
            )
            
            if bars and len(bars) > 0:
                df = bars.df.reset_index()
                df = df.rename(columns={'timestamp': 'time'})
                logger.debug(f"✅ Got {len(df)} real crypto bars for {symbol}")
                return df
                
        except Exception as e:
            logger.debug(f"Real crypto data failed for {symbol}: {e}")
            
        return None
    
    async def generate_crypto_signal(self, symbol: str, data: pd.DataFrame) -> dict:
        """Generate crypto-specific trading signals"""
        
        try:
            if len(data) < 20:
                return {'action': 'HOLD', 'symbol': symbol}
                
            current_price = data['close'].iloc[-1]
            
            # Crypto-specific technical analysis
            sma_5 = data['close'].rolling(5).mean().iloc[-1]
            sma_10 = data['close'].rolling(10).mean().iloc[-1]
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            
            # Crypto volatility analysis
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * 100  # Percentage volatility
            
            # Volume momentum (crypto loves volume)
            avg_volume = data['volume'].rolling(10).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price momentum (multiple timeframes)
            momentum_5m = (current_price - data['close'].iloc[-5]) / data['close'].iloc[-5]
            momentum_10m = (current_price - data['close'].iloc[-10]) / data['close'].iloc[-10]
            momentum_20m = (current_price - data['close'].iloc[-20]) / data['close'].iloc[-20]
            
            # RSI-like momentum
            price_changes = data['close'].diff().dropna()
            gains = price_changes.where(price_changes > 0, 0)
            losses = -price_changes.where(price_changes < 0, 0)
            avg_gain = gains.rolling(14).mean().iloc[-1]
            avg_loss = losses.rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 50
            
            # Generate crypto confidence
            confidence = 0.05  # Base crypto confidence
            
            # Trend signals (crypto trend following)
            if current_price > sma_5 > sma_10 > sma_20:
                confidence += 0.04  # Strong uptrend
            elif current_price > sma_5 > sma_10:
                confidence += 0.02  # Moderate uptrend
                
            # Volatility factor (crypto thrives on volatility)
            if volatility > 2:  # High crypto volatility
                confidence += volatility * 0.01
                
            # Volume confirmation
            if volume_ratio > 1.5:  # Volume surge
                confidence += 0.03
            elif volume_ratio > 1.2:
                confidence += 0.01
                
            # Momentum signals
            if momentum_5m > 0.01:  # 1% in 5 minutes
                confidence += abs(momentum_5m) * 3
            if momentum_10m > 0.02:  # 2% in 10 minutes  
                confidence += abs(momentum_10m) * 2
            if momentum_20m > 0.03:  # 3% in 20 minutes
                confidence += abs(momentum_20m) * 1.5
                
            # RSI factors
            if 30 < rsi < 70:  # Not overbought/oversold
                confidence += 0.01
            elif rsi < 40:  # Oversold (good buy)
                confidence += 0.02
                
            # Crypto-specific multipliers
            crypto_multipliers = {
                'BTC/USD': 1.0,   # Bitcoin baseline
                'ETH/USD': 1.1,   # Ethereum bonus
                'LTC/USD': 1.2,   # Litecoin bonus
                'BCH/USD': 1.2,   # Bitcoin Cash
                'LINK/USD': 1.3,  # DeFi tokens more volatile
                'UNI/USD': 1.4,   # DEX tokens
                'AAVE/USD': 1.3,  # DeFi lending
                'SUSHI/USD': 1.5  # Highest volatility
            }
            confidence *= crypto_multipliers.get(symbol, 1.0)
            
            # Cap confidence
            confidence = min(confidence, 0.85)
            
            # Crypto trading threshold (lower for 24/7 markets)
            if confidence > 0.12:  # 12% threshold for crypto
                return {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'confidence': confidence,
                    'volatility': volatility,
                    'volume_ratio': volume_ratio,
                    'momentum_5m': momentum_5m,
                    'rsi': rsi,
                    'data_source': 'real_alpaca_crypto',
                    'reasoning': f"Crypto signal: vol={volatility:.1f}%, mom={momentum_5m:.1%}, vol_ratio={volume_ratio:.1f}x, rsi={rsi:.0f}"
                }
                
        except Exception as e:
            logger.error(f"❌ Crypto signal error for {symbol}: {e}")
            
        return {'action': 'HOLD', 'symbol': symbol}
    
    async def execute_crypto_trade_with_adapter(self, signal: dict) -> bool:
        """Execute crypto trade using the adapter signal"""
        
        symbol = signal['symbol']
        
        try:
            if self.connected:
                # Use the platform-specific format from adapter
                platform_signal = signal.get('platform_specific', {})
                
                # Execute the trade using Alpaca
                if 'notional' in str(signal).lower() or signal.get('size'):
                    position_value = signal.get('size', 1000)  # Default $1000
                    
                    try:
                        # Method 1: Notional order (preferred for crypto)
                        order = self.api.submit_order(
                            symbol=symbol,
                            notional=round(position_value, 2),
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                        
                        logger.info(f"🪙 ADAPTER CRYPTO TRADE EXECUTED!")
                        logger.info(f"   💰 ${position_value:.2f} of {symbol}")
                        logger.info(f"   📊 Order ID: {order.id}")
                        logger.info(f"   🧠 Confidence: {signal.get('confidence', 0):.1%}")
                        logger.info(f"   🔥 ML Signal via PaperTradingAdapter!")
                        
                        return True
                        
                    except Exception as e1:
                        logger.debug(f"Adapter crypto notional failed: {e1}")
                        
                        try:
                            # Method 2: Quantity order
                            quantity = position_value / signal['price']
                            order = self.api.submit_order(
                                symbol=symbol,
                                qty=quantity,
                                side='buy',
                                type='market',
                                time_in_force='gtc'
                            )
                            
                            logger.info(f"🪙 ADAPTER CRYPTO TRADE (QTY) EXECUTED!")
                            logger.info(f"   💰 {quantity:.8f} {symbol}")
                            logger.info(f"   📊 Order ID: {order.id}")
                            
                            return True
                            
                        except Exception as e2:
                            logger.error(f"❌ Both adapter crypto methods failed: {e2}")
                            
        except Exception as e:
            logger.error(f"❌ Adapter crypto execution failed for {symbol}: {e}")
            
        return False
    
    async def execute_crypto_trade(self, signal: dict) -> bool:
        """Execute crypto trade on Alpaca"""
        
        symbol = signal['symbol']
        
        try:
            if self.connected:
                account = self.api.get_account()
                buying_power = float(account.buying_power)
                
                # Crypto position sizing (5% per trade for diversification)
                position_pct = 0.05
                position_value = buying_power * position_pct
                
                # Minimum crypto trade ($100)
                if position_value >= 100:
                    try:
                        # Method 1: Notional order (preferred for crypto)
                        order = self.api.submit_order(
                            symbol=symbol,
                            notional=round(position_value, 2),  # Round to 2 decimals
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                        
                        logger.info(f"🪙 CRYPTO TRADE EXECUTED!")
                        logger.info(f"   💰 ${position_value:.2f} of {symbol}")
                        logger.info(f"   📊 Order ID: {order.id}")
                        logger.info(f"   🧠 Confidence: {signal['confidence']:.1%}")
                        logger.info(f"   📈 Volatility: {signal.get('volatility', 0):.1f}%")
                        logger.info(f"   🔥 Real Alpaca crypto data!")
                        
                        return True
                        
                    except Exception as e1:
                        logger.debug(f"Crypto notional failed: {e1}")
                        
                        try:
                            # Method 2: Quantity order
                            quantity = position_value / signal['price']
                            order = self.api.submit_order(
                                symbol=symbol,
                                qty=quantity,
                                side='buy',
                                type='market',
                                time_in_force='gtc'
                            )
                            
                            logger.info(f"🪙 CRYPTO TRADE (QTY) EXECUTED!")
                            logger.info(f"   💰 {quantity:.8f} {symbol}")
                            logger.info(f"   📊 Order ID: {order.id}")
                            
                            return True
                            
                        except Exception as e2:
                            logger.error(f"❌ Both crypto methods failed: {e2}")
                            
        except Exception as e:
            logger.error(f"❌ Crypto execution failed for {symbol}: {e}")
            
        return False
    
    async def show_crypto_portfolio(self):
        """Show crypto portfolio status"""
        
        try:
            if self.connected:
                account = self.api.get_account()
                positions = self.api.list_positions()
                
                portfolio_value = float(account.portfolio_value)
                buying_power = float(account.buying_power)
                
                crypto_positions = [p for p in positions if '/' in p.symbol]
                
                logger.info(f"🪙 CRYPTO PORTFOLIO:")
                logger.info(f"   💰 Total Value: ${portfolio_value:,.2f}")
                logger.info(f"   💵 Buying Power: ${buying_power:,.2f}")
                logger.info(f"   📊 Crypto Positions: {len(crypto_positions)}")
                
                if crypto_positions:
                    total_crypto_value = 0
                    total_pnl = 0
                    
                    for pos in crypto_positions:
                        market_value = float(pos.market_value)
                        pnl = float(pos.unrealized_pl)
                        pnl_pct = float(pos.unrealized_plpc) * 100
                        
                        total_crypto_value += market_value
                        total_pnl += pnl
                        
                        logger.info(f"   🪙 {pos.symbol}: ${market_value:,.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
                    
                    logger.info(f"   📈 Total Crypto P&L: ${total_pnl:,.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Portfolio check failed: {e}")


async def main():
    """Run pure crypto trading"""
    
    trader = PureCryptoTrader()
    await trader.start_pure_crypto_trading()


if __name__ == "__main__":
    asyncio.run(main())
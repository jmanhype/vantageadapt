#!/usr/bin/env python3
"""
INSTANT CRYPTO TRADER - NO TRAINING, IMMEDIATE TRADES
Skip ML training, use aggressive signals, trade NOW
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


class InstantCryptoTrader:
    """
    INSTANT CRYPTO TRADING - NO TRAINING DELAYS
    """
    
    def __init__(self):
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                key_id=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url='https://paper-api.alpaca.markets'
            )
            
            self.TimeFrame = tradeapi.TimeFrame
            
            account = self.api.get_account()
            logger.info(f"⚡ INSTANT CRYPTO MODE: ${float(account.buying_power):,.2f}")
            self.connected = True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False
    
    async def instant_crypto_trading(self):
        """INSTANT crypto trading - no training delays"""
        
        print(f"""
        ⚡ INSTANT CRYPTO TRADING - ZERO DELAYS ⚡
        
        🎯 Mission: IMMEDIATE crypto positions
        🚀 Strategy: Aggressive signals, no ML training
        💰 Target: $1000+ per crypto trade  
        🔥 Speed: Trade within 10 seconds!
        
        🪙 CRYPTO TARGETS:
        • BTC/USD - Bitcoin (King Crypto)
        • ETH/USD - Ethereum (DeFi King)
        • LTC/USD - Litecoin (Fast & Cheap)
        • BCH/USD - Bitcoin Cash (Payments)
        
        ⚡ LAUNCHING INSTANT TRADES...
        """)
        
        crypto_symbols = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD']
        
        scan_count = 0
        total_trades = 0
        
        while True:
            try:
                scan_count += 1
                logger.info(f"⚡ INSTANT SCAN #{scan_count}")
                
                trades_this_scan = 0
                
                for symbol in crypto_symbols:
                    try:
                        # Get instant signal
                        signal = await self.generate_instant_signal(symbol)
                        
                        if signal and signal.get('action') == 'BUY':
                            result = await self.execute_instant_trade(signal)
                            if result:
                                trades_this_scan += 1
                                total_trades += 1
                                logger.info(f"💥 INSTANT TRADE: {symbol}")
                        
                    except Exception as e:
                        logger.debug(f"Scan error {symbol}: {e}")
                
                logger.info(f"⚡ Scan {scan_count}: {trades_this_scan} trades | Total: {total_trades}")
                
                # Show portfolio every 3 scans
                if scan_count % 3 == 0:
                    await self.show_portfolio()
                
                # Fast crypto cycles
                logger.info(f"⏰ Next scan in 15 seconds...")
                await asyncio.sleep(15)
                
            except KeyboardInterrupt:
                logger.info("🛑 Instant crypto trading stopped")
                break
            except Exception as e:
                logger.error(f"❌ Trading error: {e}")
                await asyncio.sleep(15)
    
    async def generate_instant_signal(self, symbol: str) -> dict:
        """Generate INSTANT aggressive crypto signal"""
        
        try:
            # Get real crypto data
            data = await self.get_crypto_data(symbol)
            
            if data is None or len(data) < 10:
                return {'action': 'HOLD', 'symbol': symbol}
            
            current_price = data['close'].iloc[-1]
            
            # INSTANT AGGRESSIVE SIGNALS
            confidence = 0.05  # Base confidence
            
            # Price momentum (last 10 minutes)
            if len(data) >= 10:
                price_10m_ago = data['close'].iloc[-10]
                momentum = (current_price - price_10m_ago) / price_10m_ago
                
                if momentum > 0.001:  # 0.1% up = signal
                    confidence += abs(momentum) * 10
            
            # Volume surge detection
            if len(data) >= 5:
                recent_volume = data['volume'].iloc[-5:].mean()
                older_volume = data['volume'].iloc[-10:-5].mean()
                
                if recent_volume > older_volume * 1.2:  # 20% volume increase
                    confidence += 0.08
            
            # Volatility bonus (crypto loves volatility)
            if len(data) >= 5:
                returns = data['close'].pct_change().iloc[-5:]
                volatility = returns.std()
                if volatility > 0.01:  # 1% volatility
                    confidence += volatility * 2
            
            # Random aggressive factor (instant mode)
            confidence += np.random.random() * 0.05
            
            # INSTANT TRADE THRESHOLD (very low for immediate action)
            if confidence > 0.08:  # 8% threshold for instant trades
                return {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'confidence': confidence,
                    'reasoning': f"Instant signal: {confidence:.1%} confidence",
                    'data_source': 'alpaca_real_crypto'
                }
                
        except Exception as e:
            logger.error(f"❌ Signal error for {symbol}: {e}")
            
        return {'action': 'HOLD', 'symbol': symbol}
    
    async def get_crypto_data(self, symbol: str) -> pd.DataFrame:
        """Get real crypto data quickly"""
        
        if not self.connected:
            return None
            
        try:
            start_time = datetime.now() - timedelta(hours=2)
            end_time = datetime.now()
            
            bars = self.api.get_crypto_bars(
                symbol,
                self.TimeFrame.Minute,
                start=start_time.isoformat() + 'Z',
                end=end_time.isoformat() + 'Z'
            )
            
            if bars and len(bars) > 0:
                df = bars.df.reset_index()
                return df
                
        except Exception as e:
            logger.debug(f"Data error {symbol}: {e}")
            
        return None
    
    async def execute_instant_trade(self, signal: dict) -> bool:
        """Execute INSTANT crypto trade"""
        
        symbol = signal['symbol']
        
        try:
            if self.connected:
                account = self.api.get_account()
                buying_power = float(account.buying_power)
                
                # AGGRESSIVE position sizing for instant impact
                position_value = min(buying_power * 0.2, 2000)  # 20% or $2000 max
                
                if position_value >= 100:  # Min $100 trade
                    try:
                        order = self.api.submit_order(
                            symbol=symbol,
                            notional=round(position_value, 2),
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                        
                        logger.info(f"⚡ INSTANT CRYPTO TRADE EXECUTED!")
                        logger.info(f"   💰 ${position_value:.2f} of {symbol}")
                        logger.info(f"   🎯 Price: ${signal['price']:.2f}")
                        logger.info(f"   📊 Order: {order.id}")
                        logger.info(f"   🧠 Confidence: {signal['confidence']:.1%}")
                        logger.info(f"   ⚡ INSTANT EXECUTION - NO DELAYS!")
                        
                        return True
                        
                    except Exception as e:
                        logger.error(f"❌ Trade failed: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Execution failed for {symbol}: {e}")
            
        return False
    
    async def show_portfolio(self):
        """Show current crypto portfolio"""
        
        try:
            if self.connected:
                account = self.api.get_account()
                positions = self.api.list_positions()
                
                portfolio_value = float(account.portfolio_value)
                buying_power = float(account.buying_power)
                
                crypto_positions = [p for p in positions if '/' in p.symbol]
                
                logger.info(f"⚡ INSTANT CRYPTO PORTFOLIO:")
                logger.info(f"   💰 Total Value: ${portfolio_value:,.2f}")
                logger.info(f"   💵 Buying Power: ${buying_power:,.2f}")
                logger.info(f"   🪙 Crypto Positions: {len(crypto_positions)}")
                
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
    """Launch instant crypto trading"""
    
    trader = InstantCryptoTrader()
    await trader.instant_crypto_trading()


if __name__ == "__main__":
    asyncio.run(main())
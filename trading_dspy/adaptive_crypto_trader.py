#!/usr/bin/env python3
"""
ADAPTIVE CRYPTO TRADER - TRAINS ON REAL CRYPTO DATA
Uses the 142.98% ML system but trains on actual Alpaca crypto data
"""

import asyncio
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Import ML components directly
from src.ml_trading_engine import MLTradingEngine, TradeSignal

# Set Alpaca keys
os.environ['ALPACA_API_KEY'] = "PKV0EUF7LNIUB2TJMTIK"
os.environ['ALPACA_SECRET_KEY'] = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"


class AdaptiveCryptoTrader:
    """
    ADAPTIVE CRYPTO TRADING - Trains ML models on real crypto data
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
            
            # Initialize ML trading engine
            self.ml_engine = MLTradingEngine()
            self.ml_trained = False
            
            account = self.api.get_account()
            logger.info(f"🧠 Adaptive Crypto Mode: ${float(account.buying_power):,.2f}")
            self.connected = True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.connected = False
    
    async def start_adaptive_crypto_trading(self):
        """Start adaptive crypto trading with real-time ML training"""
        
        print(f"""
        🧠 ADAPTIVE CRYPTO TRADING - REAL DATA ML TRAINING 🧠
        
        🎯 Mission: Train ML models on REAL crypto data
        🚀 Strategy: 142.98% ML system adapted to crypto markets
        💰 Target: Learn crypto patterns and trade accordingly
        🔥 Frequency: Retrain and trade every cycle!
        
        🪙 CRYPTO TARGETS:
        • BTC/USD - Bitcoin (King Crypto)
        • ETH/USD - Ethereum (DeFi King)  
        • LTC/USD - Litecoin (Fast & Cheap)
        • BCH/USD - Bitcoin Cash (Payments)
        • LINK/USD - Chainlink (Oracle Network)
        • UNI/USD - Uniswap (DEX Leader)
        • AAVE/USD - AAVE (DeFi Lending)
        • SUSHI/USD - SushiSwap (DEX)
        
        🧠 STARTING ADAPTIVE ML LEARNING...
        """)
        
        crypto_symbols = [
            'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD',
            'LINK/USD', 'UNI/USD', 'AAVE/USD', 'SUSHI/USD'
        ]
        
        scan_count = 0
        total_trades = 0
        
        # First, collect crypto data for training
        await self.collect_and_train_on_crypto_data(crypto_symbols)
        
        while True:
            try:
                scan_count += 1
                logger.info(f"🧠 ADAPTIVE SCAN #{scan_count}")
                
                trades_this_scan = 0
                
                for symbol in crypto_symbols:
                    try:
                        result = await self.scan_crypto_with_adaptive_ml(symbol)
                        if result:
                            trades_this_scan += 1
                            total_trades += 1
                            logger.info(f"💥 ADAPTIVE TRADE: {symbol}")
                        
                    except Exception as e:
                        logger.debug(f"Scan error {symbol}: {e}")
                
                logger.info(f"🧠 Scan {scan_count}: {trades_this_scan} trades | Total: {total_trades}")
                
                # Show portfolio every 5 scans
                if scan_count % 5 == 0:
                    await self.show_portfolio()
                
                # Retrain periodically on new data
                if scan_count % 10 == 0:
                    logger.info("🔄 Retraining ML models on fresh crypto data...")
                    await self.collect_and_train_on_crypto_data(crypto_symbols)
                
                # Adaptive crypto cycles
                logger.info(f"⏰ Next adaptive scan in 45 seconds...")
                await asyncio.sleep(45)
                
            except KeyboardInterrupt:
                logger.info("🛑 Adaptive crypto trading stopped")
                break
            except Exception as e:
                logger.error(f"❌ Trading error: {e}")
                await asyncio.sleep(45)
    
    async def collect_and_train_on_crypto_data(self, crypto_symbols):
        """Collect crypto data and train ML models on it"""
        
        logger.info("📊 Collecting crypto data for ML training...")
        
        # Collect substantial data from all cryptos
        all_crypto_data = []
        
        for symbol in crypto_symbols:
            try:
                # Get more historical data for training
                data = await self.get_crypto_data_for_training(symbol)
                if data is not None and len(data) > 100:
                    # Add symbol identifier
                    data['symbol'] = symbol
                    all_crypto_data.append(data)
                    logger.info(f"   ✅ {symbol}: {len(data)} bars collected")
                else:
                    logger.warning(f"   ❌ {symbol}: Insufficient data")
                    
            except Exception as e:
                logger.error(f"   ❌ {symbol}: Data collection failed - {e}")
        
        if all_crypto_data:
            # Combine all crypto data
            combined_data = pd.concat(all_crypto_data, ignore_index=True)
            logger.info(f"📈 Training ML models on {len(combined_data)} crypto data points")
            
            # Train ML models on this real crypto data
            try:
                self.ml_engine.train(combined_data)
                self.ml_trained = True
                logger.info("✅ ML models trained successfully on crypto data!")
                
            except Exception as e:
                logger.error(f"❌ ML training failed: {e}")
                self.ml_trained = False
        else:
            logger.error("❌ No crypto data collected for training")
            self.ml_trained = False
    
    async def get_crypto_data_for_training(self, symbol: str) -> pd.DataFrame:
        """Get substantial crypto data for ML training"""
        
        if not self.connected:
            return None
            
        try:
            # Get more data for training (24 hours)
            start_time = datetime.now() - timedelta(hours=24)
            end_time = datetime.now()
            
            bars = self.api.get_crypto_bars(
                symbol,
                self.TimeFrame.Minute,
                start=start_time.isoformat() + 'Z',
                end=end_time.isoformat() + 'Z'
            )
            
            if bars and len(bars) > 0:
                df = bars.df.reset_index()
                df = df.rename(columns={'timestamp': 'time'})
                
                # Add technical indicators for ML training
                df = self.add_technical_indicators(df)
                
                return df
                
        except Exception as e:
            logger.debug(f"Training data error {symbol}: {e}")
            
        return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators needed for ML training"""
        
        try:
            # Price-based indicators
            df['returns_1h'] = df['close'].pct_change(60)  # 1 hour returns
            df['returns_4h'] = df['close'].pct_change(240)  # 4 hour returns
            df['volatility_1h'] = df['returns_1h'].rolling(60).std()
            df['volatility_24h'] = df['returns_1h'].rolling(1440).std()
            
            # Volume indicators
            df['dollar_volume'] = df['close'] * df['volume']
            df['volume_ma'] = df['volume'].rolling(20).mean()
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['close'].rolling(bb_period).mean()
            bb_rolling_std = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_rolling_std * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_rolling_std * bb_std)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price ratios
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Technical indicators error: {e}")
            return df
    
    async def scan_crypto_with_adaptive_ml(self, symbol: str) -> bool:
        """Scan crypto using adaptively trained ML models"""
        
        if not self.ml_trained:
            logger.debug(f"ML not trained yet for {symbol}")
            return False
            
        try:
            # Get current crypto data
            data = await self.get_current_crypto_data(symbol)
            
            if data is not None and len(data) > 20:
                # Generate ML signal using trained models
                signal = await self.generate_ml_crypto_signal(symbol, data)
                
                if signal and signal.action == 'BUY':
                    return await self.execute_crypto_trade(signal)
                else:
                    logger.debug(f"📈 {symbol}: No ML signal")
            else:
                logger.debug(f"❌ {symbol}: Insufficient data")
                
        except Exception as e:
            logger.error(f"❌ Error scanning {symbol}: {e}")
            
        return False
    
    async def get_current_crypto_data(self, symbol: str) -> pd.DataFrame:
        """Get current crypto data for signal generation"""
        
        if not self.connected:
            return None
            
        try:
            start_time = datetime.now() - timedelta(hours=4)
            end_time = datetime.now()
            
            bars = self.api.get_crypto_bars(
                symbol,
                self.TimeFrame.Minute,
                start=start_time.isoformat() + 'Z',
                end=end_time.isoformat() + 'Z'
            )
            
            if bars and len(bars) > 0:
                df = bars.df.reset_index()
                df = df.rename(columns={'timestamp': 'time'})
                df = self.add_technical_indicators(df)
                return df
                
        except Exception as e:
            logger.debug(f"Current data error {symbol}: {e}")
            
        return None
    
    async def generate_ml_crypto_signal(self, symbol: str, data: pd.DataFrame) -> TradeSignal:
        """Generate ML signal using trained crypto models"""
        
        try:
            if not self.ml_trained:
                return None
                
            # Use the trained ML engine to generate signal
            signal = self.ml_engine.generate_signal(data)
            
            if signal:
                logger.debug(f"🧠 {symbol}: ML signal - {signal.probability:.1%} confidence")
                return signal
                
        except Exception as e:
            logger.error(f"❌ ML signal error for {symbol}: {e}")
            
        return None
    
    async def execute_crypto_trade(self, signal: TradeSignal) -> bool:
        """Execute crypto trade based on ML signal"""
        
        try:
            if self.connected:
                account = self.api.get_account()
                buying_power = float(account.buying_power)
                
                # Position sizing based on ML confidence
                base_position_pct = 0.08  # 8% base position
                confidence_multiplier = signal.probability * 2  # Scale with confidence
                position_pct = min(base_position_pct * confidence_multiplier, 0.15)  # Max 15%
                
                position_value = buying_power * position_pct
                
                if position_value >= 100:  # Min $100 trade
                    try:
                        order = self.api.submit_order(
                            symbol=signal.symbol,
                            notional=round(position_value, 2),
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                        
                        logger.info(f"🧠 ADAPTIVE ML CRYPTO TRADE EXECUTED!")
                        logger.info(f"   💰 ${position_value:.2f} of {signal.symbol}")
                        logger.info(f"   🎯 ML Confidence: {signal.probability:.1%}")
                        logger.info(f"   📊 Order ID: {order.id}")
                        logger.info(f"   🧠 Reasoning: {signal.reasoning}")
                        logger.info(f"   🔥 Trained on REAL crypto data!")
                        
                        return True
                        
                    except Exception as e:
                        logger.error(f"❌ Trade execution failed: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Trade failed: {e}")
            
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
                
                logger.info(f"🧠 ADAPTIVE CRYPTO PORTFOLIO:")
                logger.info(f"   💰 Total Value: ${portfolio_value:,.2f}")
                logger.info(f"   💵 Buying Power: ${buying_power:,.2f}")
                logger.info(f"   🪙 Crypto Positions: {len(crypto_positions)}")
                logger.info(f"   🧠 ML Training Status: {'✅ Trained' if self.ml_trained else '❌ Not Trained'}")
                
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
    """Launch adaptive crypto trading"""
    
    trader = AdaptiveCryptoTrader()
    await trader.start_adaptive_crypto_trading()


if __name__ == "__main__":
    asyncio.run(main())
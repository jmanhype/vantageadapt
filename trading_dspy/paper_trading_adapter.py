#!/usr/bin/env python3
"""
PAPER TRADING ADAPTER FOR MEGAZORD
Connects the ML signals to any paper trading platform
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger
import pandas as pd

# Import the Megazord brain
from src.kagan_megazord_coordinator import KaganMegazordCoordinator
from src.ml_trading_engine import TradeSignal


class PaperTradingAdapter:
    """
    Adapts Megazord signals for paper trading on ANY platform
    """
    
    def __init__(self, platform: str = "generic"):
        self.platform = platform
        self.active_positions = {}
        self.trade_history = []
        self.capital = 100000  # Starting capital
        
        # Initialize Megazord
        self.megazord = KaganMegazordCoordinator()
        
        logger.info(f"📄 Paper Trading Adapter initialized for {platform}")
        
    async def get_real_time_signal(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get trading signal for current market conditions
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USD", "AAPL", "EUR/USD")
            market_data: Recent price data (last 100-1000 candles)
            
        Returns:
            Trade signal with all parameters
        """
        
        # Ensure ML is trained
        if not self.megazord.ml_trained:
            logger.info("Training ML models on provided data...")
            await self.megazord._initial_ml_training()
        
        # Generate signal
        signal = await self.megazord._generate_ml_signal(market_data)
        
        if signal and signal.action == 'BUY':
            current_price = market_data['close'].iloc[-1]
            
            trade_signal = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'action': 'BUY',
                'price': current_price,
                'size': self.capital * signal.position_size,
                'stop_loss': current_price * (1 - signal.stop_loss),
                'take_profit': current_price * (1 + signal.take_profit),
                'confidence': signal.probability,
                'reasoning': signal.reasoning,
                'platform_specific': self._get_platform_format(signal, symbol, current_price)
            }
            
            logger.info(f"🎯 SIGNAL: {symbol} BUY @ ${current_price:.4f}")
            logger.info(f"   Size: ${trade_signal['size']:.2f}")
            logger.info(f"   Stop: ${trade_signal['stop_loss']:.4f}")
            logger.info(f"   Target: ${trade_signal['take_profit']:.4f}")
            logger.info(f"   Confidence: {trade_signal['confidence']:.1%}")
            
            return trade_signal
        
        return {'action': 'HOLD', 'symbol': symbol}
    
    def _get_platform_format(self, signal: TradeSignal, symbol: str, price: float) -> Dict:
        """Format signal for specific trading platforms"""
        
        if self.platform == "alpaca":
            return {
                "symbol": symbol,
                "qty": int(self.capital * signal.position_size / price),
                "side": "buy",
                "type": "limit",
                "limit_price": price,
                "time_in_force": "day",
                "order_class": "bracket",
                "stop_loss": {"stop_price": price * (1 - signal.stop_loss)},
                "take_profit": {"limit_price": price * (1 + signal.take_profit)}
            }
            
        elif self.platform == "interactive_brokers":
            return {
                "action": "BUY",
                "totalQuantity": int(self.capital * signal.position_size / price),
                "orderType": "LMT",
                "lmtPrice": price,
                "transmit": True,
                "parentId": None,
                "stopLoss": price * (1 - signal.stop_loss),
                "profitTaker": price * (1 + signal.take_profit)
            }
            
        elif self.platform == "tradingview":
            return {
                "action": "buy",
                "contracts": f"{signal.position_size * 100:.1f}%",
                "price": "market",
                "alert_message": f"ML Signal: {signal.probability:.1%} confidence"
            }
            
        else:  # Generic format
            return {
                "side": "buy",
                "size_percent": signal.position_size * 100,
                "stop_loss_percent": signal.stop_loss * 100,
                "take_profit_percent": signal.take_profit * 100
            }
    
    async def run_paper_trading_loop(self, get_market_data_func):
        """
        Main paper trading loop
        
        Args:
            get_market_data_func: Function that returns current market data
        """
        
        logger.info("🚀 Starting paper trading loop...")
        
        while True:
            try:
                # Get current market data
                symbols = ["BTC/USD", "ETH/USD", "AAPL", "TSLA"]  # Your symbols
                
                for symbol in symbols:
                    # Get latest data
                    market_data = await get_market_data_func(symbol)
                    
                    if market_data is not None and len(market_data) > 100:
                        # Check for signals
                        signal = await self.get_real_time_signal(symbol, market_data)
                        
                        if signal['action'] == 'BUY':
                            # Log the trade
                            self.active_positions[symbol] = signal
                            self.trade_history.append(signal)
                            
                            # HERE: Send to your paper trading platform
                            # Example: await send_to_alpaca(signal['platform_specific'])
                
                # Check existing positions for exits
                await self._check_exit_conditions(get_market_data_func)
                
                # Sleep before next check (adjust based on your needs)
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in paper trading loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_exit_conditions(self, get_market_data_func):
        """Check if any positions should be closed"""
        
        for symbol, position in list(self.active_positions.items()):
            current_data = await get_market_data_func(symbol)
            if current_data is None:
                continue
                
            current_price = current_data['close'].iloc[-1]
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                logger.info(f"❌ STOP LOSS: {symbol} @ ${current_price:.4f}")
                del self.active_positions[symbol]
                
            # Check take profit
            elif current_price >= position['take_profit']:
                logger.info(f"✅ TAKE PROFIT: {symbol} @ ${current_price:.4f}")
                del self.active_positions[symbol]


# Example usage for different platforms
async def example_alpaca_paper_trading():
    """Example: Alpaca paper trading"""
    
    adapter = PaperTradingAdapter(platform="alpaca")
    
    # Your function to get market data
    async def get_market_data(symbol):
        # Replace with your data source (Alpaca API, Yahoo Finance, etc.)
        # Return DataFrame with OHLCV data
        pass
    
    await adapter.run_paper_trading_loop(get_market_data)


async def example_tradingview_alerts():
    """Example: TradingView webhook alerts"""
    
    adapter = PaperTradingAdapter(platform="tradingview")
    
    # Process webhook data
    async def process_webhook(webhook_data):
        symbol = webhook_data['symbol']
        market_data = pd.DataFrame(webhook_data['candles'])
        
        signal = await adapter.get_real_time_signal(symbol, market_data)
        
        if signal['action'] == 'BUY':
            # Send alert back to TradingView
            return {
                "alert": f"BUY {symbol}",
                "message": signal['platform_specific']['alert_message']
            }


if __name__ == "__main__":
    # Run paper trading
    asyncio.run(example_alpaca_paper_trading())
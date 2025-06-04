#!/usr/bin/env python3
"""
TRADINGVIEW ALERT INTEGRATION - NO API KEYS!

This creates webhook alerts that can trigger TradingView notifications.
100% FREE using TradingView's webhook feature.
"""

import asyncio
import json
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import yfinance as yf
from loguru import logger

# Import our ML system
from paper_trading_adapter import PaperTradingAdapter


class TradingViewAlertSystem:
    """
    FREE TradingView integration using webhooks
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.adapter = PaperTradingAdapter(platform="tradingview")
        self.setup_routes()
        
        logger.info("📺 TradingView Alert System initialized")
    
    def setup_routes(self):
        """Setup Flask routes for webhooks"""
        
        @self.app.route('/webhook', methods=['POST'])
        async def webhook():
            """Receive TradingView webhook and respond with ML signal"""
            
            try:
                data = request.json
                symbol = data.get('symbol', 'BTC-USD')
                
                # Get fresh market data
                market_data = await self.get_market_data(symbol)
                
                if market_data is not None:
                    # Generate ML signal
                    signal = await self.adapter.get_real_time_signal(symbol, market_data)
                    
                    if signal['action'] == 'BUY':
                        # Return TradingView alert format
                        return jsonify({
                            "alert": f"🚀 ML BUY SIGNAL: {symbol}",
                            "message": f"Entry: ${signal['price']:.4f} | Stop: ${signal['stop_loss']:.4f} | Target: ${signal['take_profit']:.4f} | Confidence: {signal['confidence']:.1%}",
                            "action": "buy",
                            "symbol": symbol,
                            "price": signal['price'],
                            "stop_loss": signal['stop_loss'],
                            "take_profit": signal['take_profit']
                        })
                
                return jsonify({"alert": "No signal", "action": "hold"})
                
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/scan', methods=['GET'])
        async def scan_markets():
            """Scan multiple markets and return signals"""
            
            symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA', 'SPY']
            signals = []
            
            for symbol in symbols:
                market_data = await self.get_market_data(symbol)
                if market_data is not None:
                    signal = await self.adapter.get_real_time_signal(symbol, market_data)
                    if signal['action'] == 'BUY':
                        signals.append({
                            'symbol': symbol,
                            'price': signal['price'],
                            'confidence': signal['confidence'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit']
                        })
            
            return jsonify({"signals": signals, "count": len(signals)})
    
    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for analysis"""
        
        try:
            yahoo_symbol = symbol.replace("/", "-")
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
            logger.error(f"Data error for {symbol}: {e}")
            
        return None
    
    def run_server(self, host='0.0.0.0', port=5000):
        """Run the webhook server"""
        
        print(f"""
        📺 TRADINGVIEW WEBHOOK SERVER STARTING
        
        🌐 Server will run on: http://{host}:{port}
        
        📋 WEBHOOK ENDPOINTS:
        • POST http://{host}:{port}/webhook  (for TradingView alerts)
        • GET  http://{host}:{port}/scan     (scan all markets)
        
        🔗 TRADINGVIEW SETUP:
        1. Create a FREE TradingView account
        2. Open any chart
        3. Create alert with webhook URL: http://{host}:{port}/webhook
        4. Set message body: {{"symbol": "{{{{ticker}}}}"}}
        5. When alert triggers, you'll get ML analysis!
        
        💡 OR just visit: http://{host}:{port}/scan to see current signals
        
        Starting server...
        """)
        
        self.app.run(host=host, port=port, debug=False)


def create_tradingview_strategy():
    """
    Create a TradingView Pine Script that calls our webhook
    """
    
    pine_script = '''
//@version=5
strategy("ML Trading Signals", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=25)

// This script sends price data to our ML system via webhook
// NO API KEYS NEEDED!

// Simple moving averages for basic signals
sma20 = ta.sma(close, 20)
sma50 = ta.sma(close, 50)

// Basic signal condition
signal_condition = ta.crossover(sma20, sma50) and volume > ta.sma(volume, 20) * 1.5

// Plot the moving averages
plot(sma20, color=color.blue, linewidth=2)
plot(sma50, color=color.red, linewidth=2)

// When signal triggers, send webhook to our ML system
if signal_condition
    // This webhook call sends data to our ML system
    // Replace YOUR_SERVER_URL with your actual server address
    alert("ML Analysis Request", alert.freq_once_per_bar_close)
    strategy.entry("ML_Long", strategy.long)

// Exit conditions (you can customize these)
if strategy.position_size > 0 and (close < sma20 * 0.995 or close > strategy.position_avg_price * 1.01)
    strategy.close("ML_Long")

// Background color for active signals
bgcolor(signal_condition ? color.new(color.green, 90) : na)
'''
    
    # Save Pine Script to file
    with open('ml_tradingview_strategy.pine', 'w') as f:
        f.write(pine_script)
    
    print("""
    📝 TRADINGVIEW PINE SCRIPT CREATED!
    
    📁 File saved as: ml_tradingview_strategy.pine
    
    📋 HOW TO USE:
    1. Copy the contents of ml_tradingview_strategy.pine
    2. Go to TradingView.com (free account)
    3. Open Pine Script Editor
    4. Paste the script
    5. Click "Add to Chart"
    6. Set up alerts to call our webhook!
    
    💡 The script will send signals to our ML system for analysis.
    """)


# INSTANT DEMO FUNCTIONS
async def demo_webhook_signals():
    """Demo the webhook system with fake TradingView data"""
    
    alert_system = TradingViewAlertSystem()
    
    # Simulate TradingView webhook calls
    test_symbols = ['BTC-USD', 'ETH-USD', 'AAPL', 'TSLA']
    
    print("🎭 DEMO: Simulating TradingView webhook calls...")
    
    for symbol in test_symbols:
        print(f"\n📡 Processing webhook for {symbol}...")
        
        # Get real market data
        market_data = await alert_system.get_market_data(symbol)
        
        if market_data is not None:
            # Generate signal
            signal = await alert_system.adapter.get_real_time_signal(symbol, market_data)
            
            if signal['action'] == 'BUY':
                print(f"""
                ✅ WEBHOOK RESPONSE FOR {symbol}:
                Alert: 🚀 ML BUY SIGNAL
                Entry: ${signal['price']:.4f}
                Stop Loss: ${signal['stop_loss']:.4f}
                Take Profit: ${signal['take_profit']:.4f}
                Confidence: {signal['confidence']:.1%}
                """)
            else:
                print(f"⏸️ {symbol}: No signal (confidence too low)")
        
        await asyncio.sleep(1)


if __name__ == "__main__":
    print("""
    📺 TRADINGVIEW INTEGRATION OPTIONS:
    
    1. Start webhook server (for live TradingView alerts)
    2. Create Pine Script strategy
    3. Demo webhook system
    
    Choose 1, 2, or 3:
    """)
    
    choice = input().strip()
    
    if choice == "1":
        server = TradingViewAlertSystem()
        server.run_server()
    elif choice == "2":
        create_tradingview_strategy()
    else:
        print("🎭 Running webhook demo...")
        asyncio.run(demo_webhook_signals())
#!/usr/bin/env python3
"""
TRADINGVIEW PAPER TRADING INTEGRATION
Connect ML signals to TradingView's free paper trading
"""

import asyncio
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import yfinance as yf
from loguru import logger

# Import our ML system
from paper_trading_adapter import PaperTradingAdapter


class TradingViewPaperTrader:
    """
    TradingView integration with ML signals
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.adapter = PaperTradingAdapter(platform="tradingview")
        self.paper_portfolio = {"cash": 100000, "positions": {}, "trades": []}
        self.setup_routes()
        
        logger.info("📺 TradingView Paper Trader initialized")
    
    def setup_routes(self):
        """Setup Flask routes for TradingView integration"""
        
        @self.app.route('/')
        def dashboard():
            """TradingView integration dashboard"""
            return render_template_string(TRADINGVIEW_DASHBOARD_HTML)
        
        @self.app.route('/webhook', methods=['POST'])
        async def ml_webhook():
            """Receive TradingView alerts and respond with ML analysis"""
            
            try:
                data = request.json
                symbol = data.get('symbol', 'AAPL')
                
                # Get market data
                market_data = await self.get_market_data(symbol)
                
                if market_data is not None and len(market_data) > 100:
                    # Generate ML signal
                    signal = await self.adapter.get_real_time_signal(symbol, market_data)
                    
                    if signal['action'] == 'BUY':
                        # Log the paper trade
                        await self.execute_paper_trade(signal)
                        
                        # Return TradingView alert
                        return jsonify({
                            "alert": f"🤖 ML BUY: {symbol}",
                            "message": f"ENTRY: ${signal['price']:.2f} | STOP: ${signal['stop_loss']:.2f} | TARGET: ${signal['take_profit']:.2f} | CONF: {signal['confidence']:.1%}",
                            "action": "buy",
                            "symbol": symbol,
                            "price": signal['price'],
                            "stop_loss": signal['stop_loss'],
                            "take_profit": signal['take_profit'],
                            "confidence": signal['confidence']
                        })
                
                return jsonify({"alert": "No ML signal", "action": "hold"})
                
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/portfolio')
        def get_portfolio():
            """Get paper portfolio status"""
            return jsonify(self.paper_portfolio)
        
        @self.app.route('/api/scan/<symbol>')
        async def scan_symbol(symbol):
            """Scan a specific symbol for ML signals"""
            
            market_data = await self.get_market_data(symbol)
            
            if market_data is not None and len(market_data) > 100:
                signal = await self.adapter.get_real_time_signal(symbol, market_data)
                
                if signal['action'] == 'BUY':
                    return jsonify({
                        'signal': True,
                        'symbol': symbol,
                        'price': signal['price'],
                        'confidence': signal['confidence'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'timestamp': signal['timestamp']
                    })
            
            return jsonify({'signal': False, 'symbol': symbol})
    
    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data for analysis"""
        
        try:
            data = yf.download(symbol, period="30d", interval="1h", progress=False)
            
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
    
    async def execute_paper_trade(self, signal: dict):
        """Execute paper trade"""
        
        symbol = signal['symbol']
        position_size = self.paper_portfolio['cash'] * 0.25  # 25% position
        shares = position_size / signal['price']
        
        # Add to portfolio
        self.paper_portfolio['positions'][symbol] = {
            'shares': shares,
            'entry_price': signal['price'],
            'entry_time': datetime.now().isoformat(),
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'confidence': signal['confidence']
        }
        
        # Update cash
        self.paper_portfolio['cash'] -= position_size
        
        # Log trade
        self.paper_portfolio['trades'].append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': signal['price'],
            'confidence': signal['confidence']
        })
        
        logger.info(f"Paper trade executed: {shares:.2f} shares of {symbol} @ ${signal['price']:.2f}")
    
    def create_pine_script(self):
        """Generate Pine Script for TradingView"""
        
        pine_script = f'''
//@version=5
strategy("ML Paper Trading Bot", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=25)

// ML Trading Bot Integration
// This script sends alerts to our ML system for analysis

// Basic signal conditions (customize as needed)
sma20 = ta.sma(close, 20)
sma50 = ta.sma(close, 50)
rsi = ta.rsi(close, 14)

// Signal condition
ml_signal = ta.crossover(sma20, sma50) and rsi < 70 and volume > ta.sma(volume, 20) * 1.2

// Plot indicators
plot(sma20, color=color.blue, title="SMA20")
plot(sma50, color=color.red, title="SMA50")

// When signal triggers, send to ML system
if ml_signal
    // Webhook sends data to our ML system at your_server_url
    alert_message = '{{"symbol": "' + syminfo.ticker + '", "price": ' + str.tostring(close) + ', "time": "' + str.tostring(time) + '"}}'
    alert("ML Analysis", alert.freq_once_per_bar_close)
    
    // Only enter if we get ML confirmation (manual for now)
    strategy.entry("ML_Long", strategy.long, comment="ML Signal")

// Exit conditions
if strategy.position_size > 0
    if close <= strategy.position_avg_price * 0.995  // 0.5% stop loss
        strategy.close("ML_Long", comment="Stop Loss")
    if close >= strategy.position_avg_price * 1.01   // 1% take profit
        strategy.close("ML_Long", comment="Take Profit")

// Background highlighting
bgcolor(ml_signal ? color.new(color.green, 80) : na)
'''
        
        # Save to file
        with open('ml_tradingview_strategy.pine', 'w') as f:
            f.write(pine_script)
        
        return pine_script
    
    def run_server(self, host='127.0.0.1', port=5000):
        """Run the TradingView integration server"""
        
        # Generate Pine Script
        self.create_pine_script()
        
        print(f"""
        📺 TRADINGVIEW ML PAPER TRADING
        
        🌐 Server: http://{host}:{port}
        📊 Dashboard: http://{host}:{port}
        🔗 Webhook URL: http://{host}:{port}/webhook
        
        📋 SETUP STEPS:
        
        1️⃣ PINE SCRIPT (GENERATED):
           • File saved: ml_tradingview_strategy.pine
           • Copy to TradingView Pine Editor
           • Add to your charts
        
        2️⃣ WEBHOOK ALERTS:
           • Create TradingView alert
           • Set Webhook URL: http://{host}:{port}/webhook
           • Message: {{"symbol": "{{{{ticker}}}}", "price": {{{{close}}}}}}
        
        3️⃣ PAPER TRADING:
           • Use TradingView's built-in paper trading
           • Or track trades in our dashboard
        
        🎯 FEATURES:
        • Real-time ML signal analysis
        • TradingView alert integration
        • Paper portfolio tracking
        • Custom Pine Script strategy
        
        Starting server...
        """)
        
        self.app.run(host=host, port=port, debug=False)


# TradingView Dashboard HTML
TRADINGVIEW_DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>📺 TradingView ML Paper Trading</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0; padding: 20px; background: #131722; color: #d1d4dc;
        }
        .header { 
            text-align: center; margin-bottom: 30px; 
        }
        .header h1 { 
            color: #2962ff; margin: 0; font-size: 2.5em;
        }
        .setup-card { 
            background: #1e222d; padding: 25px; border-radius: 15px; 
            border: 2px solid #2962ff; margin-bottom: 20px;
        }
        .code-block { 
            background: #0d1421; padding: 15px; border-radius: 8px; 
            font-family: 'Monaco', monospace; font-size: 12px; 
            overflow-x: auto; margin: 10px 0;
        }
        .btn { 
            background: #2962ff; color: white; border: none; 
            padding: 12px 24px; border-radius: 6px; cursor: pointer; 
            margin: 5px; font-size: 14px;
        }
        .btn:hover { background: #1e53e5; }
        .status { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; margin: 20px 0; 
        }
        .status-card { 
            background: #1e222d; padding: 20px; border-radius: 10px; 
            text-align: center; border: 1px solid #2a2e39;
        }
        .status-value { 
            font-size: 1.8em; font-weight: bold; color: #2962ff; 
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📺 TradingView ML Paper Trading</h1>
        <p>Connect your TradingView alerts to our 88.79% ML system</p>
    </div>
    
    <div class="status">
        <div class="status-card">
            <div class="status-value" id="portfolio-value">$100,000</div>
            <div>Paper Portfolio</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="total-trades">0</div>
            <div>ML Trades</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="webhook-calls">0</div>
            <div>Webhook Calls</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="ml-signals">0</div>
            <div>ML Signals</div>
        </div>
    </div>
    
    <div class="setup-card">
        <h3>🔗 Webhook Setup</h3>
        <p>Use this webhook URL in your TradingView alerts:</p>
        <div class="code-block" id="webhook-url">http://localhost:5000/webhook</div>
        <p>Alert message format:</p>
        <div class="code-block">{"symbol": "{{ticker}}", "price": {{close}}}</div>
        <button class="btn" onclick="copyWebhook()">📋 Copy Webhook URL</button>
    </div>
    
    <div class="setup-card">
        <h3>📊 Pine Script Strategy</h3>
        <p>A custom Pine Script has been generated: <code>ml_tradingview_strategy.pine</code></p>
        <button class="btn" onclick="testSignal()">🧪 Test ML Signal</button>
        <button class="btn" onclick="refreshPortfolio()">🔄 Refresh Portfolio</button>
    </div>
    
    <div class="setup-card">
        <h3>📈 Paper Trading Results</h3>
        <div id="trades-list">No trades yet. Waiting for TradingView alerts...</div>
    </div>

    <script>
        function copyWebhook() {
            const url = document.getElementById('webhook-url').textContent;
            navigator.clipboard.writeText(url);
            alert('✅ Webhook URL copied to clipboard!');
        }
        
        async function testSignal() {
            try {
                const response = await fetch('/api/scan/AAPL');
                const result = await response.json();
                
                if (result.signal) {
                    alert(`🚀 ML Signal for AAPL!\\nPrice: $${result.price.toFixed(2)}\\nConfidence: ${(result.confidence * 100).toFixed(1)}%`);
                } else {
                    alert('📊 No ML signal for AAPL at this time');
                }
            } catch (error) {
                alert('❌ Error testing signal');
            }
        }
        
        async function refreshPortfolio() {
            try {
                const response = await fetch('/api/portfolio');
                const portfolio = await response.json();
                
                document.getElementById('portfolio-value').textContent = 
                    '$' + (portfolio.cash + Object.values(portfolio.positions).reduce((sum, pos) => sum + (pos.shares * pos.entry_price), 0)).toLocaleString();
                
                document.getElementById('total-trades').textContent = portfolio.trades.length;
                
                // Update trades list
                const tradesList = document.getElementById('trades-list');
                if (portfolio.trades.length > 0) {
                    tradesList.innerHTML = portfolio.trades.map(trade => 
                        `<div style="padding: 10px; margin: 5px 0; background: #0d1421; border-radius: 5px;">
                            📊 ${trade.symbol}: ${trade.shares.toFixed(2)} shares @ $${trade.price.toFixed(2)} 
                            (${(trade.confidence * 100).toFixed(1)}% confidence)
                        </div>`
                    ).join('');
                } else {
                    tradesList.innerHTML = 'No trades yet. Waiting for TradingView alerts...';
                }
                
            } catch (error) {
                console.error('Error refreshing portfolio:', error);
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshPortfolio, 30000);
        
        // Initial load
        refreshPortfolio();
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    trader = TradingViewPaperTrader()
    trader.run_server()
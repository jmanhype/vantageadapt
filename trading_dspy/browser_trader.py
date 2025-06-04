#!/usr/bin/env python3
"""
BROWSER-BASED PAPER TRADING - ZERO SETUP!

Creates a beautiful web dashboard showing live ML signals.
Just open in browser - no accounts, no APIs, nothing!
"""

import asyncio
import json
from datetime import datetime
import webbrowser
from flask import Flask, render_template_string, jsonify
import pandas as pd
import yfinance as yf
from loguru import logger
import threading

# Import our ML system
from paper_trading_adapter import PaperTradingAdapter


class BrowserTrader:
    """
    Beautiful web-based trading dashboard
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.adapter = PaperTradingAdapter(platform="browser")
        self.current_signals = []
        self.portfolio = {"cash": 100000, "positions": {}, "trades": []}
        self.setup_routes()
        
        logger.info("🌐 Browser Trader initialized")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main trading dashboard"""
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/signals')
        def get_signals():
            """Get current ML signals"""
            return jsonify(self.current_signals)
        
        @self.app.route('/api/portfolio')
        def get_portfolio():
            """Get portfolio status"""
            return jsonify(self.portfolio)
        
        @self.app.route('/api/execute/<symbol>/<action>')
        def execute_trade(symbol, action):
            """Execute a paper trade"""
            
            # Find the signal
            signal = next((s for s in self.current_signals if s['symbol'] == symbol), None)
            
            if signal and action == 'buy':
                # Execute paper trade
                shares = int(signal['size'] / signal['price'])
                cost = shares * signal['price']
                
                if cost <= self.portfolio['cash']:
                    # Add position
                    self.portfolio['positions'][symbol] = {
                        'shares': shares,
                        'entry_price': signal['price'],
                        'entry_time': datetime.now().isoformat(),
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit']
                    }
                    
                    # Update cash
                    self.portfolio['cash'] -= cost
                    
                    # Log trade
                    self.portfolio['trades'].append({
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': signal['price'],
                        'total': cost
                    })
                    
                    return jsonify({"success": True, "message": f"Bought {shares} shares of {symbol}"})
                else:
                    return jsonify({"success": False, "message": "Insufficient cash"})
            
            return jsonify({"success": False, "message": "Invalid trade"})
        
        @self.app.route('/api/scan')
        async def scan_markets():
            """Scan markets for new signals"""
            await self.update_signals()
            return jsonify({"status": "updated", "signals": len(self.current_signals)})
    
    async def update_signals(self):
        """Update current market signals"""
        
        symbols = [
            'BTC-USD', 'ETH-USD', 'SOL-USD',  # Crypto
            'AAPL', 'TSLA', 'MSFT', 'NVDA',   # Stocks
            'SPY', 'QQQ'                       # ETFs
        ]
        
        new_signals = []
        
        for symbol in symbols:
            try:
                # Get market data
                market_data = await self.get_market_data(symbol)
                
                if market_data is not None and len(market_data) > 100:
                    # Generate ML signal
                    signal = await self.adapter.get_real_time_signal(symbol, market_data)
                    
                    if signal['action'] == 'BUY':
                        # Add chart data
                        recent_prices = market_data['close'].tail(50).tolist()
                        
                        signal_data = {
                            'symbol': signal['symbol'],
                            'price': signal['price'],
                            'size': signal['size'],
                            'stop_loss': signal['stop_loss'],
                            'take_profit': signal['take_profit'],
                            'confidence': signal['confidence'],
                            'timestamp': signal['timestamp'],
                            'risk_reward': (signal['take_profit'] - signal['price']) / (signal['price'] - signal['stop_loss']),
                            'chart_data': recent_prices
                        }
                        
                        new_signals.append(signal_data)
                        
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        self.current_signals = new_signals
        logger.info(f"Updated signals: {len(new_signals)} active")
    
    async def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Get market data"""
        
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
    
    async def start_background_scanning(self):
        """Start background market scanning"""
        
        while True:
            try:
                await self.update_signals()
                await asyncio.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"Background scan error: {e}")
                await asyncio.sleep(60)
    
    def run_server(self, host='127.0.0.1', port=8080):
        """Run the web server"""
        
        # Start background scanning
        def run_async_scanning():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_background_scanning())
        
        scanning_thread = threading.Thread(target=run_async_scanning, daemon=True)
        scanning_thread.start()
        
        print(f"""
        🌐 BROWSER TRADING DASHBOARD STARTING
        
        📊 Dashboard URL: http://{host}:{port}
        
        ✨ FEATURES:
        • Live ML trading signals with charts
        • One-click paper trading execution  
        • Real-time portfolio tracking
        • Beautiful visual interface
        • Mobile-friendly design
        
        🚀 AUTO-OPENING BROWSER...
        """)
        
        # Auto-open browser
        webbrowser.open(f'http://{host}:{port}')
        
        # Run Flask server
        self.app.run(host=host, port=port, debug=False)


# HTML Template for the dashboard
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>🤖 ML Trading Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0; padding: 20px; background: #0a0e27; color: #fff;
        }
        .header { 
            text-align: center; margin-bottom: 30px; 
        }
        .header h1 { 
            color: #00ff88; margin: 0; font-size: 2.5em;
        }
        .stats { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; margin-bottom: 30px;
        }
        .stat-card { 
            background: #1a1f3a; padding: 20px; border-radius: 15px; 
            border: 2px solid #00ff88; text-align: center;
        }
        .stat-value { 
            font-size: 2em; font-weight: bold; color: #00ff88; 
        }
        .signals { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 20px; 
        }
        .signal-card { 
            background: linear-gradient(135deg, #1a1f3a, #2a2f4a); 
            padding: 25px; border-radius: 15px; border: 2px solid #ff6b35;
            box-shadow: 0 10px 30px rgba(255, 107, 53, 0.3);
        }
        .signal-header { 
            display: flex; justify-content: space-between; align-items: center; 
            margin-bottom: 15px; 
        }
        .symbol { 
            font-size: 1.5em; font-weight: bold; color: #ff6b35; 
        }
        .confidence { 
            background: #00ff88; color: #000; padding: 5px 15px; 
            border-radius: 25px; font-weight: bold; 
        }
        .price-info { 
            display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; 
            margin: 15px 0; 
        }
        .price-item { 
            text-align: center; padding: 10px; background: rgba(255,255,255,0.1); 
            border-radius: 10px; 
        }
        .price-label { 
            font-size: 0.8em; color: #aaa; display: block; 
        }
        .price-value { 
            font-size: 1.2em; font-weight: bold; 
        }
        .entry { color: #00ff88; }
        .stop { color: #ff4757; }
        .target { color: #ffd700; }
        .trade-btn { 
            width: 100%; padding: 15px; background: linear-gradient(45deg, #00ff88, #00d4aa); 
            color: #000; border: none; border-radius: 10px; font-size: 1.1em; 
            font-weight: bold; cursor: pointer; margin-top: 15px;
            transition: transform 0.2s;
        }
        .trade-btn:hover { 
            transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,255,136,0.4); 
        }
        .chart { 
            height: 60px; margin: 10px 0; background: rgba(255,255,255,0.05); 
            border-radius: 5px; position: relative; overflow: hidden;
        }
        .no-signals { 
            text-align: center; padding: 50px; color: #666; font-size: 1.2em; 
        }
        .loading { 
            text-align: center; color: #00ff88; font-size: 1.1em; 
        }
        .portfolio { 
            background: #1a1f3a; padding: 20px; border-radius: 15px; 
            border: 2px solid #00ff88; margin-bottom: 20px; 
        }
        .refresh-btn { 
            background: #ff6b35; color: #fff; border: none; padding: 10px 20px; 
            border-radius: 25px; cursor: pointer; margin: 10px; 
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 ML Trading Dashboard</h1>
        <p>Live signals from the 88.79% ML system</p>
        <button class="refresh-btn" onclick="refreshSignals()">🔄 Refresh Signals</button>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value" id="portfolio-value">$100,000</div>
            <div>Portfolio Value</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="active-signals">0</div>
            <div>Active Signals</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="total-trades">0</div>
            <div>Total Trades</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="cash">$100,000</div>
            <div>Available Cash</div>
        </div>
    </div>
    
    <div id="signals-container">
        <div class="loading">🔍 Scanning markets for ML signals...</div>
    </div>

    <script>
        let signals = [];
        let portfolio = {};
        
        async function loadSignals() {
            try {
                const response = await fetch('/api/signals');
                signals = await response.json();
                renderSignals();
            } catch (error) {
                console.error('Error loading signals:', error);
            }
        }
        
        async function loadPortfolio() {
            try {
                const response = await fetch('/api/portfolio');
                portfolio = await response.json();
                updateStats();
            } catch (error) {
                console.error('Error loading portfolio:', error);
            }
        }
        
        function renderSignals() {
            const container = document.getElementById('signals-container');
            
            if (signals.length === 0) {
                container.innerHTML = '<div class="no-signals">🔍 No ML signals detected. Markets are being scanned...</div>';
                return;
            }
            
            const signalsHTML = signals.map(signal => `
                <div class="signal-card">
                    <div class="signal-header">
                        <span class="symbol">${signal.symbol}</span>
                        <span class="confidence">${(signal.confidence * 100).toFixed(1)}%</span>
                    </div>
                    
                    <div class="price-info">
                        <div class="price-item">
                            <span class="price-label">Entry</span>
                            <div class="price-value entry">$${signal.price.toFixed(4)}</div>
                        </div>
                        <div class="price-item">
                            <span class="price-label">Stop Loss</span>
                            <div class="price-value stop">$${signal.stop_loss.toFixed(4)}</div>
                        </div>
                        <div class="price-item">
                            <span class="price-label">Take Profit</span>
                            <div class="price-value target">$${signal.take_profit.toFixed(4)}</div>
                        </div>
                    </div>
                    
                    <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                        <span>Size: $${signal.size.toFixed(0)}</span>
                        <span>R/R: 1:${signal.risk_reward.toFixed(1)}</span>
                    </div>
                    
                    <div class="chart" id="chart-${signal.symbol}"></div>
                    
                    <button class="trade-btn" onclick="executeTrade('${signal.symbol}', 'buy')">
                        🚀 BUY ${signal.symbol}
                    </button>
                    
                    <div style="font-size: 0.8em; color: #aaa; margin-top: 10px;">
                        Generated: ${new Date(signal.timestamp).toLocaleTimeString()}
                    </div>
                </div>
            `).join('');
            
            container.innerHTML = signalsHTML;
            
            // Simple chart visualization
            signals.forEach(signal => {
                drawMiniChart(signal.symbol, signal.chart_data);
            });
        }
        
        function drawMiniChart(symbol, data) {
            const chartEl = document.getElementById(`chart-${symbol}`);
            if (!chartEl || !data) return;
            
            const max = Math.max(...data);
            const min = Math.min(...data);
            const range = max - min;
            
            const points = data.map((price, i) => {
                const x = (i / (data.length - 1)) * 100;
                const y = 100 - (((price - min) / range) * 100);
                return `${x},${y}`;
            }).join(' ');
            
            chartEl.innerHTML = `
                <svg width="100%" height="100%" style="position: absolute;">
                    <polyline points="${points}" 
                              fill="none" 
                              stroke="#00ff88" 
                              stroke-width="2"/>
                </svg>
            `;
        }
        
        function updateStats() {
            document.getElementById('portfolio-value').textContent = 
                '$' + (portfolio.cash + Object.values(portfolio.positions || {}).reduce((sum, pos) => sum + (pos.shares * pos.entry_price), 0)).toLocaleString();
            
            document.getElementById('active-signals').textContent = signals.length;
            document.getElementById('total-trades').textContent = (portfolio.trades || []).length;
            document.getElementById('cash').textContent = '$' + portfolio.cash.toLocaleString();
        }
        
        async function executeTrade(symbol, action) {
            try {
                const response = await fetch(`/api/execute/${symbol}/${action}`);
                const result = await response.json();
                
                if (result.success) {
                    alert(`✅ ${result.message}`);
                    loadPortfolio();
                } else {
                    alert(`❌ ${result.message}`);
                }
            } catch (error) {
                alert('❌ Trade execution failed');
                console.error(error);
            }
        }
        
        async function refreshSignals() {
            document.getElementById('signals-container').innerHTML = '<div class="loading">🔄 Refreshing signals...</div>';
            
            try {
                await fetch('/api/scan');
                setTimeout(() => {
                    loadSignals();
                }, 2000);
            } catch (error) {
                console.error('Error refreshing signals:', error);
            }
        }
        
        // Initial load
        loadSignals();
        loadPortfolio();
        
        // Auto-refresh every 5 minutes
        setInterval(() => {
            loadSignals();
            loadPortfolio();
        }, 300000);
        
        // Update portfolio more frequently
        setInterval(loadPortfolio, 30000);
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    print("🌐 STARTING BROWSER TRADING DASHBOARD...")
    
    trader = BrowserTrader()
    trader.run_server()
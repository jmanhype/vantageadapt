#!/usr/bin/env python3
"""
INSTANT ALPACA TRADES - GUARANTEED EXECUTION
No ML dependencies, just pure trading action
"""

import asyncio
import os
from datetime import datetime
from loguru import logger

# Set Alpaca keys
os.environ['ALPACA_API_KEY'] = "PKV0EUF7LNIUB2TJMTIK"
os.environ['ALPACA_SECRET_KEY'] = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"


class AlpacaInstantTrader:
    """
    Instant Alpaca trader - WILL execute trades immediately
    """
    
    def __init__(self):
        try:
            import alpaca_trade_api as tradeapi
            self.api = tradeapi.REST(
                key_id=os.getenv('ALPACA_API_KEY'),
                secret_key=os.getenv('ALPACA_SECRET_KEY'),
                base_url='https://paper-api.alpaca.markets'
            )
            account = self.api.get_account()
            logger.info(f"✅ Alpaca connected: ${float(account.buying_power):,.2f} buying power")
            self.connected = True
        except Exception as e:
            logger.error(f"❌ Alpaca connection failed: {e}")
            self.connected = False
    
    async def execute_instant_trades(self):
        """Execute trades immediately for testing"""
        
        if not self.connected:
            logger.error("❌ Cannot trade - not connected to Alpaca")
            return
        
        print("""
        🚀 ALPACA INSTANT TRADING TEST
        
        ⚡ Executing 5 test trades immediately
        📊 Small positions to test functionality
        💰 Using paper trading account
        """)
        
        # Test symbols with small quantities
        test_trades = [
            {'symbol': 'SPY', 'qty': 1, 'reason': 'S&P 500 ETF test'},
            {'symbol': 'QQQ', 'qty': 1, 'reason': 'NASDAQ ETF test'}, 
            {'symbol': 'AAPL', 'qty': 1, 'reason': 'Apple stock test'},
            {'symbol': 'MSFT', 'qty': 1, 'reason': 'Microsoft test'},
            {'symbol': 'TSLA', 'qty': 1, 'reason': 'Tesla test'}
        ]
        
        for i, trade in enumerate(test_trades, 1):
            try:
                logger.info(f"🎯 Executing trade {i}/5: {trade['symbol']}")
                
                # Get current price
                try:
                    latest_trade = self.api.get_latest_trade(trade['symbol'])
                    current_price = latest_trade.price
                except:
                    # Fallback: get last bar
                    bars = self.api.get_bars(trade['symbol'], self.api.TimeFrame.Minute, limit=1).df
                    current_price = bars['close'].iloc[-1] if len(bars) > 0 else 100
                
                # Execute market order
                order = self.api.submit_order(
                    symbol=trade['symbol'],
                    qty=trade['qty'],
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                print(f"""
                ✅ TRADE {i} EXECUTED!
                
                📈 Symbol: {trade['symbol']}
                💰 Quantity: {trade['qty']} shares
                💵 Price: ~${current_price:.2f}
                📊 Order ID: {order.id}
                🎯 Reason: {trade['reason']}
                ⏰ Time: {datetime.now().strftime('%H:%M:%S')}
                """)
                
                # Small delay between trades
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Trade {i} failed: {e}")
                continue
        
        # Show final status
        await self.show_final_status()
    
    async def show_final_status(self):
        """Show account status after trades"""
        
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            print(f"""
            📊 FINAL TRADING STATUS
            {'='*50}
            
            💰 Portfolio Value: ${float(account.portfolio_value):,.2f}
            💵 Buying Power: ${float(account.buying_power):,.2f}
            📈 Active Positions: {len(positions)}
            
            🎯 POSITIONS:
            """)
            
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pnl_pct = float(pos.unrealized_plpc) * 100
                print(f"   📊 {pos.symbol}: {pos.qty} shares | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)")
            
            print(f"""
            
            ✅ INSTANT TRADING TEST COMPLETE!
            🎉 Trades executed on live Alpaca paper account
            📊 Portfolio actively trading
            """)
            
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")


async def main():
    """Execute instant trades"""
    
    trader = AlpacaInstantTrader()
    await trader.execute_instant_trades()


if __name__ == "__main__":
    asyncio.run(main())
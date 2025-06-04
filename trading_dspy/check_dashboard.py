#!/usr/bin/env python3
"""
Check what's actually showing on Alpaca dashboard
"""

import os
import alpaca_trade_api as tradeapi

os.environ['ALPACA_API_KEY'] = "PKV0EUF7LNIUB2TJMTIK"
os.environ['ALPACA_SECRET_KEY'] = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"

api = tradeapi.REST(
    key_id=os.environ['ALPACA_API_KEY'],
    secret_key=os.environ['ALPACA_SECRET_KEY'],
    base_url='https://paper-api.alpaca.markets'
)

print('🔍 CHECKING YOUR ALPACA DASHBOARD POSITIONS:')
print('='*60)

# Get positions
positions = api.list_positions()
print(f'📊 TOTAL POSITIONS: {len(positions)}')
print()

if positions:
    stock_positions = 0
    crypto_positions = 0
    total_value = 0
    
    for pos in positions:
        market_value = float(pos.market_value)
        total_value += market_value
        
        if '/' in pos.symbol:
            crypto_positions += 1
            print(f'🪙 CRYPTO: {pos.symbol} = {pos.qty} units (${market_value:.2f})')
        else:
            stock_positions += 1
            print(f'📈 STOCK: {pos.symbol} = {pos.qty} shares (${market_value:.2f})')
    
    print()
    print(f'📊 BREAKDOWN:')
    print(f'   📈 Stock positions: {stock_positions}')
    print(f'   🪙 Crypto positions: {crypto_positions}')
    print(f'   💰 Total position value: ${total_value:.2f}')

else:
    print('❌ No positions found on dashboard')
    print()
    print('🤔 POSSIBLE REASONS:')
    print('   • Orders are pending (not filled yet)')
    print('   • Market is closed (stock orders wait for open)')
    print('   • Trades failed to execute')

# Check recent orders
orders = api.list_orders(status='all', limit=50)
filled_orders = [o for o in orders if o.status == 'filled']
pending_orders = [o for o in orders if o.status in ['accepted', 'pending_new', 'new']]

print()
print(f'📋 ORDER STATUS:')
print(f'   ✅ Filled orders: {len(filled_orders)}')
print(f'   ⏳ Pending orders: {len(pending_orders)}')

if filled_orders:
    print()
    print('✅ RECENT FILLED ORDERS:')
    for order in filled_orders[-10:]:
        if order.filled_avg_price:
            price = float(order.filled_avg_price)
            value = float(order.qty) * price
            print(f'   {order.symbol}: {order.qty} @ ${price:.2f} = ${value:.2f}')

# Account summary
account = api.get_account()
print()
print(f'💰 ACCOUNT SUMMARY:')
print(f'   Portfolio value: ${float(account.portfolio_value):.2f}')
print(f'   Buying power: ${float(account.buying_power):.2f}')
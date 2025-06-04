#!/usr/bin/env python3
"""
TEST ALL CRYPTO TRADING METHODS - FIND THE ONE THAT WORKS
"""

import os
import alpaca_trade_api as tradeapi
from loguru import logger

# Set Alpaca keys
os.environ['ALPACA_API_KEY'] = "PKV0EUF7LNIUB2TJMTIK"
os.environ['ALPACA_SECRET_KEY'] = "XCM5z8KI1IfPBxZnPzDThDIYmTpABuXglw810IVz"

def test_all_crypto_methods():
    """Test every possible crypto trading method"""
    
    api = tradeapi.REST(
        key_id=os.getenv('ALPACA_API_KEY'),
        secret_key=os.getenv('ALPACA_SECRET_KEY'),
        base_url='https://paper-api.alpaca.markets'
    )
    
    print("🧪 TESTING ALL CRYPTO TRADING METHODS")
    
    # Test different symbol formats
    crypto_symbols = [
        'BTC/USD', 'BTCUSD', 'BTC-USD', 
        'ETH/USD', 'ETHUSD', 'ETH-USD',
        'LTC/USD', 'LTCUSD', 'LTC-USD'
    ]
    
    # Test different order methods
    test_methods = [
        # Method 1: Notional with exact 2 decimals
        {'type': 'notional_exact', 'params': {'notional': 50.00, 'side': 'buy', 'type': 'market', 'time_in_force': 'gtc'}},
        
        # Method 2: Notional with round()
        {'type': 'notional_round', 'params': {'notional': 50, 'side': 'buy', 'type': 'market', 'time_in_force': 'gtc'}},
        
        # Method 3: Quantity-based
        {'type': 'quantity', 'params': {'qty': 0.001, 'side': 'buy', 'type': 'market', 'time_in_force': 'gtc'}},
        
        # Method 4: Different time_in_force
        {'type': 'notional_day', 'params': {'notional': 50.00, 'side': 'buy', 'type': 'market', 'time_in_force': 'day'}},
        
        # Method 5: Limit order
        {'type': 'limit_order', 'params': {'notional': 50.00, 'side': 'buy', 'type': 'limit', 'limit_price': 70000, 'time_in_force': 'gtc'}},
        
        # Method 6: Different base URL
        {'type': 'live_api', 'base_url': 'https://api.alpaca.markets'},
    ]
    
    for symbol in crypto_symbols:
        print(f"\n🎯 TESTING SYMBOL: {symbol}")
        
        for method in test_methods:
            try:
                # Handle special base URL test
                if method['type'] == 'live_api':
                    test_api = tradeapi.REST(
                        key_id=os.getenv('ALPACA_API_KEY'),
                        secret_key=os.getenv('ALPACA_SECRET_KEY'),
                        base_url=method['base_url']
                    )
                    params = {'notional': 50.00, 'side': 'buy', 'type': 'market', 'time_in_force': 'gtc'}
                else:
                    test_api = api
                    params = method['params']
                
                # Submit order
                order = test_api.submit_order(symbol=symbol, **params)
                
                print(f"   ✅ SUCCESS: {method['type']} - Order ID: {order.id}")
                print(f"      Symbol: {symbol}, Method: {method['type']}")
                print(f"      Params: {params}")
                
                return True  # Stop on first success
                
            except Exception as e:
                print(f"   ❌ FAILED: {method['type']} - {e}")
                
                # Special handling for decimal places error
                if "decimal places" in str(e).lower():
                    print(f"      💡 TIP: Trying with rounded notional...")
                    try:
                        rounded_params = params.copy()
                        if 'notional' in rounded_params:
                            rounded_params['notional'] = round(rounded_params['notional'], 2)
                        order = test_api.submit_order(symbol=symbol, **rounded_params)
                        print(f"   ✅ SUCCESS WITH ROUNDING: Order ID: {order.id}")
                        return True
                    except Exception as e2:
                        print(f"      ❌ Rounding also failed: {e2}")
    
    # Test account configuration
    print(f"\n🔍 ACCOUNT CONFIGURATION ANALYSIS:")
    account = api.get_account()
    
    print(f"   Cash: ${float(account.cash):,.2f}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Trading Blocked: {account.trading_blocked}")
    print(f"   Account Blocked: {account.account_blocked}")
    print(f"   Pattern Day Trader: {account.pattern_day_trader}")
    
    # Check for crypto-specific account settings
    for attr in dir(account):
        if 'crypto' in attr.lower() or 'digital' in attr.lower():
            try:
                value = getattr(account, attr)
                print(f"   {attr}: {value}")
            except:
                pass
    
    return False

if __name__ == "__main__":
    success = test_all_crypto_methods()
    if success:
        print("\n🎉 CRYPTO TRADING METHOD FOUND!")
    else:
        print("\n😞 No working crypto method found - need to investigate further")
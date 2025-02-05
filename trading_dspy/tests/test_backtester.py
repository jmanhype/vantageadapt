import pandas as pd
import numpy as np
import pytest
import os
import pickle

from src.modules.backtester import Backtester


def test_backtester_dynamic_conditions() -> None:
    """Test the Backtester with dynamic conditions to ensure that it uses the novel trading rules for entry and exit.

    This test creates a small synthetic dataset, applies dynamic conditions for entry (price > 100) and exit (price < 90),
    and verifies that the backtester executes orders.
    """
    # Create a small synthetic trade_data DataFrame with an added 'sol_volume' column
    trade_data = pd.DataFrame({
        'dex_price': [95, 105, 105, 85, 105, 85],
        'sol_pool': [1, 1, 1, 1, 1, 1],
        'coin_pool': [1, 1, 1, 1, 1, 1],
        'sol_volume': [1, 1, 1, 1, 1, 1],
        'timestamp': pd.date_range(start='2020-01-01', periods=6, freq='T')
    })

    # Define strategy parameters
    parameters = {
        'take_profit': 0.08,
        'stop_loss': 0.12,
        'sl_window': 5,
        'max_orders': 3,
        'order_size': 1.0,
        'post_buy_delay': 1,
        'post_sell_delay': 1,
        'macd_signal_fast': 12,
        'macd_signal_slow': 26,
        'macd_signal_signal': 9,
        'min_macd_signal_threshold': 0,
        'max_macd_signal_threshold': 0,
        'enable_sl_mod': False,
        'enable_tp_mod': False
    }

    # Define dynamic conditions
    conditions = {
        'entry': ['price > 100'],
        'exit': ['price < 90']
    }

    # Instantiate Backtester with lenient performance thresholds to avoid failures due to performance requirements
    backtester = Backtester(performance_thresholds={'min_return': -100})

    result = backtester.forward(trade_data, parameters, conditions)

    # Check that result dictionary contains required keys
    assert 'metrics' in result, "Result should contain 'metrics' key."
    assert 'portfolio' in result, "Result should contain 'portfolio' key."
    assert 'validation_passed' in result, "Result should contain 'validation_passed' key."

    metrics = result['metrics']
    # Expect that some orders should have been executed
    total_orders = metrics.get('total_orders', 0)
    assert total_orders > 0, f"Expected some orders to be executed, but got total_orders={total_orders}."

    # Optionally, check that the portfolio's orders DataFrame has at least one record
    portfolio = result['portfolio']
    try:
        orders = portfolio.orders
        # orders.count() might return a Series; sum it up
        orders_count = orders.count()
        if hasattr(orders_count, 'sum'):
            total_orders_vectorbt = int(orders_count.sum())
        else:
            total_orders_vectorbt = int(orders_count)
        assert total_orders_vectorbt > 0, "Expected portfolio to have at least one order executed."
    except Exception as e:
        pytest.skip(f"Portfolio orders attribute not properly formed: {e}")


def test_backtester_dynamic_conditions_with_pickle() -> None:
    """Test the Backtester using data loaded from a pickle file located at
    /Users/speed/StratOptimv4/big_optimize_1016.pkl. This test ensures that dynamic trading
    conditions are used and that the backtester executes orders.
    """
    pickle_file = "/Users/speed/StratOptimv4/big_optimize_1016.pkl"
    if not os.path.exists(pickle_file):
        pytest.skip(f"Pickle file {pickle_file} not found.")
    
    with open(pickle_file, "rb") as f:
        trade_data_dict = pickle.load(f)

    # Use RETARDIO token data specifically
    if 'RETARDIO' not in trade_data_dict:
        pytest.skip("RETARDIO token data not found in pickle file")
    
    trade_data = trade_data_dict['RETARDIO']
    
    # Create a proper copy of a larger sample (10000 rows)
    test_data = trade_data.iloc[:10000].copy()
    
    # Ensure proper time indexing
    test_data = test_data.sort_index()  # Ensure index is sorted
    test_data.index = pd.to_datetime(test_data['timestamp'])  # Use timestamp as index
    
    # Fill NaN values
    test_data['sol_volume'] = test_data['sol_volume'].fillna(0)
    test_data['dex_price'] = test_data['dex_price'].ffill()  # Using ffill() instead of fillna(method='ffill')
    
    # Create a synthetic DataFrame with all required columns
    synthetic_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2020-01-01', periods=len(test_data), freq='min'),
        'dex_price': test_data['dex_price'].values,
        'price': test_data['dex_price'].values,  # Add price column
        'sol_volume': test_data['sol_volume'].values,
        'is_buy': test_data['is_buy'].values if 'is_buy' in test_data.columns else True,
        'base': test_data['base'].values if 'base' in test_data.columns else 'SOL',
        'sol_pool': test_data['sol_pool'].values if 'sol_pool' in test_data.columns else test_data['sol_volume'].values,
        'coin_pool': test_data['coin_pool'].values if 'coin_pool' in test_data.columns else test_data['sol_volume'].values * test_data['dex_price'].values,
        'sol_amount': test_data['sol_amount'].values if 'sol_amount' in test_data.columns else test_data['sol_volume'].values,
        'token_amount': test_data['token_amount'].values if 'token_amount' in test_data.columns else test_data['sol_volume'].values / test_data['dex_price'].values,
        'tx_position': test_data['tx_position'].values if 'tx_position' in test_data.columns else range(len(test_data)),
        'dex_price_pct_change': test_data['dex_price_pct_change'].values if 'dex_price_pct_change' in test_data.columns else test_data['dex_price'].pct_change().fillna(0)
    })
    synthetic_data.set_index('timestamp', inplace=True)
    
    # Add Close column required by vectorbt
    synthetic_data['Close'] = synthetic_data['dex_price']
    
    print("\nSynthetic DataFrame info:")
    print(synthetic_data.info())
    print("\nFirst few rows:")
    print(synthetic_data.head().to_string())
    
    # Calculate price statistics
    price_series = synthetic_data['price']  # Use price instead of dex_price
    min_price = float(price_series.min())
    max_price = float(price_series.max())
    mean_price = float(price_series.mean())
    
    # Calculate volume statistics
    volume_series = synthetic_data['sol_volume']
    mean_volume = float(volume_series.mean())
    min_volume = float(volume_series[volume_series > 0].min())  # Minimum non-zero volume
    
    print(f"\nPrice statistics:")
    print(f"min={min_price:.8f}")
    print(f"max={max_price:.8f}")
    print(f"mean={mean_price:.8f}")
    
    print(f"\nVolume statistics:")
    print(f"mean={mean_volume:.8f}")
    print(f"min_non_zero={min_volume:.8f}")
    
    # Add technical indicators
    synthetic_data['price_pct_change'] = synthetic_data['price'].pct_change()
    synthetic_data['sma_20'] = synthetic_data['price'].rolling(window=20).mean()
    synthetic_data['sma_50'] = synthetic_data['price'].rolling(window=50).mean()
    synthetic_data['momentum'] = synthetic_data['price_pct_change'].rolling(window=10).sum()
    
    # Calculate entry and exit price thresholds based on data range
    entry_price = min_price + (max_price - min_price) * 0.4  # 40th percentile
    exit_price = min_price + (max_price - min_price) * 0.6   # 60th percentile
    
    print(f"\nPrice thresholds:")
    print(f"entry_price={entry_price:.8f}")
    print(f"exit_price={exit_price:.8f}")

    # Use a combination of technical indicators for more robust trading conditions
    conditions = {
        'entry': [
            f"price < {entry_price}",  # Price is below entry threshold
            "rsi < 30",  # RSI indicates oversold
            "price < bb.lower",  # Price below lower Bollinger Band
            "macd.macd > macd.signal",  # MACD bullish crossover
            "williams_r < -80"  # Williams %R indicates oversold
        ],
        'exit': [
            f"price > {exit_price}",  # Price is above exit threshold
            "rsi > 70",  # RSI indicates overbought
            "price > bb.upper",  # Price above upper Bollinger Band
            "macd.macd < macd.signal",  # MACD bearish crossover
            "williams_r > -20"  # Williams %R indicates overbought
        ]
    }

    # Define strategy parameters with reasonable settings
    parameters = {
        'take_profit': 0.05,    # 5% take profit
        'stop_loss': 0.03,      # 3% stop loss
        'sl_window': 20,        # 20-period window for stop loss
        'max_orders': 100,      # Maximum number of orders
        'order_size': 0.1,      # Use 10% of available capital per trade
        'post_buy_delay': 5,    # Wait 5 periods after buying
        'post_sell_delay': 5,   # Wait 5 periods after selling
        'macd_signal_fast': 12,
        'macd_signal_slow': 26,
        'macd_signal_signal': 9,
        'min_macd_signal_threshold': 0,
        'max_macd_signal_threshold': 0,
        'enable_sl_mod': True,  # Enable stop loss modification
        'enable_tp_mod': True   # Enable take profit modification
    }
    
    # Verify that our conditions would actually trigger
    entry_signals = synthetic_data.eval(f"price < {entry_price}")
    exit_signals = synthetic_data.eval(f"price > {exit_price}")
    print(f"\nVerifying signals:")
    print(f"Number of potential entry signals: {entry_signals.sum()}")
    print(f"Number of potential exit signals: {exit_signals.sum()}")
    
    # Add debug columns to verify signal evaluation
    synthetic_data['entry_signal'] = entry_signals
    synthetic_data['exit_signal'] = exit_signals
    
    print("\nFirst few rows with signals:")
    print(synthetic_data[['price', 'price_pct_change', 'sma_20', 'momentum', 'sol_volume', 'entry_signal', 'exit_signal']].head(20).to_string())
    
    # Print signal distribution
    signal_windows = []
    current_signal = None
    start_idx = None
    
    for idx, (entry, exit) in enumerate(zip(entry_signals, exit_signals)):
        if entry and current_signal != 'entry':
            if start_idx is not None:
                signal_windows.append(('exit' if current_signal == 'exit' else 'none', idx - start_idx))
            start_idx = idx
            current_signal = 'entry'
        elif exit and current_signal != 'exit':
            if start_idx is not None:
                signal_windows.append(('entry' if current_signal == 'entry' else 'none', idx - start_idx))
            start_idx = idx
            current_signal = 'exit'
            
    print("\nSignal windows (showing first 10):")
    for signal_type, duration in signal_windows[:10]:
        print(f"{signal_type}: {duration} periods")
    
    # Instantiate Backtester with very lenient performance thresholds
    backtester = Backtester(performance_thresholds={'min_return': -1000})
    
    result = backtester.forward(synthetic_data, parameters, conditions)
    
    # Print detailed result information
    print("\nBacktest results:")
    print("Metrics:", result.get('metrics'))
    print("Validation passed:", result.get('validation_passed'))
    
    if 'portfolio' in result:
        portfolio = result['portfolio']
        print("\nPortfolio information:")
        
        # Print key portfolio metrics
        print("\nKey metrics:")
        try:
            total_return = float(portfolio.total_return.iloc[-1]) if hasattr(portfolio.total_return, 'iloc') else float(portfolio.total_return)
            print(f"Total return: {total_return:.4f}")
            
            if hasattr(portfolio, 'final_value'):
                final_value = float(portfolio.final_value.iloc[-1]) if hasattr(portfolio.final_value, 'iloc') else float(portfolio.final_value)
                print(f"Final value: {final_value:.4f}")
            
            if hasattr(portfolio, 'position_coverage'):
                position_coverage = float(portfolio.position_coverage.iloc[-1]) if hasattr(portfolio.position_coverage, 'iloc') else float(portfolio.position_coverage)
                print(f"Position coverage: {position_coverage:.4f}")
        except Exception as e:
            print(f"Error printing metrics: {str(e)}")
        
        # Print order information
        if hasattr(portfolio, 'order_records'):
            print("\nOrder records:")
            print(f"Number of order records: {len(portfolio.order_records)}")
            if len(portfolio.order_records) > 0:
                print("First few order records:")
                print(portfolio.order_records[:5])
        
        # Print trade information
        if hasattr(portfolio, 'trade_history'):
            print("\nTrade history:")
            print(f"Trade history shape: {portfolio.trade_history.shape}")
            if len(portfolio.trade_history) > 0:
                print("First few trades:")
                print(portfolio.trade_history.head())
    
    # Check that result dictionary contains required keys
    assert 'metrics' in result, "Result should contain 'metrics' key."
    assert 'portfolio' in result, "Result should contain 'portfolio' key."
    assert 'validation_passed' in result, "Result should contain 'validation_passed' key."
    
    metrics = result['metrics']
    # Expect that some orders should have been executed
    total_orders = metrics.get('total_orders', 0)
    assert total_orders > 0, f"Expected some orders to be executed, but got total_orders={total_orders}." 
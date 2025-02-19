"""Backtesting module with enhanced trade tracking and performance metrics."""

from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import pandas as pd
import numpy as np
import logging
import pickle
from datetime import datetime
import json
import os
import warnings
from numba.typed import Dict as NumbaDict
from numba import njit
import vectorbtpro as vbt
import ta
import itertools
from collections import namedtuple
import random
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# VBT settings
vbt.settings.plotting.use_resampler = True
vbt.settings.wrapping['freq'] = 's'

# Define trade memory structure
global trade_memory
trade_memory = None
TradeMemory = namedtuple("TradeMemory", ["trade_records", "trade_counts"])

def load_trade_data(data_path: str) -> Dict[str, pd.DataFrame]:
    """Load market data from pickle file."""
    try:
        logger.info(f"Loading trade data from {data_path}")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
        if not isinstance(data, dict):
            logger.error("Data must be a dictionary mapping asset names to DataFrames")
            return None
            
        logger.info(f"Successfully loaded {len(data)} assets")
        return data
        
    except Exception as e:
        logger.error(f"Error loading trade data: {str(e)}")
        logger.exception("Full traceback:")
        return None

def calculate_entries_and_params(trade_data_df: pd.DataFrame, p: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Calculate entry signals and parameters based on LLM-generated conditions.
    
    Args:
        trade_data_df: DataFrame containing market data
        p: Dictionary containing parameters and conditions
        
    Returns:
        DataFrame with calculated signals or None if error
    """
    try:
        # Set take profit and stop loss
        trade_data_df['take_profit'] = p.get('take_profit', 0.05)
        trade_data_df['stop_loss'] = p.get('stop_loss', 0.03)

        # Create indicators DataFrame
        df_indicators = pd.DataFrame(index=trade_data_df.index)
        
        # Add price-based indicators
        df_indicators['price'] = trade_data_df['dex_price']
        
        # Initialize default indicators that might be needed for fallback
        window = p.get('rsi_window', 14)
        df_indicators['rsi'] = ta.momentum.RSIIndicator(
            close=trade_data_df['dex_price'],
            window=window,
            fillna=True
        ).rsi()
        
        bb = ta.volatility.BollingerBands(
            close=trade_data_df['dex_price'],
            window=p.get('bb_window', 20),
            window_dev=p.get('bb_std', 2.0),
            fillna=True
        )
        df_indicators['bb_upper'] = bb.bollinger_hband()
        df_indicators['bb_lower'] = bb.bollinger_lband()
        df_indicators['bb_mid'] = bb.bollinger_mavg()
        
        # Extract conditions from parameters
        conditions = p.get('conditions', {})
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except json.JSONDecodeError:
                logger.error("Failed to parse conditions string")
                conditions = {}
                
        entry_conditions = conditions.get('entry', [])
        exit_conditions = conditions.get('exit', [])
        
        if not entry_conditions and not exit_conditions:
            # Fallback to default range trading conditions
            logger.info("Using default range trading conditions")
            entry_conditions = ["(df_indicators['rsi'] < 30) & (df_indicators['price'] <= df_indicators['bb_lower'])"]
            exit_conditions = ["(df_indicators['rsi'] > 70) | (df_indicators['price'] >= df_indicators['bb_upper'])"]
        
        # Add any additional indicators based on conditions
        for condition in entry_conditions + exit_conditions:
            if not isinstance(condition, str):
                continue
                
            # Skip if condition is not a valid Python expression
            if any(word in condition.lower() for word in ['touches', 'level', 'confidence']):
                continue
                
            condition_lower = condition.lower()
            if 'macd' in condition_lower:
                try:
                    macd = ta.trend.MACD(
                        close=trade_data_df['dex_price'],
                        window_fast=p.get('macd_fast', 12),
                        window_slow=p.get('macd_slow', 26),
                        window_sign=p.get('macd_signal', 9),
                        fillna=True
                    )
                    df_indicators['macd'] = macd.macd()
                    df_indicators['macd_signal'] = macd.macd_signal()
                except Exception as e:
                    logger.error(f"Error calculating MACD: {str(e)}")
                    df_indicators['macd'] = 0
                    df_indicators['macd_signal'] = 0
        
        # Ensure all indicators are properly filled
        df_indicators = df_indicators.ffill().bfill()
        
        # Generate entry signals from conditions
        try:
            entry_signal = pd.Series(True, index=trade_data_df.index)
            valid_conditions = []
            
            for condition in entry_conditions:
                if not isinstance(condition, str):
                    continue
                    
                # Skip if condition is not a valid Python expression
                if any(word in condition.lower() for word in ['touches', 'level', 'confidence']):
                    continue
                    
                # Replace indicator references with df_indicators
                condition = condition.replace("df['", "df_indicators['").replace("data['", "df_indicators['")
                try:
                    entry_signal &= eval(condition)
                    valid_conditions.append(condition)
                except Exception as e:
                    logger.warning(f"Invalid condition {condition}: {str(e)}")
                    
            if not valid_conditions:
                logger.warning("No valid conditions found, using defaults")
                entry_signal = (
                    (df_indicators['rsi'] < 30) & 
                    (df_indicators['price'] <= df_indicators['bb_lower'])
                )
            
            trade_data_df['entries'] = entry_signal
            
        except Exception as e:
            logger.error(f"Error evaluating entry conditions: {str(e)}")
            # Fallback to simple range trading logic
            trade_data_df['entries'] = (
                (df_indicators['rsi'] < 30) & 
                (df_indicators['price'] <= df_indicators['bb_lower'])
            )
        
        return trade_data_df
        
    except Exception as e:
        logger.error(f"Error calculating entries and params: {str(e)}")
        logger.exception("Full traceback:")
        return None

@njit
def post_signal_func_nb(c, trade_memory, fees, params):
    """Post-signal function for vectorbt backtest."""
    if vbt.pf_nb.order_filled_nb(c):
        exit_trade_records = vbt.pf_nb.get_exit_trade_records_nb(c)
        trade_memory.trade_records[: len(exit_trade_records), c.col] = exit_trade_records
        trade_memory.trade_counts[c.col] = len(exit_trade_records)

    # Set SL/TP init price to position averaged entry price
    if vbt.pf_nb.order_increased_position_nb(c):
        if params['enable_sl_mod'][0]:
            c.last_sl_info[c.col]["init_price"] = c.last_pos_info[c.col]["entry_price"]
        if params['enable_tp_mod'][0]:
            c.last_tp_info[c.col]["init_price"] = c.last_pos_info[c.col]["entry_price"]

@njit
def get_buy_price(coin_pool_value: float, sol_pool_value: float, amount_in_sol: float) -> Tuple[float, float]:
    """Calculate buy price and quantity."""
    major_quantity = coin_pool_value
    minor_quantity = sol_pool_value
    product = minor_quantity * major_quantity
    major_qty_after_execution = product / (minor_quantity + amount_in_sol)
    quantity = major_quantity - major_qty_after_execution
    price = amount_in_sol / quantity
    return price, quantity

@njit
def get_sell_price(coin_pool_value: float, sol_pool_value: float, amount_in_coin: float) -> Tuple[float, float]:
    """Calculate sell price and quantity."""
    major_quantity = coin_pool_value
    minor_quantity = sol_pool_value
    product = minor_quantity * major_quantity
    minor_qty_after_execution = product / (major_quantity + amount_in_coin)
    quantity = minor_quantity - minor_qty_after_execution
    price = quantity / amount_in_coin
    return price, quantity

@njit
def signal_func_nb(c, trade_memory, params, data, fees, size, price, last_buy_quantity, last_buy_index, last_sell_index):
    """Signal function for vectorbt backtest."""
    trade_count = trade_memory.trade_counts[c.col]
    trade_records = trade_memory.trade_records[:trade_count, c.col]

    open_trade_records = trade_records[trade_records["status"] == 0]
    closed_trade_records = trade_records[trade_records["status"] == 1]

    num_open_trades = len(open_trade_records)
    num_closed_trades = len(closed_trade_records)
    current_position_size = np.sum(
        open_trade_records["size"] * open_trade_records["entry_price"]
    )

    fees[c.i] = 0

    long_entry = data['entries'][c.i]
    long_exit = False
    size[c.i] = 0

    # Dynamic profit exit
    if num_open_trades > 0:
        current_price = c.close[c.i, 0]
        entry_price = open_trade_records["entry_price"][0]

        open_trade_return = (current_price / entry_price) - 1
        long_exit = open_trade_return > data['take_profit'][c.i]

        if current_price < entry_price:
            lookback = params['sl_window'][0]
            start_index = max(0, c.i - lookback + 1)
            stop_exit_price = np.mean(c.close[start_index:c.i+1, 0])
            open_trade_return = (stop_exit_price / entry_price) - 1.0
            long_exit = open_trade_return < -data['stop_loss'][c.i]

    # Handle exit
    if long_exit:
        size[c.i] = open_trade_records["size"][0]

    # Handle entry
    order_size_ratio = params['order_size'][0]
    order_size = order_size_ratio * data['sol_pool'][c.i]
    current_order_count = np.round(current_position_size / order_size)

    if long_entry and (current_order_count < params['max_orders'][0]):
        size[c.i] = order_size if order_size > 0.01 else 0.0
        long_entry = order_size > 0.01
    else:
        long_entry = False

    # Calculate price
    action_price = data['sol_pool'][c.i] / data['coin_pool'][c.i]
    
    if long_exit and last_buy_quantity[0][c.col] != 0:
        action_price, sell_quantity = get_sell_price(
            data['coin_pool'][c.i],
            data['sol_pool'][c.i],
            last_buy_quantity[0][c.col]
        )
        last_buy_quantity[0][c.col] = 0

    if long_entry:
        action_price, buy_quantity = get_buy_price(
            data['coin_pool'][c.i],
            data['sol_pool'][c.i],
            size[c.i][0]
        )
        last_buy_quantity[0][c.col] += buy_quantity

    price[c.i][c.col] = action_price

    # Post-trade delays
    if c.index[c.i] <= last_buy_index[0][c.col] + params['post_buy_delay'][0]:
        return False, False, False, False
    
    if long_entry and c.index[c.i] <= last_sell_index[0][c.col] + params['post_sell_delay'][0]:
        long_entry = False

    # Update indices
    if long_entry:
        last_buy_index[0][c.col] = data['slot'][c.i]

    if long_exit:
        last_sell_index[0][c.col] = data['slot'][c.i]

    return long_entry, long_exit, False, False

def init_trade_memory(target_shape: Tuple[int, int]) -> Any:
    """Initialize trade memory for vectorbt backtest."""
    global trade_memory
    if trade_memory is None:
        trade_memory = TradeMemory(
            trade_records=np.empty(target_shape, dtype=vbt.pf_enums.trade_dt),
            trade_counts=np.full(target_shape[1], 0)
        )
    return trade_memory

@vbt.parameterized(merge_func="column_stack")
def from_signals_backtest(trade_data_df: pd.DataFrame, **p) -> Any:
    """Run vectorbt backtest from entry/exit signals."""
    try:
        # Calculate entries and parameters
        trade_data_df = calculate_entries_and_params(trade_data_df, p)
        if trade_data_df is None:
            return None

        # Define parameter types
        params_dtype = [
            ('sl_window', np.int32),
            ('max_orders', np.float64),
            ('order_size', np.float64),
            ('enable_sl_mod', np.bool_),
            ('enable_tp_mod', np.bool_),
            ('post_buy_delay', np.int32),
            ('post_sell_delay', np.int32),
        ]
        params = np.array([
            (p['sl_window'], p['max_orders'], p['order_size'], p['enable_sl_mod'], p['enable_tp_mod'], p['post_buy_delay'], p['post_sell_delay'])
        ], dtype=params_dtype)

        # Define data types
        data_dtype = [
            ('entries', bool),
            ('stop_loss', np.float64),
            ('take_profit', np.float64),
            ('sol_pool', np.float64),
            ('coin_pool', np.float64),
            ('slot', np.int32),
        ]
        data = np.empty(len(trade_data_df), dtype=data_dtype)
        data['entries'] = trade_data_df['entries'].to_numpy()
        data['stop_loss'] = trade_data_df['stop_loss'].to_numpy()
        data['take_profit'] = trade_data_df['take_profit'].to_numpy()
        data['sol_pool'] = trade_data_df['sol_pool'].to_numpy()
        data['coin_pool'] = trade_data_df['coin_pool'].to_numpy()
        data['slot'] = trade_data_df.index.to_numpy()

        # Initialize arrays
        fees = np.full(len(trade_data_df), np.nan)
        size = np.full(len(trade_data_df), np.nan)
        price = trade_data_df["dex_price"].vbt.wrapper.fill().to_numpy()
        last_buy_quantity = np.array([[0]])
        last_buy_index = np.array([[0]])
        last_sell_index = np.array([[0]])

        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=trade_data_df["dex_price"],
            size=size,
            price=price,
            jitted=True,
            signal_func_nb=signal_func_nb,
            signal_args=(
                vbt.RepFunc(init_trade_memory),
                params,
                data,
                vbt.Rep("fees"),
                vbt.Rep("size"),
                vbt.Rep("price"),
                vbt.Rep("last_buy_quantity"),
                vbt.Rep("last_buy_index"),
                vbt.Rep("last_sell_index"),
            ),
            post_signal_func_nb=post_signal_func_nb,
            post_signal_args=(vbt.RepFunc(init_trade_memory), vbt.Rep("fees"), params),
            broadcast_named_args=dict(
                last_buy_quantity=last_buy_quantity,
                last_buy_index=last_buy_index,
                last_sell_index=last_sell_index
            ),
            accumulate=True,
            direction=0,
            init_cash=10,
            leverage=np.inf,
            leverage_mode="lazy",
            size_type="value",
            fees=fees,
            from_ago=0
        )
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        logger.exception("Full traceback:")
        return None

def calculate_stats(test_portfolio: Dict[str, Any], trade_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate performance statistics for backtested portfolios."""
    try:
        stats_list = []
        for asset, pf in test_portfolio.items():
            if pf is not None:
                try:
                    stats = {
                        'asset': asset,
                        'total_return': float(pf.total_return.iloc[0] if isinstance(pf.total_return, pd.Series) else pf.total_return),
                        'total_pnl': float(pf.trades.records.pnl.sum()),
                        'avg_pnl_per_trade': float(pf.trades.records['return'].mean()),
                        'total_orders': int(pf.orders.count().iloc[0] if isinstance(pf.orders.count(), pd.Series) else pf.orders.count()),
                        'total_trades': int(pf.trades.count().iloc[0] if isinstance(pf.trades.count(), pd.Series) else pf.trades.count()),
                        'sortino_ratio': float(pf.sortino_ratio.iloc[0] if isinstance(pf.sortino_ratio, pd.Series) else pf.sortino_ratio),
                    }
                    stats_list.append(stats)
                except Exception as e:
                    logger.error(f"Error calculating stats for {asset}: {str(e)}")
                    continue

        if not stats_list:
            logger.error("No valid statistics found for any asset")
            return pd.DataFrame()

        all_stats_df = pd.DataFrame(stats_list)
        all_stats_df.set_index('asset', inplace=True)
        all_stats_df = all_stats_df.sort_values('total_return', ascending=True)

        logger.info("All stats DataFrame:")
        logger.info("Top 10 assets:")
        logger.info(all_stats_df.head(10).to_string())
        logger.info("\nBottom 10 assets:")
        logger.info(all_stats_df.tail(10).to_string())

        df = all_stats_df
        name = "All assets"
        total_return_sum = df['total_return'].sum()
        total_pnl_sum = df['total_pnl'].sum()
        avg_sortino_ratio = df['sortino_ratio'].mean()
        avg_pnl_per_trade = total_return_sum / df['total_orders'].sum() if df['total_orders'].sum() > 0 else 0
        total_orders = df['total_orders'].sum()
        
        logger.info(f"\nStatistics for {name}:")
        logger.info(f"Sum of total returns: {total_return_sum:.6f}")
        logger.info(f"Sum of total pnl: {total_pnl_sum:.6f}")
        logger.info(f"Average Sortino ratio: {avg_sortino_ratio:.6f}")
        logger.info(f"Average PnL per trade: {avg_pnl_per_trade:.6f}")
        logger.info(f"Number of assets: {len(df)}")
        logger.info(f"Total orders: {total_orders}")

        return all_stats_df
        
    except Exception as e:
        logger.error(f"Error in calculate_stats: {str(e)}")
        logger.exception("Full traceback:")
        return pd.DataFrame()

def run_parameter_optimization(trade_data: Dict[str, pd.DataFrame], conditions: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
    """Run parameter optimization.
    
    Args:
        trade_data: Dictionary mapping asset names to DataFrames
        conditions: Dictionary containing entry/exit conditions
        
    Returns:
        Dictionary containing optimization results, or None if optimization fails
    """
    try:
        # Base parameter space with reduced combinations
        param_space = {
            'take_profit': [0.03, 0.05, 0.08],  # Reduced from 5 to 3
            'stop_loss': [0.02, 0.03],          # Reduced from 4 to 2
            'order_size': [0.1, 0.2],           # Reduced from 3 to 2
            'max_orders': [1, 2],               # Reduced from 3 to 2
            'sl_window': [200, 400],            # Reduced from 3 to 2
            'post_buy_delay': [1, 2],           # Reduced from 3 to 2
            'post_sell_delay': [3, 5],          # Reduced from 3 to 2
            'enable_sl_mod': [False],
            'enable_tp_mod': [False]
        }

        # Ensure conditions is a dictionary
        if not isinstance(conditions, dict):
            logger.warning("Invalid conditions format, using defaults")
            conditions = {
                'entry': ["(df_indicators['rsi'] < 30) & (df_indicators['price'] <= df_indicators['bb_lower'])"],
                'exit': ["(df_indicators['rsi'] > 70) | (df_indicators['price'] >= df_indicators['bb_upper'])"]
            }
        
        # Add indicator-specific parameters based on conditions
        all_conditions = conditions.get('entry', []) + conditions.get('exit', [])
        for condition in all_conditions:
            if not isinstance(condition, str):
                continue
                
            condition = condition.lower()
            if 'rsi' in condition:
                param_space.update({
                    'rsi_window': [14, 21],  # Reduced options
                })
            if 'macd' in condition:
                param_space.update({
                    'macd_fast': [12],
                    'macd_slow': [26],
                    'macd_signal': [9]
                })
            if 'bb' in condition or 'bollinger' in condition:
                param_space.update({
                    'bb_window': [20, 30],
                    'bb_std': [2.0, 2.5]
                })

        # Convert conditions to string
        conditions_str = json.dumps(conditions)

        # Generate limited parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Use a subset of combinations to keep optimization time reasonable
        max_combinations = 50  # Reduced from 100 to 50
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > max_combinations:
            selected_combinations = random.sample(all_combinations, max_combinations)
        else:
            selected_combinations = all_combinations

        logger.info(f"Testing {len(selected_combinations)} parameter combinations")
        
        # Run backtests with progress tracking
        results = []
        total_combinations = len(selected_combinations)
        
        for i, values in enumerate(selected_combinations, 1):
            try:
                # Progress update
                if i % 5 == 0:  # Show progress every 5 combinations
                    logger.info(f"Progress: {i}/{total_combinations} combinations tested")
                
                # Create parameters dictionary
                params = dict(zip(param_names, values))
                params['conditions'] = conditions_str
                
                # Run backtest for each asset
                portfolios = {}
                for asset, df in trade_data.items():
                    try:
                        portfolio = from_signals_backtest(df.copy(), **params)
                        if portfolio is not None:
                            portfolios[asset] = portfolio
                            
                        # Clear memory
                        gc.collect()
                        
                    except Exception as e:
                        logger.warning(f"Error in backtest for {asset}: {str(e)}")
                        continue
                
                if portfolios:
                    stats = calculate_stats(portfolios, trade_data)
                    if isinstance(stats, dict):
                        stats['parameters'] = params
                        results.append(stats)
                    
                # Clear memory after each iteration
                portfolios.clear()
                gc.collect()
                    
            except KeyboardInterrupt:
                logger.warning("Optimization interrupted by user")
                break
                
            except Exception as e:
                logger.warning(f"Error in backtest iteration: {str(e)}")
                continue
                
        if not results:
            logger.error("No successful backtest results")
            return None
            
        # Find best performing parameter set
        results_df = pd.DataFrame(results)
        results_df['score'] = (
            results_df['total_return'] * 0.4 +
            results_df['sortino_ratio'] * 0.3 +
            results_df['win_rate'] * 0.2 +
            (1 - results_df['max_drawdown'].abs()) * 0.1
        )
        
        best_result = results_df.loc[results_df['score'].idxmax()]
        
        return {
            'parameters': best_result['parameters'],
            'metrics': {
                'total_return': float(best_result['total_return']),
                'sortino_ratio': float(best_result['sortino_ratio']),
                'win_rate': float(best_result['win_rate']),
                'max_drawdown': float(best_result['max_drawdown']),
                'score': float(best_result['score'])
            }
        }
        
    except Exception as e:
        logger.error(f"Parameter optimization failed: {str(e)}")
        logger.exception("Full traceback:")
        return None

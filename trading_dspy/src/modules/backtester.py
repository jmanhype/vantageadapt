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
from .policy_optimizer import PolicyOptimizer
from numba.typed import Dict as NumbaDict
from numba import njit
import vectorbtpro as vbt
import ta
import itertools
from collections import namedtuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress debug output from dependencies
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('vectorbtpro').setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# VBT settings
vbt.settings.plotting.use_resampler = True
vbt.settings.wrapping['freq'] = 's'

# Define trade memory structure
global trade_memory
trade_memory = None
TradeMemory = namedtuple("TradeMemory", ["trade_records", "trade_counts"])

def calculate_entries_and_params(trade_data_df: pd.DataFrame, **p) -> pd.DataFrame:
    """Calculate entry signals and parameters based on multiple technical indicators."""
    try:
        # Set take profit and stop loss
        trade_data_df['take_profit'] = p['take_profit']
        trade_data_df['stop_loss'] = p['stop_loss']

        # Calculate SMA signals
        sma_20 = ta.trend.sma_indicator(trade_data_df['dex_price'], window=20)
        sma_50 = ta.trend.sma_indicator(trade_data_df['dex_price'], window=50)
        
        # Calculate RSI
        rsi = ta.momentum.rsi(trade_data_df['dex_price'], window=14)
        
        # Calculate volume metrics
        trade_data_df['sol_volume'] = trade_data_df['sol_pool']  # Use sol_pool as volume if sol_volume not present
        volume = trade_data_df['sol_volume']
        volume_sma = ta.trend.sma_indicator(volume, window=20)
        volume_increasing = volume > volume_sma
        
        # Calculate MACD for additional confirmation
        macd = vbt.MACD.run(
            trade_data_df['dex_price'], 
            fast_window=p['macd_signal_fast'], 
            slow_window=p['macd_signal_slow'], 
            signal_window=p['macd_signal_signal']
        )
        macd_signal = macd.macd.vbt.crossed_above(macd.signal)
        
        # Entry Conditions:
        # 1. Price crosses above SMA_20
        price_above_sma = trade_data_df['dex_price'].vbt.crossed_above(sma_20)
        
        # 2. RSI remains neutral or trends upwards (not oversold, trending up)
        rsi_condition = (rsi > 30) & (rsi.diff() > 0)
        
        # 3. Volume supports price movement
        volume_condition = volume_increasing
        
        # Combine all entry conditions
        trade_data_df['entries'] = (
            price_above_sma &  # Price crosses above SMA_20
            rsi_condition &    # RSI conditions met
            volume_condition & # Volume supports movement
            (macd.macd > p['min_macd_signal_threshold']) & # MACD above threshold
            macd_signal       # MACD crossover for confirmation
        )
        
        # Store indicators for exit conditions
        trade_data_df['sma_20'] = sma_20
        trade_data_df['rsi'] = rsi
        trade_data_df['volume_sma'] = volume_sma
        trade_data_df['macd'] = macd.macd
        trade_data_df['macd_signal'] = macd.signal
        trade_data_df['sol_volume'] = volume
        
        return trade_data_df
        
    except Exception as e:
        logger.error(f"Error calculating entries and params: {str(e)}")
        logger.exception("Full traceback:")
        return None

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
    """Signal function for vectorbt backtest with comprehensive entry/exit logic.
    
    Implements the following exit conditions:
    1. Price hits take profit or stop loss
    2. Price crosses below SMA_20 indicating potential trend reversal
    3. Significant drop in volume or change in buy/sell ratio
    """
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

    # Exit Conditions
    if num_open_trades > 0:
        current_price = c.close[c.i, 0]
        entry_price = open_trade_records["entry_price"][0]

        # 1. Take Profit condition
        take_profit_exit = (current_price / entry_price) - 1 > data['take_profit'][c.i]
        
        # 2. Stop Loss condition with dynamic calculation
        if current_price < entry_price:
            lookback = params['sl_window'][0]
            start_index = max(0, c.i - lookback + 1)
            stop_exit_price = np.mean(c.close[start_index:c.i+1, 0])
            stop_loss_exit = (stop_exit_price / entry_price) - 1.0 < -data['stop_loss'][c.i]
        else:
            stop_loss_exit = False
            
        # 3. Price crosses below SMA_20
        sma_cross_exit = False
        if c.i > 0:  # Check if we have previous data
            sma_cross_exit = (c.close[c.i-1, 0] > data['sma_20'][c.i-1]) and (current_price < data['sma_20'][c.i])
            
        # 4. Volume-based exit (significant drop in volume)
        volume_exit = False
        if c.i > 0:  # Check if we have previous data
            current_volume = data['sol_volume'][c.i]
            volume_sma = data['volume_sma'][c.i]
            volume_exit = (current_volume < volume_sma * 0.8)  # 20% below SMA
            
        # Combine exit conditions
        long_exit = take_profit_exit or stop_loss_exit or sma_cross_exit or volume_exit

    # Handle exit
    if long_exit:
        size[c.i] = open_trade_records["size"][0]

    # Handle entry
    order_size_ratio = params['order_size'][0]
    order_size = order_size_ratio * data['sol_volume'][c.i]  # Use sol_volume instead of sol_pool for consistency
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
    """Run vectorbt backtest from entry/exit signals with comprehensive technical indicators."""
    try:
        # Calculate entries and parameters
        trade_data_df = calculate_entries_and_params(trade_data_df, **p)
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

        # Define data types with new indicators
        data_dtype = [
            ('entries', bool),
            ('stop_loss', np.float64),
            ('take_profit', np.float64),
            ('sol_pool', np.float64),
            ('coin_pool', np.float64),
            ('slot', np.int32),
            ('sma_20', np.float64),
            ('rsi', np.float64),
            ('volume_sma', np.float64),
            ('macd', np.float64),
            ('macd_signal', np.float64),
            ('sol_volume', np.float64),
        ]
        
        # Create data array with all indicators
        data = np.empty(len(trade_data_df), dtype=data_dtype)
        data['entries'] = trade_data_df['entries'].to_numpy()
        data['stop_loss'] = trade_data_df['stop_loss'].to_numpy()
        data['take_profit'] = trade_data_df['take_profit'].to_numpy()
        data['sol_pool'] = trade_data_df['sol_pool'].to_numpy()
        data['coin_pool'] = trade_data_df['coin_pool'].to_numpy()
        data['slot'] = trade_data_df.index.to_numpy()
        data['sma_20'] = trade_data_df['sma_20'].to_numpy()
        data['rsi'] = trade_data_df['rsi'].to_numpy()
        data['volume_sma'] = trade_data_df['volume_sma'].to_numpy()
        data['macd'] = trade_data_df['macd'].to_numpy()
        data['macd_signal'] = trade_data_df['macd_signal'].to_numpy()
        data['sol_volume'] = trade_data_df['sol_volume'].to_numpy()

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

class Backtester:
    """Backtester class for running parameter optimization and backtesting."""
    
    def __init__(self, performance_thresholds: Optional[Dict[str, float]] = None):
        """Initialize backtester.
        
        Args:
            performance_thresholds: Optional dictionary of performance thresholds
        """
        self.performance_thresholds = performance_thresholds or {
            'min_return': 0.10,
            'min_trades': 10,
            'max_drawdown': 0.20
        }
        self.policy_optimizer = PolicyOptimizer()
        
    def run_parameter_optimization(self, trade_data: Dict[str, pd.DataFrame], conditions: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run parameter optimization for backtesting.
        
        Args:
            trade_data: Dictionary mapping asset names to DataFrames
            conditions: Dictionary containing entry/exit conditions
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            global trade_memory
            
            # Define parameter ranges
            params = {
                "take_profit": 0.08,
                "stop_loss": 0.12,
                "sl_window": 400,
                "max_orders": 3,
                "order_size": 0.0025,
                "post_buy_delay": 2,
                "post_sell_delay": 5,
                "macd_signal_fast": 120,
                "macd_signal_slow": 260,
                "macd_signal_signal": 90,
                "min_macd_signal_threshold": 0,
                "max_macd_signal_threshold": 0,
                "enable_sl_mod": False,
                "enable_tp_mod": False,
            }

            # Define optimization parameters with ranges suitable for multi-indicator strategy
            rand_test_params = {
                **params,
                "take_profit": vbt.Param(np.arange(0.05, 0.30, 0.02)),  # Wider range for volatile markets
                "stop_loss": vbt.Param(np.arange(0.03, 0.15, 0.01)),   # Tighter initial stops
                "macd_signal_fast": vbt.Param(np.arange(50, 500, 50)),  # Shorter ranges for faster signals
                "macd_signal_slow": vbt.Param(np.arange(100, 1000, 100)), # More moderate slow period
                "macd_signal_signal": vbt.Param(np.arange(20, 200, 20)),  # Faster signal line
                "post_buy_delay": params["post_buy_delay"],  # Keep original value
                "post_sell_delay": params["post_sell_delay"], # Keep original value
                "_random_subset": 200
            }
            
            # Run optimization on test assets
            train_portfolio = {}
            test_assets = list(trade_data.keys())[:2]
            for asset, asset_data in [(asset, trade_data[asset]) for asset in test_assets]:
                trade_memory = None
                df = asset_data.copy()
                
                # Trim length
                two_weeks_ago = asset_data['timestamp'].max() - pd.Timedelta(weeks=2)
                df = df[df['timestamp'] >= two_weeks_ago]

                pf = from_signals_backtest(df, **rand_test_params)
                train_portfolio[asset] = pf

            # Get best parameters
            portfolio = {}
            for asset, pf in train_portfolio.items():
                df = trade_data[asset].copy()
                combined_metrics = pd.DataFrame({
                    'total_return': pf.total_return,
                    'total_orders': pf.orders.count(),
                    'sortino_ratio': pf.sortino_ratio
                })
                
                combined_metrics['total_return'] = combined_metrics['total_return'] / len(df)
                combined_metrics['score'] = combined_metrics['total_orders'] * combined_metrics['sortino_ratio']
                negative_returns = combined_metrics['total_return'] < 0
                combined_metrics.loc[negative_returns, 'score'] *= 1 / combined_metrics.loc[negative_returns, 'sortino_ratio']    
                portfolio[asset] = combined_metrics

            portfolio_concat = pd.DataFrame()
            for asset, metrics in portfolio.items():
                portfolio_concat = pd.concat([portfolio_concat, metrics])

            grouped = portfolio_concat.groupby(level=portfolio_concat.index.names)
            result = grouped.agg({
                'total_return': 'sum',
                'total_orders': 'sum',
                'sortino_ratio': 'mean',
                'score': 'mean'
            })
            result.sort_values('score', ascending=False)
            result_reset = result.reset_index()
            best_score_params = result_reset.loc[result_reset['score'].idxmax()].to_dict()

            updated_params = rand_test_params.copy()
            for key, value in best_score_params.items():
                if key in updated_params:
                    if isinstance(updated_params[key], vbt.Param):
                        updated_params[key] = value
                    else:
                        pass

            updated_params.pop('score', None)
            updated_params.pop('_random_subset', None)

            # Convert float values ending in .0 to int and ensure all required parameters are included
            final_params = {
                'take_profit': updated_params['take_profit'],
                'stop_loss': updated_params['stop_loss'],
                'sl_window': updated_params['sl_window'],
                'max_orders': updated_params['max_orders'],
                'order_size': updated_params['order_size'],
                'post_buy_delay': updated_params['post_buy_delay'],
                'post_sell_delay': updated_params['post_sell_delay'],
                'macd_signal_fast': updated_params['macd_signal_fast'],
                'macd_signal_slow': updated_params['macd_signal_slow'],
                'macd_signal_signal': updated_params['macd_signal_signal'],
                'min_macd_signal_threshold': updated_params['min_macd_signal_threshold'],
                'max_macd_signal_threshold': updated_params['max_macd_signal_threshold'],
                'enable_sl_mod': updated_params['enable_sl_mod'],
                'enable_tp_mod': updated_params['enable_tp_mod']
            }

            # Convert float values ending in .0 to int
            for key, value in final_params.items():
                if isinstance(value, float) and value.is_integer():
                    final_params[key] = int(value)

            logger.info("Updated parameters:")
            logger.info(final_params)

            # Run tests on all assets
            test_portfolio = {}
            for asset, trade_data_df in trade_data.items():
                trade_memory = None
                trade_data_df = trade_data_df.copy()
                
                # Trim length
                two_weeks_ago = trade_data_df['timestamp'].max() - pd.Timedelta(weeks=2)
                trade_data_df = trade_data_df[trade_data_df['timestamp'] >= two_weeks_ago]

                pf = from_signals_backtest(trade_data_df, **final_params)
                test_portfolio[asset] = pf

            # Calculate stats
            all_stats_df = calculate_stats(test_portfolio, trade_data)
            
            # Prepare trade memory stats
            trade_memory_stats = {
                'total_trades': int(all_stats_df['total_trades'].sum()),
                'win_rate': float((all_stats_df['total_return'] > 0).mean()),
                'avg_trade_return': float(all_stats_df['total_return'].mean()),
                'avg_trade_duration': 0,  # Would need to calculate from actual trade records
                'max_drawdown': 0,  # Would need to calculate from portfolio equity curve
                'sharpe_ratio': 0,  # Would need to calculate from returns
                'sortino_ratio': float(all_stats_df['sortino_ratio'].mean()),
                'profit_factor': 0  # Would need to calculate from trade records
            }
            
            # Prepare metrics
            metrics = {
                'total_return': float(all_stats_df['total_return'].sum()),
                'total_pnl': float(all_stats_df['total_pnl'].sum()),
                'avg_pnl_per_trade': float(all_stats_df['avg_pnl_per_trade'].mean()),
                'total_trades': int(all_stats_df['total_trades'].sum()),
                'win_rate': float((all_stats_df['total_return'] > 0).mean()),
                'sortino_ratio': float(all_stats_df['sortino_ratio'].mean()),
                'asset_count': len(all_stats_df),
                'total_orders': int(all_stats_df['total_orders'].sum())
            }

            # Extract trades data from portfolios
            trades_data = {}
            for asset, pf in test_portfolio.items():
                if pf is not None and hasattr(pf, 'trades'):
                    trades_data[asset] = {
                        'records': pf.trades.records.to_dict(),
                        'stats': {
                            'total_trades': int(pf.trades.count().iloc[0]),
                            'win_rate': float((pf.trades.records['pnl'] > 0).mean()),
                            'avg_trade_return': float(pf.trades.records['return'].mean()),
                            'total_pnl': float(pf.trades.records['pnl'].sum())
                        }
                    }

            # Create backtest results structure
            backtest_results = {
                'total_return': float(all_stats_df['total_return'].sum()),
                'total_pnl': float(all_stats_df['total_pnl'].sum()),
                'sortino_ratio': float(all_stats_df['sortino_ratio'].mean()),
                'win_rate': float((all_stats_df['total_return'] > 0).mean()),
                'total_trades': int(all_stats_df['total_trades'].sum()),
                'trades': trades_data,
                'metrics': {
                    'avg_pnl_per_trade': float(all_stats_df['avg_pnl_per_trade'].mean()),
                    'asset_count': len(all_stats_df),
                    'total_orders': int(all_stats_df['total_orders'].sum()),
                    'trade_memory_stats': trade_memory_stats,
                    'per_asset_stats': {
                        'total_return': all_stats_df['total_return'].to_dict(),
                        'avg_pnl_per_trade': all_stats_df['avg_pnl_per_trade'].to_dict()
                    }
                },
                'performance_thresholds': self.performance_thresholds
            }

            # Save optimization results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results = {
                'parameters': updated_params,
                'backtest_results': backtest_results,
                'metrics': metrics,
                'trade_memory_stats': trade_memory_stats,
                'stats': {
                    'total_return': all_stats_df['total_return'].to_dict(),
                    'avg_pnl_per_trade': all_stats_df['avg_pnl_per_trade'].to_dict()
                },
                'trades': trades_data
            }
            
            with open(f'optimization_results_{timestamp}.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Optimization results saved to optimization_results_{timestamp}.json")
            return results
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {str(e)}")
            logger.exception("Full traceback:")
            return None
            
    def forward(self, trade_data: Dict[str, pd.DataFrame], parameters: Dict[str, Any], conditions: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run backtesting with parameter optimization.
        
        Args:
            trade_data: Dictionary mapping asset names to DataFrames
            parameters: Strategy parameters
            conditions: Dictionary containing entry/exit conditions
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Run parameter optimization
            results = self.run_parameter_optimization(trade_data=trade_data, conditions=conditions)
            
            if not results:
                logger.error("Parameter optimization failed")
                return {}
            
            backtest_results = results['backtest_results']
            
            # Get the final parameters from the results
            final_params = results.get('parameters', {})
            if not final_params:
                logger.error("No parameters found in optimization results")
                return {}
                
            # Store final params as class variable
            self.current_params = final_params
            
            # Use policy optimizer to adjust parameters and rules
            adjusted_params, adjusted_rules = self.policy_optimizer.feedback_loop(
                backtest_results,
                self.current_params,
                conditions
            )
            
            # Update the parameters and conditions for next iteration
            parameters.update(adjusted_params)
            conditions.update(adjusted_rules)
            
            # Check if performance meets thresholds
            total_return = backtest_results['total_return']
            total_trades = backtest_results['total_trades']
            max_drawdown = backtest_results.get('metrics', {}).get('trade_memory_stats', {}).get('max_drawdown', 0)
            
            # Use policy optimizer's current targets for validation
            validation_passed = (
                total_return >= self.policy_optimizer.performance_targets['min_return'] and
                total_trades >= self.policy_optimizer.performance_targets['min_trades'] and
                abs(max_drawdown) <= self.performance_thresholds.get('max_drawdown', 0.20)
            )
            
            results['validation_passed'] = validation_passed
            results['performance_thresholds'] = self.policy_optimizer.performance_targets
            results['optimization_history'] = self.policy_optimizer.optimization_history
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            logger.exception("Full traceback:")
            return {}

if __name__ == "__main__":
    run_parameter_optimization()

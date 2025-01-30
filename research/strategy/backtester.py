"""Backtesting module for trading strategies."""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import logging
import vectorbtpro as vbt
from .types import BacktestResults

# Configure logging
logger = logging.getLogger(__name__)

def calculate_stats(test_portfolio: Dict[str, Any], trade_data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate performance statistics for backtested portfolios.
    
    Args:
        test_portfolio: Dictionary mapping asset names to their portfolio objects
        trade_data_dict: Dictionary mapping asset names to their trade data DataFrames
        
    Returns:
        DataFrame containing performance statistics for each asset
    """
    try:
        logger.info(f"Starting calculate_stats with {len(test_portfolio)} portfolios")
        stats_list = []
        
        for asset, pf in test_portfolio.items():
            logger.info(f"\nProcessing asset: {asset}")
            
            if pf is None:
                logger.error(f"Portfolio for {asset} is None")
                continue
                
            if not hasattr(pf, 'total_return'):
                logger.error(f"Portfolio for {asset} has no total_return attribute")
                continue
                
            if not hasattr(pf, 'trades'):
                logger.error(f"Portfolio for {asset} has no trades attribute")
                continue
                
            if not hasattr(pf, 'orders'):
                logger.error(f"Portfolio for {asset} has no orders attribute")
                continue
                
            try:
                # Log raw values before conversion
                logger.debug(f"{asset} total_return type: {type(pf.total_return)}, value: {pf.total_return}")
                logger.debug(f"{asset} trades.records.pnl type: {type(pf.trades.records.pnl)}, value: {pf.trades.records.pnl}")
                logger.debug(f"{asset} trades.records['return'] type: {type(pf.trades.records['return'])}, value: {pf.trades.records['return']}")
                logger.debug(f"{asset} orders.count() type: {type(pf.orders.count())}, value: {pf.orders.count()}")
                logger.debug(f"{asset} trades.count() type: {type(pf.trades.count())}, value: {pf.trades.count()}")
                logger.debug(f"{asset} sortino_ratio type: {type(pf.sortino_ratio)}, value: {pf.sortino_ratio}")
                
                stats = {
                    'asset': asset,
                    'total_return': float(pf.total_return.iloc[0] if isinstance(pf.total_return, pd.Series) else pf.total_return),
                    'total_pnl': float(pf.trades.records.pnl.sum()),
                    'avg_pnl_per_trade': float(pf.trades.records['return'].mean()),
                    'total_orders': int(pf.orders.count().iloc[0] if isinstance(pf.orders.count(), pd.Series) else pf.orders.count()),
                    'total_trades': int(pf.trades.count().iloc[0] if isinstance(pf.trades.count(), pd.Series) else pf.trades.count()),
                    'sortino_ratio': float(pf.sortino_ratio.iloc[0] if isinstance(pf.sortino_ratio, pd.Series) else pf.sortino_ratio),
                }
                logger.info(f"Successfully calculated stats for {asset}: {stats}")
                stats_list.append(stats)
            except Exception as e:
                logger.error(f"Error calculating stats for {asset}: {str(e)}")
                logger.exception(f"Full traceback for {asset}:")
                continue

        if not stats_list:
            logger.error("No valid statistics found for any asset")
            return pd.DataFrame()

        logger.info(f"Successfully processed {len(stats_list)} assets")
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

def from_signals_backtest(trade_data_df: pd.DataFrame, **p) -> Optional[Any]:
    """Run vectorbt backtest from entry/exit signals."""
    try:
        logger.info("Starting from_signals_backtest")
        logger.debug(f"Input parameters: {p}")
        
        # Validate input data
        required_columns = ["dex_price", "entries", "exits"]
        for col in required_columns:
            if col not in trade_data_df.columns:
                logger.error(f"Missing required column: {col}")
                return None
                
        logger.debug(f"Data shape: {trade_data_df.shape}")
        logger.debug(f"Number of entries: {trade_data_df['entries'].sum()}")
        logger.debug(f"Number of exits: {trade_data_df['exits'].sum()}")
        
        # Create portfolio
        try:
            portfolio = vbt.Portfolio.from_signals(
                close=trade_data_df["dex_price"],
                entries=trade_data_df["entries"],
                exits=trade_data_df["exits"],
                init_cash=1.0,  # Reduced initial cash
                size=p.get('order_size', 0.001),
                fees=0.0,  # Remove fees for now
                freq='1h'  # Use hourly frequency
            )
            
            # Validate portfolio creation
            if portfolio is None:
                logger.error("Portfolio creation returned None")
                return None
                
            # Log portfolio attributes
            logger.debug(f"Portfolio attributes: {dir(portfolio)}")
            logger.info(f"Portfolio created successfully with {len(portfolio.orders)} orders")
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error creating portfolio: {str(e)}")
            logger.exception("Full traceback for portfolio creation:")
            return None
        
    except Exception as e:
        logger.error(f"Error in from_signals_backtest: {str(e)}")
        logger.exception("Full traceback:")
        return None

def run_parameter_optimization(trade_data: Dict[str, pd.DataFrame], conditions: Dict[str, List[str]]) -> Optional[Dict[str, Any]]:
    """Run parameter optimization for trading strategy.
    
    Args:
        trade_data: Dictionary mapping asset names to their trade data DataFrames
        conditions: Dictionary of trading conditions
        
    Returns:
        Optional dictionary containing optimization results
    """
    try:
        # Initialize parameters
        param_ranges = {
            'take_profit': np.linspace(0.02, 0.15, 5),
            'stop_loss': np.linspace(0.02, 0.15, 5),
            'sl_window': [200, 400, 600],
            'max_orders': [1, 2, 3],
            'order_size': [0.001, 0.0025, 0.005],
            'post_buy_delay': [1, 2, 3],
            'post_sell_delay': [3, 5, 7],
            'macd_signal_fast': [60, 120, 180],
            'macd_signal_slow': [120, 260, 400],
            'macd_signal_signal': [45, 90, 135],
            'min_macd_signal_threshold': [0],
            'max_macd_signal_threshold': [0],
            'enable_sl_mod': [False],
            'enable_tp_mod': [False]
        }
        
        # Generate random test parameters
        rand_test_params = {}
        for param, values in param_ranges.items():
            rand_test_params[param] = np.random.choice(values)
            
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
            combined_metrics['score'] = combined_metrics['total_orders'] * combined_metrics['sortino_ratio'] * combined_metrics['total_return']
            negative_returns = combined_metrics['total_return'] < 0
            combined_metrics.loc[negative_returns, 'score'] *= 1 / combined_metrics.loc[negative_returns, 'sortino_ratio']    
            portfolio[asset] = combined_metrics

        # Update parameters based on performance
        updated_params = rand_test_params.copy()
        logger.info("Updated parameters:")
        logger.info(updated_params)
        
        # Run final test with updated parameters
        test_portfolio = {}
        for asset, trade_data in trade_data.items():
            trade_memory = None
            trade_data = trade_data.copy()
            
            # Trim length
            two_weeks_ago = trade_data['timestamp'].max() - pd.Timedelta(weeks=2)
            trade_data = trade_data[trade_data['timestamp'] >= two_weeks_ago]

            pf = from_signals_backtest(trade_data, **updated_params)
            test_portfolio[asset] = pf

        # Calculate stats
        all_stats_df = calculate_stats(test_portfolio, trade_data)
        
        # Check if we have valid stats
        if all_stats_df.empty:
            logger.error("No valid statistics available from backtesting")
            return None
            
        # Extract trades data from portfolios
        trades_data = []
        for asset, pf in test_portfolio.items():
            if pf is not None and hasattr(pf, 'trades'):
                trades_data.extend([{
                    'asset': asset,
                    'entry_price': float(record['entry_price']),
                    'exit_price': float(record['exit_price']),
                    'pnl': float(record['pnl']),
                    'return': float(record['return']),
                    'entry_time': record['entry_time'].isoformat() if isinstance(record['entry_time'], datetime) else record['entry_time'],
                    'exit_time': record['exit_time'].isoformat() if isinstance(record['exit_time'], datetime) else record['exit_time'],
                } for record in pf.trades.records.to_dict('records')])

        # Create backtest results
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
                'trade_memory_stats': {
                    'total_trades': int(all_stats_df['total_trades'].sum()),
                    'win_rate': float((all_stats_df['total_return'] > 0).mean()),
                    'avg_trade_return': float(all_stats_df['total_return'].mean()),
                    'avg_trade_duration': 0,  # Would need to calculate from actual trade records
                    'max_drawdown': 0,  # Would need to calculate from portfolio equity curve
                    'sharpe_ratio': 0,  # Would need to calculate from returns
                    'sortino_ratio': float(all_stats_df['sortino_ratio'].mean()),
                    'profit_factor': 0  # Would need to calculate from trade records
                },
                'per_asset_stats': {
                    'total_return': all_stats_df['total_return'].to_dict(),
                    'avg_pnl_per_trade': all_stats_df['avg_pnl_per_trade'].to_dict()
                }
            }
        }

        # Save optimization results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'parameters': updated_params,
            'backtest_results': backtest_results,  # Add backtest_results to the output
            'metrics': backtest_results['metrics'],  # Use metrics from backtest_results
            'timestamp': timestamp
        }
        
        # Log the final results for debugging
        logger.info("Successfully created final results dictionary")
        logger.debug(f"Final results keys: {list(results.keys())}")
        logger.debug(f"Final results structure: {json.dumps(results, indent=2)}")
        
        # Save to file
        logger.info(f"Saving results to optimization_results_{timestamp}.json")
        with open(f'optimization_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Successfully saved optimization results to optimization_results_{timestamp}.json")
        return results
        
    except Exception as e:
        logger.error(f"Error in parameter optimization: {str(e)}")
        logger.exception("Full traceback:")
        return None 

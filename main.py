"""Strategic trading system with LLM-driven decision making and self-improvement capabilities."""

import asyncio
import logging
import argparse
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
import json
from pathlib import Path
import importlib
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from research.database import db
from research.strategy.strategy_generator import StrategicTrader
from research.analysis.trade_analyzer import TradeAnalyzer
from research.visualization.trade_visualizer import TradeVisualizer
from research.strategy.godel_agent import GodelAgent
from backtester import load_trade_data, calculate_stats, run_parameter_optimization

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Performance validation thresholds
PERFORMANCE_THRESHOLDS = {
    'min_return': 1.0,          # 100% minimum return
    'min_trades': 1000,         # Minimum number of trades
    'min_assets': 100,          # Minimum number of assets
    'min_sharpe': 1.5,          # Minimum Sharpe ratio
    'min_sortino': 2.0,         # Minimum Sortino ratio
    'max_drawdown': -0.25,      # Maximum allowed drawdown
    'min_win_rate': 0.4,        # Minimum win rate
    'max_consecutive_losses': 5, # Maximum consecutive losing trades
    'min_profit_factor': 1.2,   # Minimum profit factor
}

def validate_strategy_performance(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate if the strategy meets minimum performance requirements.
    
    Args:
        metrics: Dictionary containing strategy performance metrics
        
    Returns:
        Tuple of (success, failures) where success is a boolean indicating if all requirements were met,
        and failures is a list of strings describing which requirements failed
    """
    failures = []
    min_requirements = {
        'total_return': 0.5,  # Minimum 50% total return
        'sortino_ratio': 2.0,  # Minimum Sortino ratio
        'win_rate': 0.4,  # Minimum 40% win rate
        'total_trades': 100  # Minimum number of trades
    }
    
    if not metrics:
        failures.append("No metrics available for validation")
        return False, failures
        
    total_return = float(metrics.get('total_return', 0))
    sortino_ratio = float(metrics.get('sortino_ratio', 0))
    win_rate = float(metrics.get('win_rate', 0))
    total_trades = int(metrics.get('total_trades', 0))
    
    if total_return < min_requirements['total_return']:
        failures.append(f"Total return {total_return:.2f} below minimum requirement of {min_requirements['total_return']}")
        
    if sortino_ratio < min_requirements['sortino_ratio']:
        failures.append(f"Sortino ratio {sortino_ratio:.2f} below minimum requirement of {min_requirements['sortino_ratio']}")
        
    if win_rate < min_requirements['win_rate']:
        failures.append(f"Win rate {win_rate:.2f} below minimum requirement of {min_requirements['win_rate']}")
        
    if total_trades < min_requirements['total_trades']:
        failures.append(f"Total trades {total_trades} below minimum requirement of {min_requirements['total_trades']}")
    
    return len(failures) == 0, failures

def calculate_detailed_metrics(portfolio: Any, basic_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Calculate detailed performance metrics including trade patterns."""
    try:
        # Get trades from portfolio
        if hasattr(portfolio, 'trades'):
            trades = portfolio.trades.records_readable
        else:
            logger.error("Portfolio has no trades attribute")
            return basic_metrics
            
        # Calculate trade-level metrics
        trade_metrics = {
            'avg_trade_duration': float((trades['exit_time'] - trades['entry_time']).mean()),
            'avg_profit_per_trade': float(trades['pnl'].mean()),
            'profit_factor': float(abs(trades[trades['pnl'] > 0]['pnl'].sum() / trades[trades['pnl'] < 0]['pnl'].sum())),
            'max_consecutive_losses': calculate_max_consecutive_losses(trades),
            'win_rate': float((trades['pnl'] > 0).mean()),
            'best_trade': float(trades['pnl'].max()),
            'worst_trade': float(trades['pnl'].min()),
            'avg_win': float(trades[trades['pnl'] > 0]['pnl'].mean()),
            'avg_loss': float(trades[trades['pnl'] < 0]['pnl'].mean())
        }
        
        # Add basic metrics
        detailed_metrics = {**basic_metrics, **trade_metrics}
        
        # Add trade patterns analysis
        detailed_metrics['trade_patterns'] = analyze_trade_patterns(trades)
        detailed_metrics['market_conditions'] = analyze_market_conditions(portfolio)
        
        return detailed_metrics
        
    except Exception as e:
        logger.error(f"Error calculating detailed metrics: {str(e)}")
        return basic_metrics

def calculate_max_consecutive_losses(trades: pd.DataFrame) -> int:
    """Calculate maximum consecutive losing trades."""
    consecutive = 0
    max_consecutive = 0
    
    for pnl in trades['pnl']:
        if pnl < 0:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
            
    return max_consecutive

def analyze_trade_patterns(trades: pd.DataFrame) -> Dict[str, Any]:
    """Analyze patterns in trading behavior."""
    return {
        'time_of_day': analyze_time_patterns(trades),
        'trade_clustering': analyze_trade_clustering(trades),
        'win_loss_streaks': analyze_win_loss_streaks(trades),
        'position_sizing': analyze_position_sizing(trades)
    }

def analyze_market_conditions(portfolio: Any) -> Dict[str, Any]:
    """Analyze market conditions during trades."""
    return {
        'volatility_regime': analyze_volatility(portfolio),
        'trend_regime': analyze_trend(portfolio),
        'volume_profile': analyze_volume(portfolio),
        'liquidity_conditions': analyze_liquidity(portfolio)
    }

async def run_strategy(theme: str, trade_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """Run the trading strategy."""
    metrics = None
    best_metrics = None
    result = None
    best_result = None
    
    # Initialize trader using async factory method
    trader = await StrategicTrader.create()
    
    # Initialize memory manager
    mem0_api_key = os.getenv("MEM0_API_KEY")
    if mem0_api_key:
        from research.strategy.memory_manager import TradingMemoryManager
        memory_manager = TradingMemoryManager(api_key=mem0_api_key)
        logger.info("Memory system initialized successfully")
    else:
        memory_manager = None
        logger.warning("Memory system disabled: Missing MEM0_API_KEY")
    
    for iteration in range(5):  # Run 5 iterations
        logger.info(f"\nStarting iteration {iteration + 1}")
        
        try:
            # Analyze market context
            logger.info("Analyzing market context...")
            market_data = list(trade_data.values())[0]  # Use first asset for market analysis
            market_context = await trader.analyze_market(market_data)
            if not market_context:
                logger.error("Failed to analyze market context")
                continue
                
            logger.info(f"Market regime: {market_context.regime}")
            logger.info(f"Confidence: {market_context.confidence:.2f}")
            logger.info(f"Risk level: {market_context.risk_level}")
            
            # Query similar strategies if memory system is enabled
            if memory_manager and memory_manager.enabled:
                similar_strategies = memory_manager.query_similar_strategies(
                    market_regime=market_context.regime
                )
                if similar_strategies:
                    logger.info(f"Found {len(similar_strategies)} similar strategies")
                    for strategy in similar_strategies:
                        logger.debug(f"Similar strategy: {strategy}")
            
            # Generate strategy insights
            logger.info("\nGenerating strategy insights...")
            strategy_insights = await trader.generate_strategy(theme)
            
            if not strategy_insights:
                logger.error("Failed to generate strategy insights")
                continue

            # Generate trading rules
            logger.info("\nGenerating trading rules...")
            conditions, parameters = await trader.generate_trading_rules(strategy_insights, market_context)
            
            if not conditions or not parameters:
                logger.error("Failed to generate trading rules")
                continue
            
            # Run parameter optimization
            logger.info("\nOptimizing parameters...")
            result = run_parameter_optimization(trade_data, conditions)
            
            if result is None:
                logger.warning("Parameter optimization failed")
                continue
                
            metrics = result.get('metrics', {})
            success, failures = validate_strategy_performance(metrics)
            
            # Store strategy results in memory if enabled
            if memory_manager and memory_manager.enabled:
                from research.strategy.types import StrategyContext, BacktestResults
                context = StrategyContext(
                    market_regime=market_context.regime,
                    parameters=parameters,
                    confidence=market_context.confidence,
                    risk_level=market_context.risk_level
                )
                results = BacktestResults(
                    total_return=metrics.get('total_return', 0.0),
                    total_pnl=metrics.get('total_pnl', 0.0),
                    sortino_ratio=metrics.get('sortino_ratio', 0.0),
                    win_rate=metrics.get('win_rate', 0.0),
                    total_trades=metrics.get('total_trades', 0),
                    asset_count=len(trade_data)
                )
                memory_manager.store_strategy_results(context, results, parameters)
                logger.info("Stored strategy results in memory")
            
            # Update best result if this iteration was better
            if best_metrics is None or metrics.get('total_return', 0) > best_metrics.get('total_return', 0):
                best_metrics = metrics
                best_result = result
            
            if not success:
                logger.info("\nStrategy did not meet performance requirements")
                continue
                
        except Exception as e:
            logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
            continue
    
    if best_result is None:
        logger.warning("No successful strategy configuration found")
        return None
        
    logger.info("\nStrategy run completed")
    logger.info(f"Best metrics achieved: {json.dumps(best_metrics, indent=2)}")
    
    return best_result

async def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Run trading strategy with self-improvement')
    parser.add_argument('--theme', type=str, required=True, help='Trading strategy theme')
    parser.add_argument('--data', type=str, required=True, help='Path to trade data pickle file')
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading trade data...")
    trade_data = load_trade_data(args.data)
    if trade_data is None:
        logger.error("Failed to load trade data")
        return
    
    # Run strategy with Gödel Agent
    logger.info(f"\nRunning strategy with theme: {args.theme}")
    result = await run_strategy(args.theme, trade_data)
    
    # Log final results
    if result is None:
        logger.error("Strategy run failed")
        return
        
    if result.get('performance_ok', False):
        logger.info("Strategy optimization completed successfully!")
        logger.info(f"Final metrics: {result.get('metrics', {})}")
    else:
        logger.warning("Strategy did not meet performance requirements")
        logger.warning(f"Failures: {result.get('failures', [])}")
        
    return result

def analyze_time_patterns(trades: pd.DataFrame) -> Dict[str, Any]:
    """Analyze trading patterns based on time of day."""
    try:
        trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
        hourly_stats = trades.groupby('hour').agg({
            'pnl': ['count', 'mean', 'sum'],
            'size': 'mean'
        })
        
        return {
            'best_hour': int(hourly_stats['pnl']['mean'].idxmax()),
            'worst_hour': int(hourly_stats['pnl']['mean'].idxmin()),
            'most_active_hour': int(hourly_stats['pnl']['count'].idxmax()),
            'hourly_pnl_mean': hourly_stats['pnl']['mean'].to_dict()
        }
    except Exception as e:
        logger.error(f"Error analyzing time patterns: {str(e)}")
        return {}

def analyze_trade_clustering(trades: pd.DataFrame) -> Dict[str, Any]:
    """Analyze if trades tend to cluster together."""
    try:
        trades['time_diff'] = trades['entry_time'].diff()
        
        return {
            'avg_time_between_trades': trades['time_diff'].mean(),
            'min_time_between_trades': trades['time_diff'].min(),
            'max_time_between_trades': trades['time_diff'].max(),
            'trade_frequency_std': trades['time_diff'].std()
        }
    except Exception as e:
        logger.error(f"Error analyzing trade clustering: {str(e)}")
        return {}

def analyze_win_loss_streaks(trades: pd.DataFrame) -> Dict[str, Any]:
    """Analyze winning and losing streaks."""
    try:
        trades['is_win'] = trades['pnl'] > 0
        
        # Calculate streaks
        streak = 1
        streaks = []
        current_type = trades['is_win'].iloc[0]
        
        for is_win in trades['is_win'][1:]:
            if is_win == current_type:
                streak += 1
            else:
                streaks.append((current_type, streak))
                current_type = is_win
                streak = 1
        streaks.append((current_type, streak))
        
        # Convert to DataFrame for analysis
        streaks_df = pd.DataFrame(streaks, columns=['is_win', 'length'])
        
        return {
            'max_win_streak': int(streaks_df[streaks_df['is_win']]['length'].max()),
            'max_loss_streak': int(streaks_df[~streaks_df['is_win']]['length'].max()),
            'avg_win_streak': float(streaks_df[streaks_df['is_win']]['length'].mean()),
            'avg_loss_streak': float(streaks_df[~streaks_df['is_win']]['length'].mean())
        }
    except Exception as e:
        logger.error(f"Error analyzing win/loss streaks: {str(e)}")
        return {}

def analyze_position_sizing(trades: pd.DataFrame) -> Dict[str, Any]:
    """Analyze position sizing patterns."""
    try:
        return {
            'avg_position_size': float(trades['size'].mean()),
            'max_position_size': float(trades['size'].max()),
            'min_position_size': float(trades['size'].min()),
            'position_size_std': float(trades['size'].std()),
            'size_pnl_correlation': float(trades['size'].corr(trades['pnl']))
        }
    except Exception as e:
        logger.error(f"Error analyzing position sizing: {str(e)}")
        return {}

def analyze_volatility(portfolio: Any) -> Dict[str, Any]:
    """Analyze market volatility during trades."""
    try:
        price = portfolio.close
        returns = price.pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        
        return {
            'avg_volatility': float(volatility.mean()),
            'max_volatility': float(volatility.max()),
            'min_volatility': float(volatility.min()),
            'volatility_trend': 'increasing' if volatility.iloc[-1] > volatility.iloc[0] else 'decreasing'
        }
    except Exception as e:
        logger.error(f"Error analyzing volatility: {str(e)}")
        return {}

def analyze_trend(portfolio: Any) -> Dict[str, Any]:
    """Analyze market trend during trades."""
    try:
        price = portfolio.close
        ema_short = price.ewm(span=20).mean()
        ema_long = price.ewm(span=50).mean()
        
        trend_strength = (ema_short - ema_long) / ema_long
        
        return {
            'avg_trend_strength': float(trend_strength.mean()),
            'max_trend_strength': float(trend_strength.max()),
            'min_trend_strength': float(trend_strength.min()),
            'final_trend': 'uptrend' if trend_strength.iloc[-1] > 0 else 'downtrend',
            'trend_consistency': float((trend_strength > 0).mean())
        }
    except Exception as e:
        logger.error(f"Error analyzing trend: {str(e)}")
        return {}

def analyze_volume(portfolio: Any) -> Dict[str, Any]:
    """Analyze volume patterns during trades."""
    try:
        volume = getattr(portfolio, 'volume', None)
        if volume is None:
            return {}
            
        volume_sma = volume.rolling(window=20).mean()
        
        return {
            'avg_volume': float(volume.mean()),
            'volume_trend': 'increasing' if volume.iloc[-1] > volume.iloc[0] else 'decreasing',
            'volume_consistency': float((volume > volume_sma).mean()),
            'volume_volatility': float(volume.std() / volume.mean())
        }
    except Exception as e:
        logger.error(f"Error analyzing volume: {str(e)}")
        return {}

def analyze_liquidity(portfolio: Any) -> Dict[str, Any]:
    """Analyze market liquidity conditions."""
    try:
        price = portfolio.close
        volume = getattr(portfolio, 'volume', None)
        if volume is None:
            return {}
            
        # Calculate Amihud illiquidity ratio
        returns = price.pct_change().abs()
        illiquidity = returns / volume
        
        return {
            'avg_illiquidity': float(illiquidity.mean()),
            'max_illiquidity': float(illiquidity.max()),
            'min_illiquidity': float(illiquidity.min()),
            'illiquidity_trend': 'increasing' if illiquidity.iloc[-1] > illiquidity.iloc[0] else 'decreasing'
        }
    except Exception as e:
        logger.error(f"Error analyzing liquidity: {str(e)}")
        return {}

if __name__ == "__main__":
    asyncio.run(main())
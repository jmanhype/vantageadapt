"""Strategic trading system with LLM-driven decision making and self-improvement capabilities."""

import asyncio
import logging
from logging.handlers import RotatingFileHandler
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
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from termcolor import colored
from src.strat_optim.strategy.backtester import load_trade_data, calculate_stats, run_parameter_optimization

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

from research.database import db
from research.strategy.strategy_generator import StrategicTrader
from research.analysis.trade_analyzer import TradeAnalyzer
from research.visualization.trade_visualizer import TradeVisualizer
from research.strategy.godel_agent import GodelAgent
from research.strategy.memory_manager import TradingMemoryManager
from research.strategy.teachability import Teachability

# Configure logging
def setup_logging(theme: str) -> str:
    """Set up logging to both file and console with proper formatting.
    
    Args:
        theme: Trading strategy theme for naming the log file
        
    Returns:
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"strategy_{theme.replace(' ', '_')}_{timestamp}.log"
    
    # Create formatters and handlers
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return str(log_file)

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
    
    # Set up logging
    log_file = setup_logging(theme)
    logger.info(f"Starting strategy run - logging to {log_file}")
    logger.info(f"Theme: {theme}")
    logger.info(f"Number of assets: {len(trade_data)}")
    
    # Initialize trader using async factory method
    trader = await StrategicTrader.create()
    
    # Initialize memory and teachability
    memory_manager = TradingMemoryManager()
    teachability = Teachability(memory_client=memory_manager.client, agent_id="trading_system")
    logger.info("Memory and teachability systems initialized successfully")
    
    # Initialize GodelAgent for self-improvement
    godel_agent = GodelAgent(
        improvement_threshold=0.1,  # Only accept meaningful improvements
        max_iterations=5,          # Number of improvement attempts
        backup_dir="backups",      # Directory for code backups
        prompt_dir="prompts/trading"  # Directory for prompt templates
    )
    logger.info("GodelAgent initialized for self-improvement")
    
    # Create a summary dictionary to track overall progress
    run_summary = {
        'start_time': datetime.now().isoformat(),
        'theme': theme,
        'iterations': [],
        'best_metrics': None,
        'improvements_made': 0,
        'successful_iterations': 0
    }
        
    # Add teachability to trader
    trader.add_teachability(teachability)
    logger.info("Added teachability capability to trader")
    
    for iteration in range(5):  # Run 5 iterations
        iteration_start = datetime.now()
        logger.info(f"\nStarting iteration {iteration + 1}")
        
        # Initialize iteration summary
        iteration_summary = {
            'iteration': iteration + 1,
            'start_time': iteration_start.isoformat(),
            'market_regime': None,
            'metrics': None,
            'improvements': [],
            'success': False
        }
        
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
            
            # Update iteration summary with market context
            iteration_summary['market_regime'] = market_context.regime.value
            
            # Track performance with GodelAgent and check for improvements
            improvement_needed = godel_agent.track_performance(
                metrics=metrics,
                parameters=parameters,
                market_regime=market_context.regime
            )
            
            if improvement_needed:
                logger.info("\nAttempting strategy code improvements with GodelAgent...")
                improvements = await godel_agent.optimize_strategy(
                    module_code=trader.read_module_code(),
                    metrics=metrics,
                    context={
                        'market_regime': market_context.regime,
                        'performance_history': trader.performance_history,
                        'failures': failures
                    }
                )
                
                if improvements:
                    logger.info("GodelAgent suggested code improvements - applying changes...")
                    # Apply the improvements and update the trader
                    if await trader.apply_code_improvements(improvements):
                        run_summary['improvements_made'] += 1
                        iteration_summary['improvements'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'code_improvement',
                            'success': True
                        })
            
            # Update iteration summary with metrics
            iteration_summary['metrics'] = metrics
            iteration_summary['success'] = success
            iteration_summary['end_time'] = datetime.now().isoformat()
            
            if success:
                run_summary['successful_iterations'] += 1
            
            # Add iteration summary to run summary
            run_summary['iterations'].append(iteration_summary)
            
            # Update best metrics
            if best_metrics is None or metrics.get('total_return', 0) > best_metrics.get('total_return', 0):
                best_metrics = metrics.copy()
                best_result = result.copy()
                run_summary['best_metrics'] = best_metrics
            
            # Save current run summary to file
            summary_file = Path(log_file).parent / f"summary_{Path(log_file).stem}.json"
            with open(summary_file, 'w') as f:
                json.dump(run_summary, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
            iteration_summary['error'] = str(e)
            iteration_summary['end_time'] = datetime.now().isoformat()
            run_summary['iterations'].append(iteration_summary)
            continue
    
    # Final summary logging
    run_summary['end_time'] = datetime.now().isoformat()
    logger.info("\nStrategy Run Complete")
    logger.info(f"Total Iterations: {len(run_summary['iterations'])}")
    logger.info(f"Successful Iterations: {run_summary['successful_iterations']}")
    logger.info(f"Improvements Made: {run_summary['improvements_made']}")
    if best_metrics:
        logger.info("\nBest Metrics Achieved:")
        logger.info(f"Total Return: {best_metrics.get('total_return', 0):.2%}")
        logger.info(f"Sortino Ratio: {best_metrics.get('sortino_ratio', 0):.2f}")
        logger.info(f"Win Rate: {best_metrics.get('win_rate', 0):.2%}")
    
    # Save final summary
    summary_file = Path(log_file).parent / f"summary_{Path(log_file).stem}.json"
    with open(summary_file, 'w') as f:
        json.dump(run_summary, f, indent=2)
    
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
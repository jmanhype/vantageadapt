"""Policy optimization module implementing VSM System 5 for trading strategy adaptation."""

from typing import Dict, Any, List, Tuple
import numpy as np
from loguru import logger
from dataclasses import dataclass
from datetime import datetime
from ..utils.types import BacktestResults

@dataclass
class PerformanceMetrics:
    """Metrics tracking performance gaps against targets."""
    return_gap: float
    trade_count_gap: int
    asset_count_gap: int
    timestamp: datetime = datetime.now()

class PolicyOptimizer:
    """Policy optimization implementing VSM System 5 for trading strategy adaptation."""

    def __init__(self):
        """Initialize policy optimizer with starting targets."""
        self.performance_targets = {
            'min_return': 0.10,  # Start with proof of concept
            'min_trades': 10,
            'min_assets': 1
        }
        self.target_increment_rate = 0.1  # How fast to increase targets
        self.parameter_learning_rate = 0.1
        self.success_streak = 0
        self.required_successes = 5  # Number of consecutive successes before increasing targets
        self.optimization_history = []
        
    def analyze_performance_gap(self, results: Dict[str, Any]) -> PerformanceMetrics:
        """Calculate gaps between current performance and targets.
        
        Args:
            results: Results dictionary from strategy backtesting
            
        Returns:
            PerformanceMetrics object containing gaps
        """
        # Extract metrics from results dictionary
        backtest_results = results.get('backtest_results', {})
        total_return = float(backtest_results.get('total_return', 0.0))
        total_trades = int(backtest_results.get('total_trades', 0))
        trades = backtest_results.get('trades', {})
        
        return PerformanceMetrics(
            return_gap=self.performance_targets['min_return'] - total_return,
            trade_count_gap=self.performance_targets['min_trades'] - total_trades,
            asset_count_gap=self.performance_targets['min_assets'] - len(trades)
        )
        
    def adjust_parameters(self, current_params: Dict[str, Any], metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Adjust trading parameters based on performance gaps.
        
        Args:
            current_params: Current trading parameters
            metrics: Performance gap metrics
            
        Returns:
            Adjusted parameters dictionary
        """
        adjusted_params = current_params.copy()
        
        # If not enough trades, make entry conditions more lenient
        if metrics.trade_count_gap > 0:
            logger.info("Adjusting parameters to increase trade frequency")
            adjusted_params['post_buy_delay'] = max(1, current_params['post_buy_delay'] - 1)
            adjusted_params['post_sell_delay'] = max(1, current_params['post_sell_delay'] - 1)
            adjusted_params['macd_signal_fast'] = max(50, int(current_params['macd_signal_fast'] * 0.9))
            
        # If return is too low, adjust risk-reward
        if metrics.return_gap > 0:
            logger.info("Adjusting parameters to improve returns")
            adjusted_params['take_profit'] = min(0.15, current_params['take_profit'] * 1.1)
            adjusted_params['stop_loss'] = max(0.02, current_params['stop_loss'] * 0.9)
            adjusted_params['order_size'] = min(0.01, current_params['order_size'] * 1.1)
            
        # Log parameter adjustments
        for key, value in adjusted_params.items():
            if current_params[key] != value:
                logger.info(f"Adjusted {key}: {current_params[key]} -> {value}")
                
        return adjusted_params
        
    def optimize_trading_rules(self, rules: Dict[str, List[str]], metrics: PerformanceMetrics) -> Dict[str, List[str]]:
        """Adjust trading rules based on performance gaps.
        
        Args:
            rules: Current trading rules
            metrics: Performance gap metrics
            
        Returns:
            Adjusted trading rules dictionary
        """
        adjusted_rules = rules.copy()
        
        if metrics.trade_count_gap > 0:
            logger.info("Adjusting trading rules to increase trade frequency")
            # Make entry conditions more lenient
            entry_conditions = rules['entry']
            adjusted_entry = []
            for condition in entry_conditions:
                if 'rsi' in condition:
                    # Make RSI conditions more lenient
                    condition = condition.replace('rsi < 70', 'rsi < 75')
                    condition = condition.replace('rsi > 30', 'rsi > 25')
                elif 'macd' in condition:
                    # Make MACD conditions more lenient
                    if '>' in condition:
                        condition = condition.replace('> 0', '> -0.1')
                    elif '<' in condition:
                        condition = condition.replace('< 0', '< 0.1')
                adjusted_entry.append(condition)
            adjusted_rules['entry'] = adjusted_entry
            
            # Log rule adjustments
            logger.info("Adjusted entry conditions:")
            for old, new in zip(rules['entry'], adjusted_entry):
                if old != new:
                    logger.info(f"  {old} -> {new}")
            
        return adjusted_rules
        
    def update_targets(self, consecutive_successes: int):
        """Update performance targets if consistently meeting current targets.
        
        Args:
            consecutive_successes: Number of consecutive successful iterations
        """
        if consecutive_successes >= self.required_successes:
            old_targets = self.performance_targets.copy()
            
            # Increase targets
            self.performance_targets['min_return'] *= (1 + self.target_increment_rate)
            self.performance_targets['min_trades'] = int(self.performance_targets['min_trades'] * 1.5)
            self.performance_targets['min_assets'] = min(100, int(self.performance_targets['min_assets'] * 1.5))
            
            # Reset success streak
            self.success_streak = 0
            
            # Log target updates
            logger.info("Updated performance targets:")
            for key in self.performance_targets:
                logger.info(f"  {key}: {old_targets[key]} -> {self.performance_targets[key]}")
            
    def feedback_loop(self, results: Dict[str, Any], current_params: Dict[str, Any], 
                     current_rules: Dict[str, List[str]]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Main optimization loop implementing continuous improvement.
        
        Args:
            results: Results from strategy backtesting
            current_params: Current trading parameters
            current_rules: Current trading rules
            
        Returns:
            Tuple of (adjusted parameters, adjusted rules)
        """
        # Analyze current performance
        metrics = self.analyze_performance_gap(results)
        
        # Check if meeting current targets
        targets_met = (
            metrics.return_gap <= 0 and 
            metrics.trade_count_gap <= 0 and 
            metrics.asset_count_gap <= 0
        )
        
        if targets_met:
            self.success_streak += 1
            logger.info(f"Meeting all targets. Success streak: {self.success_streak}")
            self.update_targets(self.success_streak)
        else:
            self.success_streak = 0
            logger.info("Not meeting all targets. Resetting success streak.")
            
        # Adjust parameters and rules
        adjusted_params = self.adjust_parameters(current_params, metrics)
        adjusted_rules = self.optimize_trading_rules(current_rules, metrics)
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'targets_met': targets_met,
            'success_streak': self.success_streak,
            'performance_targets': self.performance_targets.copy(),
            'parameter_adjustments': {
                k: (current_params[k], adjusted_params[k])
                for k in current_params
                if current_params[k] != adjusted_params[k]
            }
        })
        
        return adjusted_params, adjusted_rules

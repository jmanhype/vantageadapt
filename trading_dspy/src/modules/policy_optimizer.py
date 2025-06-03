"""Policy optimization module implementing VSM System 5 for trading strategy adaptation."""

from typing import Dict, Any, List, Tuple
import numpy as np
from loguru import logger
from dataclasses import dataclass
from datetime import datetime
from ..utils.types import BacktestResults
import re
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """Metrics tracking performance gaps against targets."""
    return_gap: float
    trade_count_gap: int
    asset_count_gap: int
    win_rate_gap: float = 0.0
    timestamp: datetime = datetime.now()

class PolicyOptimizer:
    """Policy optimization implementing VSM System 5 for trading strategy adaptation."""

    def __init__(self):
        """Initialize policy optimizer with starting targets."""
        self.performance_targets = {
            'min_return': 0.10,  # Start with proof of concept
            'min_trades': 10,
            'min_assets': 1,
            'min_win_rate': 0.6  # Target win rate of 60%
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
        
        # Calculate win rate if trades exist
        win_rate = 0.0
        win_rate_gap = 0.0
        if total_trades > 0:
            win_rate = backtest_results.get('win_rate', 0.0)
            win_rate_gap = self.performance_targets.get('min_win_rate', 0.6) - win_rate
        
        return PerformanceMetrics(
            return_gap=self.performance_targets['min_return'] - total_return,
            trade_count_gap=self.performance_targets['min_trades'] - total_trades,
            asset_count_gap=self.performance_targets['min_assets'] - len(trades),
            win_rate_gap=win_rate_gap
        )
        
    def adjust_parameters(self, current_params: Dict[str, Any], metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Adjust trading parameters based on performance gaps.
        
        Uses adaptive step sizes and performance history to make intelligent adjustments.
        
        Args:
            current_params: Current trading parameters
            metrics: Performance gap metrics
            
        Returns:
            Dictionary of adjusted parameters
        """
        adjusted_params = current_params.copy()
        
        # Calculate adjustment factors based on performance history
        if not hasattr(self, 'performance_history'):
            self.performance_history = []
        
        # Store current performance
        self.performance_history.append({
            'metrics': metrics,
            'params': current_params,
            'timestamp': datetime.now()
        })
        
        # Keep last 10 iterations
        self.performance_history = self.performance_history[-10:]
        
        # Calculate trend directions
        def get_metric_trend(metric_name: str) -> float:
            """Calculate trend direction (-1 to 1) for a metric."""
            if len(self.performance_history) < 2:
                return 0
            
            values = [h['metrics'].__dict__[metric_name] for h in self.performance_history]
            # Use linear regression to get trend
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            # Normalize to -1 to 1
            return np.clip(slope * len(values), -1, 1)
        
        return_trend = get_metric_trend('return_gap')
        trade_trend = get_metric_trend('trade_count_gap')
        win_rate_trend = get_metric_trend('win_rate_gap')
        
        # Calculate adaptive step sizes
        def get_adaptive_step(base_step: float, gap: float, trend: float) -> float:
            """Calculate adaptive step size based on gap and trend."""
            # Increase step size if trend is flat or gap is growing
            if abs(trend) < 0.2 or trend > 0:
                return base_step * (1 + abs(gap))
            # Decrease step size if we're making progress
            return base_step * (1 - abs(trend))
        
        # Track parameter effectiveness
        if not hasattr(self, 'param_effectiveness'):
            self.param_effectiveness = defaultdict(lambda: {'success': 0, 'total': 0})
        
        def update_param_effectiveness(param: str, old_val: float, new_val: float):
            """Update effectiveness tracking for parameter adjustments."""
            if len(self.performance_history) < 2:
                return
            
            # Check if the last adjustment improved performance
            prev_gaps = self.performance_history[-2]['metrics']
            curr_gaps = self.performance_history[-1]['metrics']
            
            improved = (
                curr_gaps.return_gap < prev_gaps.return_gap or
                curr_gaps.trade_count_gap < prev_gaps.trade_count_gap or
                curr_gaps.win_rate_gap < prev_gaps.win_rate_gap
            )
            
            self.param_effectiveness[param]['total'] += 1
            if improved:
                self.param_effectiveness[param]['success'] += 1
        
        def get_param_effectiveness(param: str) -> float:
            """Get effectiveness score for a parameter."""
            stats = self.param_effectiveness[param]
            if stats['total'] == 0:
                return 0.5  # Default for new parameters
            return stats['success'] / stats['total']
        
        # If not enough trades, make entry conditions more lenient
        if metrics.trade_count_gap > 0:
            logger.info("Adjusting parameters to increase trade frequency")
            
            trade_step = get_adaptive_step(0.1, metrics.trade_count_gap/100, trade_trend)
            
            # Prioritize most effective parameters
            params_to_adjust = [
                ('post_buy_delay', -1),
                ('post_sell_delay', -1),
                ('macd_signal_fast', -1),
                ('lookback_window', -1),
                ('max_trades_per_day', 1)
            ]
            
            # Sort by effectiveness
            params_to_adjust.sort(key=lambda x: get_param_effectiveness(x[0]), reverse=True)
            
            for param, direction in params_to_adjust[:3]:  # Only adjust top 3 most effective
                if param in current_params:
                    old_val = current_params[param]
                    if direction > 0:
                        new_val = old_val * (1 + trade_step)
                    else:
                        new_val = old_val * (1 - trade_step)
                    
                    # Apply limits
                    if param in ['post_buy_delay', 'post_sell_delay']:
                        new_val = max(1, int(new_val))
                    elif param == 'macd_signal_fast':
                        new_val = max(40, int(new_val))
                    elif param == 'lookback_window':
                        new_val = max(10, int(new_val))
                    elif param == 'max_trades_per_day':
                        new_val = min(100, int(new_val))
                    
                    adjusted_params[param] = new_val
                    update_param_effectiveness(param, old_val, new_val)
            
        # If return is too low, adjust risk-reward
        if metrics.return_gap > 0:
            logger.info("Adjusting parameters to improve returns")
            
            return_step = get_adaptive_step(0.1, metrics.return_gap, return_trend)
            
            params_to_adjust = [
                ('take_profit', 1),
                ('stop_loss', -1),
                ('order_size', 1),
                ('trailing_stop_pct', 1)
            ]
            
            params_to_adjust.sort(key=lambda x: get_param_effectiveness(x[0]), reverse=True)
            
            for param, direction in params_to_adjust[:2]:  # Only adjust top 2 most effective
                if param in current_params:
                    old_val = current_params[param]
                    if direction > 0:
                        new_val = old_val * (1 + return_step)
                    else:
                        new_val = old_val * (1 - return_step)
                    
                    # Apply limits
                    if param == 'take_profit':
                        new_val = min(0.25, max(0.04, new_val))
                    elif param == 'stop_loss':
                        new_val = min(0.15, max(0.02, new_val))
                    elif param == 'order_size':
                        new_val = min(0.02, max(0.001, new_val))
                    elif param == 'trailing_stop_pct':
                        new_val = min(0.03, max(0.005, new_val))
                    
                    adjusted_params[param] = new_val
                    update_param_effectiveness(param, old_val, new_val)
            
            # Enable trailing stop if consistently not meeting return targets
            if return_trend > 0.5:
                adjusted_params['trailing_stop'] = True
        
        # If win rate is too low, improve risk management
        if metrics.win_rate_gap > 0:
            logger.info("Adjusting parameters to improve win rate")
            
            win_step = get_adaptive_step(0.1, metrics.win_rate_gap, win_rate_trend)
            
            params_to_adjust = [
                ('stop_loss', -1),
                ('vol_filter_threshold', -1)
            ]
            
            params_to_adjust.sort(key=lambda x: get_param_effectiveness(x[0]), reverse=True)
            
            for param, direction in params_to_adjust:
                if param in current_params:
                    old_val = current_params[param]
                    if direction > 0:
                        new_val = old_val * (1 + win_step)
                    else:
                        new_val = old_val * (1 - win_step)
                    
                    # Apply limits
                    if param == 'stop_loss':
                        new_val = min(0.15, max(0.015, new_val))
                    elif param == 'vol_filter_threshold':
                        new_val = min(2.0, max(1.0, new_val))
                    
                    adjusted_params[param] = new_val
                    update_param_effectiveness(param, old_val, new_val)
            
            # Enable dynamic sizing if win rate is consistently low
            if win_rate_trend > 0.5:
                adjusted_params['dynamic_sizing'] = True
                adjusted_params['use_volatility_filter'] = True
            
        # Log parameter adjustments
        for key, value in adjusted_params.items():
            if key in current_params and current_params[key] != value:
                logger.info(f"Adjusted {key}: {current_params[key]} -> {value}")
                
        return adjusted_params
        
    def optimize_trading_rules(self, entry_conditions: List[str], exit_conditions: List[str], metrics: PerformanceMetrics) -> Tuple[List[str], List[str]]:
        """Optimize trading rules based on performance metrics.
        
        Makes entry conditions less restrictive if not enough trades,
        and adjusts risk/reward parameters based on return gap.
        
        Args:
            entry_conditions: List of entry condition strings
            exit_conditions: List of exit condition strings
            metrics: Performance metrics to optimize for
            
        Returns:
            Tuple of (optimized entry conditions, optimized exit conditions)
        """
        optimized_entry = entry_conditions.copy()
        optimized_exit = exit_conditions.copy()
        
        # If not enough trades, make entry conditions more lenient
        if metrics.trade_count_gap > 0:
            logger.info("Adjusting trading rules to increase trade frequency")
            
            # Helper to extract numeric thresholds from condition string
            def extract_threshold(condition: str) -> Tuple[str, str, float, bool]:
                """Extract indicator, operator and threshold from condition.
                
                Returns (indicator, operator, value, is_percent)
                """
                # Match patterns like "rsi < 30" or "price < sma_20 * 0.98"
                match_direct = re.search(r'(\w+(?:\.\w+)?)\s*([<>]=?|==|!=)\s*(\d+\.?\d*)', condition)
                match_percent = re.search(r'(\w+(?:\.\w+)?)\s*([<>]=?|==|!=)\s*\w+\s*\*\s*(\d+\.?\d*)', condition)
                
                if match_direct:
                    indicator = match_direct.group(1).lower()
                    operator = match_direct.group(2)
                    value = float(match_direct.group(3))
                    is_percent = False
                    return indicator, operator, value, is_percent
                elif match_percent:
                    indicator = match_percent.group(1).lower()
                    operator = match_percent.group(2)
                    value = float(match_percent.group(3))
                    is_percent = True
                    return indicator, operator, value, is_percent
                
                # Handle compound conditions
                if "and" in condition or "or" in condition:
                    parts = re.split(r'\s+(?:and|or)\s+', condition)
                    for part in parts:
                        match_direct = re.search(r'(\w+(?:\.\w+)?)\s*([<>]=?|==|!=)\s*(\d+\.?\d*)', part)
                        if match_direct:
                            indicator = match_direct.group(1).lower()
                            operator = match_direct.group(2)
                            value = float(match_direct.group(3))
                            is_percent = False
                            return indicator, operator, value, is_percent
                        
                        match_percent = re.search(r'(\w+(?:\.\w+)?)\s*([<>]=?|==|!=)\s*\w+\s*\*\s*(\d+\.?\d*)', part)
                        if match_percent:
                            indicator = match_percent.group(1).lower()
                            operator = match_percent.group(2)
                            value = float(match_percent.group(3))
                            is_percent = True
                            return indicator, operator, value, is_percent
                
                return None, None, 0, False
            
            # Helper to modify condition based on extracted parts
            def modify_condition(condition: str, indicator: str, operator: str, value: float, is_percent: bool, new_value: float) -> str:
                """Modify a condition with new threshold value."""
                if not indicator or not operator:
                    return condition
                
                if is_percent:
                    pattern = rf'({re.escape(indicator)})\s*({re.escape(operator)})\s*(\w+)\s*\*\s*{value}'
                    if '*' in condition:  # Only replace if it contains multiplication
                        replacement = f'\\1 \\2 \\3 * {new_value}'
                        return re.sub(pattern, replacement, condition)
                else:
                    pattern = rf'({re.escape(indicator)})\s*({re.escape(operator)})\s*{value}'
                    replacement = f'\\1 \\2 {new_value}'
                    return re.sub(pattern, replacement, condition)
                
                return condition
            
            # Choose a more aggressive adjustment factor based on trade count gap
            gap_magnitude = min(1.0, max(0.1, metrics.trade_count_gap / 50))
            
            # For extreme trade count gaps, use our emergency method to create very lenient conditions
            if metrics.trade_count_gap > 40:
                logger.info(f"CRITICAL TRADE COUNT GAP: {metrics.trade_count_gap}. Applying emergency entry conditions")
                return self._create_lenient_entry_conditions(optimized_entry, metrics), optimized_exit
            
            # Modify conditions to be more lenient
            for i, condition in enumerate(optimized_entry):
                indicator, operator, value, is_percent = extract_threshold(condition)
                
                if not indicator:
                    continue
                
                # Make thresholds more lenient based on operator
                if indicator.lower() == 'rsi':
                    if operator == '<':
                        # Increase the RSI threshold to allow more trades
                        new_threshold = min(90, int(value + 5 * gap_magnitude))
                        optimized_entry[i] = modify_condition(condition, indicator, operator, value, is_percent, new_threshold)
                        logger.info(f"  {indicator} {operator} {value} -> {indicator} {operator} {new_threshold}")
                    elif operator == '>':
                        # Decrease the RSI threshold to allow more trades
                        new_threshold = max(10, int(value - 5 * gap_magnitude))
                        optimized_entry[i] = modify_condition(condition, indicator, operator, value, is_percent, new_threshold)
                        logger.info(f"  {indicator} {operator} {value} -> {indicator} {operator} {new_threshold}")
                elif 'sma' in indicator.lower() or 'price' in indicator.lower():
                    if (operator == '<' and not is_percent) or (operator == '>' and is_percent):
                        # For price < SMA comparisons, increase the threshold
                        new_value = value * (1 + 0.02 * gap_magnitude)
                        optimized_entry[i] = modify_condition(condition, indicator, operator, value, is_percent, round(new_value, 4))
                        logger.info(f"  {condition} -> {optimized_entry[i]}")
                    elif (operator == '>' and not is_percent) or (operator == '<' and is_percent):
                        # For price > SMA comparisons, decrease the threshold
                        new_value = value * (1 - 0.02 * gap_magnitude)
                        optimized_entry[i] = modify_condition(condition, indicator, operator, value, is_percent, round(new_value, 4))
                        logger.info(f"  {condition} -> {optimized_entry[i]}")
                elif 'macd' in indicator.lower():
                    if operator == '>':
                        # Decrease the MACD threshold to allow more trades
                        new_threshold = max(-0.1, value - 0.005 * gap_magnitude)
                        optimized_entry[i] = modify_condition(condition, indicator, operator, value, is_percent, round(new_threshold, 5))
                        logger.info(f"  {condition} -> {optimized_entry[i]}")
                    elif operator == '<':
                        # Increase the MACD threshold to allow more trades
                        new_threshold = min(0.1, value + 0.005 * gap_magnitude)
                        optimized_entry[i] = modify_condition(condition, indicator, operator, value, is_percent, round(new_threshold, 5))
                        logger.info(f"  {condition} -> {optimized_entry[i]}")
                
                # If the trade count gap is very large, consider adding more lenient conditions
                if metrics.trade_count_gap > 20 and len(optimized_entry) < 5:
                    if 'rsi' not in ' '.join(optimized_entry).lower():
                        # Add RSI condition if none exists
                        optimized_entry.append("rsi < 75")
                        logger.info(f"  Added new condition: rsi < 75")
                    elif 'price' not in ' '.join(optimized_entry).lower() and 'sma' not in ' '.join(optimized_entry).lower():
                        # Add price relative to SMA condition if none exists
                        optimized_entry.append("price > sma_50 * 0.97")
                        logger.info(f"  Added new condition: price > sma_50 * 0.97")
                    elif 'macd' not in ' '.join(optimized_entry).lower():
                        # Add MACD condition if none exists
                        optimized_entry.append("macd.macd > -0.0001")
                        logger.info(f"  Added new condition: macd.macd > -0.0001")
            
            # Add at least one more lenient condition if trade gap is severe
            if metrics.trade_count_gap > 30 and len(optimized_entry) < 3:
                optimized_entry.append("price > 0")  # Always true condition to increase trade frequency
                logger.info(f"  Added always-true condition: price > 0")
                
        # Adjust exit conditions if return is too low
        if metrics.return_gap > 0:
            # TODO: Improve exit condition optimization
            pass
            
        return optimized_entry, optimized_exit
        
    def _create_lenient_entry_conditions(self, current_conditions: List[str], metrics: PerformanceMetrics) -> List[str]:
        """Create extremely lenient entry conditions when trade count is critically low.
        
        Args:
            current_conditions: Current entry conditions
            metrics: Performance metrics
            
        Returns:
            List of lenient entry conditions that will generate more trades
        """
        logger.warning("Creating extremely lenient entry conditions due to severe trade count gap")
        
        # Start with a clean slate while preserving any existing conditions
        lenient_conditions = []
        
        # Determine which indicators are currently used
        indicators_used = {ind.split('.')[0].lower().split(' ')[0] for cond in current_conditions for ind in cond.split(' ') if '>' in cond or '<' in cond}
        
        # Keep any conditions that use indicators not in our standard set
        for condition in current_conditions:
            indicator = condition.split(' ')[0].lower()
            if indicator not in ['rsi', 'macd', 'price', 'sma', 'ema', 'volume']:
                lenient_conditions.append(condition)
                logger.info(f"  Keeping specialized condition: {condition}")
        
        # Add always true condition to ensure trades happen
        lenient_conditions.append("price > 0")
        logger.info(f"  Added base condition: price > 0")
        
        # Add extremely lenient RSI condition
        if 'rsi' in indicators_used:
            lenient_conditions.append("rsi < 85")
            logger.info(f"  Added lenient condition: rsi < 85")
        
        # Add lenient MACD condition
        if 'macd' in indicators_used:
            lenient_conditions.append("macd.macd > -0.01")
            logger.info(f"  Added lenient condition: macd.macd > -0.01")
            
        # Add lenient SMA condition
        if 'sma' in indicators_used or 'price' in indicators_used:
            lenient_conditions.append("price > sma_50 * 0.95")
            logger.info(f"  Added lenient condition: price > sma_50 * 0.95")
        
        if metrics.trade_count_gap > 50:
            # For extremely severe gaps, add trade-forcing conditions
            logger.warning("Adding trade-forcing conditions due to extreme trade count gap")
            lenient_conditions = ["price > 0"]  # Reset to just the always-true condition
            logger.info(f"  Added guaranteed trade condition: price > 0")
        
        logger.info(f"Final lenient conditions: {lenient_conditions}")
        return lenient_conditions
        
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
            self.performance_targets['min_win_rate'] = min(0.9, self.performance_targets['min_win_rate'] + 0.05)
            
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
        adjusted_rules = self.optimize_trading_rules(current_rules['entry'], current_rules['exit'], metrics)
        
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
        
        return adjusted_params, {
            'entry': adjusted_rules[0],
            'exit': adjusted_rules[1]
        }

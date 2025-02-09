"""Policy optimization module implementing VSM System 5 for trading strategy adaptation."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime
from ..utils.types import BacktestResults

@dataclass
class Position:
    """Represents a trading position with enhanced tracking."""
    entry_price: float
    size: float
    stop_price: float
    highest_price: float = 0.0
    profits_taken: List[bool] = field(default_factory=lambda: [False, False, False])
    active: bool = True

@dataclass
class EnhancedPerformanceMetrics:
    """Enhanced metrics tracking performance gaps and risk metrics."""
    return_gap: float
    trade_count_gap: int
    asset_count_gap: int
    sortino_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    trade_frequency: float
    timestamp: datetime = field(default_factory=datetime.now)

    def calculate_score(self) -> float:
        """Calculate comprehensive performance score."""
        return (
            0.25 * (1.0 - max(0, self.return_gap)) +
            0.25 * self.sortino_ratio / 2.0 +
            0.20 * (1.0 + self.sharpe_ratio) / 2.0 +
            0.15 * (1.0 - abs(self.max_drawdown)) +
            0.15 * self.win_rate
        )

class RiskManager:
    """Manages risk limits and position sizing."""
    
    def __init__(self):
        self.max_drawdown = 0.15
        self.daily_loss_limit = 0.05
        self.position_size_base = 0.01
        self.vol_lookback = 20
        
    def calculate_position_size(self, volatility: float, account_value: float) -> float:
        """Calculate position size based on volatility."""
        vol_scale = 1.0 / (1.0 + volatility)
        return min(
            self.position_size_base * vol_scale * account_value,
            account_value * 0.05
        )
    
    def check_risk_limits(self, metrics: EnhancedPerformanceMetrics) -> bool:
        """Check if current metrics are within risk limits."""
        return (
            abs(metrics.max_drawdown) <= self.max_drawdown and
            metrics.return_gap >= -self.daily_loss_limit
        )

class TradeManager:
    """Manages trade entries, exits, and position sizing."""
    
    def __init__(self):
        self.trailing_stop_pct = 0.02
        self.profit_taking_levels = [0.5, 0.75, 1.0]
        self.confirmation_required = 2
        
    def should_enter(self, signals: Dict[str, bool], market_conditions: Dict[str, Any]) -> bool:
        """Determine if entry conditions are met."""
        confirmations = sum(1 for signal in signals.values() if signal)
        return (
            confirmations >= self.confirmation_required and
            market_conditions['volatility'] < market_conditions['volatility_threshold'] and
            market_conditions['volume'] > market_conditions['volume_threshold']
        )
    
    def manage_position(self, position: Position, current_price: float) -> Tuple[bool, float]:
        """Manage existing position including trailing stops and profit taking."""
        if not position.active:
            return False, 0.0
            
        trail_price = position.entry_price * (1 - self.trailing_stop_pct)
        if current_price > position.highest_price:
            position.highest_price = current_price
            trail_price = position.highest_price * (1 - self.trailing_stop_pct)
        
        unrealized_pnl = (current_price - position.entry_price) / position.entry_price
        for level, target in enumerate(self.profit_taking_levels):
            if unrealized_pnl >= target and not position.profits_taken[level]:
                position.size *= 0.75
                position.profits_taken[level] = True
                
        return True, trail_price

class RegimeAwareOptimizer:
    """Optimizes parameters based on market regime."""
    
    def __init__(self):
        self.regime_params = {
            'TRENDING': {
                'take_profit': (0.03, 0.22),
                'stop_loss': (0.015, 0.17),
                'momentum': 0.85,
                'vol_threshold': 0.02
            },
            'RANGING': {
                'take_profit': (0.02, 0.15),
                'stop_loss': (0.01, 0.12),
                'momentum': 0.75,
                'vol_threshold': 0.015
            }
        }
    
    def get_regime_params(self, regime: str, confidence: float) -> Dict[str, Any]:
        """Get optimized parameters for current market regime."""
        base_params = self.regime_params[regime]
        scale = 0.5 + (confidence * 0.5)
        
        return {
            'take_profit': self.scale_range(base_params['take_profit'], scale),
            'stop_loss': self.scale_range(base_params['stop_loss'], scale),
            'momentum': base_params['momentum'] * scale,
            'vol_threshold': base_params['vol_threshold'] * (2 - scale)
        }
    
    @staticmethod
    def scale_range(param_range: Tuple[float, float], scale: float) -> Tuple[float, float]:
        """Scale parameter range based on confidence."""
        mid = (param_range[0] + param_range[1]) / 2
        spread = (param_range[1] - param_range[0]) * scale
        return (
            max(param_range[0], mid - spread/2),
            min(param_range[1], mid + spread/2)
        )

class PolicyOptimizer:
    """Enhanced policy optimization implementing VSM System 5 for trading strategy adaptation."""

    def __init__(self):
        """Initialize policy optimizer with enhanced components."""
        self.risk_manager = RiskManager()
        self.trade_manager = TradeManager()
        self.regime_optimizer = RegimeAwareOptimizer()
        
        # Performance targets
        self.performance_targets = {
            'min_return': 0.10,
            'min_trades': 10,
            'min_assets': 1,
            'min_sortino': 2.0,
            'min_sharpe': 1.5,
            'max_drawdown': -0.15,
            'min_win_rate': 0.55
        }
        
        # Learning parameters
        self.target_increment_rate = 0.05
        self.parameter_learning_rate = 0.05
        self.momentum = 0.8
        self.required_successes = 4
        
        # Parameter bounds
        self.param_bounds = {
            'take_profit': (0.02, 0.22),
            'stop_loss': (0.01, 0.17),
            'order_size': (0.001, 0.012),
            'macd_signal_fast': (45, 550)
        }
        
        # State tracking
        self.best_params = None
        self.best_performance = None
        self.success_streak = 0
        self.optimization_history = []

    def analyze_performance_gap(self, results: Dict[str, Any]) -> EnhancedPerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        backtest_results = results.get('backtest_results', {})
        total_return = float(backtest_results.get('total_return', 0.0))
        total_trades = int(backtest_results.get('total_trades', 0))
        trades = backtest_results.get('trades', {})
        
        return EnhancedPerformanceMetrics(
            return_gap=self.performance_targets['min_return'] - total_return,
            trade_count_gap=self.performance_targets['min_trades'] - total_trades,
            asset_count_gap=self.performance_targets['min_assets'] - len(trades),
            sortino_ratio=float(backtest_results.get('sortino_ratio', 0.0)),
            sharpe_ratio=float(backtest_results.get('sharpe_ratio', 0.0)),
            max_drawdown=float(backtest_results.get('max_drawdown', 0.0)),
            win_rate=float(backtest_results.get('win_rate', 0.0)),
            avg_trade_duration=float(backtest_results.get('avg_trade_duration', 0.0)),
            trade_frequency=float(backtest_results.get('trade_frequency', 0.0))
        )

    def adjust_parameters(self, current_params: Dict[str, Any], metrics: EnhancedPerformanceMetrics,
                         market_regime: str, regime_confidence: float) -> Dict[str, Any]:
        """Adjust parameters with regime awareness and risk management."""
        adjusted_params = current_params.copy()
        current_score = metrics.calculate_score()
        
        # Get regime-specific parameters
        regime_params = self.regime_optimizer.get_regime_params(market_regime, regime_confidence)
        
        # Calculate adjustment scale
        adjustment_scale = 1.0
        if self.best_performance and current_score >= self.best_performance * 0.9:
            adjustment_scale = 0.7
            logger.info("Good performance detected, making smaller adjustments")
        
        # Apply regime-specific bounds
        for param, value in adjusted_params.items():
            if param in regime_params:
                min_val, max_val = regime_params[param]
                adjusted_params[param] = np.clip(value, min_val, max_val)
        
        # Adjust parameters based on performance gaps
        if metrics.trade_count_gap > 0:
            self._adjust_frequency_params(adjusted_params, adjustment_scale)
        
        if metrics.return_gap > 0:
            self._adjust_return_params(adjusted_params, adjustment_scale)
        
        # Apply momentum and reversion
        if self.best_params:
            for param in adjusted_params:
                if param in self.best_params:
                    adjusted_params[param] = (
                        0.6 * self.best_params[param] +
                        0.4 * adjusted_params[param]
                    )
        
        # Update best parameters if current performance is best
        if self.best_performance is None or current_score > self.best_performance:
            self.best_params = current_params.copy()
            self.best_performance = current_score
            logger.info("New best parameters found!")
        
        return adjusted_params

    def _adjust_frequency_params(self, params: Dict[str, Any], scale: float):
        """Adjust parameters affecting trade frequency."""
        logger.info("Adjusting parameters to increase trade frequency")
        for param, value in params.items():
            if param in ['post_buy_delay', 'post_sell_delay']:
                current_change = max(1, value - 1)
                params[param] = max(1, int(
                    value + self.momentum * current_change * scale
                ))
            elif param == 'macd_signal_fast':
                min_val, max_val = self.param_bounds[param]
                new_val = max(min_val, int(value * (1 - self.parameter_learning_rate * scale)))
                params[param] = min(max_val, new_val)

    def _adjust_return_params(self, params: Dict[str, Any], scale: float):
        """Adjust parameters affecting returns."""
        logger.info("Adjusting parameters to improve returns")
        for param in ['take_profit', 'stop_loss', 'order_size']:
            if param in params:
                min_val, max_val = self.param_bounds[param]
                multiplier = 1.1 if param == 'take_profit' else 0.9
                change = (multiplier - 1.0) * self.parameter_learning_rate * scale
                new_val = params[param] * (1 + change)
                params[param] = min(max_val, max(min_val, new_val))

    def optimize_trading_rules(self, rules: Dict[str, List[str]], metrics: EnhancedPerformanceMetrics,
                             market_regime: str) -> Dict[str, List[str]]:
        """Optimize trading rules with regime awareness."""
        adjusted_rules = rules.copy()
        
        if metrics.trade_count_gap > 0:
            logger.info("Adjusting trading rules to increase trade frequency")
            entry_conditions = rules['entry']
            adjusted_entry = []
            
            for condition in entry_conditions:
                if 'rsi' in condition:
                    # Adjust RSI thresholds based on regime
                    if market_regime == 'RANGING':
                        condition = condition.replace('rsi < 70', 'rsi < 75')
                        condition = condition.replace('rsi > 30', 'rsi > 25')
                    else:  # TRENDING
                        condition = condition.replace('rsi < 70', 'rsi < 80')
                        condition = condition.replace('rsi > 30', 'rsi > 20')
                elif 'macd' in condition:
                    # Adjust MACD thresholds based on regime
                    threshold = '-0.1' if market_regime == 'RANGING' else '-0.05'
                    if '>' in condition:
                        condition = condition.replace('> 0', f'> {threshold}')
                    elif '<' in condition:
                        condition = condition.replace('< 0', f'< {threshold}')
                adjusted_entry.append(condition)
                
            adjusted_rules['entry'] = adjusted_entry
            
            # Log adjustments
            logger.info("Adjusted entry conditions:")
            for old, new in zip(rules['entry'], adjusted_entry):
                if old != new:
                    logger.info(f"  {old} -> {new}")
        
        return adjusted_rules

    def feedback_loop(self, results: Dict[str, Any], current_params: Dict[str, Any],
                     current_rules: Dict[str, List[str]]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """Enhanced feedback loop with regime awareness and risk management."""
        # Calculate comprehensive metrics
        metrics = self.analyze_performance_gap(results)
        current_score = metrics.calculate_score()
        
        # Check risk limits
        if not self.risk_manager.check_risk_limits(metrics):
            logger.warning("Risk limits exceeded, reverting to conservative parameters")
            return self._get_conservative_params(), current_rules
        
        # Get market regime information
        market_regime = results.get('market_regime', 'RANGING')
        regime_confidence = results.get('regime_confidence', 0.5)
        
        # Adjust parameters with regime awareness
        adjusted_params = self.adjust_parameters(
            current_params,
            metrics,
            market_regime,
            regime_confidence
        )
        
        # Optimize trading rules
        adjusted_rules = self.optimize_trading_rules(
            current_rules,
            metrics,
            market_regime
        )
        
        # Update optimization history
        self._update_optimization_history(
            metrics,
            current_score,
            current_params,
            adjusted_params,
            market_regime
        )
        
        return adjusted_params, adjusted_rules

    def _get_conservative_params(self) -> Dict[str, Any]:
        """Get conservative parameters for risk management."""
        return {
            'take_profit': self.param_bounds['take_profit'][0],
            'stop_loss': self.param_bounds['stop_loss'][1],
            'order_size': self.param_bounds['order_size'][0],
            'macd_signal_fast': self.param_bounds['macd_signal_fast'][0]
        }

    def _update_optimization_history(self, metrics: EnhancedPerformanceMetrics,
                                   current_score: float,
                                   current_params: Dict[str, Any],
                                   adjusted_params: Dict[str, Any],
                                   market_regime: str):
        """Update optimization history with enhanced tracking."""
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'strategy_score': current_score,
            'market_regime': market_regime,
            'parameter_adjustments': {
                k: (current_params[k], adjusted_params[k])
                for k in current_params
                if current_params[k] != adjusted_params[k]
            },
            'performance_targets': self.performance_targets.copy()
        })

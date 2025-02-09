"""Tests for the policy optimization module."""

import pytest
from datetime import datetime
from src.modules.policy_optimizer import PolicyOptimizer, PerformanceMetrics
from src.utils.types import BacktestResults

def test_policy_optimizer_initialization():
    """Test PolicyOptimizer initialization with default values."""
    optimizer = PolicyOptimizer()
    
    assert optimizer.performance_targets['min_return'] == 0.10
    assert optimizer.performance_targets['min_trades'] == 10
    assert optimizer.performance_targets['min_assets'] == 1
    assert optimizer.target_increment_rate == 0.1
    assert optimizer.success_streak == 0
    assert optimizer.required_successes == 5

def test_analyze_performance_gap():
    """Test performance gap analysis."""
    optimizer = PolicyOptimizer()
    
    # Create test backtest results
    results = BacktestResults(
        total_return=0.05,  # Below target of 0.10
        total_pnl=500.0,
        sortino_ratio=1.5,
        win_rate=0.6,
        total_trades=5,  # Below target of 10
        trades={'asset1': {}},  # One asset
        metrics={},
        timestamp=datetime.now()
    )
    
    metrics = optimizer.analyze_performance_gap(results)
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.return_gap == pytest.approx(0.05)  # 0.10 - 0.05
    assert metrics.trade_count_gap == 5  # 10 - 5
    assert metrics.asset_count_gap == 0  # 1 - 1

def test_adjust_parameters():
    """Test parameter adjustment based on performance gaps."""
    optimizer = PolicyOptimizer()
    
    current_params = {
        'take_profit': 0.05,
        'stop_loss': 0.03,
        'post_buy_delay': 2,
        'post_sell_delay': 2,
        'macd_signal_fast': 100,
        'order_size': 0.005
    }
    
    metrics = PerformanceMetrics(
        return_gap=0.05,  # Not meeting return target
        trade_count_gap=5,  # Not meeting trade count target
        asset_count_gap=0,
        timestamp=datetime.now()
    )
    
    adjusted = optimizer.adjust_parameters(current_params, metrics)
    
    # Should adjust for more trades
    assert adjusted['post_buy_delay'] == 1  # Reduced from 2
    assert adjusted['post_sell_delay'] == 1  # Reduced from 2
    assert adjusted['macd_signal_fast'] < 100  # Should be reduced
    
    # Should adjust for better returns
    assert adjusted['take_profit'] > 0.05  # Increased
    assert adjusted['stop_loss'] < 0.03  # Decreased
    assert adjusted['order_size'] > 0.005  # Increased

def test_optimize_trading_rules():
    """Test trading rules optimization."""
    optimizer = PolicyOptimizer()
    
    current_rules = {
        'entry': [
            "rsi < 70",
            "macd.macd > 0",
            "price > sma_20"
        ],
        'exit': [
            "rsi > 30",
            "macd.macd < 0"
        ]
    }
    
    metrics = PerformanceMetrics(
        return_gap=0.0,  # Meeting return target
        trade_count_gap=5,  # Not meeting trade count target
        asset_count_gap=0,
        timestamp=datetime.now()
    )
    
    adjusted = optimizer.optimize_trading_rules(current_rules, metrics)
    
    # Should make entry conditions more lenient
    assert "rsi < 75" in adjusted['entry']  # Increased from 70
    assert "macd.macd > -0.1" in adjusted['entry']  # More lenient MACD condition

def test_update_targets():
    """Test performance target updates."""
    optimizer = PolicyOptimizer()
    initial_return = optimizer.performance_targets['min_return']
    initial_trades = optimizer.performance_targets['min_trades']
    
    # Not enough successes
    optimizer.update_targets(3)
    assert optimizer.performance_targets['min_return'] == initial_return
    assert optimizer.performance_targets['min_trades'] == initial_trades
    
    # Enough successes
    optimizer.update_targets(5)
    assert optimizer.performance_targets['min_return'] > initial_return
    assert optimizer.performance_targets['min_trades'] > initial_trades

def test_feedback_loop():
    """Test the complete feedback loop."""
    optimizer = PolicyOptimizer()
    
    results = BacktestResults(
        total_return=0.15,  # Above target
        total_pnl=1000.0,
        sortino_ratio=2.0,
        win_rate=0.7,
        total_trades=15,  # Above target
        trades={'asset1': {}, 'asset2': {}},
        metrics={},
        timestamp=datetime.now()
    )
    
    current_params = {
        'take_profit': 0.05,
        'stop_loss': 0.03,
        'post_buy_delay': 2,
        'post_sell_delay': 2,
        'macd_signal_fast': 100,
        'order_size': 0.005
    }
    
    current_rules = {
        'entry': ["rsi < 70", "macd.macd > 0"],
        'exit': ["rsi > 30", "macd.macd < 0"]
    }
    
    adjusted_params, adjusted_rules = optimizer.feedback_loop(
        results, current_params, current_rules
    )
    
    # Should have optimization history
    assert len(optimizer.optimization_history) == 1
    assert optimizer.success_streak == 1  # Meeting all targets
    
    # Parameters should be unchanged since we're meeting targets
    assert adjusted_params == current_params
    assert adjusted_rules == current_rules

def test_consecutive_successes():
    """Test behavior with consecutive successes."""
    optimizer = PolicyOptimizer()
    initial_targets = optimizer.performance_targets.copy()
    
    results = BacktestResults(
        total_return=0.15,  # Above target
        total_pnl=1000.0,
        sortino_ratio=2.0,
        win_rate=0.7,
        total_trades=15,  # Above target
        trades={'asset1': {}},
        metrics={},
        timestamp=datetime.now()
    )
    
    params = {'take_profit': 0.05}
    rules = {'entry': ["rsi < 70"]}
    
    # Run feedback loop multiple times
    for _ in range(6):  # More than required_successes
        optimizer.feedback_loop(results, params, rules)
    
    # Targets should have increased
    assert optimizer.performance_targets['min_return'] > initial_targets['min_return']
    assert optimizer.performance_targets['min_trades'] > initial_targets['min_trades']
    assert len(optimizer.optimization_history) == 6

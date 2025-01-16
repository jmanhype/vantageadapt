"""Benchmarking module for strategy performance evaluation and comparison."""

from typing import Dict, List, Any, Optional, Tuple
from .market_benchmarks import calculate_market_benchmarks
from .strategy_benchmarks import compare_strategies, generate_comparison_report
from .performance_metrics import calculate_advanced_metrics

__all__ = [
    'calculate_market_benchmarks',
    'compare_strategies',
    'calculate_advanced_metrics',
    'generate_comparison_report'
] 
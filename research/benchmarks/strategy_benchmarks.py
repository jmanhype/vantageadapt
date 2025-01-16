"""Strategy comparison benchmarks and analysis tools."""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from .performance_metrics import calculate_advanced_metrics
from .market_benchmarks import calculate_market_benchmarks


def compare_strategies(
    strategies: List[Dict[str, Any]],
    price_data: pd.DataFrame,
    initial_capital: float = 10.0
) -> Dict[str, Any]:
    """Compare multiple strategies against each other and market benchmarks.
    
    Args:
        strategies: List of strategy results dictionaries
        price_data: DataFrame containing price data
        initial_capital: Initial capital for calculations
        
    Returns:
        Dictionary containing comparison metrics
    """
    comparison = {
        'strategies': {},
        'rankings': {},
        'correlations': pd.DataFrame(),
        'market_comparison': None
    }
    
    # Calculate market benchmarks
    market_metrics = calculate_market_benchmarks(price_data, initial_capital=initial_capital)
    comparison['market_comparison'] = market_metrics
    
    # Process each strategy
    for strategy in strategies:
        strategy_name = strategy.get('name', 'Unknown Strategy')
        metrics = calculate_advanced_metrics(strategy.get('portfolio'), strategy.get('trades'))
        comparison['strategies'][strategy_name] = metrics
    
    # Calculate strategy rankings
    metrics_to_rank = ['total_return', 'sharpe_ratio', 'sortino_ratio', 'win_rate']
    rankings = {}
    
    for metric in metrics_to_rank:
        values = [(name, strat.get(metric, 0)) for name, strat in comparison['strategies'].items()]
        sorted_strategies = sorted(values, key=lambda x: x[1], reverse=True)
        rankings[metric] = {strat[0]: rank + 1 for rank, strat in enumerate(sorted_strategies)}
    
    comparison['rankings'] = rankings
    
    # Calculate return correlations if possible
    returns_data = {}
    for strategy_name, strategy in comparison['strategies'].items():
        if 'returns' in strategy:
            returns_data[strategy_name] = strategy['returns']
    
    if returns_data:
        returns_df = pd.DataFrame(returns_data)
        comparison['correlations'] = returns_df.corr()
    
    return comparison


def generate_comparison_report(comparison_results: Dict[str, Any]) -> str:
    """Generate a formatted report of strategy comparisons.
    
    Args:
        comparison_results: Results from compare_strategies
        
    Returns:
        Formatted string containing the comparison report
    """
    report = []
    report.append("Strategy Comparison Report")
    report.append("=" * 25)
    
    # Market benchmark comparison
    market = comparison_results['market_comparison']
    report.append("\nMarket Benchmark:")
    report.append(f"Buy & Hold Return: {market['buy_hold_return']:.2%}")
    report.append(f"Market Volatility: {market['annualized_volatility']:.2%}")
    report.append(f"Market Sharpe Ratio: {market['sharpe_ratio']:.2f}")
    
    # Strategy performance summary
    report.append("\nStrategy Performance Summary:")
    for strategy_name, metrics in comparison_results['strategies'].items():
        report.append(f"\n{strategy_name}:")
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    # Rankings
    report.append("\nStrategy Rankings:")
    for metric, ranks in comparison_results['rankings'].items():
        report.append(f"\n{metric.replace('_', ' ').title()}:")
        sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])
        for strategy, rank in sorted_ranks:
            report.append(f"{rank}. {strategy}")
    
    return "\n".join(report) 
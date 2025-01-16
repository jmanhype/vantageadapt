"""Test script for trade analysis and visualization."""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.io as pio

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from research.analysis.trade_analyzer import TradeAnalyzer
from research.visualization.trade_visualizer import TradeVisualizer

def generate_sample_data(n_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample price and trade data for testing.
    
    Args:
        n_days: Number of days of data to generate
        
    Returns:
        Tuple of (price_data, trades_df)
    """
    # Generate timestamps
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=n_days),
        end=datetime.now(),
        freq='1H'
    )
    
    # Generate price data
    np.random.seed(42)
    price = 100 * (1 + np.random.randn(len(timestamps)).cumsum() * 0.02)
    volume = np.random.randint(1000, 10000, len(timestamps))
    
    price_data = pd.DataFrame({
        'price': price,
        'volume': volume
    }, index=timestamps)
    
    # Generate trade data
    n_trades = 100
    trade_times = pd.date_range(
        start=timestamps[0],
        end=timestamps[-1],
        periods=n_trades
    )
    
    trades_df = pd.DataFrame({
        'entry_price': [price_data['price'][price_data.index.get_indexer([t], method='nearest')[0]] for t in trade_times],
        'exit_price': [price_data['price'][price_data.index.get_indexer([t + timedelta(hours=2)], method='nearest')[0]] for t in trade_times],
        'pnl': np.random.randn(n_trades) * 10  # Random PnL
    }, index=trade_times)
    
    return price_data, trades_df

def main():
    """Run analysis and visualization demo."""
    print("Generating sample data...")
    price_data, trades_df = generate_sample_data()
    
    print("\nAnalyzing trades...")
    analyzer = TradeAnalyzer(price_data, trades_df)
    trade_stats = analyzer.get_trade_stats()
    
    print("\nTrade Statistics:")
    print(f"Win Rate: {trade_stats.win_rate:.2%}")
    print(f"Profit Factor: {trade_stats.profit_factor:.2f}")
    print(f"Recovery Factor: {trade_stats.recovery_factor:.2f}")
    print(f"Average Win: {trade_stats.avg_win_size:.2f}")
    print(f"Average Loss: {trade_stats.avg_loss_size:.2f}")
    print(f"Largest Win: {trade_stats.largest_win:.2f}")
    print(f"Largest Loss: {trade_stats.largest_loss:.2f}")
    
    print("\nMarket Conditions Performance:")
    for condition, stats in trade_stats.market_conditions.items():
        print(f"\n{condition}:")
        for metric, value in stats.items():
            print(f"  {metric}: {value:.2f}")
    
    print("\nGenerating visualizations...")
    visualizer = TradeVisualizer(price_data, trades_df, trade_stats)
    
    # Create output directory
    output_dir = Path("output/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save plots
    plots = {
        'price_trades': visualizer.plot_price_with_trades(),
        'pnl_distribution': visualizer.plot_pnl_distribution(),
        'market_conditions': visualizer.plot_market_conditions_performance(),
        'equity_curve': visualizer.plot_equity_curve(),
        'streaks': visualizer.plot_win_loss_streaks()
    }
    
    for name, fig in plots.items():
        output_path = output_dir / f"{name}.html"
        fig.write_html(str(output_path))
        print(f"Saved {name} plot to {output_path}")
    
    print("\nAnalysis complete! Check the output/analysis directory for visualization files.")

if __name__ == "__main__":
    main() 
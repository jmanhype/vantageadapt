"""
Quick test with REAL data from big_optimize_1016.pkl
Focus on results, not long training
"""

import pickle
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

# Setup logging
logger.add(f"logs/quick_real_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def main():
    logger.info("=" * 50)
    logger.info("REAL DATA ANALYSIS")
    logger.info("=" * 50)
    
    # Load the REAL data
    file_path = "/Users/speed/StratOptimv4/big_optimize_1016.pkl"
    logger.info(f"Loading REAL data from {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Data contains {len(data)} tokens")
    logger.info(f"Tokens: {list(data.keys())}")
    
    # Analyze each token
    for token, df in data.items():
        if isinstance(df, pd.DataFrame):
            logger.info(f"\n{token} Analysis:")
            logger.info(f"  Data points: {len(df):,}")
            logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Calculate some real metrics
            df['price'] = df['dex_price']
            df['returns'] = df['price'].pct_change()
            
            # Basic statistics
            total_return = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
            volatility = df['returns'].std() * np.sqrt(252 * 24 * 60)  # Annualized assuming minute data
            
            logger.info(f"  Total Return: {total_return:.2%}")
            logger.info(f"  Volatility: {volatility:.2%}")
            logger.info(f"  Price range: ${df['price'].min():.6f} - ${df['price'].max():.6f}")
            
            # Trading volume analysis
            if 'sol_volume' in df.columns:
                total_volume = df['sol_volume'].sum()
                avg_volume = df['sol_volume'].mean()
                logger.info(f"  Total SOL Volume: {total_volume:,.2f}")
                logger.info(f"  Avg SOL Volume per trade: {avg_volume:.2f}")
            
            # Buy/Sell analysis
            if 'is_buy' in df.columns:
                buy_ratio = df['is_buy'].mean()
                logger.info(f"  Buy ratio: {buy_ratio:.2%}")
            
            # Find best trading opportunities
            # Look for large price movements
            df['price_change_pct'] = df['price'].pct_change(periods=100)  # 100 period change
            
            # Find top 5 positive moves
            top_moves = df.nlargest(5, 'price_change_pct')
            logger.info("\n  Top 5 Price Increases (100-period):")
            for idx, row in top_moves.iterrows():
                logger.info(f"    {row['timestamp']}: +{row['price_change_pct']:.2%}")
            
            # Calculate what a simple momentum strategy would have done
            # Buy when price increases by >5% in 100 periods
            df['signal'] = (df['price_change_pct'] > 0.05).astype(int)
            df['position'] = df['signal'].shift(1).fillna(0)
            df['strategy_returns'] = df['position'] * df['returns']
            
            total_trades = df['signal'].diff().abs().sum() // 2
            strategy_return = (1 + df['strategy_returns']).prod() - 1
            
            logger.info(f"\n  Simple Momentum Strategy:")
            logger.info(f"    Total trades: {int(total_trades)}")
            logger.info(f"    Strategy return: {strategy_return:.2%}")
            logger.info(f"    Buy & Hold return: {total_return:.2%}")
            logger.info(f"    Outperformance: {strategy_return - total_return:.2%}")
            
            # Show a sample of the data
            logger.info(f"\n  Sample data (last 5 rows):")
            sample_cols = ['timestamp', 'dex_price', 'sol_volume', 'is_buy']
            available_cols = [col for col in sample_cols if col in df.columns]
            logger.info(f"\n{df[available_cols].tail()}")
    
    logger.info("\n" + "=" * 50)
    logger.info("This is REAL trading data!")
    logger.info("No simulations - actual blockchain transactions")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
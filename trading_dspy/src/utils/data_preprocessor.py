"""Data preprocessing utilities for the trading system."""

from typing import Dict, Any
import pandas as pd
from loguru import logger

def preprocess_market_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Preprocess raw market data into a format suitable for the trading system.
    
    Args:
        df: DataFrame containing market data
        
    Returns:
        Preprocessed market data dictionary
    """
    try:
        # Calculate additional indicators
        df['returns'] = df['dex_price_pct_change']  # Already calculated in the data
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['sma_20'] = df['dex_price'].rolling(window=20).mean()
        df['sma_50'] = df['dex_price'].rolling(window=50).mean()
        
        # Calculate RSI using price changes
        delta = df['dex_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Package data for the trading system
        market_data = {
            'dates': df['timestamp'].tolist(),
            'prices': df['dex_price'].tolist(),
            'volumes': df['sol_volume'].tolist(),
            'indicators': {
                'sma_20': df['sma_20'].tolist(),
                'sma_50': df['sma_50'].tolist(),
                'rsi': df['rsi'].tolist(),
                'volatility': df['volatility'].tolist(),
                'buy_sell_ratio': df['rolling_buy_sell_ratio'].tolist(),
                'volume_trend': df['rolling_sol_volume'].tolist(),
                'buy_sell_ratio_1000': df['rolling_buy_sell_ratio_1000'].tolist(),
                'volume_trend_1000': df['rolling_sol_volume_1000'].tolist(),
                'buy_sell_ratio_7000': df['rolling_buy_sell_ratio_7000'].tolist(),
                'volume_trend_7000': df['rolling_sol_volume_7000'].tolist(),
                'buy_sell_ratio_10000': df['rolling_buy_sell_ratio_10000'].tolist(),
                'volume_trend_10000': df['rolling_sol_volume_10000'].tolist(),
                'price_to_cum_vol_ratio': df['price_to_cum_vol_ratio_50000'].tolist()
            },
            'raw_data': df  # Keep raw data for backtesting
        }
        
        logger.info("Preprocessed market data successfully")
        logger.info(f"Data points: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        return market_data
        
    except Exception as e:
        logger.error(f"Error preprocessing market data: {str(e)}")
        raise 
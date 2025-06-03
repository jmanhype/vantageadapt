"""Data preprocessing utilities for the trading system."""

from typing import Dict, Any
import pandas as pd
import numpy as np
from loguru import logger


class DataPreprocessor:
    """Data preprocessor for market data with technical indicators."""
    
    def __init__(self, use_all_features: bool = True):
        self.use_all_features = use_all_features
        
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and features to market data."""
        df = df.copy()
        
        # Ensure we have the required columns
        if 'close' not in df.columns and 'dex_price' in df.columns:
            df['close'] = df['dex_price']
        if 'volume' not in df.columns and 'sol_volume' in df.columns:
            df['volume'] = df['sol_volume']
        if 'high' not in df.columns:
            df['high'] = df['close'].rolling(20).max()
        if 'low' not in df.columns:
            df['low'] = df['close'].rolling(20).min()
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1)
            
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['sma_20']
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Handle infinities and NaN values properly
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        # Final check for any remaining problematic values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        return df


def preprocess_market_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Preprocess raw market data into a format suitable for the trading system.
    
    Args:
        df: DataFrame containing market data
        
    Returns:
        Preprocessed market data dictionary
    """
    try:
        # Calculate additional indicators based on available columns
        if 'dex_price_pct_change' in df.columns:
            df['returns'] = df['dex_price_pct_change']
        elif 'returns' not in df.columns:
            price_col = 'dex_price' if 'dex_price' in df.columns else 'close'
            df['returns'] = df[price_col].pct_change()
            
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Use the appropriate price column
        price_col = 'dex_price' if 'dex_price' in df.columns else 'close'
        df['sma_20'] = df[price_col].rolling(window=20).mean()
        df['sma_50'] = df[price_col].rolling(window=50).mean()
        
        # Calculate RSI using price changes
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Clean up any infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        # Package data for the trading system
        # Handle timestamp - could be index or column
        if 'timestamp' in df.columns:
            dates = df['timestamp'].tolist()
        else:
            dates = df.index.tolist()
            
        market_data = {
            'dates': dates,
            'prices': df['dex_price'].tolist() if 'dex_price' in df.columns else df['close'].tolist(),
            'volumes': df['sol_volume'].tolist() if 'sol_volume' in df.columns else df['volume'].tolist(),
            'indicators': {
                'sma_20': df['sma_20'].tolist(),
                'sma_50': df['sma_50'].tolist(),
                'rsi': df['rsi'].tolist(),
                'volatility': df['volatility'].tolist(),
                'buy_sell_ratio': df['rolling_buy_sell_ratio'].tolist() if 'rolling_buy_sell_ratio' in df.columns else [0] * len(df),
                'volume_trend': df['rolling_sol_volume'].tolist() if 'rolling_sol_volume' in df.columns else [0] * len(df),
                'buy_sell_ratio_1000': df['rolling_buy_sell_ratio_1000'].tolist() if 'rolling_buy_sell_ratio_1000' in df.columns else [0] * len(df),
                'volume_trend_1000': df['rolling_sol_volume_1000'].tolist() if 'rolling_sol_volume_1000' in df.columns else [0] * len(df),
                'buy_sell_ratio_7000': df['rolling_buy_sell_ratio_7000'].tolist() if 'rolling_buy_sell_ratio_7000' in df.columns else [0] * len(df),
                'volume_trend_7000': df['rolling_sol_volume_7000'].tolist() if 'rolling_sol_volume_7000' in df.columns else [0] * len(df),
                'buy_sell_ratio_10000': df['rolling_buy_sell_ratio_10000'].tolist() if 'rolling_buy_sell_ratio_10000' in df.columns else [0] * len(df),
                'volume_trend_10000': df['rolling_sol_volume_10000'].tolist() if 'rolling_sol_volume_10000' in df.columns else [0] * len(df),
                'price_to_cum_vol_ratio': df['price_to_cum_vol_ratio_50000'].tolist() if 'price_to_cum_vol_ratio_50000' in df.columns else [0] * len(df)
            },
            'raw_data': df  # Keep raw data for backtesting
        }
        
        logger.info("Preprocessed market data successfully")
        logger.info(f"Data points: {len(df)}")
        if 'timestamp' in df.columns:
            logger.info(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        else:
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return market_data
        
    except Exception as e:
        logger.error(f"Error preprocessing market data: {str(e)}")
        raise 
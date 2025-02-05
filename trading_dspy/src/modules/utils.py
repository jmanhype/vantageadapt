"""Utility functions for data loading and management."""

import os
from typing import List, Optional, Dict, Any
import pandas as pd
from loguru import logger

def get_available_assets() -> List[str]:
    """Get list of available assets from the data directory.
    
    Returns:
        List of asset symbols
    """
    data_dir = os.path.join(os.getcwd(), "data")
    assets = []
    
    try:
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return assets
            
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                assets.append(file.replace(".csv", ""))
                logger.info(f"Found asset: {file.replace('.csv', '')}")
                
    except Exception as e:
        logger.error(f"Error reading data directory: {str(e)}")
        
    return assets

def load_trade_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load trading data for a given asset.
    
    Args:
        symbol: Asset symbol to load data for
        
    Returns:
        DataFrame containing the asset's trading data, or None if loading fails
    """
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        
        if df.empty:
            logger.warning(f"Empty data file: {file_path}")
            return None
            
        # Clean data
        df = clean_trading_data(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {str(e)}")
        return None
        
def clean_trading_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess trading data.
    
    Args:
        df: Raw trading data DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    try:
        # Remove rows with NaN values
        original_rows = len(df)
        df = df.dropna()
        rows_removed = original_rows - len(df)
        
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows with NaN values")
            
        # Ensure required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"Successfully cleaned data: {len(df)} rows remaining")
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise 
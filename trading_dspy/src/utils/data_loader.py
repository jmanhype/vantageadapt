"""Data loading utilities for the trading system."""

import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional


class DataLoader:
    """Utility class for loading market data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        
    def load_parquet(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from a parquet file."""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.data_dir / path
                
            if path.exists():
                df = pd.read_parquet(path)
                # Ensure timestamp index
                if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df = df.set_index('timestamp')
                return df
            else:
                logger.warning(f"File not found: {path}")
                return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None
            
    def load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load data from a CSV file."""
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.data_dir / path
                
            if path.exists():
                df = pd.read_csv(path)
                # Convert timestamp if present
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                return df
            else:
                logger.warning(f"File not found: {path}")
                return None
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return None
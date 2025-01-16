"""Database package initialization."""
from typing import List

from .connection import DatabaseConnection
from .models.base import Base
from .models.trading import Strategy, Backtest

db = DatabaseConnection()

__all__: List[str] = ["db", "Base", "Strategy", "Backtest"]

__version__ = '0.1.0' 
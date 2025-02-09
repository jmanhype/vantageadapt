"""Type definitions for the trading system."""

from typing import Dict, List, Any, TypedDict, Literal, Optional
from datetime import datetime

class StrategyContext(TypedDict):
    """Type definition for strategy generation context."""
    market_regime: str
    regime_confidence: float
    timeframe: str
    asset_type: str
    risk_profile: str
    performance_history: Optional[Dict[str, float]]
    constraints: Dict[str, Any]

from enum import Enum

class MarketRegime(str, Enum):
    """Enum for different market regimes."""
    TRENDING = 'TRENDING'
    RANGING = 'RANGING'
    RANGING_LOW_VOL = 'RANGING_LOW_VOL'
    RANGING_HIGH_VOL = 'RANGING_HIGH_VOL'
    TRENDING_UP = 'TRENDING_UP'
    TRENDING_DOWN = 'TRENDING_DOWN'
    UNKNOWN = 'UNKNOWN'

class MarketContext(TypedDict):
    """Type definition for market analysis context."""
    regime: MarketRegime
    confidence: float
    timestamp: datetime
    metrics: Dict[str, float]

class BacktestResults(TypedDict):
    """Type definition for backtesting results."""
    total_return: float
    total_pnl: float
    total_trades: int
    win_rate: float
    sortino_ratio: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    trade_frequency: float
    trades: Dict[str, List[Dict[str, Any]]]
    timestamp: datetime

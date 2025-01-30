"""Type definitions for the trading strategy system."""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

class MarketRegime(Enum):
    """Market regime enumeration."""
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"
    UNKNOWN = "UNKNOWN"

@dataclass
class StrategyContext:
    """Context information for strategy generation."""
    market_regime: MarketRegime
    confidence: float
    risk_level: str
    parameters: Dict[str, Any]
    opportunity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "market_regime": self.market_regime.value,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "parameters": self.parameters,
            "opportunity_score": self.opportunity_score
        }

@dataclass
class BacktestResults:
    """Results from strategy backtesting."""
    total_return: float
    total_pnl: float
    sortino_ratio: float
    win_rate: float
    total_trades: int
    trades: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    timestamp: datetime = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_return": self.total_return,
            "total_pnl": self.total_pnl,
            "sortino_ratio": self.sortino_ratio,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "trades": self.trades,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        } 
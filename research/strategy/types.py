"""Type definitions for the strategy module."""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

class MarketRegime(Enum):
    """Enum for different market regimes."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"

@dataclass
class StrategyContext:
    """Context for a trading strategy, including market regime and parameters."""
    market_regime: MarketRegime
    parameters: Dict[str, Any]
    confidence: float
    risk_level: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the strategy context to a dictionary for serialization."""
        return {
            "market_regime": self.market_regime.name,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "risk_level": self.risk_level
        }

@dataclass
class BacktestResults:
    """Results from backtesting a strategy."""
    total_return: float
    total_pnl: float
    sortino_ratio: float
    win_rate: float
    total_trades: int
    asset_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_return": float(self.total_return),
            "total_pnl": float(self.total_pnl),
            "sortino_ratio": float(self.sortino_ratio),
            "win_rate": float(self.win_rate),
            "total_trades": int(self.total_trades),
            "asset_count": int(self.asset_count)
        } 
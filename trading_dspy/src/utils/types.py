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
    regime: MarketRegime
    confidence: float
    risk_level: str
    parameters: Dict[str, Any]
    opportunity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # Convert parameters to be JSON serializable
        serializable_params = {}
        for key, value in self.parameters.items():
            if isinstance(value, (list, dict, str, int, float, bool)):
                serializable_params[key] = value
            elif value is None:
                serializable_params[key] = None
            else:
                # Convert any other type to string representation
                serializable_params[key] = str(value)
        
        return {
            "regime": self.regime.value,
            "confidence": float(self.confidence),
            "risk_level": str(self.risk_level),
            "parameters": serializable_params,
            "opportunity_score": float(self.opportunity_score)
        }
        
    @classmethod
    def from_market_context(cls, market_context: Any, parameters: Optional[Dict[str, Any]] = None) -> 'StrategyContext':
        """Create a StrategyContext from a MarketContext.
        
        Args:
            market_context: The MarketContext object to convert
            parameters: Optional parameters dictionary. If not provided, will create from market_context attributes
            
        Returns:
            A new StrategyContext object
        """
        if parameters is None:
            parameters = {
                "volatility_level": market_context.volatility_level,
                "trend_strength": market_context.trend_strength,
                "volume_profile": market_context.volume_profile,
                "key_levels": market_context.key_levels,
                "analysis": market_context.analysis
            }
            
        return cls(
            regime=market_context.regime,
            confidence=market_context.confidence,
            risk_level=market_context.risk_level,
            parameters=parameters,
            opportunity_score=0.0  # Default since MarketContext doesn't have this
        )

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
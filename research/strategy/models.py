"""Models for trading strategy generation."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

class MarketRegime(Enum):
    """Market regime types with detailed characteristics."""
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"
    REVERSAL = "REVERSAL"
    UNKNOWN = "UNKNOWN"

@dataclass
class MarketContext:
    """Rich market context for strategic decisions."""
    regime: MarketRegime
    confidence: float
    volatility_level: float
    trend_strength: float
    volume_profile: str
    risk_level: str
    key_levels: Dict[str, List[float]]
    analysis: Dict[str, str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'volatility_level': self.volatility_level,
            'trend_strength': self.trend_strength,
            'volume_profile': self.volume_profile,
            'risk_level': self.risk_level,
            'key_levels': self.key_levels,
            'analysis': self.analysis
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketContext':
        """Create from dictionary with error handling."""
        try:
            regime = MarketRegime(data.get('regime', 'UNKNOWN'))
        except ValueError:
            regime = MarketRegime.UNKNOWN
            
        return cls(
            regime=regime,
            confidence=float(data.get('confidence', 0.0)),
            volatility_level=float(data.get('volatility_level', 0.0)),
            trend_strength=float(data.get('trend_strength', 0.0)),
            volume_profile=str(data.get('volume_profile', 'stable')),
            risk_level=str(data.get('risk_level', 'medium')),
            key_levels=data.get('key_levels', {'support': [], 'resistance': []}),
            analysis=data.get('analysis', {
                'price_action': '',
                'volume_analysis': '',
                'momentum': '',
                'volatility': ''
            })
        )

@dataclass
class StrategyInsight:
    """Strategic trading insights and recommendations."""
    regime_change_probability: float
    suggested_position_size: float
    volatility_adjustment: Dict[str, float]
    regime_specific_rules: Dict[str, Dict[str, float]]
    key_indicators: Dict[str, List[str]]
    risk_management: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime_change_probability': self.regime_change_probability,
            'suggested_position_size': self.suggested_position_size,
            'volatility_adjustment': self.volatility_adjustment,
            'regime_specific_rules': self.regime_specific_rules,
            'key_indicators': self.key_indicators,
            'risk_management': self.risk_management
        } 
"""Market regime classification module."""

from typing import Dict, Any
from loguru import logger
import time
import dspy
from dspy import Module

from ..utils.types import MarketRegime


class MarketRegimeClassifier(Module):
    """Classifier for market regime using direct prediction."""

    def __init__(self) -> None:
        """Initialize market regime classifier."""
        logger.info("Initializing MarketRegimeClassifier")
        super().__init__()

    def forward(self, market_data: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """Classify market regime.
        
        Args:
            market_data: Dictionary containing market data
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary containing regime classification
        """
        try:
            start_time = time.time()
            logger.info("Starting regime classification")
            
            # Extract recent data points
            prices = market_data['prices'][-20:]  # Reduced from 50 to 20 points
            volumes = market_data['volumes'][-20:]
            indicators = market_data['indicators']
            
            # Calculate basic trend indicators
            logger.info("Calculating trend indicators")
            sma_20 = indicators['sma_20'][-1]
            sma_50 = indicators['sma_50'][-1]
            current_price = prices[-1]
            
            # Calculate volatility
            logger.info("Calculating volatility")
            volatility = indicators['volatility'][-1] if 'volatility' in indicators else None
            
            # Determine regime
            logger.info("Determining market regime")
            if volatility and volatility > 0.02:  # High volatility threshold
                regime = MarketRegime.RANGING_HIGH_VOL
                confidence = min(volatility * 50, 1.0)  # Scale confidence with volatility
            elif current_price > sma_20 and sma_20 > sma_50:
                regime = MarketRegime.TRENDING_UP
                confidence = min((current_price - sma_50) / sma_50 * 5, 1.0)
            elif current_price < sma_20 and sma_20 < sma_50:
                regime = MarketRegime.TRENDING_DOWN
                confidence = min((sma_50 - current_price) / sma_50 * 5, 1.0)
            else:
                regime = MarketRegime.RANGING
                confidence = 0.6  # Default confidence for ranging market

            response = {
                "market_context": {
                    "regime": regime,
                    "confidence": float(confidence)
                },
                "analysis_text": f"Market is in {regime} regime with {confidence:.2f} confidence",
                "risk_level": "high" if regime == MarketRegime.RANGING_HIGH_VOL else "moderate"
            }
            
            total_duration = time.time() - start_time
            logger.info("Regime classification completed in {:.2f} seconds", total_duration)
            logger.info("Classification results: regime={}, risk_level={}, confidence={:.2f}",
                       response["market_context"]["regime"],
                       response["risk_level"],
                       response["market_context"]["confidence"])
            
            return response

        except Exception as e:
            logger.error("Error in regime classification: {}", str(e))
            logger.exception("Full traceback:")
            return {
                "market_context": {"regime": MarketRegime.UNKNOWN, "confidence": 0.0},
                "analysis_text": f"Classification failed: {str(e)}",
                "risk_level": "unknown"
            }

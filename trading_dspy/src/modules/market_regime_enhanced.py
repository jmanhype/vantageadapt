"""Enhanced market regime classifier with improved confidence scoring and advanced detection."""

from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from loguru import logger
import time
from enum import Enum
import math

from ..utils.types import MarketRegime

class EnhancedMarketRegimeClassifier:
    """Enhanced market regime classifier with improved confidence scoring."""
    
    def __init__(self, 
                sma_fast_window: int = 20, 
                sma_slow_window: int = 50,
                volatility_window: int = 20,
                rsi_window: int = 14,
                vol_threshold_high: float = 0.002,
                vol_threshold_low: float = 0.001,
                trend_strength_threshold: float = 0.05):
        """Initialize the enhanced market regime classifier.
        
        Args:
            sma_fast_window: Window size for fast SMA
            sma_slow_window: Window size for slow SMA
            volatility_window: Window size for volatility calculation
            rsi_window: Window size for RSI
            vol_threshold_high: Threshold for high volatility
            vol_threshold_low: Threshold for low volatility
            trend_strength_threshold: Threshold for trend strength
        """
        logger.info("Initializing EnhancedMarketRegimeClassifier")
        self.sma_fast_window = sma_fast_window
        self.sma_slow_window = sma_slow_window
        self.volatility_window = volatility_window
        self.rsi_window = rsi_window
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
        self.trend_strength_threshold = trend_strength_threshold
        
    def forward(self, market_data: pd.DataFrame, timeframe: str = None) -> Dict[str, Any]:
        """Classify market regime with enhanced confidence calculation.
        
        Args:
            market_data: Market data dataframe with price and volume information
            timeframe: Timeframe for analysis (ignored in enhanced classifier)
            
        Returns:
            Dictionary with regime, confidence, and risk level
        """
        logger.info("Starting regime classification")
        start_time = time.time()
        
        # Extract price data - handle both DataFrame and dictionary
        if isinstance(market_data, pd.DataFrame):
            # Direct DataFrame input
            if 'dex_price' in market_data.columns:
                price_data = market_data['dex_price']
            else:
                price_data = market_data['close'] if 'close' in market_data.columns else market_data['price']
        elif isinstance(market_data, dict) and 'raw_data' in market_data:
            # Dictionary with DataFrame in 'raw_data' key
            df = market_data['raw_data']
            if 'dex_price' in df.columns:
                price_data = df['dex_price']
            else:
                price_data = df['close'] if 'close' in df.columns else df['price']
        else:
            logger.error("Invalid market data format. Expected DataFrame or dict with 'raw_data' key.")
            # Return default values if data format is invalid
            return {
                "regime": MarketRegime.RANGING_LOW_VOL,
                "confidence": 0.5,
                "risk_level": "moderate",
                "metrics": {}
            }
        
        # Calculate trend indicators
        logger.info("Calculating trend indicators")
        sma_fast = price_data.rolling(window=self.sma_fast_window).mean()
        sma_slow = price_data.rolling(window=self.sma_slow_window).mean()
        
        # Enhanced trend detection
        price_to_sma_fast = price_data / sma_fast - 1  # Calculate percentage difference
        sma_fast_to_sma_slow = sma_fast / sma_slow - 1  # Calculate SMA alignment
        
        # Calculate price percent change
        price_pct_change = price_data.pct_change()
        
        # Calculate volatility as standard deviation of returns
        logger.info("Calculating volatility")
        volatility = price_pct_change.rolling(window=self.volatility_window).std()
        
        # Calculate RSI for overbought/oversold detection
        delta = price_pct_change
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Get current values
        current_price = price_data.iloc[-1]
        current_sma_fast = sma_fast.iloc[-1]
        current_sma_slow = sma_slow.iloc[-1]
        current_volatility = volatility.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_price_to_sma_fast = price_to_sma_fast.iloc[-1]
        current_sma_fast_to_sma_slow = sma_fast_to_sma_slow.iloc[-1]
        
        # Determining market regime
        logger.info("Determining market regime")
        trending = (current_sma_fast_to_sma_slow > self.trend_strength_threshold)
        bullish = current_price > current_sma_fast > current_sma_slow
        bearish = current_price < current_sma_fast < current_sma_slow
        high_volatility = current_volatility > self.vol_threshold_high
        low_volatility = current_volatility < self.vol_threshold_low
        
        # Enhanced regime detection
        if trending:
            if bullish:
                regime = MarketRegime.TRENDING_BULLISH
                # Calculate trend confidence
                trend_confidence = self._calculate_trend_confidence(
                    price_to_sma=current_price_to_sma_fast,
                    sma_alignment=current_sma_fast_to_sma_slow,
                    volatility=current_volatility,
                    rsi=current_rsi,
                    bullish=True
                )
                risk_level = "moderate" if high_volatility else "low"
            elif bearish:
                regime = MarketRegime.TRENDING_BEARISH
                # Calculate trend confidence
                trend_confidence = self._calculate_trend_confidence(
                    price_to_sma=current_price_to_sma_fast,
                    sma_alignment=current_sma_fast_to_sma_slow,
                    volatility=current_volatility,
                    rsi=current_rsi,
                    bullish=False
                )
                risk_level = "high" if high_volatility else "moderate"
            else:
                # Mixed signals - could be transition
                regime = MarketRegime.RANGING_HIGH_VOL if high_volatility else MarketRegime.RANGING_LOW_VOL
                trend_confidence = 0.5 + (abs(current_sma_fast_to_sma_slow) * 5)  # Adjust confidence based on partial trend
                risk_level = "moderate"
        else:
            # Not trending - range-bound market
            if high_volatility:
                regime = MarketRegime.RANGING_HIGH_VOL
                trend_confidence = 0.70 + (current_volatility / self.vol_threshold_high * 0.2)  # Higher confidence with higher volatility
                risk_level = "high"
            else:
                regime = MarketRegime.RANGING_LOW_VOL
                trend_confidence = 0.60 + ((self.vol_threshold_low - current_volatility) / self.vol_threshold_low * 0.3)  # Higher confidence with lower volatility
                risk_level = "moderate"
                
        # Cap confidence at 0.95
        confidence = min(0.95, trend_confidence)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Regime classification completed in {elapsed_time:.2f} seconds")
        logger.info(f"Classification results: regime={regime}, risk_level={risk_level}, confidence={confidence:.2f}")
        
        # Create classification result
        return {
            "regime": regime,
            "confidence": confidence,
            "risk_level": risk_level,
            "metrics": {
                "price_to_sma_fast": current_price_to_sma_fast,
                "sma_fast_to_sma_slow": current_sma_fast_to_sma_slow,
                "volatility": current_volatility,
                "rsi": current_rsi,
                "trending": trending,
                "bullish": bullish,
                "bearish": bearish
            }
        }
        
    def _calculate_trend_confidence(self, 
                                   price_to_sma: float,
                                   sma_alignment: float,
                                   volatility: float,
                                   rsi: float,
                                   bullish: bool) -> float:
        """Calculate confidence score for trending markets.
        
        Args:
            price_to_sma: Price to SMA fast ratio
            sma_alignment: SMA fast to SMA slow alignment
            volatility: Current volatility
            rsi: Current RSI
            bullish: Whether the trend is bullish
            
        Returns:
            Confidence score between 0 and 1
        """
        # Start with base confidence
        base_confidence = 0.6
        
        # Add confidence based on price-to-SMA distance
        # The further the price is from SMA in the trend direction, the stronger the trend
        price_sma_score = abs(price_to_sma) * 5  # Scale up for meaningful contribution
        price_sma_score = min(0.2, price_sma_score)  # Cap contribution
        
        # Check if price-to-SMA direction matches trend direction
        if (bullish and price_to_sma > 0) or (not bullish and price_to_sma < 0):
            base_confidence += price_sma_score
        else:
            base_confidence -= price_sma_score  # Reduce confidence if direction doesn't match
        
        # Add confidence based on SMA alignment
        # The further apart the SMAs are in the trend direction, the stronger the trend
        sma_alignment_score = abs(sma_alignment) * 7.5  # Scale up for meaningful contribution
        sma_alignment_score = min(0.15, sma_alignment_score)  # Cap contribution
        
        # Check if SMA alignment direction matches trend direction
        if (bullish and sma_alignment > 0) or (not bullish and sma_alignment < 0):
            base_confidence += sma_alignment_score
        else:
            base_confidence -= sma_alignment_score
            
        # Adjust confidence based on RSI for extreme readings
        rsi_score = 0
        if bullish and rsi > 70:  # Overbought in bullish trend adds confidence
            rsi_score = 0.1 * (rsi - 70) / 30  # Scale between 0 and 0.1
        elif not bullish and rsi < 30:  # Oversold in bearish trend adds confidence
            rsi_score = 0.1 * (30 - rsi) / 30  # Scale between 0 and 0.1
        base_confidence += rsi_score
        
        # Volatility should be moderate for strong trends
        # Too high volatility can indicate unstable trends
        # Too low volatility might not be trend-following enough
        volatility_score = 0.0
        if self.vol_threshold_low < volatility < self.vol_threshold_high:
            # Ideal volatility range
            volatility_position = (volatility - self.vol_threshold_low) / (self.vol_threshold_high - self.vol_threshold_low)
            # We want a bell curve with maximum at 0.5 (middle of range)
            volatility_score = 0.1 * (1 - abs(volatility_position - 0.5) * 2)
        else:
            # Outside ideal range, penalize
            volatility_score = -0.1
        base_confidence += volatility_score
        
        # Ensure confidence is between 0.05 and 0.95
        return max(0.05, min(0.95, base_confidence))
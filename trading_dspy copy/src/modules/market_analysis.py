"""Market analysis module using DSPy."""

from typing import Dict, Any, Optional, Union
import pandas as pd
from loguru import logger
import dspy
from dspy import Signature, InputField, OutputField, Module, ChainOfThought, Predict
import time
import json

from ..utils.prompt_manager import PromptManager
from ..utils.types import MarketRegime
from .market_regime import MarketRegimeClassifier


class MarketAnalyzer(ChainOfThought):
    """Market analysis module using chain-of-thought reasoning."""

    def __init__(self, prompt_manager: PromptManager):
        """Initialize market analyzer.
        
        Args:
            prompt_manager: Manager for handling prompts
        """
        logger.info("Initializing MarketAnalyzer with ChainOfThought reasoning")
        
        # Define predictor with proper signature
        signature = dspy.Signature(
            "market_data: dict, timeframe: str, prompt: str -> "
            "regime: str, confidence: float, risk_level: str, analysis: str",
            instructions=(
                "Given the financial market data, timeframe, and prompt, analyze the market conditions and provide detailed insights. "
                "Return the market regime, confidence level, risk assessment, and detailed analysis."
            )
        )
        self.predictor = Predict(signature)
        
        # Initialize with signature
        super().__init__(signature)
        self.prompt_manager = prompt_manager
        
        # Initialize regime classifier
        logger.info("Initializing regime classifier")
        self.regime_classifier = MarketRegimeClassifier()

    def _prepare_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare market data for analysis by reducing data points.
        
        Args:
            market_data: Raw market data dictionary
            
        Returns:
            Prepared market data with reduced points
        """
        # Take last 20 points instead of 50
        recent_data = {
            'prices': market_data['prices'][-20:],
            'volumes': market_data['volumes'][-20:],
            'indicators': {
                k: v[-20:] if isinstance(v, list) else v
                for k, v in market_data['indicators'].items()
            }
        }
        
        # Calculate key statistics
        price_change = (recent_data['prices'][-1] - recent_data['prices'][0]) / recent_data['prices'][0] * 100
        avg_volume = sum(recent_data['volumes']) / len(recent_data['volumes'])
        
        # Add summary statistics
        recent_data['summary'] = {
            'price_change_pct': price_change,
            'avg_volume': avg_volume,
            'current_price': recent_data['prices'][-1],
            'current_volume': recent_data['volumes'][-1]
        }
        
        return recent_data

    def _parse_market_context(self, result: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse market context from predictor response.

        Args:
            result: The response from the predictor, either as a string or dict.

        Returns:
            Dict containing the parsed market context with regime, confidence,
            risk level and analysis text.
        """
        try:
            # If result is already a dict with the expected fields, use it directly
            if isinstance(result, dict):
                return {
                    "regime": result.get("regime", MarketRegime.UNKNOWN.value),
                    "confidence": result.get("confidence", 0.0),
                    "risk_level": result.get("risk_level", "unknown"),
                    "analysis_text": result.get("analysis", "No analysis provided")
                }

            # Clean the text response
            cleaned_text = result.strip()
            
            # Remove code block markers if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            
            # Parse JSON response
            try:
                parsed = json.loads(cleaned_text)
                return {
                    "regime": parsed.get("regime", MarketRegime.UNKNOWN.value),
                    "confidence": parsed.get("confidence", 0.0),
                    "risk_level": parsed.get("risk_level", "unknown"),
                    "analysis_text": parsed.get("analysis", "No analysis provided")
                }
            except json.JSONDecodeError as e:
                logger.error("Error parsing JSON response: {}", str(e))
                return {
                    "regime": MarketRegime.UNKNOWN.value,
                    "confidence": 0.0,
                    "risk_level": "unknown",
                    "analysis_text": f"Error parsing response: {str(e)}"
                }
                
        except Exception as e:
            logger.error("Error parsing market context: {}", str(e))
            return {
                "regime": MarketRegime.UNKNOWN.value,
                "confidence": 0.0,
                "risk_level": "unknown",
                "analysis_text": f"Error in analysis: {str(e)}"
            }

    def forward(self, market_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Analyze market data and generate context.

        Args:
            market_data: Dictionary containing market data.
            timeframe: Timeframe for analysis.

        Returns:
            Dictionary containing market context and analysis.
        """
        try:
            # Get market analysis prompt template
            logger.info("Starting market analysis for timeframe: {}", timeframe)
            logger.info("Fetching market analysis prompt template")
            prompt_template = self.prompt_manager.get_prompt("market_analysis")
            if not prompt_template:
                raise ValueError("Market analysis prompt template not found")
            
            # Extract and prepare recent market data
            logger.info("Extracting and preparing recent market data")
            recent_data = self._prepare_market_data(market_data)
            logger.debug("Prepared data summary: price_change={:.2f}%, avg_volume={:.2f}", 
                        recent_data["summary"]["price_change_pct"],
                        recent_data["summary"]["avg_volume"])
            
            # Get initial regime classification
            logger.info("Getting initial regime classification")
            regime_result = self.regime_classifier.forward(market_data=market_data, timeframe=timeframe)
            regime_context = regime_result.get("market_context", {})
            
            # Format prompt with prepared market data and regime
            logger.info("Formatting prompt with prepared market data")
            try:
                formatted_prompt = self.prompt_manager.format_prompt(
                    template=prompt_template,
                    timeframe=timeframe,
                    current_regime=regime_context.get("regime", MarketRegime.UNKNOWN.value),
                    prices=recent_data['prices'],
                    volumes=recent_data['volumes'],
                    indicators=recent_data['indicators'],
                    price_change_pct=recent_data['summary']['price_change_pct'],
                    avg_volume=recent_data['summary']['avg_volume'],
                    current_price=recent_data['summary']['current_price'],
                    current_volume=recent_data['summary']['current_volume']
                )
            except KeyError as e:
                logger.error(f"Error formatting prompt: {e}")
                logger.error("Recent data keys: {}", list(recent_data.keys()))
                logger.error("Summary keys: {}", list(recent_data.get('summary', {}).keys()))
                raise
            
            # Log prompt length for debugging
            logger.debug("Formatted prompt length: {}", len(formatted_prompt))
            
            # Call predictor with formatted prompt
            logger.info("Calling predictor with formatted prompt")
            result = self.predictor(
                market_data=recent_data,
                timeframe=timeframe,
                prompt=formatted_prompt
            )
            
            # Get the response text from the result
            if hasattr(result, 'regime'):
                # Result is already in the expected format
                market_context = {
                    "regime": result.regime,
                    "confidence": float(result.confidence),
                    "risk_level": result.risk_level,
                    "analysis_text": result.analysis
                }
                return {
                    "market_context": {
                        "regime": market_context["regime"],
                        "confidence": market_context["confidence"],
                        "risk_level": market_context["risk_level"]
                    },
                    "analysis_text": market_context["analysis_text"]
                }
            elif hasattr(result, 'text'):
                response_text = result.text
            elif hasattr(result, 'analysis_text'):
                response_text = result.analysis_text
            elif isinstance(result, str):
                response_text = result
            elif isinstance(result, dict):
                # If result is already a dict with the expected fields, use it directly
                market_context = {
                    "regime": result.get("regime", MarketRegime.UNKNOWN.value),
                    "confidence": result.get("confidence", 0.0),
                    "risk_level": result.get("risk_level", "unknown"),
                    "analysis_text": result.get("analysis", "No analysis provided")
                }
                return {
                    "market_context": {
                        "regime": market_context["regime"],
                        "confidence": market_context["confidence"],
                        "risk_level": market_context["risk_level"]
                    },
                    "analysis_text": market_context["analysis_text"]
                }
            else:
                response_text = json.dumps({
                    "regime": MarketRegime.UNKNOWN.value,
                    "confidence": 0.0,
                    "risk_level": "unknown",
                    "analysis": "Failed to get response text from predictor"
                })
            
            # Parse market context from response text
            market_context = self._parse_market_context(response_text)
            
            # Create response
            response = {
                "market_context": {
                    "regime": market_context["regime"],
                    "confidence": market_context["confidence"],
                    "risk_level": market_context["risk_level"]
                },
                "analysis_text": market_context["analysis_text"]
            }
            
            logger.info("Market analysis completed successfully")
            logger.info("Regime: {}, Confidence: {:.2f}, Risk Level: {}", 
                       response["market_context"]["regime"],
                       response["market_context"]["confidence"],
                       response["market_context"]["risk_level"])
            
            return response
            
        except Exception as e:
            logger.error("Error in market analysis: {}", str(e))
            logger.exception("Full traceback:")
            return {
                "market_context": {
                    "regime": MarketRegime.UNKNOWN.value,
                    "confidence": 0.0,
                    "risk_level": "unknown"
                },
                "analysis_text": f"Analysis failed: {str(e)}"
            }


class MarketRegimeClassifier(Module):
    """Classifier for market regime using direct prediction."""

    def __init__(self):
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
            logger.info("Starting market regime classification for timeframe: {}", timeframe)
            
            # Get recent price data
            logger.info("Extracting recent market data")
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
                regime = MarketRegime.RANGING_HIGH_VOL.value
                confidence = min(volatility * 50, 1.0)  # Scale confidence with volatility
            elif current_price > sma_20 and sma_20 > sma_50:
                regime = MarketRegime.TRENDING_BULLISH.value
                confidence = min((current_price - sma_50) / sma_50 * 5, 1.0)
            elif current_price < sma_20 and sma_20 < sma_50:
                regime = MarketRegime.TRENDING_BEARISH.value
                confidence = min((sma_50 - current_price) / sma_50 * 5, 1.0)
            else:
                regime = MarketRegime.RANGING_LOW_VOL.value
                confidence = 0.6  # Default confidence for ranging market

            response = {
                "market_context": {
                    "regime": regime,
                    "confidence": float(confidence)
                },
                "analysis_text": f"Market is in {regime} regime with {confidence:.2f} confidence",
                "risk_level": "high" if regime == MarketRegime.RANGING_HIGH_VOL.value else "moderate"
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
                "market_context": {"regime": MarketRegime.UNKNOWN.value, "confidence": 0.0},
                "analysis_text": f"Classification failed: {str(e)}",
                "risk_level": "unknown"
            } 
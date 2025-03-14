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
                # Extract the fields, ensuring proper types
                regime = result.get("regime", MarketRegime.UNKNOWN.value)
                
                # Handle confidence as float
                try:
                    confidence = float(result.get("confidence", 0.0))
                except (ValueError, TypeError):
                    logger.warning("Invalid confidence value, defaulting to 0.0")
                    confidence = 0.0
                
                risk_level = result.get("risk_level", "unknown")
                analysis = result.get("analysis", "No analysis provided")
                
                return {
                    "regime": regime,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "analysis_text": analysis
                }

            # If result is not a dict, process it as a string
            if not isinstance(result, str):
                # Convert to string if it's some other type
                result = str(result)
                
            # Clean the text response
            cleaned_text = result.strip()
            
            # Try to extract JSON from various formats
            json_content = None
            
            # Check for code blocks with JSON
            if "```json" in cleaned_text:
                # Extract content between ```json and ```
                parts = cleaned_text.split("```json", 1)[1].split("```", 1)
                if parts:
                    json_content = parts[0].strip()
            elif "```" in cleaned_text:
                # Extract content between ``` and ```
                parts = cleaned_text.split("```", 2)
                if len(parts) >= 2:
                    json_content = parts[1].strip()
            
            # If no code blocks found, try to find JSON objects directly
            if not json_content and cleaned_text.strip().startswith("{") and cleaned_text.strip().endswith("}"):
                # The entire text might be JSON
                json_content = cleaned_text
            
            # Try to find JSON object anywhere in the text using regex
            if not json_content:
                import re
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                matches = re.findall(json_pattern, cleaned_text)
                if matches:
                    # Use the largest JSON object found
                    json_content = max(matches, key=len)
            
            # Parse JSON response if found
            if json_content:
                try:
                    parsed = json.loads(json_content)
                    
                    # Extract fields with type checking
                    regime = parsed.get("regime", MarketRegime.UNKNOWN.value)
                    
                    # Handle confidence as float
                    try:
                        confidence = float(parsed.get("confidence", 0.0))
                    except (ValueError, TypeError):
                        logger.warning("Invalid confidence value in JSON, defaulting to 0.0")
                        confidence = 0.0
                    
                    risk_level = parsed.get("risk_level", "unknown")
                    analysis = parsed.get("analysis", "No analysis provided")
                    
                    return {
                        "regime": regime,
                        "confidence": confidence,
                        "risk_level": risk_level,
                        "analysis_text": analysis
                    }
                except json.JSONDecodeError as e:
                    logger.error("Error parsing JSON response: {}", str(e))
                    # Fall through to text parsing as backup
            
            # If JSON parsing failed or no JSON found, try to extract information from text
            logger.warning("Falling back to text parsing")
            
            # Try to extract regime from text
            regime = MarketRegime.UNKNOWN.value
            for r in [r.value for r in MarketRegime]:
                if r in cleaned_text:
                    regime = r
                    break
            
            # Try to extract confidence from text
            confidence = 0.0
            confidence_patterns = [
                r'confidence[:\s]+([0-9]*\.?[0-9]+)',
                r'confidence[:\s]+(high|medium|low)',
                r'([0-9]*\.?[0-9]+)[:\s]+confidence'
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if value.lower() == 'high':
                        confidence = 0.8
                    elif value.lower() == 'medium':
                        confidence = 0.5
                    elif value.lower() == 'low':
                        confidence = 0.2
                    else:
                        try:
                            confidence = float(value)
                            # Ensure confidence is in [0, 1]
                            confidence = max(0.0, min(1.0, confidence))
                        except ValueError:
                            pass
                    break
            
            # Try to extract risk level from text
            risk_level = "unknown"
            if 'risk' in cleaned_text.lower():
                if 'high risk' in cleaned_text.lower():
                    risk_level = "high"
                elif 'medium risk' in cleaned_text.lower() or 'moderate risk' in cleaned_text.lower():
                    risk_level = "moderate"
                elif 'low risk' in cleaned_text.lower():
                    risk_level = "low"
            
            return {
                "regime": regime,
                "confidence": confidence,
                "risk_level": risk_level,
                "analysis_text": cleaned_text[:500]  # Truncate analysis to avoid very long text
            }
                
        except Exception as e:
            logger.error("Error parsing market context: {}", str(e))
            logger.exception("Full traceback:")
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
            try:
                # Try using the predictor directly
                result = self.predictor(
                    market_data=recent_data,
                    timeframe=timeframe,
                    prompt=formatted_prompt
                )
                
                # Check if result has expected attributes
                logger.debug(f"Predictor returned result: {result}")
                if isinstance(result, dict):
                    logger.debug(f"Result is a dict with keys: {result.keys()}")
                
                # If we get a result but it's empty or doesn't have the expected keys, provide fallback analysis
                if (isinstance(result, dict) and (not result or set(result.keys()) != {'regime', 'confidence', 'risk_level', 'analysis'})) or \
                   (not isinstance(result, dict) and (not hasattr(result, 'regime') or not result.regime)):
                    logger.warning("LLM returned empty or invalid response, using fallback from regime classifier")
                    # Use the regime classifier result as fallback
                    regime_type = regime_result["market_context"]["regime"]
                    confidence = regime_result["market_context"]["confidence"]
                    risk_level = regime_result["risk_level"]
                    
                    # Create a fallback response
                    result = type('obj', (object,), {
                        'regime': regime_type,
                        'confidence': confidence,
                        'risk_level': risk_level,
                        'analysis': f"Market is in {regime_type} regime with {confidence:.2f} confidence. Risk level is {risk_level}."
                    })
            except Exception as e:
                logger.error(f"Error calling predictor: {str(e)}")
                # Fallback to regime classifier result
                logger.warning("Using fallback analysis from regime classifier")
                regime_type = regime_result["market_context"]["regime"]
                confidence = regime_result["market_context"]["confidence"]
                risk_level = regime_result["risk_level"]
                
                # Create a fallback response
                result = type('obj', (object,), {
                    'regime': regime_type,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'analysis': f"Market is in {regime_type} regime with {confidence:.2f} confidence. Risk level is {risk_level}."
                })
            
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
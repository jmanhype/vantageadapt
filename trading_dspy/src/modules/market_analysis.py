"""Market analysis module using DSPy."""

from typing import Dict, Any, Optional, Union
import pandas as pd
from loguru import logger
import dspy
from dspy import Signature, InputField, OutputField, Module, ChainOfThought, Predict
import time
import json
import random

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
        
        # Flag for optimization mode
        self.is_optimizing = False

    def _prepare_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare market data for analysis by reducing data points.
        
        During optimization, this method will use varying window sizes and random
        starting points to ensure diverse training examples. It also applies
        various data sampling strategies to further enhance diversity.
        
        Args:
            market_data: Raw market data dictionary
            
        Returns:
            Prepared market data with reduced points
        """
        # Check if market_data is nested under 'summary' key
        if 'summary' in market_data and 'prices' in market_data['summary']:
            # Extract data from nested structure
            logger.debug("Found nested market data structure with 'summary' key")
            prices = market_data['summary']['prices']
            volumes = market_data['summary'].get('volumes', [])
            indicators = market_data['summary'].get('indicators', {})
        else:
            # Use direct structure
            logger.debug("Using direct market data structure")
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [])
            indicators = market_data.get('indicators', {})
            
        # Ensure we have data to work with
        if not prices:
            logger.error("No price data found in market_data")
            logger.error(f"Market data keys: {list(market_data.keys())}")
            if 'summary' in market_data:
                logger.error(f"Summary keys: {list(market_data['summary'].keys())}")
            raise ValueError("No price data available for analysis")
            
        # Determine sampling strategy for optimization
        if self.is_optimizing:
            # Use multiple sampling strategies for diversity
            sampling_strategies = ["random_window", "segment_based", "volatility_biased", "recent_biased"]
            chosen_strategy = random.choice(sampling_strategies)
            logger.debug(f"Using sampling strategy: {chosen_strategy} for optimization")
            
            # Get available data size
            data_length = len(prices)
            
            if chosen_strategy == "random_window":
                # Enhanced random window with larger size variation
                window_size = random.randint(20, 100)  # Increased from 10-50 to 20-100
                if data_length > window_size + 20:
                    start_idx = random.randint(0, data_length - window_size)
                    end_idx = start_idx + window_size
                else:
                    start_idx = 0
                    end_idx = min(window_size, data_length)
                    
            elif chosen_strategy == "segment_based":
                # Divide data into 3 segments (early, middle, late) and sample from one
                segment_size = data_length // 3
                segment = random.choice(["early", "middle", "late"])
                
                if segment == "early":
                    start_range = (0, segment_size)
                elif segment == "middle":
                    start_range = (segment_size, 2 * segment_size)
                else:  # late
                    start_range = (2 * segment_size, data_length - 30)
                    
                # Choose a random window size and starting point within the segment
                window_size = random.randint(30, 60)
                start_idx = random.randint(start_range[0], max(start_range[0], min(start_range[1], data_length - window_size)))
                end_idx = start_idx + window_size
                
            elif chosen_strategy == "volatility_biased":
                # Attempt to find a more volatile section of data
                # This is a simple approach - calculate price changes and find higher volatility areas
                window_size = random.randint(30, 70)
                
                if data_length > window_size + 30:
                    # Calculate simplified volatility for chunks of data
                    chunk_size = 20
                    volatilities = []
                    
                    for i in range(0, data_length - chunk_size, 10):
                        chunk = prices[i:i+chunk_size]
                        if len(chunk) > 1:
                            # Use max-min range as a simple volatility measure
                            volatility = (max(chunk) - min(chunk)) / min(chunk) if min(chunk) > 0 else 0
                            volatilities.append((i, volatility))
                    
                    if volatilities:
                        # Sort by volatility and pick from top half with some randomness
                        volatilities.sort(key=lambda x: x[1], reverse=True)
                        top_volatilities = volatilities[:len(volatilities)//2]
                        chosen_start, _ = random.choice(top_volatilities)
                        
                        start_idx = chosen_start
                        end_idx = min(start_idx + window_size, data_length)
                    else:
                        # Fallback to random window
                        start_idx = random.randint(0, max(0, data_length - window_size))
                        end_idx = start_idx + window_size
                else:
                    # Fallback for small datasets
                    start_idx = 0
                    end_idx = min(window_size, data_length)
                    
            else:  # recent_biased - bias toward more recent data
                window_size = random.randint(25, 75)
                
                # Generate a biased random number that favors recent data
                # This will more often pick from the second half of the data
                bias = random.random() ** 2  # Square to bias toward 1.0
                start_idx = int((data_length - window_size) * bias)
                end_idx = start_idx + window_size
                
                # Ensure indices are valid
                start_idx = max(0, min(start_idx, data_length - window_size))
                end_idx = min(end_idx, data_length)
                
        else:
            # In production, use fixed window of 20 points from the end
            window_size = 20
            start_idx = -window_size
            end_idx = None
        
        # Extract data window
        recent_data = {
            'prices': prices[start_idx:end_idx],
            'volumes': volumes[start_idx:end_idx] if volumes else [],
            'indicators': {
                k: v[start_idx:end_idx] if isinstance(v, list) else v
                for k, v in indicators.items()
            } if indicators else {}
        }
        
        # Calculate key statistics
        price_change = (recent_data['prices'][-1] - recent_data['prices'][0]) / recent_data['prices'][0] * 100
        
        # Handle empty volumes list to prevent ZeroDivisionError
        if recent_data['volumes'] and len(recent_data['volumes']) > 0:
            avg_volume = sum(recent_data['volumes']) / len(recent_data['volumes'])
            current_volume = recent_data['volumes'][-1]
        else:
            avg_volume = 0.0
            current_volume = 0.0
        
        # Add summary statistics
        recent_data['summary'] = {
            'price_change_pct': price_change,
            'avg_volume': avg_volume,
            'current_price': recent_data['prices'][-1],
            'current_volume': current_volume,
            'window_size': window_size,
            'data_start_idx': start_idx,
            'sampling_strategy': chosen_strategy if self.is_optimizing else "fixed_recent"
        }
        
        # Log additional information during optimization
        if self.is_optimizing:
            logger.debug(f"Prepared market data: strategy={chosen_strategy}, window_size={window_size}, " 
                        f"start_idx={start_idx}, price_change={price_change:.2f}%")
        
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
            
            # Check if market_data has the required keys
            if 'prices' not in market_data:
                # Check if it's nested under 'summary'
                if 'summary' in market_data and 'prices' in market_data['summary']:
                    logger.debug("Found prices in nested structure under 'summary'")
                    market_data = market_data['summary']
                else:
                    logger.error("Market data missing 'prices' field: {}", list(market_data.keys()))
                    # Return default response instead of raising error
                    return {
                        "market_context": {"regime": MarketRegime.UNKNOWN.value, "confidence": 0.0},
                        "analysis_text": "Classification failed: Missing required price data",
                        "risk_level": "unknown"
                    }
            
            # Check if market_data contains indicators in different formats
            if 'indicators' not in market_data:
                # First check if data is nested under 'summary'
                if 'summary' in market_data and 'indicators' in market_data['summary']:
                    logger.debug("Found indicators in nested structure under 'summary'")
                    market_data['indicators'] = market_data['summary']['indicators']
                elif all(k in market_data for k in ['sma_20', 'sma_50', 'rsi']):
                    # Indicators are directly in the market_data
                    logger.debug("Found indicators as direct fields in market_data, restructuring")
                    market_data['indicators'] = {
                        'sma_20': [market_data['sma_20']] if not isinstance(market_data['sma_20'], list) else market_data['sma_20'],
                        'sma_50': [market_data['sma_50']] if not isinstance(market_data['sma_50'], list) else market_data['sma_50'],
                        'rsi': [market_data['rsi']] if not isinstance(market_data['rsi'], list) else market_data['rsi']
                    }
                    # Add other indicators if they exist
                    for indicator in ['macd', 'volatility', 'bollinger_upper', 'bollinger_lower']:
                        if indicator in market_data:
                            market_data['indicators'][indicator] = [market_data[indicator]] if not isinstance(market_data[indicator], list) else market_data[indicator]
                else:
                    logger.error("Market data missing 'indicators' field: {}", list(market_data.keys()))
                    # Create empty indicators dict instead of raising error
                    market_data['indicators'] = {}
                
            # Get recent price data
            logger.info("Extracting recent market data")
            
            # Handle different price data structures
            if 'prices' not in market_data:
                logger.error("Market data missing 'prices' field: {}", list(market_data.keys()))
                if 'summary' in market_data and 'prices' in market_data['summary']:
                    logger.debug("Found prices in nested structure under 'summary'")
                    market_data['prices'] = market_data['summary']['prices']
                elif 'current_price' in market_data:
                    # Just use current price if that's all we have
                    logger.debug("No historical prices, using current_price")
                    market_data['prices'] = [market_data['current_price']] * 20  # Create synthetic history
                else:
                    logger.error("No price data found in any expected location")
                    raise ValueError("No price data available for analysis")
            
            # Ensure we have at least one price point
            if not market_data['prices']:
                logger.error("Empty prices list in market data")
                raise ValueError("Empty prices list in market data")
                
            # Take the last 20 points
            prices = market_data['prices'][-20:]  # Reduced from 50 to 20 points
            volumes = market_data.get('volumes', [])[-20:] if market_data.get('volumes') else []
            indicators = market_data['indicators']
            
            # Calculate basic trend indicators
            logger.info("Calculating trend indicators")
            # Check for required indicators
            if 'sma_20' not in indicators or 'sma_50' not in indicators:
                logger.warning("Market data missing required indicators sma_20 or sma_50")
                # Create default values if missing
                if 'sma_20' not in indicators:
                    indicators['sma_20'] = [sum(prices[:min(20, len(prices))]) / min(20, len(prices))]
                    logger.debug("Created default sma_20 from prices")
                if 'sma_50' not in indicators:
                    indicators['sma_50'] = [sum(prices[:min(50, len(prices))]) / min(50, len(prices))]
                    logger.debug("Created default sma_50 from prices")
                
            sma_20 = indicators['sma_20'][-1] if indicators['sma_20'] else prices[-1] * 0.98
            sma_50 = indicators['sma_50'][-1] if indicators['sma_50'] else prices[-1] * 0.95
            current_price = prices[-1]
            
            # Calculate volatility
            logger.info("Calculating volatility")
            if 'volatility' in indicators and indicators['volatility']:
                volatility = indicators['volatility'][-1]
            else:
                # Calculate simple volatility if not provided
                if len(prices) > 5:
                    price_changes = [abs(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
                    volatility = sum(price_changes) / len(price_changes)
                    logger.debug("Calculated default volatility: {}", volatility)
                else:
                    volatility = 0.01  # Default low volatility
                    logger.debug("Using default volatility value")
            
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
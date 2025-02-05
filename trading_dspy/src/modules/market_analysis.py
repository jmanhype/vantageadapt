"""Market analysis module using DSPy."""

from typing import Dict, Any, Optional
import pandas as pd
from loguru import logger
import dspy
from dspy import Signature, InputField, OutputField, Module, ChainOfThought, Predict
import time
import json

from ..utils.prompt_manager import PromptManager


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
            "market_data, timeframe -> market_context, analysis_text, risk_level",
            instructions=(
                "Given the financial market data and timeframe, analyze the market conditions and provide detailed insights. "
                "Return market_context as a JSON string containing 'regime' and 'confidence' fields."
            )
        )
        self.predictor = Predict(signature)
        
        # Initialize with signature
        super().__init__(signature)
        self.prompt_manager = prompt_manager

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

    def _parse_market_context(self, context_str: str) -> Dict[str, Any]:
        """Parse market context from string to dictionary.
        
        Args:
            context_str: Market context as a string
            
        Returns:
            Parsed market context dictionary
        """
        try:
            # Try to parse as JSON first
            return json.loads(context_str)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract regime and confidence from text
            logger.warning("Failed to parse market context as JSON, falling back to text extraction")
            context = {"regime": "unknown", "confidence": 0.0}
            
            # Look for regime in text
            if "uptrend" in context_str.lower():
                context["regime"] = "uptrend"
            elif "downtrend" in context_str.lower():
                context["regime"] = "downtrend"
            elif "ranging" in context_str.lower():
                context["regime"] = "ranging"
            elif "high_volatility" in context_str.lower():
                context["regime"] = "high_volatility"
                
            # Look for confidence value
            import re
            confidence_match = re.search(r'confidence[:\s]+(\d+\.?\d*)', context_str.lower())
            if confidence_match:
                context["confidence"] = float(confidence_match.group(1))
                
            return context

    def forward(self, market_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Analyze market data and generate context.
        
        Args:
            market_data: Dictionary containing market data
            timeframe: Analysis timeframe
            
        Returns:
            Dictionary containing market analysis results
        """
        try:
            start_time = time.time()
            logger.info("Starting market analysis for timeframe: {}", timeframe)
            
            # Get and format the prompt
            logger.info("Fetching market analysis prompt template")
            prompt = self.prompt_manager.get_prompt("market_analysis")
            if not prompt:
                raise ValueError("Market analysis prompt not found")
                
            # Extract and prepare recent data
            logger.info("Extracting and preparing recent market data")
            prepared_data = self._prepare_market_data(market_data)
            logger.debug("Prepared data summary: price_change={:.2f}%, avg_volume={:.2f}",
                        prepared_data['summary']['price_change_pct'],
                        prepared_data['summary']['avg_volume'])
                
            # Format prompt with reduced data
            logger.info("Formatting prompt with prepared market data")
            formatted_prompt = self.prompt_manager.format_prompt(
                prompt,
                prices=prepared_data['prices'],
                volumes=prepared_data['volumes'],
                indicators=prepared_data['indicators'],
                summary=prepared_data['summary'],
                timeframe=timeframe
            )
            logger.debug("Formatted prompt length: {} characters", len(formatted_prompt))

            # Generate analysis using predictor
            logger.info("Making API call to GPT for market analysis...")
            api_start_time = time.time()
            result = self.predictor(
                market_data=prepared_data,
                timeframe=timeframe
            )
            api_duration = time.time() - api_start_time
            logger.info("API call completed in {:.2f} seconds", api_duration)

            # Extract results from prediction
            logger.info("Processing API response")
            
            # Parse market context from string
            market_context = self._parse_market_context(result.market_context)
            
            # Create response using parsed results
            response = {
                "market_context": market_context,
                "analysis_text": result.analysis_text,
                "risk_level": result.risk_level or "unknown"
            }
            
            total_duration = time.time() - start_time
            logger.info("Market analysis completed in {:.2f} seconds", total_duration)
            logger.info("Analysis results: regime={}, risk_level={}, confidence={}",
                       response["market_context"]["regime"],
                       response["risk_level"],
                       response["market_context"]["confidence"])
            
            return response

        except Exception as e:
            logger.error("Error in market analysis: {}", str(e))
            logger.exception("Full traceback:")
            return {
                "market_context": {},
                "analysis_text": f"Analysis failed: {str(e)}",
                "risk_level": "unknown"
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
                regime = "high_volatility"
                confidence = min(volatility * 50, 1.0)  # Scale confidence with volatility
            elif current_price > sma_20 and sma_20 > sma_50:
                regime = "uptrend"
                confidence = min((current_price - sma_50) / sma_50 * 5, 1.0)
            elif current_price < sma_20 and sma_20 < sma_50:
                regime = "downtrend"
                confidence = min((sma_50 - current_price) / sma_50 * 5, 1.0)
            else:
                regime = "ranging"
                confidence = 0.6  # Default confidence for ranging market

            response = {
                "market_context": {
                    "regime": regime,
                    "confidence": float(confidence)
                },
                "analysis_text": f"Market is in {regime} regime with {confidence:.2f} confidence",
                "risk_level": "high" if regime == "high_volatility" else "moderate"
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
                "market_context": {"regime": "unknown", "confidence": 0.0},
                "analysis_text": f"Classification failed: {str(e)}",
                "risk_level": "unknown"
            } 
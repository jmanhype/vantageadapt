"""LLM interface for trading strategy generation."""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import openai
from prompts.prompt_manager import PromptManager
from research.strategy.models import MarketContext, StrategyInsight

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface for LLM-based trading strategy generation."""

    def __init__(self, prompt_manager: PromptManager):
        """Initialize the LLM interface."""
        self.prompt_manager = prompt_manager
        self.default_rules = {
            "conditions": {
                "entry": ["price > ma_20", "volume > volume_ma"],
                "exit": ["price < ma_20", "drawdown > max_drawdown"]
            },
            "parameters": {
                "take_profit": 0.1,
                "stop_loss": 0.05,
                "order_size": 0.001,
                "max_orders": 3
            }
        }

    def chat_completion(
        self,
        messages: List[Dict],
        model: str = "gpt-4-1106-preview",
        temperature: float = 0.7,
        response_format: Dict = None,
        functions: List[Dict] = None,
        function_call: Dict = None
    ) -> Any:
        """Send chat completion request using OpenAI API."""
        try:
            client = openai.OpenAI()
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            if functions:
                kwargs["functions"] = functions
            if function_call:
                kwargs["function_call"] = function_call
            if response_format and not functions:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            if not response.choices:
                return None

            # If using function calling, return the entire response object
            if functions:
                return response
            
            # For regular completions, return just the content
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            return None

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            return None

    async def analyze_market(
        self, market_data: pd.DataFrame
    ) -> Optional[MarketContext]:
        """Analyze market data and return context."""
        try:
            # Calculate market metrics
            price_change = market_data["price"].pct_change()
            volatility = price_change.std()
            trend_strength = abs(market_data["price"].iloc[-1] - market_data["price"].iloc[-20]) / market_data["price"].iloc[-20]
            volume_profile = market_data["sol_volume"].mean() if "sol_volume" in market_data else 0.0
            
            market_summary = {
                "price": float(market_data["price"].iloc[-1]),
                "price_change": float(price_change.iloc[-1]),
                "volume": float(volume_profile),
                "volatility": float(volatility),
                "trend_strength": float(trend_strength),
                "recent_high": float(market_data["price"].rolling(20).max().iloc[-1]),
                "recent_low": float(market_data["price"].rolling(20).min().iloc[-1])
            }

            system_prompt = self.prompt_manager.get_prompt_content(
                "trading/market_analysis", "system"
            )
            user_prompt = self.prompt_manager.get_prompt_content(
                "trading/market_analysis", "user"
            )

            if not system_prompt or not user_prompt:
                raise ValueError("Failed to load market analysis prompts")

            # Format user prompt with market data
            formatted_user_prompt = user_prompt.format(
                price=market_summary["price"],
                price_change=market_summary["price_change"],
                volume=market_summary["volume"],
                volatility=market_summary["volatility"],
                trend_strength=market_summary["trend_strength"],
                recent_high=market_summary["recent_high"],
                recent_low=market_summary["recent_low"]
            )

            # Define the function schema for market analysis
            functions = [{
                "name": "analyze_market_context",
                "description": "Analyze market data and provide market context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "regime": {
                            "type": "string",
                            "enum": ["RANGING_LOW_VOL", "RANGING_HIGH_VOL", "TRENDING"],
                            "description": "The current market regime"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence level in the regime classification (0-1)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "volatility_level": {
                            "type": "number",
                            "description": "Current market volatility level (0-1)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "trend_strength": {
                            "type": "number",
                            "description": "Strength of the current trend (0-1)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "volume_profile": {
                            "type": "number",
                            "description": "Volume profile analysis (0-1)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "risk_level": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Current market risk level"
                        }
                    },
                    "required": ["regime", "confidence", "volatility_level", "trend_strength", "volume_profile", "risk_level"]
                }
            }]

            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_user_prompt}
                ],
                temperature=0.1,
                functions=functions,
                function_call={"name": "analyze_market_context"}
            )

            if not response or not response.choices:
                logger.error("No response from LLM for market analysis")
                return None

            function_call = response.choices[0].message.function_call
            if not function_call or function_call.name != "analyze_market_context":
                logger.error("Invalid function call response")
                return None

            try:
                market_data = json.loads(function_call.arguments)
                return MarketContext.from_dict(market_data)
            except Exception as e:
                logger.error(f"Error creating MarketContext: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return None

    async def generate_strategy(
        self, theme: str, market_context: MarketContext
    ) -> Optional[StrategyInsight]:
        """Generate strategic trading insights using a structured approach."""
        try:
            # Define the function schema for structured generation
            functions = [{
                "name": "generate_strategy_parameters",
                "description": "Generate trading strategy parameters based on market context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "regime_change_probability": {
                            "type": "number",
                            "description": "Probability of market regime change",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "suggested_position_size": {
                            "type": "number",
                            "description": "Suggested position size as a fraction",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "volatility_adjustment": {
                            "type": "object",
                            "properties": {
                                "entry_zone": {"type": "number", "minimum": 0},
                                "exit_zone": {"type": "number", "minimum": 0},
                                "stop_loss": {"type": "number", "minimum": 0}
                            },
                            "required": ["entry_zone", "exit_zone", "stop_loss"]
                        },
                        "regime_specific_rules": {
                            "type": "object",
                            "properties": {
                                "ranging_low_vol": {
                                    "type": "object",
                                    "properties": {
                                        "position_size_mult": {"type": "number", "minimum": 0},
                                        "stop_loss_mult": {"type": "number", "minimum": 0},
                                        "take_profit_mult": {"type": "number", "minimum": 0}
                                    },
                                    "required": ["position_size_mult", "stop_loss_mult", "take_profit_mult"]
                                },
                                "ranging_high_vol": {
                                    "type": "object",
                                    "properties": {
                                        "position_size_mult": {"type": "number", "minimum": 0},
                                        "stop_loss_mult": {"type": "number", "minimum": 0},
                                        "take_profit_mult": {"type": "number", "minimum": 0}
                                    },
                                    "required": ["position_size_mult", "stop_loss_mult", "take_profit_mult"]
                                },
                                "trending": {
                                    "type": "object",
                                    "properties": {
                                        "position_size_mult": {"type": "number", "minimum": 0},
                                        "stop_loss_mult": {"type": "number", "minimum": 0},
                                        "take_profit_mult": {"type": "number", "minimum": 0}
                                    },
                                    "required": ["position_size_mult", "stop_loss_mult", "take_profit_mult"]
                                }
                            },
                            "required": ["ranging_low_vol", "ranging_high_vol", "trending"]
                        },
                        "key_indicators": {
                            "type": "object",
                            "properties": {
                                "primary": {"type": "array", "items": {"type": "string"}},
                                "confirmation": {"type": "array", "items": {"type": "string"}},
                                "exit": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["primary", "confirmation", "exit"]
                        },
                        "risk_management": {
                            "type": "object",
                            "properties": {
                                "max_position_size": {"type": "number", "minimum": 0, "maximum": 1},
                                "max_correlation": {"type": "number", "minimum": 0, "maximum": 1},
                                "max_drawdown": {"type": "number", "minimum": 0, "maximum": 1}
                            },
                            "required": ["max_position_size", "max_correlation", "max_drawdown"]
                        }
                    },
                    "required": [
                        "regime_change_probability",
                        "suggested_position_size",
                        "volatility_adjustment",
                        "regime_specific_rules",
                        "key_indicators",
                        "risk_management"
                    ]
                }
            }]

            system_prompt = """You are a trading strategy generator that provides strategy parameters based on market context.
Your task is to analyze the market context and generate appropriate numeric parameters for the trading strategy."""

            user_prompt = f"""Generate trading strategy parameters for theme: {theme}
Market Context:
- Regime: {market_context.regime.value}
- Volatility Level: {market_context.volatility_level}
- Trend Strength: {market_context.trend_strength}
- Volume Profile: {market_context.volume_profile}
- Risk Level: {market_context.risk_level}"""

            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Lower temperature for more deterministic output
                functions=functions,
                function_call={"name": "generate_strategy_parameters"}
            )

            if not response or not response.choices:
                logger.error("No response from LLM")
                return None

            function_call = response.choices[0].message.function_call
            if not function_call or function_call.name != "generate_strategy_parameters":
                logger.error("Invalid function call response")
                return None

            try:
                strategy_data = json.loads(function_call.arguments)
                
                # Create StrategyInsight with validated data
                return StrategyInsight(
                    regime_change_probability=strategy_data["regime_change_probability"],
                    suggested_position_size=strategy_data["suggested_position_size"],
                    volatility_adjustment=strategy_data["volatility_adjustment"],
                    regime_specific_rules=strategy_data["regime_specific_rules"],
                    key_indicators=strategy_data["key_indicators"],
                    risk_management=strategy_data["risk_management"]
                )
                
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.error(f"Error processing function call response: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None

    async def generate_trading_rules(
        self,
        strategy_insights: StrategyInsight,
        market_context: MarketContext,
        performance_analysis: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """Generate specific trading rules and parameters."""
        try:
            logger.info("Starting trading rules generation...")

            system_prompt = """You are a trading rules generator. Output ONLY a JSON object with exactly this structure:
{
    "conditions": {
        "entry": ["condition1", "condition2"],
        "exit": ["condition1", "condition2"]
    },
    "parameters": {
        "take_profit": number,
        "stop_loss": number,
        "order_size": number,
        "max_orders": number
    }
}
Do not include any other text, explanations, or formatting."""

            # Convert insights to dict for formatting
            insights_dict = {
                "regime_change_probability": strategy_insights.regime_change_probability,
                "suggested_position_size": strategy_insights.suggested_position_size,
                "volatility_adjustment": strategy_insights.volatility_adjustment,
                "regime_specific_rules": strategy_insights.regime_specific_rules,
                "key_indicators": strategy_insights.key_indicators,
                "risk_management": strategy_insights.risk_management
            }

            # Convert market context to dict for formatting
            context_dict = {
                "regime": market_context.regime.value,
                "confidence": market_context.confidence,
                "volatility_level": market_context.volatility_level,
                "trend_strength": market_context.trend_strength,
                "volume_profile": market_context.volume_profile,
                "risk_level": market_context.risk_level,
                "key_levels": market_context.key_levels,
                "analysis": market_context.analysis
            }

            user_prompt = f"""Based on:
Strategy Insights: {json.dumps(insights_dict)}
Market Context: {json.dumps(context_dict)}

Generate trading rules and parameters. Return ONLY a JSON object with no additional text."""

            # Try multiple times with different temperature settings
            for temp in [0.1, 0.05]:  # Use very low temperatures for more deterministic output
                try:
                    response = self.chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temp,
                        response_format={"type": "json_object"}
                    )

                    if not response:
                        logger.warning(f"No response from LLM at temperature {temp}")
                        continue

                    # Clean and validate JSON
                    try:
                        # Remove any non-JSON content
                        response = response.strip()
                        if '```' in response:
                            response = response.split('```')[1]
                            if response.startswith('json'):
                                response = response[4:]
                            response = response.strip('`')
                        
                        # Parse JSON
                        rules_data = json.loads(response)
                        
                        # Validate structure
                        if not isinstance(rules_data, dict):
                            raise ValueError("Response is not a dictionary")
                        
                        if not all(k in rules_data for k in ["conditions", "parameters"]):
                            raise ValueError("Missing required top-level keys")
                        
                        conditions = rules_data["conditions"]
                        parameters = rules_data["parameters"]
                        
                        if not isinstance(conditions, dict) or not all(k in conditions for k in ["entry", "exit"]):
                            raise ValueError("Invalid conditions structure")
                        
                        if not isinstance(parameters, dict):
                            raise ValueError("Invalid parameters structure")
                        
                        # Convert parameters to float
                        validated_params = {}
                        for key, value in parameters.items():
                            try:
                                validated_params[key] = float(value)
                            except (ValueError, TypeError):
                                validated_params[key] = self.default_rules["parameters"].get(key, 0.1)
                        
                        # Validate conditions
                        validated_conditions = {}
                        for key in ["entry", "exit"]:
                            if key in conditions and isinstance(conditions[key], list):
                                validated_conditions[key] = [str(cond) for cond in conditions[key]]
                            else:
                                validated_conditions[key] = self.default_rules["conditions"][key]
                        
                        logger.info(f"Successfully generated trading rules at temperature {temp}")
                        return validated_conditions, validated_params

                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Error validating JSON at temperature {temp}: {str(e)}")
                        continue

                except Exception as e:
                    logger.warning(f"Unexpected error at temperature {temp}: {str(e)}")
                    continue

            # If all attempts failed, return defaults
            logger.warning("All attempts to generate rules failed, using defaults")
            return self.default_rules["conditions"], self.default_rules["parameters"]

        except Exception as e:
            logger.error(f"Error in generate_trading_rules: {str(e)}")
            return self.default_rules["conditions"], self.default_rules["parameters"]

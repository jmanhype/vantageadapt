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
                "entry": [
                    "price > ma_20 and volume > volume_ma",
                    "macd_signal > min_macd_signal_threshold",
                    "volatility < 0.2"  # Add volatility check for ranging markets
                ],
                "exit": [
                    "price < ma_20",
                    "macd_signal < max_macd_signal_threshold",
                    "drawdown > max_drawdown"
                ]
            },
            "parameters": {
                "take_profit": 0.03,  # Reduced from 0.1 for ranging market
                "stop_loss": 0.02,    # Reduced from 0.05 for ranging market
                "order_size": 0.001,  # Reduced for more conservative sizing
                "max_orders": 3,
                "sl_window": 200,     # Reduced window for faster adaptation
                "post_buy_delay": 2,
                "post_sell_delay": 5,
                "macd_signal_fast": 700,
                "macd_signal_slow": 2100,
                "macd_signal_signal": 900,
                "min_macd_signal_threshold": 0,
                "max_macd_signal_threshold": 0,
                "enable_sl_mod": True,  # Enable dynamic stop loss
                "enable_tp_mod": True   # Enable dynamic take profit
            }
        }
        
        # Add regime-specific parameter adjustments
        self.regime_adjustments = {
            "RANGING_LOW_VOL": {
                "take_profit_mult": 0.8,  # Reduce take profit targets
                "stop_loss_mult": 0.7,    # Tighter stops
                "position_size_mult": 0.6  # Smaller positions
            },
            "RANGING_HIGH_VOL": {
                "take_profit_mult": 1.2,
                "stop_loss_mult": 1.3,
                "position_size_mult": 0.4
            },
            "TRENDING": {
                "take_profit_mult": 1.5,
                "stop_loss_mult": 1.0,
                "position_size_mult": 1.0
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

            return response

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

            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_user_prompt}
                ]
            )

            if not response:
                logger.error("No response from LLM for market analysis")
                return None

            market_data = self.parse_json_response(response)
            if not market_data:
                logger.error("Failed to parse market analysis response")
                return None

            # Validate required fields
            required_fields = ["regime", "confidence", "volatility_level", "trend_strength", "volume_profile", "risk_level"]
            for field in required_fields:
                if field not in market_data:
                    logger.error(f"Missing required field in market analysis response: {field}")
                    return None

            # Create MarketContext from response
            try:
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
        """Generate strategic trading insights."""
        try:
            # Define the function schema
            functions = [
                {
                    "name": "generate_strategy_parameters",
                    "description": "Generate trading strategy parameters based on market context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "regime_change_probability": {
                                "type": "number",
                                "description": "Probability of market regime change (0-1)",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "suggested_position_size": {
                                "type": "number",
                                "description": "Suggested position size as a fraction (0-1)",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "volatility_adjustment": {
                                "type": "object",
                                "properties": {
                                    "entry_zone": {
                                        "type": "number",
                                        "description": "Entry zone multiplier",
                                        "minimum": 0
                                    },
                                    "exit_zone": {
                                        "type": "number",
                                        "description": "Exit zone multiplier",
                                        "minimum": 0
                                    },
                                    "stop_loss": {
                                        "type": "number",
                                        "description": "Stop loss multiplier",
                                        "minimum": 0
                                    }
                                },
                                "required": ["entry_zone", "exit_zone", "stop_loss"]
                            },
                            "regime_specific_rules": {
                                "type": "object",
                                "properties": {
                                    "ranging_low_vol": {
                                        "type": "object",
                                        "properties": {
                                            "position_size_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            },
                                            "stop_loss_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            },
                                            "take_profit_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            }
                                        },
                                        "required": ["position_size_mult", "stop_loss_mult", "take_profit_mult"]
                                    },
                                    "ranging_high_vol": {
                                        "type": "object",
                                        "properties": {
                                            "position_size_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            },
                                            "stop_loss_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            },
                                            "take_profit_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            }
                                        },
                                        "required": ["position_size_mult", "stop_loss_mult", "take_profit_mult"]
                                    },
                                    "trending": {
                                        "type": "object",
                                        "properties": {
                                            "position_size_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            },
                                            "stop_loss_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            },
                                            "take_profit_mult": {
                                                "type": "number",
                                                "minimum": 0
                                            }
                                        },
                                        "required": ["position_size_mult", "stop_loss_mult", "take_profit_mult"]
                                    }
                                },
                                "required": ["ranging_low_vol", "ranging_high_vol", "trending"]
                            },
                            "key_indicators": {
                                "type": "object",
                                "properties": {
                                    "primary": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "confirmation": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "exit": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["primary", "confirmation", "exit"]
                            },
                            "risk_management": {
                                "type": "object",
                                "properties": {
                                    "max_position_size": {
                                        "type": "number",
                                        "description": "Maximum position size (0-1)",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "max_correlation": {
                                        "type": "number",
                                        "description": "Maximum correlation between positions (0-1)",
                                        "minimum": 0,
                                        "maximum": 1
                                    },
                                    "max_drawdown": {
                                        "type": "number",
                                        "description": "Maximum allowed drawdown (0-1)",
                                        "minimum": 0,
                                        "maximum": 1
                                    }
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
                }
            ]

            system_prompt = """You are a trading strategy generator that provides strategy parameters based on market context.
Your task is to analyze the market context and generate appropriate numeric parameters for the trading strategy."""

            user_prompt = f"""Generate trading strategy parameters for theme: {theme}
Market Context:
- Regime: {market_context.regime.value}
- Volatility Level: {market_context.volatility_level}
- Trend Strength: {market_context.trend_strength}
- Volume Profile: {market_context.volume_profile}
- Risk Level: {market_context.risk_level}"""

            if not system_prompt or not user_prompt:
                raise ValueError("Failed to load strategy generation prompts")

            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
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

            # Get regime-specific adjustments
            regime = market_context.regime.value
            regime_adjustments = self.regime_adjustments.get(regime, {})
            
            # Apply regime multipliers to base parameters
            base_params = self.default_rules["parameters"].copy()
            if regime_adjustments:
                base_params["take_profit"] *= regime_adjustments.get("take_profit_mult", 1.0)
                base_params["stop_loss"] *= regime_adjustments.get("stop_loss_mult", 1.0)
                base_params["order_size"] *= regime_adjustments.get("position_size_mult", 1.0)

            # Adjust for volatility
            if market_context.volatility_level > 0.2:
                volatility_factor = 0.2 / market_context.volatility_level
                base_params["order_size"] *= volatility_factor
                base_params["stop_loss"] *= (1 + (1 - volatility_factor))
                base_params["take_profit"] *= (1 + (1 - volatility_factor))

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
        "max_orders": number,
        "sl_window": number,
        "post_buy_delay": number,
        "post_sell_delay": number,
        "macd_signal_fast": number,
        "macd_signal_slow": number,
        "macd_signal_signal": number,
        "min_macd_signal_threshold": number,
        "max_macd_signal_threshold": number,
        "enable_sl_mod": boolean,
        "enable_tp_mod": boolean
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
Base Parameters: {json.dumps(base_params)}

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
                        
                        # Merge with base parameters and apply regime adjustments
                        validated_params = base_params.copy()
                        for key, value in parameters.items():
                            try:
                                if key in validated_params:
                                    # Convert to float except for boolean parameters
                                    if key not in ['enable_sl_mod', 'enable_tp_mod']:
                                        validated_params[key] = float(value)
                                    else:
                                        validated_params[key] = bool(value)
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid value for parameter {key}: {value}")
                        
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

            # If all attempts failed, return defaults with regime adjustments
            logger.warning("All attempts to generate rules failed, using defaults with regime adjustments")
            return self.default_rules["conditions"], base_params

        except Exception as e:
            logger.error(f"Error in generate_trading_rules: {str(e)}")
            return self.default_rules["conditions"], base_params

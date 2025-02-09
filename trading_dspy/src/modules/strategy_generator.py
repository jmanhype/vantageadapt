"""Strategy generation module using DSPy for autonomous trading strategy creation."""

from typing import Dict, Any, Optional, Tuple, List
from loguru import logger
import dspy
from dspy import Module
import json

from ..utils.prompt_manager import PromptManager
from ..utils.memory_manager import TradingMemoryManager


class StrategyGenerator(Module):
    """Strategy generation module using chain-of-thought reasoning for autonomous strategy creation."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        memory_manager: TradingMemoryManager
    ):
        """Initialize strategy generator.
        
        Args:
            prompt_manager: Manager for handling prompts
            memory_manager: Manager for handling strategy memory
        """
        super().__init__()
        self.prompt_manager = prompt_manager
        self.memory_manager = memory_manager
        
        # Enhanced signature for autonomous strategy generation
        signature = dspy.Signature(
            """
            market_context: dict, theme: str, base_parameters: dict, prompt: str -> 
            reasoning: str, trade_signal: str, parameters: dict, parameter_ranges: dict,
            confidence: float, entry_conditions: list, exit_conditions: list, indicators: list
            """,
            instructions=(
                "Generate a complete trading strategy based on market context and theme. "
                "Return parameters must include stop_loss (0-1), take_profit (0-5), and position_size (0-1). "
                "Parameter_ranges should specify min/max values for optimization. "
                "Entry and exit conditions should be executable Python expressions. "
                "Indicators should list all technical indicators needed."
            )
        )
        self.predictor = dspy.Predict(signature)

    def forward(
        self,
        market_context: Dict[str, Any],
        theme: str,
        base_parameters: Optional[Dict[str, Any]] = None,
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """Generate trading strategy with autonomous improvements.
        
        Args:
            market_context: Market context from analysis
            theme: Trading strategy theme
            base_parameters: Optional base parameters
            timeframe: Trading timeframe
            
        Returns:
            Dictionary containing strategy details
        """
        try:
            # Get and format the prompt
            prompt = self.prompt_manager.get_prompt("strategy_generator")
            if not prompt:
                raise ValueError("Strategy generator prompt not found")
                
            # Format market context for prompt
            if not market_context or not isinstance(market_context, dict):
                market_context = {}
                
            # Get recent performance data
            recent_performance = self._get_recent_performance()
                
            context_summary = {
                'regime': market_context.get('regime', 'unknown'),
                'confidence': market_context.get('confidence', 0.0),
                'risk_level': market_context.get('risk_level', 'unknown'),
                'analysis': market_context.get('analysis_text', ''),
                'recent_performance': recent_performance
            }

            # Prepare default values for required fields with parameter ranges
            default_values = {
                'market_context': json.dumps(context_summary),
                'trading_theme': theme,
                'base_parameters': json.dumps(base_parameters or {}),
                'recent_performance': json.dumps(recent_performance),
                'parameters': json.dumps({
                    'stop_loss': '',
                    'take_profit': '',
                    'position_size': '',
                    'additional_params': {}
                }),
                'parameter_ranges': json.dumps({
                    'stop_loss': [0.02, 0.15],
                    'take_profit': [0.04, 0.30],
                    'position_size': [0.1, 1.0],
                    'sl_window': [200, 600],
                    'max_orders': [1, 5],
                    'order_size': [0.001, 0.005],
                    'macd_signal_fast': [50, 500],
                    'macd_signal_slow': [100, 1000],
                    'macd_signal_signal': [20, 200]
                }),
                'reasoning': '',
                'entry_conditions': json.dumps([]),
                'exit_conditions': json.dumps([]),
                'indicators': json.dumps([]),
                'trade_signal': '',
                'confidence': ''
            }
                
            formatted_prompt = self.prompt_manager.format_prompt(
                prompt,
                **default_values
            )

            # Generate strategy using predictor
            try:
                result = self.predictor(
                    market_context=context_summary,
                    theme=theme,
                    base_parameters=base_parameters or {},
                    prompt=formatted_prompt
                )
                
                # Extract result fields with parameter ranges
                # Create strategy context matching StrategyContext type
                strategy = {
                    "market_regime": context_summary['regime'],
                    "regime_confidence": result.confidence,
                    "timeframe": timeframe,
                    "asset_type": "crypto",
                    "risk_profile": context_summary.get('risk_level', 'unknown'),
                    "performance_history": self._get_recent_performance(),
                    "constraints": {
                        **result.parameters,
                        "parameter_ranges": result.parameter_ranges,
                        "entry_conditions": result.entry_conditions,
                        "exit_conditions": result.exit_conditions,
                        "indicators": result.indicators,
                        "strategy_type": theme,
                        "reasoning": result.reasoning,
                        "trade_signal": result.trade_signal
                    }
                }
                
                # Format the strategy for prompt
                default_values.update({
                    "reasoning": strategy["constraints"]["reasoning"],
                    "parameters": json.dumps(strategy["constraints"]),
                    "parameter_ranges": json.dumps(strategy["constraints"]["parameter_ranges"]),
                    "entry_conditions": json.dumps(strategy["constraints"]["entry_conditions"]),
                    "exit_conditions": json.dumps(strategy["constraints"]["exit_conditions"])
                })
                
                # Store the strategy
                self._store_strategy(strategy)
                
                return strategy
                
            except Exception as e:
                logger.error(f"Error in DSPy predictor: {str(e)}")
                return self._get_default_strategy(theme, base_parameters, context_summary['regime'], timeframe)

        except Exception as e:
            logger.error(f"Error in strategy generation: {str(e)}")
            logger.exception("Full traceback:")
            return self._get_default_strategy(theme, base_parameters, "unknown", timeframe)

    def validate_strategy(self, strategy: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate generated strategy.
        
        Args:
            strategy: Generated strategy
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            if not strategy:
                return False, "Empty strategy"
                
            # Validate required fields for StrategyContext
            required_fields = [
                'market_regime', 'regime_confidence', 'timeframe', 'asset_type',
                'risk_profile', 'constraints'
            ]
            missing_fields = [f for f in required_fields if f not in strategy]
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}"

            # Validate regime confidence
            confidence = float(strategy['regime_confidence'])
            if not 0 <= confidence <= 1:
                return False, f"Invalid regime confidence value: {confidence}"

            # Get constraints
            constraints = strategy['constraints']
            if not isinstance(constraints, dict):
                return False, "Constraints must be a dictionary"

            # Validate parameters in constraints
            required_params = ['stop_loss', 'take_profit', 'position_size']
            missing_params = [p for p in required_params if p not in constraints]
            if missing_params:
                return False, f"Missing required parameters in constraints: {', '.join(missing_params)}"

            # Validate parameter values
            try:
                stop_loss = float(constraints['stop_loss'])
                take_profit = float(constraints['take_profit'])
                position_size = float(constraints['position_size'])

                if not 0 < stop_loss < 1:
                    return False, f"Invalid stop loss: {stop_loss}"

                if not 0 < take_profit < 5:
                    return False, f"Invalid take profit: {take_profit}"

                if not 0 < position_size <= 1:
                    return False, f"Invalid position size: {position_size}"
            except (ValueError, TypeError) as e:
                return False, f"Invalid parameter value format: {str(e)}"

            # Validate parameter ranges in constraints
            ranges = constraints.get('parameter_ranges', {})
            if not isinstance(ranges, dict):
                return False, "Parameter ranges must be a dictionary"

            required_ranges = {
                'stop_loss': (0.02, 0.15),
                'take_profit': (0.04, 0.30),
                'position_size': (0.1, 1.0)
            }

            for param, (min_val, max_val) in required_ranges.items():
                if param not in ranges:
                    ranges[param] = [min_val, max_val]
                else:
                    range_values = ranges[param]
                    if not isinstance(range_values, list) or len(range_values) != 2:
                        ranges[param] = [min_val, max_val]
                    elif range_values[0] >= range_values[1] or range_values[0] < min_val or range_values[1] > max_val:
                        ranges[param] = [min_val, max_val]

            # Update ranges in constraints
            constraints['parameter_ranges'] = ranges

            # Validate trade signal
            valid_signals = ['BUY', 'SELL', 'HOLD']
            if constraints.get('trade_signal') not in valid_signals:
                return False, f"Invalid trade signal: {constraints.get('trade_signal')}"

            # Validate conditions
            entry_conditions = constraints.get('entry_conditions', [])
            if not isinstance(entry_conditions, list) or not entry_conditions:
                return False, "Entry conditions must be a non-empty list"

            exit_conditions = constraints.get('exit_conditions', [])
            if not isinstance(exit_conditions, list) or not exit_conditions:
                return False, "Exit conditions must be a non-empty list"

            # Validate indicators
            indicators = constraints.get('indicators', [])
            if not isinstance(indicators, list) or not indicators:
                return False, "Indicators must be a non-empty list"

            # Update validated conditions and indicators
            constraints['entry_conditions'] = entry_conditions
            constraints['exit_conditions'] = exit_conditions
            constraints['indicators'] = indicators
                
            return True, "Strategy is valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _validate_conditions(self, conditions: List[str]) -> List[str]:
        """Validate and clean trading conditions.
        
        Args:
            conditions: List of condition expressions
            
        Returns:
            List of validated conditions
        """
        if not conditions:
            return []
            
        valid_conditions = []
        for condition in conditions:
            # Remove any obviously dangerous operations
            if any(dangerous in condition.lower() for dangerous in ['exec', 'eval', 'import', 'os.', 'system']):
                continue
            # Ensure condition references valid indicators
            if 'df_indicators' in condition:
                valid_conditions.append(condition)
        return valid_conditions

    def _get_recent_performance(self) -> Dict[str, Any]:
        """Get recent strategy performance from memory.
        
        Returns:
            Dictionary containing recent performance metrics
        """
        try:
            return self.memory_manager.get_recent_performance()
        except Exception as e:
            logger.error(f"Error getting recent performance: {str(e)}")
            return {}

    def _store_strategy(self, strategy: Dict[str, Any]) -> None:
        """Store strategy in memory for future reference.
        
        Args:
            strategy: Strategy to store
        """
        try:
            # Get constraints with defaults
            constraints = strategy.get("constraints", {})
            
            # Default values for required fields
            default_entry_conditions = [
                "df_indicators['rsi'] < 70",
                "df_indicators['sma_20'] > df_indicators['sma_50']"
            ]
            default_exit_conditions = [
                "df_indicators['rsi'] > 30",
                "df_indicators['sma_20'] < df_indicators['sma_50']"
            ]
            default_indicators = [
                "rsi",
                "sma_20",
                "sma_50",
                "macd"
            ]
            
            # Ensure constraints has all required fields with non-empty values
            constraints["entry_conditions"] = (
                constraints.get("entry_conditions", []) or default_entry_conditions
            )
            constraints["exit_conditions"] = (
                constraints.get("exit_conditions", []) or default_exit_conditions
            )
            constraints["indicators"] = (
                constraints.get("indicators", []) or default_indicators
            )
            constraints["trade_signal"] = constraints.get("trade_signal", "HOLD")
            
            # Validate constraints first
            if not isinstance(constraints.get("entry_conditions"), list) or not constraints["entry_conditions"]:
                logger.error("Entry conditions must be a non-empty list")
                return
            if not isinstance(constraints.get("exit_conditions"), list) or not constraints["exit_conditions"]:
                logger.error("Exit conditions must be a non-empty list")
                return
            if not isinstance(constraints.get("indicators"), list) or not constraints["indicators"]:
                logger.error("Indicators must be a non-empty list")
                return

            try:
                # Format strategy to match StrategyContext requirements with explicit type conversion
                formatted_strategy = {
                    "market_regime": str(strategy.get("market_regime", "UNKNOWN")),
                    "regime_confidence": float(strategy.get("regime_confidence", 0.0)),
                    "timeframe": str(strategy.get("timeframe", "1h")),
                    "asset_type": str(strategy.get("asset_type", "crypto")),
                    "risk_profile": str(strategy.get("risk_profile", "unknown")),
                    "performance_history": strategy.get("performance_history", {}),
                    "constraints": constraints
                }

                # Log the formatted strategy for debugging
                logger.debug("Formatted strategy:")
                logger.debug(f"Market Regime: {formatted_strategy['market_regime']}")
                logger.debug(f"Confidence: {formatted_strategy['regime_confidence']}")
                logger.debug(f"Timeframe: {formatted_strategy['timeframe']}")
                logger.debug(f"Asset Type: {formatted_strategy['asset_type']}")
                logger.debug(f"Risk Profile: {formatted_strategy['risk_profile']}")
                logger.debug(f"Constraints: {json.dumps(formatted_strategy['constraints'], indent=2)}")

                # Store the formatted strategy
                success = self.memory_manager.store_strategy(formatted_strategy)
                if not success:
                    logger.error("Failed to store strategy in memory manager")
                    return
                logger.info("Successfully stored strategy in memory")

            except (ValueError, TypeError) as e:
                logger.error(f"Error formatting strategy: {str(e)}")
                logger.debug("Strategy content:", strategy)
                return
                
        except Exception as e:
            logger.error(f"Error storing strategy: {str(e)}")
            logger.debug("Strategy content:", strategy)

    def _get_default_strategy(
        self,
        theme: str,
        base_parameters: Optional[Dict[str, Any]],
        regime: str,
        timeframe: str = "1h"
    ) -> Dict[str, Any]:
        """Get default strategy when generation fails.
        
        Args:
            theme: Strategy theme
            base_parameters: Optional base parameters
            regime: Market regime
            timeframe: Trading timeframe
            
        Returns:
            Default strategy dictionary
        """
        # Create default strategy matching StrategyContext type
        return {
            "market_regime": regime,
            "regime_confidence": 0.0,
            "timeframe": timeframe,
            "asset_type": "crypto",
            "risk_profile": "unknown",
            "performance_history": self._get_recent_performance(),
            "constraints": {
                **(base_parameters if base_parameters is not None else {
                    "stop_loss": 0.02,
                    "take_profit": 0.04,
                    "position_size": 0.1
                }),
                "parameter_ranges": {
                    "stop_loss": [0.02, 0.15],
                    "take_profit": [0.04, 0.30],
                    "position_size": [0.1, 1.0],
                    "sl_window": [200, 600],
                    "max_orders": [1, 5],
                    "order_size": [0.001, 0.005]
                },
                "entry_conditions": [
                    "df_indicators['rsi'] < 70",
                    "df_indicators['sma_20'] > df_indicators['sma_50']"
                ],
                "exit_conditions": [
                    "df_indicators['rsi'] > 30",
                    "df_indicators['sma_20'] < df_indicators['sma_50']"
                ],
                "indicators": [
                    "rsi",
                    "sma_20",
                    "sma_50",
                    "macd"
                ],
                "strategy_type": theme,
                "reasoning": "Error in strategy generation",
                "trade_signal": "HOLD"
            }
        }

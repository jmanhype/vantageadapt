"""Strategy generation module using DSPy."""

from typing import Dict, Any, Optional, Tuple
from loguru import logger
import dspy
from dspy import Module

from ..utils.prompt_manager import PromptManager
from ..utils.memory_manager import TradingMemoryManager


class StrategyGenerator(Module):
    """Strategy generation module using chain-of-thought reasoning."""

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
        
        # Use O3 mini model with explicit signature
        signature = dspy.Signature(
            "market_context: dict, theme: str, base_parameters: dict, prompt: str -> reasoning: str, trade_signal: str, parameters: dict, confidence: float",
            instructions=(
                "Generate a trading strategy based on market context and theme. "
                "Return parameters must include stop_loss (0-1), take_profit (0-5), and position_size (0-1)."
            )
        )
        self.predictor = dspy.Predict(signature)

    def forward(
        self,
        market_context: Dict[str, Any],
        theme: str,
        base_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate trading strategy.
        
        Args:
            market_context: Market context from analysis
            theme: Trading strategy theme
            base_parameters: Optional base parameters
            
        Returns:
            Dictionary containing strategy details
        """
        try:
            # Get and format the prompt
            prompt = self.prompt_manager.get_prompt("strategy_generator")
            if not prompt:
                raise ValueError("Strategy generator prompt not found")
                
            # Format market context for prompt
            context_summary = {
                'regime': market_context.get('regime', 'unknown'),
                'confidence': market_context.get('confidence', 0.0),
                'risk_level': market_context.get('risk_level', 'unknown'),
                'analysis': market_context.get('analysis_text', '')
            }
                
            formatted_prompt = self.prompt_manager.format_prompt(
                prompt,
                market_context=context_summary,
                trading_theme=theme,
                base_parameters=base_parameters or {}
            )

            # Generate strategy using predictor
            result = self.predictor(
                market_context=market_context,
                theme=theme,
                base_parameters=base_parameters or {},
                prompt=formatted_prompt
            )

            # Extract and validate parameters
            parameters = result.parameters
            if not isinstance(parameters, dict):
                parameters = {}
            
            # Ensure required parameters exist with valid values
            required_params = {
                'stop_loss': (0.0, 1.0),
                'take_profit': (0.0, 5.0),
                'position_size': (0.0, 1.0)
            }
            
            # Validate and fix parameters
            for param, (min_val, max_val) in required_params.items():
                # Try to get parameter value
                try:
                    if param in parameters:
                        value = float(parameters[param])
                        # Clamp value to valid range
                        parameters[param] = max(min_val, min(value, max_val))
                    elif base_parameters and param in base_parameters:
                        value = float(base_parameters[param])
                        if min_val <= value <= max_val:
                            parameters[param] = value
                    else:
                        # Use default values if missing
                        parameters[param] = {
                            'stop_loss': 0.02,  # 2% default stop loss
                            'take_profit': 0.04,  # 4% default take profit
                            'position_size': 0.1   # 10% default position size
                        }[param]
                except (ValueError, TypeError):
                    # Use defaults if conversion fails
                    parameters[param] = {
                        'stop_loss': 0.02,
                        'take_profit': 0.04,
                        'position_size': 0.1
                    }[param]

            # Create response using prediction results
            response = {
                "reasoning": result.reasoning,
                "trade_signal": result.trade_signal,
                "parameters": parameters,
                "confidence": result.confidence
            }

            return response

        except Exception as e:
            logger.error(f"Error in strategy generation: {str(e)}")
            return {}

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
                
            required_fields = ['reasoning', 'trade_signal', 'parameters', 'confidence']
            missing_fields = [f for f in required_fields if f not in strategy]
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}"
                
            # Validate trade signal
            valid_signals = ['BUY', 'SELL', 'HOLD']
            if strategy['trade_signal'] not in valid_signals:
                return False, f"Invalid trade signal: {strategy['trade_signal']}"
                
            # Validate confidence
            confidence = float(strategy['confidence'])
            if not 0 <= confidence <= 1:
                return False, f"Invalid confidence value: {confidence}"
                
            # Validate parameters
            params = strategy['parameters']
            if not isinstance(params, dict):
                return False, "Parameters must be a dictionary"
                
            required_params = ['stop_loss', 'take_profit', 'position_size']
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                return False, f"Missing required parameters: {', '.join(missing_params)}"
                
            # Validate parameter values
            if not 0 < float(params['stop_loss']) < 1:
                return False, f"Invalid stop loss: {params['stop_loss']}"
                
            if not 0 < float(params['take_profit']) < 5:
                return False, f"Invalid take profit: {params['take_profit']}"
                
            if not 0 < float(params['position_size']) <= 1:
                return False, f"Invalid position size: {params['position_size']}"
                
            return True, "Strategy is valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}" 
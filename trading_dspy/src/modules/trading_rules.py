"""Trading rules generation module using DSPy."""

from typing import Dict, Any, List, Optional
from loguru import logger
import dspy
from dspy import Module
import json
import pandas as pd
import numpy as np
import ta

from ..utils.prompt_manager import PromptManager


class TradingRulesGenerator(Module):
    """Trading rules generation module using DSPy predictor."""

    def __init__(self, prompt_manager: PromptManager):
        """Initialize trading rules generator.
        
        Args:
            prompt_manager: Manager for handling prompts
        """
        super().__init__()
        self.prompt_manager = prompt_manager
        
        # Define predictor with proper signature
        signature = dspy.Signature(
            "strategy_insights: dict, market_context: dict, prompt: str -> entry_conditions: list[str], exit_conditions: list[str], parameters: dict, reasoning: str, indicators: list[str]",
            instructions=(
                "Generate detailed trading rules based on strategy insights and market context.\n\n"
                "YOU MUST RETURN ALL 5 FIELDS. NO EXCEPTIONS. ALWAYS INCLUDE:\n"
                "1. entry_conditions: A list of at least 2-3 specific entry rules\n"
                "2. exit_conditions: A list of at least 2 exit rules including stop loss and take profit\n"
                "3. parameters: A dict with stop_loss, take_profit, and position_size values\n"
                "4. reasoning: A detailed explanation (100+ words) for the rules\n"
                "5. indicators: A list of ALL indicators used (e.g., ['price', 'sma_20', 'rsi', 'macd.macd'])\n\n"
                "CRITICAL: The 'indicators' field is MANDATORY. Even if no special indicators are used, "
                "return at least ['price']. Common indicators include: 'price', 'sma_20', 'sma_50', 'rsi', "
                "'macd.macd', 'macd.signal', 'bb.upper', 'bb.lower', 'volume', 'williams_r'.\n\n"
                "FAILURE TO INCLUDE ALL 5 FIELDS WILL CAUSE A SYSTEM ERROR."
            )
        )
        self.predictor = dspy.Predict(signature)

    def __deepcopy__(self, memo):
        """Create a deep copy of this object.
        
        Args:
            memo: Dictionary of already copied objects
            
        Returns:
            A deep copy of this object
        """
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Copy attributes
        for k, v in self.__dict__.items():
            if k == 'prompt_manager':
                # For prompt_manager just create a reference, don't deepcopy
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
                
        return result
    
    def deepcopy(self, memo=None):
        """Public interface for deep copying this object.
        
        Args:
            memo: Optional dictionary of already copied objects
            
        Returns:
            A deep copy of this object
        """
        if memo is None:
            memo = {}
        return self.__deepcopy__(memo)

    def predictors(self) -> List[dspy.Predict]:
        """Return the list of predictors used by this module.
        
        Returns:
            List of predictors
        """
        return [self.predictor]

    def forward(
        self,
        strategy_insights: Dict[str, Any],
        market_context: Dict[str, Any],
        performance_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate trading rules.
        
        Args:
            strategy_insights: Strategy insights from generator
            market_context: Current market context
            performance_analysis: Optional performance analysis
            
        Returns:
            Dictionary containing trading rules
        """
        try:
            # Get and format the prompt
            prompt = self.prompt_manager.get_prompt("trading_rules")
            if not prompt:
                raise ValueError("Trading rules prompt not found")
                
            # Ensure strategy_insights has parameters
            if 'parameters' not in strategy_insights:
                strategy_insights['parameters'] = {}
                
            # Set default parameters if not provided
            default_params = {
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'position_size': 0.1
            }
            
            for param, default_value in default_params.items():
                if param not in strategy_insights['parameters']:
                    strategy_insights['parameters'][param] = default_value
                # Ensure parameters are floats
                else:
                    try:
                        strategy_insights['parameters'][param] = float(strategy_insights['parameters'][param])
                    except (ValueError, TypeError):
                        strategy_insights['parameters'][param] = default_value

            # Create a copy of strategy_insights for prompt formatting
            prompt_strategy_insights = strategy_insights.copy()
            prompt_strategy_insights = self._convert_sets(prompt_strategy_insights)

            # Format dictionaries for prompt
            strategy_insights_str = json.dumps(prompt_strategy_insights, indent=2)
            market_context_str = json.dumps(market_context, indent=2)
            
            # Define available technical indicators for conditions
            available_indicators = {
                'price': 'Current price',
                'sma_20': '20-period Simple Moving Average',
                'sma_50': '50-period Simple Moving Average',
                'rsi': 'Relative Strength Index (14)',
                'macd.macd': 'MACD line',
                'macd.signal': 'MACD signal line',
                'bb.upper': 'Upper Bollinger Band',
                'bb.lower': 'Lower Bollinger Band',
                'williams_r': 'Williams %R'
            }
            
            # Add available indicators to prompt parameters
            prompt_params = {
                'parameters': json.dumps(self._convert_sets(strategy_insights["parameters"]), indent=2),
                'reasoning': strategy_insights.get("reasoning", ""),
                'available_indicators': json.dumps(available_indicators, indent=2),
                'strategy_insights': strategy_insights_str,
                'market_context': market_context_str,
                'entry_conditions': json.dumps(strategy_insights.get("entry_conditions", []), indent=2),
                'exit_conditions': json.dumps(strategy_insights.get("exit_conditions", []), indent=2)
            }

            # Format the prompt with available indicators
            formatted_prompt = self.prompt_manager.format_prompt(
                prompt,
                **prompt_params
            )

            # Generate rules using predictor
            result = self.predictor(
                strategy_insights=strategy_insights,
                market_context=market_context,
                prompt=formatted_prompt
            )

            # Parse the result - it MUST have the right structure
            if hasattr(result, 'entry_conditions'):
                # Direct access to result fields
                result_dict = {
                    'entry_conditions': result.entry_conditions,
                    'exit_conditions': result.exit_conditions,
                    'parameters': result.parameters,
                    'reasoning': result.reasoning,
                    'indicators': result.indicators
                }
            else:
                # This should NEVER happen with DSPy
                raise ValueError(f"Predictor returned invalid result type: {type(result)}. Expected Prediction object with all required fields.")

            # Get conditions from result - ALL fields MUST be present
            entry_conditions = result_dict['entry_conditions']
            exit_conditions = result_dict['exit_conditions']
            reasoning = result_dict['reasoning']
            indicators = result_dict['indicators']
            parameters = result_dict['parameters']
            
            # Validate indicators is not empty
            if not indicators or not isinstance(indicators, list):
                raise ValueError(f"LLM failed to return valid indicators list. Got: {indicators}")

            # If conditions are not lists, convert them into lists
            if not isinstance(entry_conditions, list):
                entry_conditions = [entry_conditions]
            if not isinstance(exit_conditions, list):
                exit_conditions = [exit_conditions]
            
            # Standardize and filter conditions using a validity check
            entry_conditions = [
                self._standardize_condition(str(c).strip())
                for c in entry_conditions
                if c and str(c).strip() and self._is_valid_condition(str(c).strip())
            ]
            exit_conditions = [
                self._standardize_condition(str(c).strip())
                for c in exit_conditions
                if c and str(c).strip() and self._is_valid_condition(str(c).strip())
            ]

            # Validate that we have conditions
            if not entry_conditions or not exit_conditions:
                raise ValueError(f"LLM returned empty conditions. Entry: {entry_conditions}, Exit: {exit_conditions}")
            
            conditions = {
                'entry': entry_conditions,
                'exit': exit_conditions
            }

            # Create response with validated data
            response = {
                'status': '',  # Not used by DSPy predictor
                'conditions': conditions,
                'parameters': parameters,  # Use the parameters from the LLM response
                'reasoning': str(reasoning),
                'indicators': indicators  # Already validated to be non-empty list
            }

            response = self._convert_sets(response)
            return response

        except Exception as e:
            logger.error(f"CRITICAL ERROR in trading rules generation: {str(e)}")
            logger.error("This error should NEVER happen. The LLM failed to return all required fields.")
            logger.error("Required fields: entry_conditions, exit_conditions, parameters, reasoning, indicators")
            # Re-raise the error - DO NOT HIDE IT
            raise e

    def _standardize_condition(self, condition: str) -> str:
        """Standardize a condition string.
        
        Args:
            condition: The condition string to standardize
            
        Returns:
            Standardized condition string
        """
        # Remove bullet markers if present
        if condition.startswith("- "):
            condition = condition[2:]
        
        # Remove any surrounding quotes
        condition = condition.strip('"\'')
        
        # Replace any newlines with spaces
        condition = condition.replace('\n', ' ')
        
        # Remove multiple spaces
        condition = ' '.join(condition.split())
        
        return condition

    def _convert_sets(self, obj: Any) -> Any:
        """Recursively convert sets to lists."""
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_sets(item) for item in obj]
        else:
            return obj 

    def _is_valid_condition(self, condition: str) -> bool:
        """Check if a condition string is valid.
        
        Args:
            condition: The condition string to check
            
        Returns:
            True if the condition contains a valid comparison operator and references valid indicators
        """
        # Check for comparison operators
        operators = ['>', '<', '==', '!=', '>=', '<=']
        has_operator = any(op in condition for op in operators)
        
        # Check for valid indicators
        valid_indicators = [
            'price', 'sma_20', 'sma_50', 'rsi', 
            'macd.macd', 'macd.signal', 
            'bb.upper', 'bb.lower', 'williams_r'
        ]
        has_indicator = any(indicator in condition for indicator in valid_indicators)
        
        return has_operator and has_indicator

    def _get_default_conditions(self) -> Dict[str, Any]:
        """Get default trading conditions.
        
        Returns:
            Dictionary containing default conditions
        """
        return {
            'entry_conditions': [
                "price > sma_20",
                "rsi < 70",
                "macd.macd > macd.signal"
            ],
            'exit_conditions': [
                "price < sma_20",
                "rsi > 30",
                "macd.macd < macd.signal"
            ],
            'parameters': {
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'position_size': 0.1
            },
            'reasoning': "Default strategy using trend following with momentum confirmation",
            'indicators': ["sma_20", "rsi", "macd"]
        }

def apply_trading_rules(
    data: pd.DataFrame,
    conditions: Optional[Dict[str, List[str]]] = None,
    **params
) -> Optional[Dict[str, Any]]:
    """Apply trading rules to price data.
    
    Args:
        data: DataFrame containing price data
        conditions: Dictionary of entry and exit conditions
        **params: Additional parameters for trading rules
        
    Returns:
        Dictionary containing trading results
    """
    try:
        # Add technical indicators
        data = add_indicators(data)
        
        # Initialize signals
        data["buy_signal"] = False
        data["sell_signal"] = False
        
        # Apply entry conditions
        if conditions and "entry" in conditions:
            for condition in conditions["entry"]:
                try:
                    data.loc[eval(f"data.{condition}"), "buy_signal"] = True
                except Exception as e:
                    logger.warning(f"Error evaluating entry condition '{condition}': {str(e)}")
                    
        # Apply exit conditions
        if conditions and "exit" in conditions:
            for condition in conditions["exit"]:
                try:
                    data.loc[eval(f"data.{condition}"), "sell_signal"] = True
                except Exception as e:
                    logger.warning(f"Error evaluating exit condition '{condition}': {str(e)}")
                    
        # Apply trading logic
        position = 0
        entry_price = 0
        trades = []
        total_pnl = 0
        
        for i in range(len(data)):
            if position == 0 and data.iloc[i]["buy_signal"]:
                position = 1
                entry_price = data.iloc[i]["close"]
                trades.append({
                    "type": "buy",
                    "price": entry_price,
                    "timestamp": data.iloc[i]["timestamp"]
                })
            elif position == 1 and data.iloc[i]["sell_signal"]:
                exit_price = data.iloc[i]["close"]
                pnl = (exit_price - entry_price) / entry_price
                total_pnl += pnl
                position = 0
                trades.append({
                    "type": "sell",
                    "price": exit_price,
                    "timestamp": data.iloc[i]["timestamp"],
                    "pnl": pnl
                })
                
        # Calculate statistics
        if trades:
            win_trades = len([t for t in trades if t.get("pnl", 0) > 0])
            total_trades = len([t for t in trades if "pnl" in t])
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            # Calculate Sortino ratio
            returns = pd.Series([t.get("pnl", 0) for t in trades if "pnl" in t])
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1
            sortino_ratio = (returns.mean() * np.sqrt(252)) / downside_std if downside_std != 0 else 0
            
            results = {
                "total_return": total_pnl,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "sortino_ratio": sortino_ratio,
                "trades": trades
            }
            
            return results
            
        return None
        
    except Exception as e:
        logger.error(f"Error applying trading rules: {str(e)}")
        return None
        
def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to price data.
    
    Args:
        data: DataFrame containing price data
        
    Returns:
        DataFrame with added indicators
    """
    try:
        # Add SMA indicators
        data["sma_20"] = ta.trend.sma_indicator(data["close"], window=20)
        data["sma_50"] = ta.trend.sma_indicator(data["close"], window=50)
        data["sma_200"] = ta.trend.sma_indicator(data["close"], window=200)
        
        # Add RSI
        data["rsi"] = ta.momentum.rsi(data["close"], window=14)
        
        # Add MACD
        macd = ta.trend.MACD(data["close"])
        data["macd"] = macd.macd()
        data["macd_signal"] = macd.macd_signal()
        data["macd_diff"] = macd.macd_diff()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data["close"])
        data["bb_high"] = bollinger.bollinger_hband()
        data["bb_low"] = bollinger.bollinger_lband()
        data["bb_mid"] = bollinger.bollinger_mavg()
        
        return data
        
    except Exception as e:
        logger.error(f"Error adding indicators: {str(e)}")
        return data 
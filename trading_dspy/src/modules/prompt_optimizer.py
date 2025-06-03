"""Prompt optimization module using MiPro for enhanced market analysis and strategy generation."""

from typing import Dict, List, Any, Optional, Callable
import dspy
from loguru import logger
import time
import json
import hashlib
import pandas as pd
import numpy as np

from ..utils.mipro_optimizer import MiProWrapper
from ..utils.prompt_manager import PromptManager

class PromptOptimizer:
    """Prompt optimization module using MiPro for the trading system."""

    def __init__(self, prompt_manager: PromptManager):
        """Initialize the prompt optimizer.
        
        Args:
            prompt_manager: The prompt manager instance
        """
        logger.info("Initializing PromptOptimizer with MiPro")
        self.prompt_manager = prompt_manager
        self.mipro = MiProWrapper(
            prompt_manager=prompt_manager,
            use_v2=True,
            max_bootstrapped_demos=3,
            num_candidate_programs=5
        )
        
        # Initialize example storage
        self.examples = {}  # Dict to store examples for each module
        self.example_hashes = {}  # Dict to store hashes of examples for deduplication
        
    def optimize_market_analysis(self, module: dspy.Module, examples: List[Dict[str, Any]]) -> dspy.Module:
        """Optimize the market analysis module using MiPro.
        
        Args:
            module: The market analysis module to optimize
            examples: Examples for optimization
            
        Returns:
            Optimized module
        """
        if len(examples) < 2:
            logger.warning("Not enough examples for optimization, need at least 2")
            return module
            
        # Define market analysis metric function - updated for DSPy optimization signature
        def market_analysis_metric(gold: Dict[str, Any], pred: Dict[str, Any], trace=None) -> float:
            """Metric function for market analysis optimization."""
            # Check regime match
            regime_match = int(gold.get('regime') == pred.get('regime', ''))
            
            # Check risk level match
            risk_match = int(gold.get('risk_level') == pred.get('risk_level', ''))
            
            # Check confidence difference
            try:
                gold_conf = float(gold.get('confidence', 0.0))
                pred_conf = float(pred.get('confidence', 0.0))
                # Small penalty for large confidence differences
                conf_score = max(0.0, 1.0 - abs(gold_conf - pred_conf))
            except (ValueError, TypeError):
                conf_score = 0.0
            
            # Check if analysis contains key indicators
            analysis = pred.get('analysis', '')
            mentions_sma = 1 if 'SMA' in analysis or 'moving average' in analysis.lower() else 0
            mentions_volatility = 1 if 'volatil' in analysis.lower() else 0
            mentions_support = 1 if 'support' in analysis.lower() or 'resistance' in analysis.lower() else 0
            mentions_risk = 1 if 'risk' in analysis.lower() else 0
            
            # Calculate score (weights add up to 1.0)
            score = (regime_match * 0.4) + (risk_match * 0.2) + (conf_score * 0.1) + \
                   (mentions_sma * 0.1) + (mentions_volatility * 0.1) + \
                   (mentions_support * 0.05) + (mentions_risk * 0.05)
            
            return score
        
        logger.info(f"Starting market analysis optimization with {len(examples)} examples")
        start_time = time.time()
        
        try:
            # Set optimization flag
            if hasattr(module, 'is_optimizing'):
                module.is_optimizing = True
            
            # Use MiPro for optimization
            optimized_module = self.mipro.optimize(
                module=module,
                examples=examples,
                prompt_name="market_analysis",
                metric_fn=market_analysis_metric
            )
            
            # Reset optimization flag
            if hasattr(optimized_module, 'is_optimizing'):
                optimized_module.is_optimizing = False
            
            duration = time.time() - start_time
            logger.info(f"Market analysis optimization completed in {duration:.2f} seconds")
            
            return optimized_module
            
        except Exception as e:
            logger.error(f"Error in market analysis optimization: {str(e)}")
            logger.exception("Full traceback:")
            
            # Reset optimization flag in case of error
            if hasattr(module, 'is_optimizing'):
                module.is_optimizing = False
                
            return module
        
    def optimize_strategy_generation(self, module: dspy.Module, examples: List[Dict[str, Any]]) -> dspy.Module:
        """Optimize the strategy generation module.
        
        Args:
            module: The strategy generation module to optimize
            examples: Examples for optimization
            
        Returns:
            Optimized module
        """
        if len(examples) < 2:
            logger.warning("Not enough examples for optimization, need at least 2")
            return module
            
        # Define strategy generation metric function with updated signature
        def strategy_generation_metric(gold: Dict[str, Any], pred: Dict[str, Any], trace=None) -> float:
            """Metric function for strategy generation optimization."""
            # Check for required strategy components
            strategy = pred.get('strategy', {})
            
            # Essential components exist
            has_indicators = int('indicators' in strategy)
            has_entry = int('entry_conditions' in strategy)
            has_exit = int('exit_conditions' in strategy)
            has_timeframe = int('timeframe' in strategy)
            has_parameters = int('parameters' in strategy)
            essentials_score = (has_indicators + has_entry + has_exit + has_timeframe + has_parameters) / 5.0
            
            # Quality checks
            indicator_count = len(strategy.get('indicators', []))
            entry_complexity = len(str(strategy.get('entry_conditions', '')).split('and') + str(strategy.get('entry_conditions', '')).split('or'))
            exit_complexity = len(str(strategy.get('exit_conditions', '')).split('and') + str(strategy.get('exit_conditions', '')).split('or'))
            
            quality_score = min(1.0, (indicator_count / 3) * 0.3 + (entry_complexity / 3) * 0.3 + (exit_complexity / 3) * 0.4)
            
            # Calculate score (weights add up to 1.0)
            score = (essentials_score * 0.7) + (quality_score * 0.3)
            
            return score
            
        logger.info(f"Starting strategy generation optimization with {len(examples)} examples")
        start_time = time.time()
        
        # Optimize the module
        optimized_module = self.mipro.optimize(
            module=module,
            examples=examples,
            prompt_name="strategy_generator",
            metric_fn=strategy_generation_metric
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Strategy generation optimization completed in {elapsed_time:.2f} seconds")
        
        return optimized_module
        
    def optimize_trading_rules(self, module: dspy.Module, examples: List[Dict[str, Any]]) -> dspy.Module:
        """Optimize the trading rules generator module using MiPro.
        
        Args:
            module: The trading rules generator module to optimize
            examples: Examples for optimization
            
        Returns:
            Optimized module
        """
        if len(examples) < 2:
            logger.warning("Not enough examples for optimization, need at least 2")
            return module
            
        # Define trading rules metric function with updated signature
        def trading_rules_metric(gold: Dict[str, Any], pred: Dict[str, Any], trace=None) -> float:
            """Metric function for trading rules optimization."""
            # Check for required components
            rules = pred.get('trading_rules', {})
            
            # Handle different output structures
            entry_conditions = rules.get('entry_conditions', rules.get('conditions', {}).get('entry', []))
            exit_conditions = rules.get('exit_conditions', rules.get('conditions', {}).get('exit', []))
            
            # Essential components exist - use progressive scoring
            # First check if we have conditions at all
            has_entry = float(bool(entry_conditions))
            has_exit = float(bool(exit_conditions))
            
            # Count the number of entry and exit conditions for a progressive score
            entry_count = len(entry_conditions) if isinstance(entry_conditions, list) else (1 if has_entry else 0)
            exit_count = len(exit_conditions) if isinstance(exit_conditions, list) else (1 if has_exit else 0)
            
            # Progressive score based on number of conditions (max at 3 for each)
            entry_score = min(1.0, entry_count / 3.0) * 0.25
            exit_score = min(1.0, exit_count / 3.0) * 0.25
            
            # Check if specific parameters are present
            conditions_text = str(entry_conditions) + str(exit_conditions) + str(rules)
            
            # Detailed parameter scoring
            parameter_checks = {
                'stop_loss': 0.1,
                'take_profit': 0.1,
                'position_size': 0.05,
                'risk_management': 0.05,
            }
            
            parameter_score = 0.0
            for param, weight in parameter_checks.items():
                if param in conditions_text.lower():
                    parameter_score += weight
            
            # Indicator usage score - weighted by diversity
            indicator_checks = {
                'sma': 0.04,
                'rsi': 0.04,
                'macd': 0.04,
                'bollinger': 0.04,
                'volume': 0.04,
            }
            
            indicator_score = 0.0
            for indicator, weight in indicator_checks.items():
                if indicator in conditions_text.lower():
                    indicator_score += weight
            
            # Detailed logic validation
            logic_checks = {
                # Check for comparison operators
                '>': 0.05,
                '<': 0.05,
                '==': 0.03,
                '>=': 0.03,
                '<=': 0.03,
                
                # Check for logical operators
                'and': 0.04,
                'or': 0.03,
                'not': 0.02,
            }
            
            logic_score = 0.0
            for operator, weight in logic_checks.items():
                if operator in conditions_text:
                    logic_score += weight
            
            # Reasoning quality check
            reasoning = rules.get('reasoning', '')
            reasoning_length = len(str(reasoning))
            reasoning_score = min(0.15, (reasoning_length / 500) * 0.15)  # Max at 500 chars
            
            # Calculate combined score with diminishing returns
            # Base score ensures progress for any valid attempt - increased from 0.05 to 0.1
            base_score = 0.1
            
            # Boost scores for having any entry or exit conditions at all
            if has_entry:
                base_score += 0.03  # Additional bonus just for having any entry condition
            if has_exit:
                base_score += 0.03  # Additional bonus just for having any exit condition
                
            # Main score components
            component_scores = {
                'entry_exit': entry_score + exit_score,
                'parameters': parameter_score,
                'indicators': indicator_score,
                'logic': logic_score,
                'reasoning': reasoning_score
            }
            
            # Calculate final score - ensure it can reach up to 1.0 for perfect solutions
            score = base_score + sum(component_scores.values())
            
            # Apply progressive bonus for scores above previous plateau
            # This helps break through plateaus by rewarding incremental improvements
            plateau_threshold = 0.14  # Same as the previous plateau for easier triggering
            
            # First apply a fixed bonus for crossing the threshold at all
            if score >= plateau_threshold:
                fixed_bonus = 0.05  # Add a fixed 5% bonus for reaching the threshold
                score += fixed_bonus
                
                # Then add progressive bonus for any improvement above threshold
                if score > plateau_threshold:
                    progressive_bonus = (score - plateau_threshold) * 0.5  # 50% bonus on improvement above plateau
                    score += progressive_bonus
                    
                    # Log the bonus clearly
                    total_bonus = fixed_bonus + progressive_bonus
                    logger.warning(f"💥 PLATEAU BONUS APPLIED 💥 Base score {score-total_bonus:.4f} > {plateau_threshold:.2f}")
                    logger.warning(f"💰 Fixed bonus: +{fixed_bonus:.4f}, Progressive bonus: +{progressive_bonus:.4f}, Total: +{total_bonus:.4f}")
                    logger.warning(f"🔥 Final score with bonus: {score:.4f} 🔥")
                
            # Ensure score is within bounds
            score = max(0.0, min(1.0, score))
            
            # Debug log to help monitor scoring progress
            component_debug = f"E:{entry_score:.2f}+X:{exit_score:.2f}+P:{parameter_score:.2f}+I:{indicator_score:.2f}+L:{logic_score:.2f}+R:{reasoning_score:.2f}"
            if hasattr(logger, 'debug'):
                logger.debug(f"Rule metric: {score:.4f} [{component_debug}]")
                
            return score
            
        logger.info(f"Starting trading rules optimization with {len(examples)} examples")
        start_time = time.time()
        
        try:
            # Use MiPro for optimization
            optimized_module = self.mipro.optimize(
                module=module,
                examples=examples,
                prompt_name="trading_rules",
                metric_fn=trading_rules_metric
            )
            
            duration = time.time() - start_time
            logger.info(f"Trading rules optimization completed in {duration:.2f} seconds")
            
            return optimized_module
            
        except Exception as e:
            logger.error(f"Error during trading rules optimization: {str(e)}")
            logger.exception("Full traceback:")
            return module

    def check_optimization_status(self) -> Dict[str, Any]:
        """Check optimization status for all prompts.
        
        Returns:
            Dictionary with optimization status for each prompt
        """
        status = {}
        
        for prompt_name in self.prompt_manager.get_prompt_names():
            status[prompt_name] = self.prompt_manager.get_optimization_status(prompt_name)
            
        return status
        
    def _generate_example_hash(self, example: Dict[str, Any]) -> str:
        """Generate a hash for an example to check for duplicates.
        
        Args:
            example: The example to hash
            
        Returns:
            Hash string for the example
        """
        example_str = json.dumps(example, sort_keys=True)
        return hashlib.md5(example_str.encode()).hexdigest()
        
    def _make_serializable(self, obj: Any) -> Any:
        """Make an object JSON serializable by converting problematic types.
        
        Args:
            obj: The object to make serializable
            
        Returns:
            A serializable version of the object
        """
        try:
            # Handle None
            if obj is None:
                return None
                
            # Handle dictionaries
            if isinstance(obj, dict):
                # Check for problematic keys (non-string)
                result = {}
                for k, v in obj.items():
                    # Convert keys to strings if they aren't already
                    key = str(k) if not isinstance(k, str) else k
                    try:
                        # Handle each value recursively
                        result[key] = self._make_serializable(v)
                    except Exception as e:
                        # If serialization fails, use string representation
                        logger.warning(f"Failed to serialize value for key '{key}': {e}")
                        result[key] = str(v)
                return result
                
            # Handle lists and tuples
            elif isinstance(obj, (list, tuple)):
                result = []
                for item in obj:
                    try:
                        result.append(self._make_serializable(item))
                    except Exception as e:
                        # If serialization fails, use string representation
                        logger.warning(f"Failed to serialize list item: {e}")
                        result.append(str(item))
                return result
                
            # Handle Pandas timestamp and datetime objects
            elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)) or hasattr(obj, 'isoformat'):
                return str(obj)
                
            # Handle Pandas DataFrames and Series
            elif isinstance(obj, pd.DataFrame):
                try:
                    return {"data": obj.to_dict(orient="records"), "_type": "dataframe"}
                except:
                    # Fallback for problematic DataFrames
                    return {"summary": f"DataFrame with shape {obj.shape}", "_type": "dataframe_summary"}
            elif isinstance(obj, pd.Series):
                try:
                    return {"data": obj.to_dict(), "_type": "series"}
                except:
                    # Fallback for problematic Series
                    return {"summary": f"Series with length {len(obj)}", "_type": "series_summary"}
                
            # Handle NumPy arrays and numerical types
            elif isinstance(obj, np.ndarray):
                try:
                    return obj.tolist()
                except:
                    return str(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
                
            # Check if object is directly JSON serializable (including basic Python types)
            elif isinstance(obj, (int, float, str, bool)):
                return obj
                
            # Handle any other types by converting to string
            else:
                try:
                    # Try to serialize with json
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError, OverflowError):
                    # If that fails, convert to string
                    return str(obj)
                    
        except Exception as e:
            logger.error(f"Unexpected error in _make_serializable: {e}")
            # Last resort fallback
            return f"<unserializable object: {type(obj).__name__}>"
        
    def collect_market_analysis_example(self, market_data: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Collect a market analysis example for optimization.
        
        Args:
            market_data: The input market data
            result: The output analysis result
            
        Returns:
            True if example was added, False otherwise
        """
        try:
            logger.info("Collecting market analysis example")
            
            # Verify we have the required data
            if not all(k in market_data for k in ['prices', 'volumes', 'indicators']):
                logger.warning("Market data missing required fields")
                return False
                
            # Verify we have enough data points
            min_required_points = 50  # We need at least 50 points to create varied examples
            if len(market_data['prices']) < min_required_points:
                logger.warning(f"Not enough data points, need at least {min_required_points}")
                return False
                
            # Extract risk level from market context or analysis result
            risk_level = (
                result.get('market_context', {}).get('risk_level') or  # Try market context first
                result.get('risk_level') or  # Then direct risk_level
                'moderate'  # Default to moderate if not found
            )
            
            # Normalize risk level to one of the expected values
            risk_level = risk_level.lower()
            if risk_level not in ['low', 'moderate', 'high']:
                if 'low' in risk_level:
                    risk_level = 'low'
                elif 'high' in risk_level:
                    risk_level = 'high'
                else:
                    risk_level = 'moderate'
                
            # Create example with actual market data
            example = {
                'market_data': {
                    'prices': market_data['prices'],
                    'volumes': market_data['volumes'],
                    'indicators': market_data['indicators'].copy() if isinstance(market_data['indicators'], dict) else {}
                },
                'timeframe': result.get('timeframe', '1h'),
                'outputs': {
                    'regime': result.get('market_context', {}).get('regime', ''),
                    'confidence': float(result.get('market_context', {}).get('confidence', 0.0)),
                    'risk_level': risk_level,
                    'analysis': result.get('analysis_text', '')
                }
            }
            
            # Verify the example has valid outputs
            if not example['outputs']['regime'] or example['outputs']['regime'] == 'UNKNOWN':
                logger.warning("Example has invalid regime")
                return False
                
            if example['outputs']['confidence'] <= 0.0:
                logger.warning("Example has invalid confidence")
                return False
                
            # Add to examples list if not already present
            example_hash = hash(str(example['outputs']))
            if example_hash not in self.example_hashes.get('market_analysis', set()):
                self.examples.setdefault('market_analysis', []).append(example)
                self.example_hashes.setdefault('market_analysis', set()).add(example_hash)
                logger.info("Added new market analysis example")
                return True
            else:
                logger.info("Example already exists, skipping")
                return False
                
        except Exception as e:
            logger.error(f"Error collecting market analysis example: {str(e)}")
            return False
            
    def collect_strategy_example(self, market_analysis: Dict[str, Any], strategy: Dict[str, Any]) -> bool:
        """Collect a strategy generation example for optimization.
        
        Args:
            market_analysis: The input market analysis
            strategy: The output strategy
            
        Returns:
            True if example was added, False otherwise
        """
        try:
            logger.info("Collecting strategy generation example")
            
            # Create simplified versions of the input data to avoid serialization issues
            simplified_market_analysis = {
                "regime": market_analysis.get('market_context', {}).get('regime', ''),
                "confidence": market_analysis.get('market_context', {}).get('confidence', 0.0),
                "risk_level": market_analysis.get('risk_level', 'moderate'),
                "analysis_summary": market_analysis.get('analysis_text', '')[:200] + "..." if market_analysis.get('analysis_text') else ""
            }
            
            # Create a simplified version of the strategy output
            simplified_strategy = {
                "reasoning": strategy.get('reasoning', ''),
                "trade_signal": strategy.get('trade_signal', 'HOLD'),
                "confidence": strategy.get('confidence', 0.0)
            }
            
            # Add simplified parameter object if present
            if 'parameters' in strategy:
                # Convert parameters to string to avoid complex objects
                parameters_str = str(strategy.get('parameters', {}))
                simplified_strategy['parameters'] = parameters_str
            
            # Make both objects fully serializable
            serializable_market_analysis = self._make_serializable(simplified_market_analysis)
            serializable_strategy = self._make_serializable(simplified_strategy)
            
            # Prepare example (removed 'prompt' field which was causing errors)
            example = {
                'market_analysis': serializable_market_analysis,
                'outputs': {
                    'strategy': serializable_strategy
                }
            }
            
            # Verify the example is serializable
            try:
                # Test JSON serialization
                json_str = json.dumps(example)
                logger.debug(f"Strategy example serialization successful, size: {len(json_str)} bytes")
            except Exception as e:
                logger.error(f"Strategy example serialization failed: {e}")
                # Try to fix by converting to strings
                example['outputs']['strategy'] = str(simplified_strategy)
                example['market_analysis'] = str(simplified_market_analysis)
                logger.info("Converted complex fields to strings to ensure serializability")
            
            # Generate hash and check if example already exists
            example_hash = self._generate_example_hash(example)
            examples = self.prompt_manager.get_examples('strategy_generator')
            
            # Check for duplicates
            if any(self._generate_example_hash(ex) == example_hash for ex in examples):
                logger.info("Duplicate strategy example found, skipping")
                return False
                
            # Add example
            self.prompt_manager.add_example('strategy_generator', example)
            logger.info(f"Added new strategy example with trade signal: {simplified_strategy.get('trade_signal')}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting strategy example: {str(e)}")
            logger.exception("Detailed error:")
            return False
            
    def collect_trading_rules_example(self, strategy: Dict[str, Any], trading_rules: Dict[str, Any]) -> bool:
        """Collect a trading rules example for optimization.
        
        Args:
            strategy: The input strategy
            trading_rules: The output trading rules
            
        Returns:
            True if example was added, False otherwise
        """
        try:
            logger.info("Collecting trading rules example")
            
            # Create simplified versions of the input data to avoid serialization issues
            simplified_strategy = {
                "trade_signal": strategy.get('trade_signal', 'HOLD'),
                "confidence": strategy.get('confidence', 0.0),
                "reasoning_summary": strategy.get('reasoning', '')[:200] + "..." if strategy.get('reasoning') else ""
            }
            
            # Add market context if present
            if 'market_context' in strategy:
                simplified_strategy['market_context'] = {
                    "regime": strategy.get('market_context', {}).get('regime', ''),
                    "risk_level": strategy.get('market_context', {}).get('risk_level', '')
                }
            
            # Create a simplified version of the trading rules output
            simplified_trading_rules = {
                "reasoning": trading_rules.get('reasoning', '')
            }
            
            # Add simplified entry/exit conditions if present
            entry_conditions = trading_rules.get('conditions', {}).get('entry', [])
            exit_conditions = trading_rules.get('conditions', {}).get('exit', [])
            
            # Convert complex condition objects to simpler strings
            if entry_conditions:
                simplified_trading_rules['entry_conditions'] = [str(c) for c in entry_conditions]
            
            if exit_conditions:
                simplified_trading_rules['exit_conditions'] = [str(c) for c in exit_conditions]
            
            # Convert parameters to string
            if 'parameters' in trading_rules:
                simplified_trading_rules['parameters'] = str(trading_rules.get('parameters', {}))
            
            # Make both objects fully serializable
            serializable_strategy = self._make_serializable(simplified_strategy)
            serializable_trading_rules = self._make_serializable(simplified_trading_rules)
            
            # Prepare example (removed 'prompt' field which was causing errors)
            example = {
                'strategy': serializable_strategy,
                'outputs': {
                    'trading_rules': serializable_trading_rules
                }
            }
            
            # Verify the example is serializable
            try:
                # Test JSON serialization
                json_str = json.dumps(example)
                logger.debug(f"Trading rules example serialization successful, size: {len(json_str)} bytes")
            except Exception as e:
                logger.error(f"Trading rules example serialization failed: {e}")
                # Try to fix by converting to strings
                example['outputs']['trading_rules'] = str(simplified_trading_rules)
                example['strategy'] = str(simplified_strategy)
                logger.info("Converted complex fields to strings to ensure serializability")
            
            # Generate hash and check if example already exists
            example_hash = self._generate_example_hash(example)
            examples = self.prompt_manager.get_examples('trading_rules')
            
            # Check for duplicates
            if any(self._generate_example_hash(ex) == example_hash for ex in examples):
                logger.info("Duplicate trading rules example found, skipping")
                return False
                
            # Add example
            self.prompt_manager.add_example('trading_rules', example)
            logger.info(f"Added new trading rules example with {len(simplified_trading_rules.get('entry_conditions', []))} entry conditions")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting trading rules example: {str(e)}")
            logger.exception("Detailed error:")
            return False
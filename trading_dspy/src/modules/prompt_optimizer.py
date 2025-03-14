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
        
    def optimize_market_analysis(self, module: dspy.Module, examples: List[Dict[str, Any]]) -> dspy.Module:
        """Optimize the market analysis module.
        
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
            
            # Check if analysis contains key indicators
            analysis = pred.get('analysis', '')
            mentions_sma = 1 if 'SMA' in analysis or 'moving average' in analysis.lower() else 0
            mentions_volatility = 1 if 'volatil' in analysis.lower() else 0
            mentions_support = 1 if 'support' in analysis.lower() or 'resistance' in analysis.lower() else 0
            
            # Calculate score (weights add up to 1.0)
            score = (regime_match * 0.5) + (risk_match * 0.3) + (mentions_sma * 0.1) + (mentions_volatility * 0.05) + (mentions_support * 0.05)
            
            return score
        
        logger.info(f"Starting market analysis optimization with {len(examples)} examples")
        start_time = time.time()
        
        try:
            # Create direct DSPy examples instead of going through the prepare_examples method
            dspy_examples = []
            for ex in examples:
                try:
                    # Extract outputs
                    outputs = ex.get('outputs', {})
                    
                    # First create empty example
                    example = dspy.Example()
                    
                    # Set input attributes directly with proper structure to avoid KeyError
                    example.market_data = {
                        "prices": [100.0, 101.0, 102.0, 101.5, 102.5],
                        "volumes": [1000, 1200, 900, 1100, 1300],
                        "indicators": {
                            "sma_20": [99.0, 100.0, 101.0, 101.2, 101.8],
                            "sma_50": [95.0, 96.0, 97.0, 98.0, 99.0],
                            "rsi": [55, 60, 65, 58, 62],
                            "volatility": [0.02, 0.025, 0.022, 0.018, 0.02]
                        }
                    }
                    example.timeframe = ex.get('timeframe', '1h')
                    
                    # Mark which ones are inputs - removed 'prompt' as it's causing errors
                    example = example.with_inputs('market_data', 'timeframe')
                    
                    # Set output attributes directly
                    example.regime = outputs.get('regime', '')
                    example.confidence = float(outputs.get('confidence', 0.0))
                    example.risk_level = outputs.get('risk_level', '')
                    example.analysis = outputs.get('analysis', '')
                    
                    # Verify the example has at least some input fields set
                    input_fields = example.inputs()
                    if len(input_fields) == 0:
                        logger.warning(f"Example has no input fields, skipping")
                        continue
                    
                    dspy_examples.append(example)
                    
                except Exception as e:
                    logger.error(f"Error creating direct example: {e}")
                    # Continue to next example
                
            logger.info(f"Created {len(dspy_examples)} direct examples for market analysis")
            
            # Use train/validation split
            if len(dspy_examples) <= 3:
                trainset = dspy_examples[:1]
                valset = dspy_examples[1:]
            else:
                split_idx = max(1, int(len(dspy_examples) * 0.7))
                trainset = dspy_examples[:split_idx]
                valset = dspy_examples[split_idx:]
                
            # Run direct MiPro optimization
            optimizer = dspy.teleprompt.MIPROv2(
                metric=market_analysis_metric,
                num_candidates=5,
                init_temperature=0.7,
            )
            
            # Run optimization directly
            optimized_module = optimizer.compile(
                student=module,
                trainset=trainset,
                valset=valset,
                minibatch_size=max(1, len(valset)-1),
                requires_permission_to_run=False,
                num_trials=5,
                minibatch_full_eval_steps=2
            )
            
            # Store the optimized prompt
            if hasattr(optimized_module, 'prompt') and self.prompt_manager is not None:
                optimized_prompt = getattr(optimized_module, 'prompt')
                if optimized_prompt and isinstance(optimized_prompt, str):
                    # Store the optimized prompt
                    self.prompt_manager.update_prompt("market_analysis", optimized_prompt)
                    logger.info(f"Updated prompt 'market_analysis' with optimized version")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Direct market analysis optimization completed in {elapsed_time:.2f} seconds")
            return optimized_module
            
        except Exception as e:
            logger.error(f"Error in direct optimization: {str(e)}")
            logger.info("Falling back to generic optimization")
            
            # Optimize the module using the general method as fallback
            optimized_module = self.mipro.optimize(
                module=module,
                examples=examples,
                prompt_name="market_analysis",
                metric_fn=market_analysis_metric
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Market analysis optimization completed in {elapsed_time:.2f} seconds")
            
            return optimized_module
        
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
        """Optimize the trading rules generator module.
        
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
            
            # Essential components exist
            has_entry = int('entry_rule' in rules)
            has_exit = int('exit_rule' in rules)
            has_stop_loss = int('stop_loss' in rules) or "stop_loss" in str(rules.get('exit_rule', ''))
            has_take_profit = int('take_profit' in rules) or "take_profit" in str(rules.get('exit_rule', ''))
            
            essentials_score = (has_entry + has_exit + has_stop_loss + has_take_profit) / 4.0
            
            # Validate rules are executable
            try:
                # Check for basic syntax components - not a full check but helps
                entry_valid = int('>' in str(rules.get('entry_rule', '')) or '<' in str(rules.get('entry_rule', '')) or '==' in str(rules.get('entry_rule', '')))
                exit_valid = int('>' in str(rules.get('exit_rule', '')) or '<' in str(rules.get('exit_rule', '')) or '==' in str(rules.get('exit_rule', '')))
                validity_score = (entry_valid + exit_valid) / 2.0
            except:
                validity_score = 0.0
                
            # Calculate score (weights add up to 1.0)
            score = (essentials_score * 0.7) + (validity_score * 0.3)
            
            return score
            
        logger.info(f"Starting trading rules optimization with {len(examples)} examples")
        start_time = time.time()
        
        # Optimize the module
        optimized_module = self.mipro.optimize(
            module=module,
            examples=examples,
            prompt_name="trading_rules",
            metric_fn=trading_rules_metric
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Trading rules optimization completed in {elapsed_time:.2f} seconds")
        
        return optimized_module

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
            
            # Create a simplified version of market data with proper structure matching what market_analysis expects
            simplified_market_data = {
                "prices": market_data.get('prices', [])[-10:] if 'prices' in market_data else [100.0, 101.0, 102.0, 101.5, 102.5],
                "volumes": market_data.get('volumes', [])[-10:] if 'volumes' in market_data else [1000, 1200, 900, 1100, 1300],
                "indicators": {
                    "sma_20": market_data.get('indicators', {}).get('sma_20', [])[-10:] if 'indicators' in market_data and 'sma_20' in market_data.get('indicators', {}) else [99.0, 100.0, 101.0, 101.2, 101.8],
                    "sma_50": market_data.get('indicators', {}).get('sma_50', [])[-10:] if 'indicators' in market_data and 'sma_50' in market_data.get('indicators', {}) else [95.0, 96.0, 97.0, 98.0, 99.0],
                    "rsi": market_data.get('indicators', {}).get('rsi', [])[-10:] if 'indicators' in market_data and 'rsi' in market_data.get('indicators', {}) else [55, 60, 65, 58, 62],
                    "volatility": market_data.get('indicators', {}).get('volatility', [])[-10:] if 'indicators' in market_data and 'volatility' in market_data.get('indicators', {}) else [0.02, 0.025, 0.022, 0.018, 0.02]
                }
            }
            
            # Create serializable versions of the data
            serializable_market_data = self._make_serializable(simplified_market_data)
            
            # Extract outputs, ensuring they are properly formatted
            regime = result.get('market_context', {}).get('regime', '')
            confidence = float(result.get('market_context', {}).get('confidence', 0.0))
            risk_level = result.get('risk_level', 'moderate')
            analysis = result.get('analysis_text', '')
            
            # Prepare example (removed 'prompt' field which was causing errors)
            example = {
                'market_data': serializable_market_data,
                'timeframe': '1h',
                'outputs': {
                    'regime': regime,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'analysis': analysis
                }
            }
            
            # Verify the example is serializable
            try:
                # Test JSON serialization
                json_str = json.dumps(example)
                logger.debug(f"Example serialization successful, size: {len(json_str)} bytes")
            except Exception as e:
                logger.error(f"Example serialization failed: {e}")
                # Try to fix by converting problematic fields to strings
                example['outputs']['confidence'] = str(confidence)
                example['market_data'] = str(simplified_market_data)
                logger.info("Converted complex fields to strings to ensure serializability")
            
            # Generate hash and check if example already exists
            example_hash = self._generate_example_hash(example)
            examples = self.prompt_manager.get_examples('market_analysis')
            
            # Check for duplicates
            if any(self._generate_example_hash(ex) == example_hash for ex in examples):
                logger.info("Duplicate market analysis example found, skipping")
                return False
                
            # Add example
            self.prompt_manager.add_example('market_analysis', example)
            logger.info(f"Added new market analysis example with regime: {regime}, confidence: {confidence}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting market analysis example: {str(e)}")
            logger.exception("Detailed error:")
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
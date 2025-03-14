"""MiPro (Mixed Prompt Optimization) wrapper for trading system.

This module provides a wrapper around DSPy's built-in MiPro optimizer.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import time
from loguru import logger
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2  # MIPRO is now deprecated in newer DSPy versions

class MiProWrapper:
    """Wrapper for DSPy's MiPro optimizer for enhancing trading system modules."""
    
    def __init__(self, 
                prompt_manager=None,
                use_v2: bool = True,
                max_bootstrapped_demos: int = 3,
                num_candidate_programs: int = 10,
                temperature: float = 0.7):
        """Initialize the MiPro wrapper.
        
        Args:
            prompt_manager: Optional prompt manager to store optimized prompts
            use_v2: Whether to use MIPROv2 (recommended) or the original MIPRO
            max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
            num_candidate_programs: Number of candidate programs to generate
            temperature: Temperature for generation
        """
        self.prompt_manager = prompt_manager
        self.use_v2 = use_v2
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.num_candidate_programs = num_candidate_programs
        self.temperature = temperature
        self.optimized_modules = {}
        
    def prepare_examples(self, examples: List[Dict[str, Any]]) -> List[dspy.Example]:
        """Prepare examples for optimization.
        
        Args:
            examples: List of examples with inputs and outputs
            
        Returns:
            List of DSPy Example objects
        """
        # Validate examples
        if not examples or len(examples) < 2:
            raise ValueError("Need at least 2 examples for optimization")
            
        # Create dspy examples
        dspy_examples = []
        for ex in examples:
            try:
                # Check for market analysis examples format
                if 'market_data' in ex and 'timeframe' in ex:
                    # Handle market analysis example specifically
                    outputs_dict = ex.get('outputs', {})
                    
                    # First create example with inputs only
                    example = dspy.Example()
                    
                    # First set the attributes directly
                    example.market_data = {"summary": "Market data summary..."}
                    example.timeframe = ex.get('timeframe', '1h')
                    example.prompt = ex.get('prompt', '')
                    
                    # Then mark which ones are inputs
                    example = example.with_inputs('market_data', 'timeframe', 'prompt')
                    
                    # Set output attributes directly
                    example.regime = outputs_dict.get('regime', '')
                    example.confidence = float(outputs_dict.get('confidence', 0.0))
                    example.risk_level = outputs_dict.get('risk_level', '')
                    example.analysis = outputs_dict.get('analysis', '')
                
                # Check for strategy generation examples format
                elif 'market_analysis' in ex:
                    # Strategy example format
                    outputs_dict = ex.get('outputs', {}).get('strategy', {})
                    
                    # First create example with inputs only
                    example = dspy.Example()
                    
                    # Set input attributes directly
                    example.market_context = {"summary": "Market context summary..."}
                    example.theme = 'default'
                    example.base_parameters = None
                    
                    # Mark inputs
                    example = example.with_inputs('market_context', 'theme', 'base_parameters')
                    
                    # Set output attributes directly
                    example.reasoning = outputs_dict.get('reasoning', '')
                    example.trade_signal = outputs_dict.get('trade_signal', 'HOLD')
                    example.parameters = str(outputs_dict.get('parameters', {}))  # Convert to string to avoid complex object issues
                    example.confidence = float(outputs_dict.get('confidence', 0.0))
                
                # Check for trading rules examples format
                elif 'strategy' in ex:
                    # Trading rules example format
                    outputs_dict = ex.get('outputs', {}).get('trading_rules', {})
                    
                    # First create example with inputs only
                    example = dspy.Example()
                    
                    # Set input attributes directly
                    example.strategy_insights = {"summary": "Strategy insights summary..."}
                    example.market_context = {"summary": "Market context summary..."}
                    example.performance_analysis = None
                    
                    # Mark inputs
                    example = example.with_inputs('strategy_insights', 'market_context', 'performance_analysis')
                    
                    # Set output attributes directly
                    example.conditions = str(outputs_dict.get('conditions', {'entry': [], 'exit': []}))  # Convert to string
                    example.parameters = str(outputs_dict.get('parameters', {}))  # Convert to string
                    example.reasoning = outputs_dict.get('reasoning', '')
                
                else:
                    # Generic fallback - not ideal but prevents crashes
                    logger.warning(f"Unknown example format, attempting best-effort processing")
                    
                    # Extract inputs (exclude outputs and prompt)
                    ex_inputs = {k: v for k, v in ex.items() if k != 'output' and k != 'outputs' and k != 'prompt'}
                    
                    # Extract outputs
                    outputs = ex.get('outputs', ex.get('output', {}))
                    
                    # Create example
                    example = dspy.Example()
                    
                    # Process and set inputs to avoid complex objects
                    input_keys = []
                    for k, v in ex_inputs.items():
                        if isinstance(v, (dict, list)):
                            # Convert complex objects to strings
                            setattr(example, k, str(v))
                        else:
                            setattr(example, k, v)
                        input_keys.append(k)
                    
                    # Set inputs if we have any
                    if input_keys:
                        example = example.with_inputs(*input_keys)
                    
                    # Process and set outputs
                    if isinstance(outputs, dict):
                        for k, v in outputs.items():
                            if isinstance(v, (dict, list)):
                                # Convert complex objects to strings
                                setattr(example, k, str(v))
                            elif isinstance(v, (float, int)):
                                # Ensure numerical values are proper
                                setattr(example, k, float(v))
                            else:
                                setattr(example, k, v)
            
                # Verify the example has at least some input fields set
                input_fields = example.inputs()
                if len(input_fields) == 0:
                    logger.warning(f"Example has no input fields, skipping")
                    continue
                
                dspy_examples.append(example)
                
            except Exception as e:
                logger.error(f"Error creating example: {str(e)}")
                # Continue with next example
        
        logger.info(f"Created {len(dspy_examples)} examples for optimization")
        return dspy_examples

    def optimize(self, 
                module: dspy.Module, 
                examples: List[Dict[str, Any]], 
                prompt_name: str,
                metric_fn: Callable) -> dspy.Module:
        """Optimize a module using MiPro.
        
        Args:
            module: The DSPy module to optimize
            examples: List of examples with inputs and outputs
            prompt_name: Name of the prompt to optimize
            metric_fn: Function to evaluate the module's performance
            
        Returns:
            Optimized DSPy module
        """
        # Prepare examples
        logger.info(f"Starting MiPro{'v2' if self.use_v2 else ''} optimization for {prompt_name}")
        
        try:
            dspy_examples = self.prepare_examples(examples)
            
            # Ensure we have enough examples
            if len(dspy_examples) < 2:
                logger.warning(f"Not enough valid examples for {prompt_name}, need at least 2")
                return module

            # Create train/validation split - ensuring at least one example in each set
            if len(dspy_examples) <= 3:
                trainset = dspy_examples[:1]
                valset = dspy_examples[1:]
            else:
                split_idx = max(1, int(len(dspy_examples) * 0.7))
                trainset = dspy_examples[:split_idx]
                valset = dspy_examples[split_idx:]
            
            # Log the split details
            logger.info(f"Split examples into {len(trainset)} training and {len(valset)} validation examples")
            
            # Set up the optimizer - always use MIPROv2 since MIPRO is deprecated
            optimizer = MIPROv2(
                metric=metric_fn,
                num_candidates=self.num_candidate_programs,
                init_temperature=self.temperature,
            )
            
            # Run optimization
            start_time = time.time()
            
            # Calculate safe minibatch sizes
            minibatch_size = max(1, min(2, len(valset)))
            eval_steps = max(1, min(2, len(valset)))
            
            logger.info(f"Running MiProv2 with minibatch_size={minibatch_size}, eval_steps={eval_steps}")
            optimized_module = optimizer.compile(
                student=module,
                trainset=trainset,
                valset=valset,
                minibatch_size=minibatch_size,
                requires_permission_to_run=False,
                num_trials=5,  # Reduced from 10 to speed up
                minibatch_full_eval_steps=eval_steps
            )
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            return module
        
        try:
            # Store optimization result
            self.optimized_modules[prompt_name] = optimized_module
            optimization_time = time.time() - start_time
            logger.info(f"MiPro optimization completed in {optimization_time:.2f}s")
            
            # Store the optimized prompt if prompt manager is available
            if hasattr(optimized_module, 'prompt') and self.prompt_manager is not None:
                optimized_prompt = getattr(optimized_module, 'prompt')
                if optimized_prompt and isinstance(optimized_prompt, str):
                    # Store the optimized prompt
                    self.prompt_manager.update_prompt(prompt_name, optimized_prompt)
                    logger.info(f"Updated prompt '{prompt_name}' with optimized version")
            
            return optimized_module
        except Exception as e:
            logger.error(f"Error storing optimization results: {str(e)}")
            return module
    
    def evaluate(self, module: dspy.Module, examples: List[Dict[str, Any]], metric_fn: Callable) -> Dict[str, float]:
        """Evaluate a module.
        
        Args:
            module: Module to evaluate
            examples: Examples to evaluate on
            metric_fn: Metric function to use
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare examples for evaluation
        dspy_examples = self.prepare_examples(examples)
            
        # Evaluate the module
        evaluator = Evaluate(
            metric=metric_fn,
            eval_examples=dspy_examples,
        )
        
        result = evaluator(module)
        logger.info(f"Evaluation result: {result}")
        return result
    
    def get_optimized_module(self, prompt_name: str) -> Optional[dspy.Module]:
        """Get an optimized module by prompt name.
        
        Args:
            prompt_name: Name of the prompt
            
        Returns:
            The optimized DSPy module or None if not found
        """
        return self.optimized_modules.get(prompt_name)
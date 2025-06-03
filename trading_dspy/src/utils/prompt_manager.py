"""
Centralized Prompt Management System
Implements Kagan's vision: "All the prompts should be in one place"
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class PromptTemplate:
    """Single prompt template with metadata"""
    name: str
    content: str
    category: str
    description: str
    variables: List[str]
    last_modified: str
    performance_score: Optional[float] = None
    usage_count: int = 0
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class CentralizedPromptManager:
    """
    Kagan's Vision: "All the prompts should be in one place. 
    I should be able to change the way that it's assessing every single LLM."
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Central registry file
        self.registry_file = self.prompts_dir / "prompt_registry.json"
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Initialize prompt categories
        self.categories = {
            'market_analysis': 'Market condition analysis and regime classification',
            'strategy_generation': 'Trading strategy creation and optimization',
            'trading_rules': 'Entry/exit signal generation and timing',
            'risk_management': 'Position sizing and risk assessment',
            'optimization': 'MiPro optimization and improvement prompts',
            'evaluation': 'Performance assessment and analysis'
        }
        
        # Initialize prompts structure  
        self.prompts_dir = Path(prompts_dir)
        self.prompts = {}
        self.optimized_prompts_dir = self.prompts_dir / "optimized"
        
        # Create optimized prompts directory if it doesn't exist
        os.makedirs(self.optimized_prompts_dir, exist_ok=True)
        
        self.load_prompts()
        
    def load_prompts(self) -> None:
        """Load all prompt templates from the prompts directory."""
        try:
            # Load text prompts
            for prompt_file in self.prompts_dir.glob('*.txt'):
                name = prompt_file.stem
                with open(prompt_file, 'r') as f:
                    self.prompts[name] = {
                        'template': f.read(),
                        'examples': [],
                        'optimized': False,
                        'last_optimized': None
                    }
            
            # Check for optimized versions
            for prompt_file in self.optimized_prompts_dir.glob('*.txt'):
                name = prompt_file.stem
                if name in self.prompts:
                    with open(prompt_file, 'r') as f:
                        optimized_template = f.read()
                        self.prompts[name]['optimized_template'] = optimized_template
                        self.prompts[name]['optimized'] = True
                        self.prompts[name]['last_optimized'] = os.path.getmtime(prompt_file)
                        logger.info(f"Loaded optimized version of prompt '{name}'")
            
            # Load examples if they exist
            examples_file = self.prompts_dir / 'examples.json'
            if examples_file.exists():
                with open(examples_file, 'r') as f:
                    examples = json.load(f)
                    for name, ex in examples.items():
                        if name in self.prompts:
                            self.prompts[name]['examples'] = ex
            
            logger.info(f"Loaded {len(self.prompts)} prompt templates")
            
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
            
    def get_prompt(self, name: str, use_optimized: bool = True) -> Optional[str]:
        """Get a prompt template by name.
        
        Args:
            name: Name of the prompt template
            use_optimized: Whether to use optimized version if available
            
        Returns:
            Prompt template string if found, None otherwise
        """
        try:
            if name not in self.prompts:
                logger.error(f"Prompt template '{name}' not found")
                return None
                
            if use_optimized and self.prompts[name].get('optimized', False):
                logger.info(f"Using optimized version of prompt '{name}'")
                return self.prompts[name]['optimized_template']
            else:
                return self.prompts[name]['template']
        except KeyError:
            logger.error(f"Prompt template '{name}' not found")
            return None
            
    def get_examples(self, name: str) -> List[Dict[str, Any]]:
        """Get examples for a prompt template.
        
        Args:
            name: Name of the prompt template
            
        Returns:
            List of example dictionaries
        """
        try:
            return self.prompts[name].get('examples', [])
        except KeyError:
            logger.error(f"Examples for prompt '{name}' not found")
            return []
            
    def format_prompt(self, template: str, **kwargs: Any) -> str:
        """Format a prompt template with provided values.
        
        Args:
            template: Prompt template string
            **kwargs: Values to format the template with
            
        Returns:
            Formatted prompt string
        """
        try:
            logger.debug(f"format_prompt called with kwargs: {kwargs}")

            # Clean up any newlines in the values and handle JSON formatting
            cleaned_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (list, dict)):
                    # Convert to JSON string with proper escaping
                    json_str = json.dumps(v, indent=2)
                    cleaned_kwargs[k] = json_str
                else:
                    cleaned_kwargs[k] = str(v).replace('\n', ' ').strip()

            # Format the template with cleaned kwargs
            template = template.replace('{', '{{').replace('}', '}}')  # Escape all braces
            
            # Unescape template variables for market analysis
            variables = [
                'timeframe', 'current_regime', 'prices', 'volumes', 'indicators',
                'price_change_pct', 'avg_volume', 'current_price', 'current_volume'
            ]
            for var in variables:
                template = template.replace('{{' + var + '}}', '{' + var + '}')
            
            return template.format(**cleaned_kwargs)
            
        except KeyError as e:
            error_key = str(e).strip('"').strip()
            logger.error(f"Missing required value for prompt formatting: {error_key}")
            raise
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise
            
    def _calculate_similarity(self, example1: Dict[str, Any], example2: Dict[str, Any], example_type: str = "default") -> float:
        """Calculate similarity between two examples.
        
        This function implements specialized similarity calculations for different types of examples.
        
        Args:
            example1: First example dictionary
            example2: Second example dictionary
            example_type: Type of example (market_analysis, strategy_generator, trading_rules, default)
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Extract comparison keys based on example type
            if example_type == "market_analysis":
                # For market analysis, focus on output regime, confidence
                if 'outputs' in example1 and 'outputs' in example2:
                    # Calculate similarity based on regime matching and confidence difference
                    regime1 = example1.get('outputs', {}).get('regime', '')
                    regime2 = example2.get('outputs', {}).get('regime', '')
                    
                    # If regimes don't match, consider them different examples
                    if regime1 != regime2:
                        return 0.0
                    
                    # If confidence values are very close, consider them similar
                    conf1 = example1.get('outputs', {}).get('confidence', 0.0)
                    conf2 = example2.get('outputs', {}).get('confidence', 0.0)
                    
                    # Calculate confidence similarity - within 0.1 is very similar
                    conf_similarity = max(0.0, 1.0 - abs(conf1 - conf2))
                    
                    # Calculate similarity using regime match and confidence
                    return 0.7 + (0.3 * conf_similarity)
                    
                # Fall back to comparing analysis_text/analysis_summary if outputs not available
                elif 'analysis_text' in example1 and 'analysis_text' in example2:
                    text1 = example1.get('analysis_text', '')
                    text2 = example2.get('analysis_text', '')
                    # Use Jaccard similarity on words
                    words1 = set(text1.lower().split()[:50])  # Use first 50 words
                    words2 = set(text2.lower().split()[:50])
                    
                    if not words1 or not words2:
                        return 0.0
                    
                    # Calculate Jaccard similarity
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    
                    return intersection / union if union > 0 else 0.0
                    
            elif example_type == "strategy_generator":
                # For strategy generation, focus on trade signal, parameters, reasoning
                if 'outputs' in example1 and 'outputs' in example2:
                    strategy1 = example1.get('outputs', {}).get('strategy', {})
                    strategy2 = example2.get('outputs', {}).get('strategy', {})
                    
                    # If trade signals are different, they're different strategies
                    signal1 = strategy1.get('trade_signal', '')
                    signal2 = strategy2.get('trade_signal', '')
                    
                    if signal1 != signal2:
                        return 0.0
                    
                    # Check parameter similarity (position_size, stop_loss, take_profit)
                    params1 = strategy1.get('parameters', {})
                    params2 = strategy2.get('parameters', {})
                    
                    # Normalize parameters to dict if they're strings
                    if isinstance(params1, str):
                        try:
                            params1 = eval(params1)
                        except:
                            params1 = {}
                    
                    if isinstance(params2, str):
                        try:
                            params2 = eval(params2)
                        except:
                            params2 = {}
                    
                    # Extract key parameters
                    position_size1 = params1.get('position_size', 0)
                    position_size2 = params2.get('position_size', 0)
                    stop_loss1 = params1.get('stop_loss', 0)
                    stop_loss2 = params2.get('stop_loss', 0)
                    take_profit1 = params1.get('take_profit', 0)
                    take_profit2 = params2.get('take_profit', 0)
                    
                    # Check parameter similarity
                    param_diff = (
                        abs(position_size1 - position_size2) / max(0.1, max(position_size1, position_size2)) +
                        abs(stop_loss1 - stop_loss2) / max(0.01, max(stop_loss1, stop_loss2)) +
                        abs(take_profit1 - take_profit2) / max(0.01, max(take_profit1, take_profit2))
                    ) / 3.0
                    
                    param_similarity = max(0.0, 1.0 - min(1.0, param_diff))
                    
                    # Check reasoning similarity using simple word comparison
                    reasoning1 = strategy1.get('reasoning', '')
                    reasoning2 = strategy2.get('reasoning', '')
                    
                    # Use first 50 words for comparison
                    words1 = set(reasoning1.lower().split()[:50])
                    words2 = set(reasoning2.lower().split()[:50])
                    
                    if words1 and words2:
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        reasoning_similarity = intersection / union if union > 0 else 0.0
                    else:
                        reasoning_similarity = 0.0
                    
                    # Combined similarity score - weigh parameters more than reasoning
                    return 0.6 * param_similarity + 0.4 * reasoning_similarity
                    
            elif example_type == "trading_rules":
                # For trading rules, focus on entry/exit conditions and parameters
                if 'outputs' in example1 and 'outputs' in example2:
                    rules1 = example1.get('outputs', {}).get('trading_rules', {})
                    rules2 = example2.get('outputs', {}).get('trading_rules', {})
                    
                    # Check entry conditions similarity
                    entry1 = str(rules1.get('entry_conditions', rules1.get('conditions', {}).get('entry', [])))
                    entry2 = str(rules2.get('entry_conditions', rules2.get('conditions', {}).get('entry', [])))
                    
                    # Simple string overlap ratio for entry conditions
                    entry_similarity = 0.0
                    if entry1 and entry2:
                        common_len = 0
                        for i in range(min(len(entry1), len(entry2))):
                            if entry1[i] == entry2[i]:
                                common_len += 1
                        entry_similarity = common_len / max(len(entry1), len(entry2))
                    
                    # Check exit conditions similarity
                    exit1 = str(rules1.get('exit_conditions', rules1.get('conditions', {}).get('exit', [])))
                    exit2 = str(rules2.get('exit_conditions', rules2.get('conditions', {}).get('exit', [])))
                    
                    # Simple string overlap ratio for exit conditions
                    exit_similarity = 0.0
                    if exit1 and exit2:
                        common_len = 0
                        for i in range(min(len(exit1), len(exit2))):
                            if exit1[i] == exit2[i]:
                                common_len += 1
                        exit_similarity = common_len / max(len(exit1), len(exit2))
                    
                    # Combined similarity
                    return 0.5 * entry_similarity + 0.5 * exit_similarity
            
            # Default fallback to string comparison
            key1 = str(example1.get('output', example1.get('outputs', example1)))
            key2 = str(example2.get('output', example2.get('outputs', example2)))
            
            # Simple string similarity ratio (Jaccard for words)
            words1 = set(key1.lower().split()[:100])  # Limit to first 100 words
            words2 = set(key2.lower().split()[:100])
            
            if not words1 or not words2:
                return 0.0
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating example similarity: {str(e)}")
            logger.exception("Detailed error:")
            return 0.0  # On error, assume not similar
    
    def add_example(self, name: str, example: Dict[str, Any], check_duplicates: bool = True) -> None:
        """Add an example to a prompt template.
        
        Args:
            name: Name of the prompt template
            example: Example dictionary
            check_duplicates: Whether to check for duplicates before adding
        """
        try:
            if name not in self.prompts:
                raise KeyError(f"Prompt template '{name}' not found")
            
            # Verify the example is JSON serializable
            try:
                json.dumps(example)
            except (TypeError, ValueError, OverflowError) as e:
                logger.error(f"Example is not JSON serializable: {e}")
                
                # Try to make it serializable by converting complex objects to strings
                fixed_example = self._make_serializable(example)
                logger.info("Converted non-serializable fields to strings")
                example = fixed_example
            
            # Check for duplicates if requested
            if check_duplicates:
                # Advanced duplication check using similarity calculation
                current_examples = self.prompts[name]['examples']
                is_duplicate = False
                
                # Determine similarity threshold based on prompt type
                similarity_threshold = 0.9  # Default strict threshold
                
                # Adjust threshold by prompt type
                if "market_analysis" in name:
                    similarity_threshold = 0.75  # More permissive for market analysis
                    example_type = "market_analysis"
                elif "strategy_generator" in name:
                    similarity_threshold = 0.85  # Somewhat permissive for strategies
                    example_type = "strategy_generator"
                elif "trading_rules" in name:
                    similarity_threshold = 0.8  # Medium permissive for trading rules
                    example_type = "trading_rules"
                else:
                    example_type = "default"
                
                logger.debug(f"Using similarity threshold {similarity_threshold} for {name} examples")
                
                # Check against existing examples
                for existing in current_examples:
                    # Calculate similarity between examples
                    similarity = self._calculate_similarity(example, existing, example_type)
                    
                    logger.debug(f"Example similarity: {similarity:.4f} (threshold: {similarity_threshold})")
                    
                    # Check if similarity exceeds threshold
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        logger.info(f"Skipping duplicate example for prompt '{name}' (similarity: {similarity:.4f})")
                        break
                
                if is_duplicate:
                    return
            
            # Add the example
            self.prompts[name]['examples'].append(example)
            
            # Save to examples file
            examples_file = self.prompts_dir / 'examples.json'
            examples = {}
            if examples_file.exists():
                try:
                    with open(examples_file, 'r') as f:
                        examples = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Existing examples file was corrupted, creating new one")
                    examples = {}
                    
            examples[name] = self.prompts[name]['examples']
            
            try:
                with open(examples_file, 'w') as f:
                    json.dump(examples, f, indent=2)
                    
                logger.info(f"Added example to prompt '{name}'")
            except Exception as e:
                logger.error(f"Error writing examples file: {e}")
                # Try a more robust approach by writing just this example's collection
                try:
                    with open(self.prompts_dir / f'examples_{name}.json', 'w') as f:
                        json.dump({name: self.prompts[name]['examples']}, f, indent=2)
                    logger.info(f"Saved examples for '{name}' to separate file")
                except Exception as nested_e:
                    logger.error(f"Failed to save examples even to separate file: {nested_e}")
            
        except Exception as e:
            logger.error(f"Error adding example: {str(e)}")
            logger.exception("Detailed error:")
            
    def _make_serializable(self, obj: Any) -> Any:
        """Make an object JSON serializable by converting problematic types.
        
        Args:
            obj: The object to make serializable
            
        Returns:
            A serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # Handle datetime objects
            return str(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert any other types to string
            return str(obj)
            
    def add_successful_example(self, name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Add a successful example to the prompt template.
        
        Args:
            name: Name of the prompt template
            inputs: Input dictionary
            outputs: Output dictionary
        """
        example = {**inputs, 'outputs': outputs}
        self.add_example(name, example, check_duplicates=True)
    
    def update_prompt(self, name: str, new_template: str) -> None:
        """Update a prompt template with an optimized version.
        
        Args:
            name: Name of the prompt template
            new_template: New optimized template
        """
        try:
            if name not in self.prompts:
                raise KeyError(f"Prompt template '{name}' not found")
                
            # Store the optimized template
            self.prompts[name]['optimized_template'] = new_template
            self.prompts[name]['optimized'] = True
            self.prompts[name]['last_optimized'] = time.time()
            
            # Save to optimized prompts directory
            optimized_file = self.optimized_prompts_dir / f"{name}.txt"
            with open(optimized_file, 'w') as f:
                f.write(new_template)
                
            logger.info(f"Updated prompt '{name}' with optimized version")
            
        except Exception as e:
            logger.error(f"Error updating prompt: {str(e)}")
            raise
    
    def get_optimization_status(self, name: str) -> Dict[str, Any]:
        """Get optimization status for a prompt.
        
        Args:
            name: Name of the prompt template
            
        Returns:
            Dictionary with optimization status information
        """
        try:
            if name not in self.prompts:
                raise KeyError(f"Prompt template '{name}' not found")
                
            optimized = self.prompts[name].get('optimized', False)
            last_optimized = self.prompts[name].get('last_optimized', None)
            
            return {
                'optimized': optimized,
                'last_optimized': last_optimized,
                'example_count': len(self.prompts[name].get('examples', []))
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization status: {str(e)}")
            raise
            
    def get_prompt_names(self) -> List[str]:
        """Get list of available prompt names.
        
        Returns:
            List of prompt template names
        """
        return list(self.prompts.keys())
    
    def need_optimization(self, name: str, min_examples: int = 3, max_age_hours: int = 24) -> bool:
        """Check if a prompt needs optimization.
        
        Args:
            name: Name of the prompt template
            min_examples: Minimum number of examples required
            max_age_hours: Maximum age of optimization in hours
            
        Returns:
            True if prompt needs optimization, False otherwise
        """
        try:
            if name not in self.prompts:
                return False
                
            # Check if there are enough examples
            examples_count = len(self.prompts[name].get('examples', []))
            if examples_count < min_examples:
                return False
                
            # Check if already optimized
            optimized = self.prompts[name].get('optimized', False)
            if not optimized:
                return True
                
            # Check optimization age
            last_optimized = self.prompts[name].get('last_optimized', 0)
            if not last_optimized:
                return True
                
            age_hours = (time.time() - last_optimized) / 3600
            return age_hours > max_age_hours
            
        except Exception as e:
            logger.error(f"Error checking optimization need: {str(e)}")
            return False 
    
    def list_prompts(self) -> List[str]:
        """Alias for get_prompt_names for compatibility.
        
        Returns:
            List of prompt template names
        """
        return list(self.prompts.keys())
    
    def get_best_performing_prompts(self, category: str, limit: int = 3) -> List[Any]:
        """Get best performing prompts for a category.
        
        Args:
            category: Prompt category to search
            limit: Maximum number of prompts to return
            
        Returns:
            List of best performing prompt objects
        """
        # Simple implementation - return prompts from category
        matching_prompts = []
        for name, prompt_data in self.prompts.items():
            if category in name.lower():
                # Create a simple prompt object
                class PromptObj:
                    def __init__(self, content):
                        self.content = content
                
                template = prompt_data.get('optimized_template', prompt_data.get('template', ''))
                if template:
                    matching_prompts.append(PromptObj(template))
        
        return matching_prompts[:limit]
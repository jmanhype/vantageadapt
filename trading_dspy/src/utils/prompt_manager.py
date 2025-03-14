"""Prompt management utilities for the trading system with MiPro optimization support."""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import os
import time
from loguru import logger

class PromptManager:
    """Manager for handling prompts and their templates with optimization support."""

    def __init__(self, prompts_dir: str):
        """Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt templates
        """
        from os import path
        logger.info(f"Initializing PromptManager from file: {__file__}")
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
                # Simple duplication check - assuming examples with similar outputs are duplicates
                current_examples = self.prompts[name]['examples']
                is_duplicate = False
                
                # Extract a key for comparison (usually output fields)
                if 'output' in example:
                    example_key = str(example['output'])
                elif 'outputs' in example:
                    example_key = str(example['outputs'])
                else:
                    # If no output fields, use all fields
                    example_key = str(example)
                
                # Check against existing examples
                for existing in current_examples:
                    if 'output' in existing:
                        existing_key = str(existing['output'])
                    elif 'outputs' in existing:
                        existing_key = str(existing['outputs'])
                    else:
                        existing_key = str(existing)
                    
                    # Check similarity (simple string equality for now)
                    if example_key == existing_key:
                        is_duplicate = True
                        logger.info(f"Skipping duplicate example for prompt '{name}'")
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
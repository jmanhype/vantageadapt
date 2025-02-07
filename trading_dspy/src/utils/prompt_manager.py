"""Prompt management utilities for the trading system."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from loguru import logger

class PromptManager:
    """Manager for handling prompts and their templates."""

    def __init__(self, prompts_dir: str):
        """Initialize prompt manager.
        
        Args:
            prompts_dir: Directory containing prompt templates
        """
        from os import path
        logger.info(f"Initializing PromptManager from file: {__file__}")
        self.prompts_dir = Path(prompts_dir)
        self.prompts = {}
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
                        'examples': []
                    }
            
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
            
    def get_prompt(self, name: str) -> Optional[str]:
        """Get a prompt template by name.
        
        Args:
            name: Name of the prompt template
            
        Returns:
            Prompt template string if found, None otherwise
        """
        try:
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
            
    def add_example(self, name: str, example: Dict[str, Any]) -> None:
        """Add an example to a prompt template.
        
        Args:
            name: Name of the prompt template
            example: Example dictionary
        """
        try:
            if name not in self.prompts:
                raise KeyError(f"Prompt template '{name}' not found")
                
            self.prompts[name]['examples'].append(example)
            
            # Save to examples file
            examples_file = self.prompts_dir / 'examples.json'
            examples = {}
            if examples_file.exists():
                with open(examples_file, 'r') as f:
                    examples = json.load(f)
                    
            examples[name] = self.prompts[name]['examples']
            
            with open(examples_file, 'w') as f:
                json.dump(examples, f, indent=2)
                
            logger.info(f"Added example to prompt '{name}'")
            
        except Exception as e:
            logger.error(f"Error adding example: {str(e)}")
            raise
            
    def get_prompt_names(self) -> List[str]:
        """Get list of available prompt names.
        
        Returns:
            List of prompt template names
        """
        return list(self.prompts.keys()) 
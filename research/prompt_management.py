"""Prompt management system for the strategy generator.

This module provides functionality for managing and versioning prompts used in the
strategy generation pipeline.
"""
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompts for the strategy generation pipeline."""
    
    def __init__(self, base_path: str = "prompts"):
        """Initialize the prompt manager.
        
        Args:
            base_path: Base directory for prompts
        """
        self.base_path = Path(base_path)
        self.prompts: Dict[str, Dict[str, str]] = {}
        self.version_history: Dict[str, list] = {}
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load all prompts from the prompts directory."""
        try:
            for category in ['strategy', 'analysis', 'optimization', 'evaluation']:
                category_path = self.base_path / category
                if not category_path.exists():
                    logger.warning(f"Category directory not found: {category}")
                    continue
                
                self.prompts[category] = {}
                for prompt_file in category_path.glob('*.prompt'):
                    prompt_name = prompt_file.stem
                    with open(prompt_file, 'r') as f:
                        self.prompts[category][prompt_name] = f.read()
                        
            logger.info(f"Loaded prompts from {self.base_path}")
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
    
    def get_prompt(self, category: str, name: str) -> Optional[str]:
        """Get a prompt by category and name.
        
        Args:
            category: Prompt category (strategy, analysis, etc.)
            name: Name of the prompt
            
        Returns:
            The prompt text if found, None otherwise
        """
        return self.prompts.get(category, {}).get(name)
    
    def update_prompt(self, category: str, name: str, content: str) -> None:
        """Update a prompt and save its history.
        
        Args:
            category: Prompt category
            name: Prompt name
            content: New prompt content
        """
        try:
            # Save current version to history
            if category not in self.version_history:
                self.version_history[category] = {}
            if name not in self.version_history[category]:
                self.version_history[category][name] = []
            
            current_version = self.get_prompt(category, name)
            if current_version:
                self.version_history[category][name].append({
                    'content': current_version,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            # Update prompt
            if category not in self.prompts:
                self.prompts[category] = {}
            self.prompts[category][name] = content
            
            # Save to file
            prompt_path = self.base_path / category / f"{name}.prompt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(prompt_path, 'w') as f:
                f.write(content)
                
            logger.info(f"Updated prompt: {category}/{name}")
        except Exception as e:
            logger.error(f"Error updating prompt: {str(e)}")
            raise
    
    def get_prompt_history(self, category: str, name: str) -> list:
        """Get version history for a prompt.
        
        Args:
            category: Prompt category
            name: Prompt name
            
        Returns:
            List of historical versions with timestamps
        """
        return self.version_history.get(category, {}).get(name, [])
    
    def format_prompt(self, category: str, name: str, **kwargs: Any) -> Optional[str]:
        """Format a prompt with provided variables.
        
        Args:
            category: Prompt category
            name: Prompt name
            **kwargs: Variables to format into the prompt
            
        Returns:
            Formatted prompt if found, None otherwise
        """
        prompt = self.get_prompt(category, name)
        if prompt is None:
            return None
            
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required variable in prompt: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None
    
    def list_prompts(self, category: Optional[str] = None) -> Dict[str, list]:
        """List available prompts.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Dictionary of categories and their prompts
        """
        if category:
            return {category: list(self.prompts.get(category, {}).keys())}
        return {cat: list(prompts.keys()) for cat, prompts in self.prompts.items()}
    
    def export_prompts(self, output_path: str) -> None:
        """Export all prompts and their history to JSON.
        
        Args:
            output_path: Path to save the export
        """
        try:
            export_data = {
                'prompts': self.prompts,
                'version_history': self.version_history,
                'exported_at': datetime.utcnow().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Exported prompts to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting prompts: {str(e)}")
            raise
    
    def import_prompts(self, input_path: str) -> None:
        """Import prompts from a JSON export.
        
        Args:
            input_path: Path to the import file
        """
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Update prompts
            for category, prompts in import_data['prompts'].items():
                for name, content in prompts.items():
                    self.update_prompt(category, name, content)
            
            # Update history
            self.version_history.update(import_data.get('version_history', {}))
            
            logger.info(f"Imported prompts from {input_path}")
        except Exception as e:
            logger.error(f"Error importing prompts: {str(e)}")
            raise 
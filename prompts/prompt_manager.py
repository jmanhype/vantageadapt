"""Centralized prompt management system."""

import os
from typing import Dict, Any, Optional
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages loading and accessing prompts from YAML files."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the prompt manager.
        
        Args:
            base_dir: Base directory for prompts. Defaults to the 'prompts' directory.
        """
        if base_dir is None:
            # Get the absolute path to the prompts directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the root directory
            root_dir = os.path.dirname(current_dir)
            base_dir = os.path.join(root_dir, 'prompts')
        self.base_dir = base_dir
        self.prompts: Dict[str, Any] = {}
        self._load_all_prompts()
        
    def _load_all_prompts(self) -> None:
        """Load all prompt YAML files from the prompts directory structure."""
        try:
            # Walk through all directories under base_dir
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    if file.endswith('.yaml') or file.endswith('.yml'):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, self.base_dir)
                        # Create prompt key from path (e.g., 'trading/market_analysis')
                        prompt_key = os.path.splitext(relative_path)[0].replace(os.sep, '/')
                        
                        try:
                            with open(file_path, 'r') as f:
                                content = yaml.safe_load(f)
                                if content and isinstance(content, dict):
                                    self.prompts[prompt_key] = content
                                    logger.debug(f"Loaded prompt: {prompt_key}")
                                else:
                                    logger.warning(f"Invalid prompt file format: {file_path}")
                        except Exception as e:
                            logger.error(f"Error loading prompt file {file_path}: {str(e)}")
                            
        except Exception as e:
            logger.error(f"Error loading prompts: {str(e)}")
            raise
        
    def get_prompt(self, prompt_key: str) -> Optional[Dict[str, Any]]:
        """Get a prompt by its key.
        
        Args:
            prompt_key: The key for the prompt (e.g., 'trading/market_analysis')
            
        Returns:
            The prompt dictionary if found, None otherwise
        """
        return self.prompts.get(prompt_key)
        
    def reload_prompts(self) -> None:
        """Reload all prompts from files."""
        self.prompts.clear()
        self._load_all_prompts()
        
    def get_all_prompt_keys(self) -> list[str]:
        """Get a list of all available prompt keys.
        
        Returns:
            List of prompt keys
        """
        return list(self.prompts.keys())
        
    def get_prompt_content(self, prompt_key: str, role: str = 'system') -> Optional[str]:
        """Get the content of a specific prompt role.
        
        Args:
            prompt_key: The key for the prompt
            role: The role to get content for ('system', 'user', etc.)
            
        Returns:
            The prompt content if found, None otherwise
        """
        prompt = self.get_prompt(prompt_key)
        if prompt and isinstance(prompt, dict):
            return prompt.get(role)
        return None 
# Additional methods for CentralizedPromptManager compatibility

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

def update_prompt_content(self, name: str, content: str = None, last_modified: str = None) -> None:
    """Update prompt with new content.
    
    Args:
        name: Name of the prompt
        content: New content (optional)
        last_modified: Last modified timestamp (optional)
    """
    if name in self.prompts and content:
        self.prompts[name]['optimized_template'] = content
        self.prompts[name]['optimized'] = True
        if last_modified:
            self.prompts[name]['last_modified'] = last_modified
        print(f"Updated prompt: {name}")
"""Memory system configuration."""

from typing import Dict, Any
import os
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Memory system configuration
MEM0_CONFIG = {
    "api_key": None,  # Will be loaded from environment
    "org_id": None,  # Will be loaded from environment
    "project_id": None,  # Will be loaded from environment
    "user_id": "vantageadapt_user",
    "success_criteria": {
        "min_return": 0.5,  # 50% minimum return
        "min_sortino": 1.5,  # Minimum Sortino ratio
        "min_win_rate": 0.4  # Minimum win rate
    },
    "metadata_types": {
        "strategy_results": "strategy_results",
        "market_analysis": "market_analysis",
        "performance_metrics": "performance_metrics"
    },
    "search_limit": 10  # Maximum number of similar strategies to return
}

def validate_config() -> bool:
    """Validate the memory system configuration.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Update config with environment variables
        MEM0_CONFIG["api_key"] = os.getenv("MEM0_API_KEY")
        MEM0_CONFIG["org_id"] = os.getenv("MEM0_ORG_ID", "jay6-default-org")
        MEM0_CONFIG["project_id"] = os.getenv("MEM0_PROJECT_ID", "default-project")
        
        # Check API key
        if not MEM0_CONFIG["api_key"]:
            logger.error("MEM0_API_KEY not found in environment")
            return False
            
        # Log configuration (excluding sensitive data)
        logger.debug("Memory configuration:")
        logger.debug(f"- org_id: {MEM0_CONFIG['org_id']}")
        logger.debug(f"- project_id: {MEM0_CONFIG['project_id']}")
        logger.debug(f"- user_id: {MEM0_CONFIG['user_id']}")
        
        # Validate success criteria
        criteria = MEM0_CONFIG.get("success_criteria", {})
        if not all(k in criteria for k in ["min_return", "min_sortino", "min_win_rate"]):
            logger.error("Missing required success criteria")
            return False
            
        # Validate metadata types
        metadata = MEM0_CONFIG.get("metadata_types", {})
        if not all(k in metadata for k in ["strategy_results", "market_analysis", "performance_metrics"]):
            logger.error("Missing required metadata types")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        return False 
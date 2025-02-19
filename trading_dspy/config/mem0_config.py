"""Configuration for the mem0 memory system."""

from typing import Dict, Any
import os
from pathlib import Path
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

MEM0_CONFIG = {
    "api_key": os.getenv("MEM0_API_KEY", ""),
    "user_id": os.getenv("MEM0_USER_ID", "trading_system"),
    "search_limit": 10,  # Maximum number of similar strategies to retrieve
    "metadata_types": {
        "strategy_results": "strategy_results",
        "market_analysis": "market_analysis",
        "performance_metrics": "performance_metrics"
    },
    "score_weights": {
        "total_return": 0.4,
        "sortino_ratio": 0.3,
        "win_rate": 0.3
    },
    "success_thresholds": {
        "min_score": 1.0,  # Minimum composite score for "successful" strategy
        "min_return": 0.10,  # Minimum 10% return
        "min_trades": 10,  # Minimum number of trades
        "min_win_rate": 0.5,  # Minimum 50% win rate
        "min_sortino": 1.0  # Minimum Sortino ratio
    }
}

def validate_config() -> bool:
    """Validate the memory system configuration.
    
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check required environment variables
        if not MEM0_CONFIG["api_key"]:
            logger.error("MEM0_API_KEY not found in environment")
            return False
            
        # Check user_id
        if not MEM0_CONFIG["user_id"]:
            logger.error("MEM0_USER_ID not found in environment")
            return False
            
        # Validate score weights sum to 1
        weights = MEM0_CONFIG["score_weights"]
        weight_sum = sum(weights.values())
        if not 0.99 <= weight_sum <= 1.01:  # Allow for floating point imprecision
            logger.error(f"Score weights must sum to 1.0, got {weight_sum}")
            return False
            
        # Validate thresholds are reasonable
        thresholds = MEM0_CONFIG["success_thresholds"]
        if not 0 <= thresholds["min_win_rate"] <= 1:
            logger.error("min_win_rate must be between 0 and 1")
            return False
            
        if thresholds["min_return"] < 0:
            logger.error("min_return cannot be negative")
            return False
            
        if thresholds["min_trades"] < 1:
            logger.error("min_trades must be at least 1")
            return False
            
        if thresholds["min_sortino"] < 0:
            logger.error("min_sortino cannot be negative")
            return False
            
        logger.info("Memory system configuration validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        return False

def get_config() -> Dict[str, Any]:
    """Get the current configuration.
    
    Returns:
        Dict containing configuration settings
    """
    return MEM0_CONFIG.copy()

def update_config(updates: Dict[str, Any]) -> bool:
    """Update configuration settings.
    
    Args:
        updates: Dictionary of settings to update
        
    Returns:
        bool: True if update was successful
    """
    try:
        # Update configuration
        for key, value in updates.items():
            if key in MEM0_CONFIG:
                if isinstance(MEM0_CONFIG[key], dict) and isinstance(value, dict):
                    MEM0_CONFIG[key].update(value)
                else:
                    MEM0_CONFIG[key] = value
                    
        # Validate new configuration
        if not validate_config():
            logger.error("Configuration update failed validation")
            return False
            
        logger.info("Configuration updated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return False 
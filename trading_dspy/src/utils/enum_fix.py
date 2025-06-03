"""Fix for enum parsing issues in the trading pipeline."""

import re
from typing import Any, Union
from src.utils.types import MarketRegime

def clean_regime_string(regime_str: Any) -> str:
    """Clean regime string by removing extra quotes and whitespace.
    
    Args:
        regime_str: The regime string that might have extra quotes
        
    Returns:
        Cleaned regime string
    """
    if not isinstance(regime_str, str):
        return str(regime_str)
    
    # Remove surrounding whitespace
    cleaned = regime_str.strip()
    
    # Remove surrounding quotes (both single and double)
    while cleaned and cleaned[0] in ['"', "'"] and cleaned[-1] in ['"', "'"]:
        cleaned = cleaned[1:-1].strip()
    
    # Remove escaped quotes
    cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")
    
    return cleaned

def safe_market_regime(regime_value: Any) -> MarketRegime:
    """Safely convert a value to MarketRegime enum.
    
    Args:
        regime_value: The value to convert (can be string with quotes, enum, etc.)
        
    Returns:
        MarketRegime enum value
    """
    # If already a MarketRegime, return it
    if isinstance(regime_value, MarketRegime):
        return regime_value
    
    # Clean the string
    cleaned_value = clean_regime_string(regime_value)
    
    # Try to match with enum values
    try:
        return MarketRegime(cleaned_value)
    except ValueError:
        # If exact match fails, try case-insensitive match
        for regime in MarketRegime:
            if regime.value.upper() == cleaned_value.upper():
                return regime
        
        # If still no match, return UNKNOWN
        return MarketRegime.UNKNOWN

def fix_market_context(market_context: dict) -> dict:
    """Fix market context by cleaning regime values.
    
    Args:
        market_context: Market context dictionary
        
    Returns:
        Fixed market context dictionary
    """
    if 'regime' in market_context:
        market_context['regime'] = safe_market_regime(market_context['regime']).value
    
    return market_context
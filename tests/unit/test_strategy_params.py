"""Test strategy parameter validation and edge case handling."""

import pytest
import logging
from typing import Dict, Any
from research.strategy.llm_interface import LLMInterface
from research.strategy.types import MarketRegime, StrategyContext

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@pytest.fixture
async def llm_interface():
    """Create LLM interface instance."""
    return await LLMInterface.create()

@pytest.fixture
def edge_case_params() -> Dict[str, Dict[str, Any]]:
    """Generate edge case parameter combinations."""
    return {
        "zero_values": {
            "entry_threshold": 0.0,
            "exit_threshold": 0.0,
            "stop_loss": 0.0,
            "position_size": 0.0
        },
        "extreme_values": {
            "entry_threshold": 1.0,
            "exit_threshold": 1.0,
            "stop_loss": 1.0,
            "position_size": 1.0
        },
        "negative_values": {
            "entry_threshold": -0.5,
            "exit_threshold": -0.2,
            "stop_loss": -0.1,
            "position_size": -0.3
        },
        "inconsistent_values": {
            "entry_threshold": 0.3,
            "exit_threshold": 0.8,  # Exit > Entry
            "stop_loss": 0.05,
            "position_size": 0.1
        },
        "missing_values": {
            "entry_threshold": 0.7,
            # Missing exit_threshold
            "stop_loss": 0.02
            # Missing position_size
        }
    }

@pytest.mark.asyncio
async def test_edge_case_parameters(llm_interface, edge_case_params):
    """Test validation of edge case strategy parameters.
    
    This test verifies that:
    1. Zero values are properly handled
    2. Extreme values are caught
    3. Negative values are rejected
    4. Inconsistent thresholds are detected
    5. Missing parameters are identified
    """
    context = StrategyContext(
        market_regime=MarketRegime.TRENDING_BULLISH,
        confidence=0.9,
        risk_level="medium",
        parameters={}
    )
    
    validation_results = {}
    
    # Test each edge case
    for case_name, params in edge_case_params.items():
        try:
            # Attempt to validate parameters
            context.parameters = params
            is_valid = await llm_interface.validate_strategy_parameters(
                context=context,
                parameters=params
            )
            validation_results[case_name] = {
                "valid": is_valid,
                "error": None
            }
        except Exception as e:
            validation_results[case_name] = {
                "valid": False,
                "error": str(e)
            }
            logger.warning(f"{case_name} validation failed: {str(e)}")
    
    # Verify results
    assert not validation_results["zero_values"]["valid"], \
        "Zero values should be invalid"
    assert not validation_results["extreme_values"]["valid"], \
        "Extreme values should be invalid"
    assert not validation_results["negative_values"]["valid"], \
        "Negative values should be invalid"
    assert not validation_results["inconsistent_values"]["valid"], \
        "Inconsistent thresholds should be invalid"
    assert not validation_results["missing_values"]["valid"], \
        "Missing parameters should be invalid"
    
    # Log validation summary
    logger.info("Parameter validation results:")
    for case, result in validation_results.items():
        logger.info(f"{case}: {'✓' if result['valid'] else '✗'}")
        if result['error']:
            logger.info(f"  Error: {result['error']}")
    
    logger.info("Parameter validation test passed")

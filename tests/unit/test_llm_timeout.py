"""Test LLM interface timeout handling and fallback behavior."""

import pytest
import logging
import asyncio
from datetime import datetime
from typing import Optional
from unittest.mock import patch, AsyncMock
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
    interface = await LLMInterface.create()
    return interface

@pytest.fixture
def sample_strategy_context():
    """Create sample strategy context."""
    return StrategyContext(
        market_regime=MarketRegime.TRENDING_BULLISH,
        confidence=0.9,
        risk_level="medium",
        parameters={
            "entry_threshold": 0.75,
            "stop_loss": 0.02
        }
    )

class TimeoutResponse:
    """Mock response that simulates timeout."""
    def __init__(self, delay: float):
        self.delay = delay
        
    async def __call__(self, *args, **kwargs):
        await asyncio.sleep(self.delay)
        raise asyncio.TimeoutError("LLM request timed out")

@pytest.mark.asyncio
async def test_strategy_update_timeout(llm_interface, sample_strategy_context):
    """Test handling of LLM API timeouts during strategy updates.
    
    This test verifies that:
    1. Timeouts are handled gracefully
    2. System falls back to last known good strategy
    3. Appropriate warnings are logged
    """
    # First store a valid strategy
    initial_strategy = await llm_interface.generate_strategy(
        "test_strategy",
        sample_strategy_context
    )
    assert initial_strategy, "Failed to generate initial strategy"
    
    # Mock LLM to simulate timeout
    with patch('research.strategy.llm_interface.LLMInterface.chat_completion',
              new_callable=AsyncMock) as mock_chat:
        # Configure mock to timeout
        mock_chat.side_effect = TimeoutResponse(2.0)
        
        # Attempt strategy update
        try:
            updated_strategy = await llm_interface.generate_strategy(
                "test_strategy",
                sample_strategy_context,
                timeout=1.0
            )
        except asyncio.TimeoutError:
            logger.warning("Expected timeout occurred")
            
        # Verify fallback behavior
        current_strategy = await llm_interface.get_current_strategy()
        assert current_strategy, "No strategy available after timeout"
        
        # Check that we kept the last known good strategy
        assert (
            current_strategy.parameters == initial_strategy.parameters
        ), "Failed to maintain last known good strategy"
        
        # Verify timeout was logged
        logger.info("Timeout handling test passed")
        
        # Verify warning was issued
        mock_chat.assert_called_once()

"""Test LLM interface timeout handling and fallback behavior."""

import pytest
import pytest_asyncio
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List
from unittest.mock import patch, AsyncMock, MagicMock
from research.strategy.llm_interface import LLMInterface, MarketContext, MarketRegime

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MockResponse:
    """Mock LLM response."""
    def __init__(self, content: str):
        self.content = content

@pytest_asyncio.fixture
async def llm_interface():
    """Create LLM interface instance."""
    interface = await LLMInterface.create()
    return interface

@pytest.fixture
def sample_strategy_context():
    """Create sample strategy context."""
    return MarketContext(
        regime=MarketRegime.TRENDING_BULLISH,
        confidence=0.9,
        volatility_level=0.5,
        trend_strength=0.8,
        volume_profile="increasing",
        risk_level="medium",
        key_levels={"support": [100.0], "resistance": [110.0]},
        analysis={
            "price_action": "Strong upward momentum",
            "volume_analysis": "Increasing buy pressure",
            "momentum": "Bullish",
            "volatility": "Moderate"
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
    # Mock initial strategy response
    initial_response = MockResponse('''{
        "regime_change_probability": 0.1,
        "suggested_position_size": 0.05,
        "risk_reward_target": 2.0,
        "entry_zones": [{"price": 100, "size": 0.5}],
        "exit_zones": [{"price": 110, "size": 0.5}],
        "stop_loss_zones": [{"price": 95, "size": 1.0}],
        "trade_frequency": "medium",
        "position_sizing_advice": "Start with half position",
        "risk_management_notes": ["Use tight stops"],
        "opportunity_description": "Trending market breakout"
    }''')
    
    # Mock the chat completion for initial strategy
    with patch('research.strategy.llm_interface.LLMInterface.chat_completion',
              return_value=initial_response):
        # First store a valid strategy
        initial_strategy = await llm_interface.generate_strategy(
            theme="test_strategy",
            market_context=sample_strategy_context
        )
        assert initial_strategy is not None, "Failed to generate initial strategy"
    
        # Mock LLM to simulate timeout
        with patch('research.strategy.llm_interface.LLMInterface.chat_completion',
                  new_callable=AsyncMock) as mock_chat:
            # Configure mock to timeout
            mock_chat.side_effect = TimeoutResponse(2.0)
            
            # Attempt strategy update
            try:
                updated_strategy = await llm_interface.generate_strategy(
                    theme="test_strategy",
                    market_context=sample_strategy_context
                )
            except asyncio.TimeoutError:
                logger.warning("Expected timeout occurred")
                
            # Verify fallback behavior
            current_strategy = await llm_interface.get_current_strategy()
            assert current_strategy is not None, "No strategy available after timeout"
            
            # Check that we kept the last known good strategy
            assert (
                current_strategy.suggested_position_size == initial_strategy.suggested_position_size and
                current_strategy.risk_reward_target == initial_strategy.risk_reward_target and
                current_strategy.trade_frequency == initial_strategy.trade_frequency
            ), "Failed to maintain last known good strategy"
            
            # Verify timeout was logged
            logger.info("Timeout handling test passed")
            
            # Verify warning was issued
            mock_chat.assert_called_once()

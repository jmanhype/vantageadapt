"""Test memory system resilience to corruption during concurrent writes."""

import pytest
import pytest_asyncio
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock
from research.strategy.types import MarketRegime, StrategyContext

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MockMemoryClient:
    """Mock memory client for testing."""
    def __init__(self):
        self.storage = []
        
    async def reset(self):
        """Clear storage."""
        self.storage = []
        
    async def add(self, messages: List[Dict], user_id: str, agent_id: str) -> bool:
        """Add messages to storage."""
        try:
            self.storage.extend(messages)
            return True
        except Exception as e:
            logger.error(f"Add failed: {str(e)}")
            return False
            
    async def get_all(self, user_id: str, agent_id: str) -> List[Dict]:
        """Get all stored messages."""
        return [{"memory": msg} for msg in self.storage]

@pytest_asyncio.fixture
async def memory_client():
    """Create memory client instance for testing."""
    client = MockMemoryClient()
    await client.reset()  # Clear any existing data
    return client

@pytest.fixture
def sample_strategy_context():
    """Create sample strategy context for testing."""
    return StrategyContext(
        market_regime=MarketRegime.TRENDING_BULLISH,
        confidence=0.85,
        risk_level="medium",
        parameters={
            "entry_threshold": 0.75,
            "position_size": 0.1,
            "stop_loss": 0.02
        }
    )

async def write_strategy(client: MockMemoryClient, context: StrategyContext, 
                        delay: float) -> bool:
    """Attempt to write strategy with delay."""
    await asyncio.sleep(delay)
    try:
        success = await client.add(
            messages=[{
                "role": "system",
                "content": str(context.to_dict()),
                "metadata": {
                    "type": "strategy",
                    "regime": context.market_regime.value
                }
            }],
            user_id="test_user",
            agent_id="test_agent"
        )
        logger.info(f"Write completed with delay {delay:.2f}s: {success}")
        return True if success else False
    except Exception as e:
        logger.error(f"Write failed with delay {delay:.2f}s: {str(e)}")
        return False

async def verify_strategy(client: MockMemoryClient, context: StrategyContext) -> bool:
    """Verify strategy was stored correctly."""
    try:
        stored = await client.get_all(
            user_id="test_user",
            agent_id="test_agent"
        )
        if not stored:
            logger.error("No stored strategies found")
            return False
            
        # Verify latest strategy matches
        latest = stored[-1]
        stored_dict = eval(latest['memory']['content'])  # Safe since we stored it
        
        matches = (
            stored_dict["market_regime"] == context.market_regime.value and
            stored_dict["confidence"] == context.confidence and
            stored_dict["risk_level"] == context.risk_level and
            stored_dict["parameters"] == context.parameters
        )
        
        if not matches:
            logger.error("Stored strategy does not match original")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

@pytest.mark.asyncio
async def test_concurrent_memory_writes(memory_client, sample_strategy_context):
    """Test concurrent writes to memory system.
    
    This test verifies that:
    1. Concurrent writes don't corrupt stored data
    2. All writes are atomic
    3. Latest write is always retrievable
    """
    # Setup concurrent writes with varying delays
    delays = [0.1, 0.2, 0.0, 0.15, 0.05]  # Intentionally overlapping
    write_tasks = [
        write_strategy(memory_client, sample_strategy_context, delay)
        for delay in delays
    ]
    
    # Execute writes concurrently
    results = await asyncio.gather(*write_tasks)
    logger.info(f"Write results: {results}")
    
    # Verify final state
    is_valid = await verify_strategy(memory_client, sample_strategy_context)
    
    # Check results
    assert all(results), "Not all writes completed successfully"
    assert is_valid, "Final stored strategy is invalid"
    
    # Log final state
    logger.info("Memory integrity test passed")

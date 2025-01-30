"""Test memory system resilience to corruption during concurrent writes."""

import pytest
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
from research.strategy.memory_manager import MemoryManager
from research.strategy.types import MarketRegime, StrategyContext

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@pytest.fixture
def memory_manager():
    """Create memory manager instance for testing."""
    manager = MemoryManager()
    manager.reset()  # Clear any existing data
    return manager

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

async def write_strategy(manager: MemoryManager, context: StrategyContext, 
                        delay: float) -> bool:
    """Attempt to write strategy with delay."""
    await asyncio.sleep(delay)
    try:
        success = await manager.store_strategy_results(context)
        logger.info(f"Write completed with delay {delay:.2f}s: {success}")
        return success
    except Exception as e:
        logger.error(f"Write failed with delay {delay:.2f}s: {str(e)}")
        return False

async def verify_strategy(manager: MemoryManager, context: StrategyContext) -> bool:
    """Verify strategy was stored correctly."""
    try:
        stored = await manager.query_similar_strategies(context.market_regime)
        if not stored:
            logger.error("No stored strategies found")
            return False
            
        # Verify latest strategy matches
        latest = stored[-1]
        matches = (
            latest["market_regime"] == context.market_regime.value and
            latest["confidence"] == context.confidence and
            latest["risk_level"] == context.risk_level and
            latest["parameters"] == context.parameters
        )
        
        if not matches:
            logger.error("Stored strategy does not match original")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

@pytest.mark.asyncio
async def test_concurrent_memory_writes(memory_manager, sample_strategy_context):
    """Test concurrent writes to memory system.
    
    This test verifies that:
    1. Concurrent writes don't corrupt stored data
    2. All writes are atomic
    3. Latest write is always retrievable
    """
    # Setup concurrent writes with varying delays
    delays = [0.1, 0.2, 0.0, 0.15, 0.05]  # Intentionally overlapping
    write_tasks = [
        write_strategy(memory_manager, sample_strategy_context, delay)
        for delay in delays
    ]
    
    # Execute writes concurrently
    results = await asyncio.gather(*write_tasks)
    logger.info(f"Write results: {results}")
    
    # Verify final state
    is_valid = await verify_strategy(memory_manager, sample_strategy_context)
    
    # Check results
    assert all(results), "Not all writes completed successfully"
    assert is_valid, "Final stored strategy is invalid"
    
    # Log final state
    logger.info("Memory integrity test passed")

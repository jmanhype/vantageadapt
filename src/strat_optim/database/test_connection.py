"""Test database connection."""

import asyncio
import logging
from .connection import DatabaseConnection

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_connection():
    """Test database connection."""
    db = DatabaseConnection()
    try:
        await db.init()
        logger.info("Database connection successful")
        
        # Test creating a strategy
        strategy_id = await db.create_strategy("Test Strategy")
        logger.info(f"Created strategy with ID: {strategy_id}")
        
        # Test getting all strategies
        strategies = await db.get_all_strategies()
        logger.info(f"Found {len(strategies)} strategies")
        
        await db.close()
        logger.info("Database connection closed")
        
    except Exception as e:
        logger.error(f"Error testing database connection: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_connection()) 
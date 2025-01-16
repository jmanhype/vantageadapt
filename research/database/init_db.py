"""Database initialization script."""

import asyncio
import logging
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy_utils import database_exists, create_database
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from research.database.models.trading import Base
from research.database.config.settings import DATABASE_URL, SYNC_DATABASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def init_db():
    """Initialize database and create tables."""
    engine = None
    try:
        # Load environment variables
        load_dotenv(os.path.join(project_root, 'research', '.env'))
        logger.info("Loaded environment variables")
        
        logger.info(f"Using database URL: {DATABASE_URL}")
        
        # Create engine
        engine = create_async_engine(
            DATABASE_URL,
            echo=True  # Enable SQL logging
        )
        logger.info("Created async engine")
        
        # Create database if it doesn't exist
        logger.info(f"Checking if database exists at: {SYNC_DATABASE_URL}")
        
        try:
            if not database_exists(SYNC_DATABASE_URL):
                create_database(SYNC_DATABASE_URL)
                logger.info("Created new database")
            else:
                logger.info("Database already exists")
        except Exception as e:
            logger.error(f"Error checking/creating database: {str(e)}")
            raise
        
        # Create tables
        try:
            async with engine.begin() as conn:
                logger.info("Dropping existing tables...")
                await conn.run_sync(Base.metadata.drop_all)
                
                logger.info("Creating new tables...")
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
        
    finally:
        if engine:
            await engine.dispose()
            logger.info("Disposed database engine")

if __name__ == "__main__":
    asyncio.run(init_db()) 
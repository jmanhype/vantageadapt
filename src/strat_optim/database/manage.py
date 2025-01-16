#!/usr/bin/env python3
"""Database management script.

This script provides functionality for managing database migrations and setup.
"""
import asyncio
import logging
from typing import Optional
import typer
from alembic.config import Config
from alembic import command
import os
import sys
from sqlalchemy import create_engine

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from research.database import db
from research.database.models.base import Base
from research.database.config.settings import SYNC_DATABASE_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Typer app
app = typer.Typer()

def get_alembic_config() -> Config:
    """Get Alembic configuration."""
    config = Config()
    config.set_main_option("script_location", "research/database/migrations")
    return config

@app.command()
def create():
    """Create database tables."""
    try:
        # Create synchronous engine
        engine = create_engine(SYNC_DATABASE_URL)
        
        # Create tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise typer.Exit(1)

@app.command()
def migrate(revision: Optional[str] = None):
    """Run database migrations."""
    try:
        config = get_alembic_config()
        if revision:
            command.upgrade(config, revision)
        else:
            command.upgrade(config, "head")
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.error(f"Failed to run database migrations: {str(e)}")
        raise typer.Exit(1)

@app.command()
def rollback(revision: str):
    """Rollback database migrations."""
    try:
        config = get_alembic_config()
        command.downgrade(config, revision)
        logger.info(f"Database rolled back to revision {revision}")
    except Exception as e:
        logger.error(f"Failed to rollback database: {str(e)}")
        raise typer.Exit(1)

@app.command()
def revision(message: str):
    """Create a new migration revision."""
    try:
        config = get_alembic_config()
        command.revision(config, autogenerate=True, message=message)
        logger.info("Created new migration revision")
    except Exception as e:
        logger.error(f"Failed to create migration revision: {str(e)}")
        raise typer.Exit(1)

def main():
    """Main entry point."""
    try:
        app()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
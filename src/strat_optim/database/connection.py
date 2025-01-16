"""Database connection module."""

import os
import json
import logging
from typing import Optional, List, Dict, Any
import asyncpg
from .config.settings import DATABASE_CONFIG

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager."""
    
    def __init__(self):
        """Initialize database connection."""
        self.pool = None
        self._is_connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected
        
    async def init(self) -> None:
        """Initialize database connection."""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=DATABASE_CONFIG["host"],
                port=DATABASE_CONFIG["port"],
                user=DATABASE_CONFIG["user"],
                password=DATABASE_CONFIG["password"],
                database=DATABASE_CONFIG["database"],
                min_size=1,
                max_size=10,
                max_queries=50000,
                max_inactive_connection_lifetime=300.0
            )
            
            # Create tables if they don't exist
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        id SERIAL PRIMARY KEY,
                        theme TEXT NOT NULL,
                        description TEXT,
                        conditions JSONB,
                        parameters JSONB,
                        market_context JSONB,
                        strategy_insights JSONB,
                        status TEXT NOT NULL DEFAULT 'inactive',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );

                    CREATE TABLE IF NOT EXISTS performance (
                        id SERIAL PRIMARY KEY,
                        strategy_id INTEGER REFERENCES strategies(id),
                        total_return DECIMAL,
                        sharpe_ratio DECIMAL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            
            self._is_connected = True
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database connection: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close database connection."""
        try:
            if self.pool:
                await self.pool.close()
            self._is_connected = False
            logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
            raise
            
    async def get_all_strategies(self) -> List[Dict[str, Any]]:
        """Get all strategies with their performance metrics."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT s.*, 
                       COALESCE(json_build_object(
                           'total_return', p.total_return,
                           'sharpe_ratio', p.sharpe_ratio
                       ), json_build_object(
                           'total_return', 0,
                           'sharpe_ratio', 0
                       )) as performance
                FROM strategies s
                LEFT JOIN performance p ON s.id = p.strategy_id
                ORDER BY s.created_at DESC
            """)
            return [dict(row) for row in rows]
            
    async def get_strategy(self, strategy_id: int) -> Optional[Dict[str, Any]]:
        """Get a single strategy by ID with its performance metrics."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT s.*, 
                       COALESCE(json_build_object(
                           'total_return', p.total_return,
                           'sharpe_ratio', p.sharpe_ratio
                       ), json_build_object(
                           'total_return', 0,
                           'sharpe_ratio', 0
                       )) as performance
                FROM strategies s
                LEFT JOIN performance p ON s.id = p.strategy_id
                WHERE s.id = $1
            """, strategy_id)
            return dict(row) if row else None
            
    async def get_active_strategies(self) -> List[Dict[str, Any]]:
        """Get all active strategies."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM strategies WHERE status = 'active'
            """)
            return [dict(row) for row in rows]
            
    async def create_strategy(self, theme: str) -> int:
        """Create a new strategy."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO strategies (theme, status)
                VALUES ($1, 'inactive')
                RETURNING id
            """, theme)
            return row['id']
            
    async def update_strategy_status(self, strategy_id: int, status: str):
        """Update strategy status."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE strategies 
                SET status = $1
                WHERE id = $2
            """, status, strategy_id)

    async def save_strategy(
        self,
        theme: str,
        description: str,
        conditions: Dict[str, List[str]],
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        market_context: Dict[str, Any],
        strategy_insights: Dict[str, Any],
        performance_analysis: Dict[str, Any]
    ) -> int:
        """Save complete strategy data to database.
        
        Args:
            theme: Strategy theme
            description: Strategy description
            conditions: Trading conditions
            parameters: Strategy parameters
            metrics: Performance metrics
            market_context: Market context data
            strategy_insights: Strategy insights
            performance_analysis: Performance analysis data
            
        Returns:
            ID of the created strategy
        """
        try:
            async with self.pool.acquire() as conn:
                # Convert dictionaries to JSON strings
                conditions_json = json.dumps(conditions)
                parameters_json = json.dumps(parameters)
                market_context_json = json.dumps(market_context)
                strategy_insights_json = json.dumps(strategy_insights)
                
                # Insert strategy
                strategy_row = await conn.fetchrow("""
                    INSERT INTO strategies (
                        theme, description, conditions, parameters,
                        market_context, strategy_insights, status
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, 'inactive')
                    RETURNING id
                """, theme, description, conditions_json, parameters_json,
                     market_context_json, strategy_insights_json)
                
                strategy_id = strategy_row['id']
                
                # Insert performance metrics
                await conn.execute("""
                    INSERT INTO performance (
                        strategy_id, total_return, sharpe_ratio
                    )
                    VALUES ($1, $2, $3)
                """, strategy_id, metrics.get('total_return', 0.0),
                     metrics.get('sharpe_ratio', 0.0))
                
                return strategy_id
                
        except Exception as e:
            logger.error(f"Error saving strategy: {str(e)}")
            raise

# Create global database instance
db = DatabaseConnection()
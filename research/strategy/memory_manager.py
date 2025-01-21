"""Memory manager for trading strategies."""

from typing import Dict, List, Optional, Any, Tuple
import os
import logging
from datetime import datetime
from mem0 import MemoryClient
from .types import MarketRegime, StrategyContext, BacktestResults
import json

logger = logging.getLogger(__name__)

class TradingMemoryManager:
    """Manages memory for trading strategies using mem0ai."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the trading memory manager.
        
        Args:
            api_key: Optional API key for mem0ai. If not provided, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        self.enabled = False
        self.client = None
        
        if not self.api_key:
            logger.warning("No MEM0_API_KEY provided, memory system will be disabled")
            return

        try:
            self.client = MemoryClient(api_key=self.api_key)
            self.enabled = True
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            self.client = None

    def store_strategy_results(self, context: StrategyContext, results: BacktestResults, iteration: Optional[int] = None) -> bool:
        """Store strategy results in memory.
        
        Args:
            context: The strategy context containing market regime and parameters
            results: The backtest results containing performance metrics
            iteration: Optional iteration number for the strategy run
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.client or not self.enabled:
            logger.debug("Memory system disabled, skipping storage")
            return False

        try:
            # Convert context and results to a structured memory format
            memory_content = {
                "market_regime": context.market_regime.value,
                "confidence": context.confidence,
                "risk_level": context.risk_level,
                "parameters": context.parameters,
                "performance": {
                    "total_return": float(results.total_return),
                    "total_pnl": float(results.total_pnl),
                    "sortino_ratio": float(results.sortino_ratio),
                    "win_rate": float(results.win_rate),
                    "total_trades": results.total_trades
                },
                "timestamp": datetime.utcnow().isoformat(),
                "iteration": iteration
            }

            # Add memory with metadata
            response = self.client.add(
                messages=[{
                    "role": "system",
                    "content": json.dumps(memory_content),
                    "metadata": {
                        "type": "strategy_results",
                        "market_regime": context.market_regime.value,
                        "success": results.total_return > 0,
                        "iteration": iteration
                    }
                }],
                user_id="trading_system",
                agent_id="strategy_optimizer"
            )
            logger.info(f"Stored strategy results in memory: {response}")
            return True
        except Exception as e:
            logger.error(f"Failed to store strategy results: {e}")
            return False

    def query_similar_strategies(self, market_regime: MarketRegime) -> List[Dict[str, Any]]:
        """Query for similar successful strategies based on market regime.
        
        Args:
            market_regime: The current market regime to find strategies for
            
        Returns:
            List of similar successful strategies with their parameters and results
        """
        if not self.client or not self.enabled:
            logger.debug("Memory system disabled, returning empty list")
            return []

        try:
            # Get all memories for our trading system
            memories = self.client.get_all(
                user_id="trading_system",
                agent_id="strategy_optimizer"
            )
            
            # Debug: Log memory structure
            logger.debug(f"Memory response type: {type(memories)}")
            if memories:
                logger.debug(f"First memory structure: {memories[0]}")

            # Filter and parse memories
            similar_strategies = []
            for memory in memories:
                try:
                    # Try to get content directly from memory dict
                    content = json.loads(memory.get("content", "{}"))
                    if (content.get("market_regime") == market_regime.value and 
                        content.get("performance", {}).get("total_return", 0) > 0):
                        similar_strategies.append(content)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse memory: {e}")
                    continue

            logger.info(f"Found {len(similar_strategies)} similar successful strategies")
            return similar_strategies
        except Exception as e:
            logger.error(f"Failed to query similar strategies: {e}")
            return []

    def reset(self) -> bool:
        """Reset the memory store.
        
        Returns:
            bool: True if reset was successful, False otherwise
        """
        if not self.client or not self.enabled:
            logger.debug("Memory system disabled, skipping reset")
            return False

        try:
            # Get all memories first
            memories = self.client.get_all(
                user_id="trading_system",
                agent_id="strategy_optimizer"
            )
            
            # Delete each memory by ID
            success = True
            for memory in memories:
                try:
                    # Memory ID is in memory["id"]
                    memory_id = memory.get("id")
                    if memory_id:
                        self.client.delete(memory_id)
                except Exception as e:
                    logger.warning(f"Failed to delete memory {memory.get('id', 'unknown')}: {e}")
                    success = False
                    
            if success:
                logger.info("Memory store reset successfully")
            else:
                logger.warning("Memory store reset completed with some errors")
            return success
        except Exception as e:
            logger.error(f"Failed to reset memory store: {e}")
            return False 
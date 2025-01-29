"""Memory manager for trading strategies."""

from typing import Dict, List, Optional, Any, Tuple, Union
import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from mem0 import MemoryClient
from .types import MarketRegime, StrategyContext, BacktestResults
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from config.mem0_config import MEM0_CONFIG, validate_config
import json

# Set up logging first
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class TradingMemoryManager:
    """Manages memory for trading strategies using mem0ai."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the trading memory manager.
        
        Args:
            api_key: Optional API key for mem0ai. If not provided, will use config.
        """
        # Load environment variables first
        load_dotenv()
        
        # Use provided API key or from environment or config
        self.api_key = api_key or os.getenv("MEM0_API_KEY") or MEM0_CONFIG["api_key"]
        self.enabled = False
        self.client = None
        
        logger.debug("Initializing memory manager...")
        
        # Early validation
        if not self.api_key:
            logger.error("No API key provided")
            return
            
        # Validate configuration
        if not validate_config():
            logger.error("Invalid memory system configuration")
            return

        try:
            # Create client without testing connection
            self.client = MemoryClient(api_key=self.api_key)
            self.enabled = True
            logger.info("Memory system initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
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
            # Check if results meet success criteria
            success = (
                results.total_return >= MEM0_CONFIG["success_criteria"]["min_return"] and
                results.sortino_ratio >= MEM0_CONFIG["success_criteria"]["min_sortino"] and
                results.win_rate >= MEM0_CONFIG["success_criteria"]["min_win_rate"]
            )

            # Create a more descriptive conversation
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Store trading strategy results for {context.market_regime.value} market regime:\n"
                        f"- Confidence: {context.confidence}\n"
                        f"- Risk Level: {context.risk_level}\n"
                        f"- Parameters: {json.dumps(context.parameters, indent=2)}"
                    )
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
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
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "iteration": iteration
                    }, indent=2)
                }
            ]

            # Add memory with metadata
            response = self.client.add(
                messages=messages,
                user_id=MEM0_CONFIG["user_id"],
                metadata={
                    "type": MEM0_CONFIG["metadata_types"]["strategy_results"],
                    "market_regime": context.market_regime.value,
                    "success": success,
                    "iteration": iteration,
                    "confidence": context.confidence,
                    "risk_level": context.risk_level,
                    "total_return": float(results.total_return),
                    "total_pnl": float(results.total_pnl),
                    "sortino_ratio": float(results.sortino_ratio),
                    "win_rate": float(results.win_rate),
                    "total_trades": results.total_trades,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

            # Check if the response is valid
            if response is not None:
                logger.info("Successfully stored strategy results in memory")
                return True
            else:
                logger.warning("No valid response from memory storage")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store strategy results: {e}")
            logger.debug(f"Exception details:", exc_info=True)
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
            # Search for similar strategies
            query = f"Strategy execution in {market_regime.value} market regime"
            memories = self.client.search(
                query=query,
                user_id=MEM0_CONFIG["user_id"],
                limit=MEM0_CONFIG["search_limit"]
            )
            
            logger.debug(f"Search response: {memories}")
            
            # Parse and filter memories
            similar_strategies = []
            for memory in memories:
                try:
                    metadata = memory.get("metadata", {})
                    
                    logger.debug(f"Memory: {memory}")
                    logger.debug(f"Metadata: {metadata}")
                    
                    # Check if this is a successful strategy for the right market regime
                    if (metadata.get("market_regime") == market_regime.value and 
                        metadata.get("success", False) and
                        metadata.get("type") == MEM0_CONFIG["metadata_types"]["strategy_results"]):
                        
                        # Create strategy details from metadata
                        strategy_details = {
                            "market_regime": metadata.get("market_regime"),
                            "confidence": metadata.get("confidence"),
                            "risk_level": metadata.get("risk_level"),
                            "performance": {
                                "total_return": metadata.get("total_return"),
                                "total_pnl": metadata.get("total_pnl"),
                                "sortino_ratio": metadata.get("sortino_ratio"),
                                "win_rate": metadata.get("win_rate"),
                                "total_trades": metadata.get("total_trades")
                            }
                        }
                        similar_strategies.append(strategy_details)
                except (KeyError, AttributeError) as e:
                    logger.warning(f"Failed to parse memory: {e}")
                    continue
            
            logger.info(f"Found {len(similar_strategies)} similar successful strategies")
            return similar_strategies
        except Exception as e:
            logger.error(f"Failed to query similar strategies: {e}")
            logger.debug("Exception details:", exc_info=True)
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
            memories = self.client.get_all(user_id=MEM0_CONFIG["user_id"])
            
            if not memories:
                logger.info("No memories to reset")
                return True
                
            # Delete each memory
            success = True
            for memory in memories:
                try:
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
            logger.debug(f"Exception details:", exc_info=True)
            return False 
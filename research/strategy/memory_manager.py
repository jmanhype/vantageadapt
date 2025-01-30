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
            context: Strategy context
            results: Backtest results
            iteration: Optional iteration number
            
        Returns:
            True if storage was successful
        """
        try:
            if not self.enabled:
                logger.warning("Memory system not enabled")
                return False
                
            # Calculate a weighted score for the strategy
            score = (
                (0.4 * (1 + float(results.total_return))) +  # Normalize to prevent negative scores
                (0.3 * (1 + float(results.sortino_ratio))) +
                (0.3 * float(results.win_rate))
            )
            
            # More lenient success criteria based on composite score
            success = score > 1.0  # This means it's better than average in some aspects
            
            # Create a more descriptive conversation
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Store trading strategy results for {context.regime.value} market regime:\n"
                        f"- Confidence: {context.confidence}\n"
                        f"- Risk Level: {context.risk_level}\n"
                        f"- Parameters: {json.dumps(context.parameters, indent=2)}"
                    )
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "regime": context.regime.value,
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
                        "score": score,
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
                    "regime": context.regime.value,  # Use consistent key name
                    "success": success,
                    "score": score,
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
                logger.info(f"Successfully stored strategy results in memory with score {score}")
                return True
            else:
                logger.warning("No valid response from memory storage")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store strategy results: {e}")
            logger.debug(f"Exception details:", exc_info=True)
            return False

    def query_similar_strategies(self, market_regime: MarketRegime) -> List[Dict[str, Any]]:
        """Query for similar strategies based on market regime, including partially successful ones.
        
        Args:
            market_regime: The current market regime to find strategies for
            
        Returns:
            List of similar strategies with their parameters and results, sorted by effectiveness
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
                    
                    # Check if this is a strategy for the right market regime
                    if (metadata.get("regime") == market_regime.value and  # Fixed key name
                        metadata.get("type") == MEM0_CONFIG["metadata_types"]["strategy_results"]):
                        
                        # Use pre-calculated score if available, otherwise calculate it
                        score = metadata.get("score")
                        if score is None:
                            total_return = metadata.get("total_return", 0)
                            sortino_ratio = metadata.get("sortino_ratio", 0)
                            win_rate = metadata.get("win_rate", 0)
                            
                            score = (
                                (0.4 * (1 + total_return)) +
                                (0.3 * (1 + sortino_ratio)) +
                                (0.3 * win_rate)
                            )
                        
                        # Create strategy details from metadata
                        strategy_details = {
                            "market_regime": metadata.get("regime"),  # Fixed key name
                            "confidence": metadata.get("confidence"),
                            "risk_level": metadata.get("risk_level"),
                            "performance": {
                                "total_return": metadata.get("total_return", 0),
                                "total_pnl": metadata.get("total_pnl", 0),
                                "sortino_ratio": metadata.get("sortino_ratio", 0),
                                "win_rate": metadata.get("win_rate", 0),
                                "total_trades": metadata.get("total_trades", 0)
                            },
                            "score": score,
                            "parameters": metadata.get("parameters", {})
                        }
                        similar_strategies.append(strategy_details)
                except (KeyError, AttributeError) as e:
                    logger.warning(f"Failed to parse memory: {e}")
                    continue
            
            # Sort strategies by score in descending order
            similar_strategies.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top N strategies
            top_n = min(5, len(similar_strategies))
            similar_strategies = similar_strategies[:top_n]
            
            logger.info(f"Found {len(similar_strategies)} similar strategies for {market_regime.value}")
            for i, strat in enumerate(similar_strategies):
                logger.info(f"Strategy {i+1}: Score={strat['score']:.3f}, Return={strat['performance']['total_return']:.3f}")
            
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
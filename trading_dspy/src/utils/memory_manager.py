"""Memory manager for trading strategies."""

from typing import Dict, List, Optional, Any, Tuple, Union
import os
import logging
from datetime import datetime, timezone, timedelta
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

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Suppress debug output from dependencies
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class TradingMemoryManager:
    """Manages memory for trading strategies using mem0ai."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the trading memory manager.
        
        Args:
            api_key: Optional API key for mem0ai. If not provided, will use config.
        """
        # Load environment variables first
        load_dotenv()
        
        # Use provided API key or from config
        self.api_key = api_key or MEM0_CONFIG["api_key"]
        self.enabled = False
        self.client = None
        
        logger.info("Initializing memory manager...")
        
        # Early validation
        if not self.api_key:
            logger.error("No API key provided")
            return
            
        # Validate configuration
        if not validate_config():
            logger.error("Invalid memory system configuration")
            return

        try:
            # Create client and test connection
            self.client = MemoryClient(api_key=self.api_key)
            # Test connection by making a simple query
            test_response = self.client.search(
                query="test",
                user_id=MEM0_CONFIG["user_id"],
                limit=1,
                output_format="v1.1"  # Use latest format
            )
            if test_response is not None:
                self.enabled = True
                logger.info("Memory system initialized and connection tested successfully")
            else:
                logger.error("Memory system connection test failed")
                self.client = None
                
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {str(e)}")
            logger.debug("Exception details:", exc_info=True)
            self.client = None

    def store_strategy_results(self, context: StrategyContext, results: Union[BacktestResults, Dict[str, Any]], iteration: Optional[int] = None) -> bool:
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
                
            # Convert dict to BacktestResults if needed
            if isinstance(results, dict):
                if 'backtest_results' in results:
                    br = results['backtest_results']
                    if isinstance(br, dict):
                        results = BacktestResults(
                            total_return=float(br.get('total_return', 0.0)),
                            total_pnl=float(br.get('total_pnl', 0.0)),
                            sortino_ratio=float(br.get('sortino_ratio', 0.0)),
                            win_rate=float(br.get('win_rate', 0.0)),
                            total_trades=int(br.get('total_trades', 0)),
                            trades=br.get('trades', []),
                            metrics=br.get('metrics', {})
                        )
                    
            # Calculate a weighted score for the strategy
            score = (
                (0.4 * (1 + float(results.total_return))) +  # Normalize to prevent negative scores
                (0.3 * (1 + float(results.sortino_ratio))) +
                (0.3 * float(results.win_rate))
            )
            
            # More lenient success criteria based on composite score
            success = score > 0.8  # Lower threshold to allow more strategies to be stored
            
            # Create strategy content
            strategy_content = {
                "strategy": {
                    "regime": context.regime.value,
                    "confidence": context.confidence,
                    "risk_level": context.risk_level,
                    "parameters": context.parameters,
                    "performance": {
                        "total_return": float(results.total_return),
                        "total_pnl": float(results.total_pnl),
                        "sortino_ratio": float(results.sortino_ratio),
                        "win_rate": float(results.win_rate),
                        "total_trades": results.total_trades,
                        "trades": list(results.trades)[:5] if results.trades else []  # Store only first 5 trades as example
                    },
                    "score": score,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "iteration": iteration
                }
            }
            
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
                    "content": json.dumps(strategy_content, indent=2)
                }
            ]
            
            # Add memory with metadata and categories
            try:
                # Prepare metadata with proper error handling
                metadata = {
                    "type": MEM0_CONFIG["metadata_types"]["strategy_results"],
                    "regime": str(context.regime.value),
                    "success": bool(success),
                    "score": float(score),
                    "iteration": iteration if iteration is not None else 0,
                    "confidence": float(context.confidence),
                    "risk_level": str(context.risk_level),
                    "total_return": float(results.total_return),
                    "total_pnl": float(results.total_pnl),
                    "sortino_ratio": float(results.sortino_ratio),
                    "win_rate": float(results.win_rate),
                    "total_trades": int(results.total_trades),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                
                # Parameters might be complex, convert to string
                params_str = json.dumps(context.parameters)
                metadata["parameters_json"] = params_str
                
                # Ensure all metadata is JSON serializable
                for k, v in metadata.items():
                    if not isinstance(v, (str, int, float, bool, type(None))):
                        metadata[k] = str(v)
                
                # Convert categories to strings
                categories = ["trading_strategy"]
                if hasattr(context.regime, 'value') and context.regime.value:
                    categories.append(str(context.regime.value).lower())
                
                # Serialize content to JSON string
                content_json = json.dumps(strategy_content)
                
                # Add memory
                response = self.client.add(
                    messages=messages,
                    user_id=MEM0_CONFIG["user_id"],
                    content=content_json,  # Add structured content as JSON string
                    categories=categories,  # Add categories for better filtering
                    metadata=metadata,
                    output_format="v1.1"  # Use latest format
                )
            except Exception as add_err:
                logger.error(f"Error in client.add for strategy results: {add_err}")
                # Try without content as fallback
                try:
                    response = self.client.add(
                        messages=messages,
                        user_id=MEM0_CONFIG["user_id"],
                        metadata=metadata
                    )
                except Exception as fallback_err:
                    logger.error(f"Fallback also failed: {fallback_err}")
                    return False

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

    def query_similar_strategies(
        self,
        market_regime: Union[MarketRegime, str],
        page: int = 1,
        page_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Query for similar strategies based on market regime, including partially successful ones.
        
        Args:
            market_regime: The current market regime to find strategies for (can be MarketRegime enum or string)
            page: Page number for pagination (1-based)
            page_size: Number of results per page
            
        Returns:
            List of similar strategies with their parameters and results, sorted by effectiveness
        """
        if not self.client or not self.enabled:
            logger.debug("Memory system disabled, returning empty list")
            return []
            
        # Convert string regime to enum if needed
        if isinstance(market_regime, str):
            try:
                market_regime = MarketRegime[market_regime]
            except KeyError:
                logger.error(f"Invalid market regime string: {market_regime}")
                return []
            
        if market_regime == MarketRegime.UNKNOWN:
            logger.debug("Unknown market regime, returning empty list")
            return []

        try:
            # Search for similar strategies with metadata filter using v2 search
            query = f"Trading strategy for {market_regime.value} market conditions"
            logger.debug(f"Searching with query: {query}")
            
            # Calculate offset for pagination
            offset = (page - 1) * page_size
            
            # Use v2 search with advanced filtering
            memories = self.client.search(
                query=query,
                user_id=MEM0_CONFIG["user_id"],
                limit=page_size,
                offset=offset,
                version="v2",
                output_format="v1.1",
                metadata_filter={
                    "AND": [
                        {"type": {"eq": MEM0_CONFIG["metadata_types"]["strategy_results"]}},
                        {"regime": {"eq": market_regime.value}},
                        {"success": {"eq": True}},
                        {"score": {"gte": 0.5}}  # Only get strategies with decent scores
                    ]
                }
            )
            
            logger.debug(f"Search response: {memories}")
            
            # Parse and filter memories
            similar_strategies = []
            for memory in memories:
                try:
                    metadata = memory.get("metadata", {})
                    content = memory.get("content", {})
                    
                    logger.debug(f"Memory: {memory}")
                    logger.debug(f"Metadata: {metadata}")
                    
                    # Extract strategy details from content if available
                    strategy_data = content.get("strategy", {}) if isinstance(content, dict) else {}
                    
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
                    
                    # Create strategy details from metadata and content
                    strategy_details = {
                        "memory_id": memory.get("id"),  # Include memory ID
                        "market_regime": metadata.get("regime"),
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
                        "parameters": strategy_data.get("parameters", metadata.get("parameters", {})),
                        "timestamp": metadata.get("timestamp")
                    }
                    similar_strategies.append(strategy_details)
                except (KeyError, AttributeError) as e:
                    logger.warning(f"Failed to parse memory: {e}")
                    continue
            
            # Sort strategies by score in descending order
            similar_strategies.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Found {len(similar_strategies)} similar strategies for {market_regime.value} (page {page})")
            for i, strat in enumerate(similar_strategies):
                logger.info(f"Strategy {i+1}: Score={strat['score']:.3f}, Return={strat['performance']['total_return']:.3f}")
            
            return similar_strategies
        except Exception as e:
            logger.error(f"Failed to query similar strategies: {e}")
            logger.debug("Exception details:", exc_info=True)
            return []

    def get_strategy_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get the history of changes for a specific strategy.
        
        Args:
            memory_id: The ID of the memory to get history for
            
        Returns:
            List of historical changes
        """
        if not self.enabled:
            logger.debug("Memory system disabled, returning empty list")
            return []
            
        try:
            history = self.client.history(memory_id)
            return history if history else []
        except Exception as e:
            logger.error(f"Failed to get strategy history: {e}")
            logger.debug("Exception details:", exc_info=True)
            return []

    def batch_update_strategies(self, updates: List[Dict[str, Any]]) -> bool:
        """Update multiple strategies in a single call.
        
        Args:
            updates: List of dictionaries containing memory_id and updated data
            
        Returns:
            True if all updates were successful
        """
        if not self.enabled:
            logger.debug("Memory system disabled, returning False")
            return False
            
        try:
            response = self.client.batch_update(updates)
            return response is not None
        except Exception as e:
            logger.error(f"Failed to batch update strategies: {e}")
            logger.debug("Exception details:", exc_info=True)
            return False

    def batch_delete_strategies(self, memory_ids: List[str]) -> bool:
        """Delete multiple strategies in a single call.
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            True if all deletions were successful
        """
        if not self.enabled:
            logger.debug("Memory system disabled, returning False")
            return False
            
        try:
            delete_memories = [{"memory_id": mid} for mid in memory_ids]
            response = self.client.batch_delete(delete_memories)
            return response is not None
        except Exception as e:
            logger.error(f"Failed to batch delete strategies: {e}")
            logger.debug("Exception details:", exc_info=True)
            return False

    def get_all_strategies(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """Get all stored strategies with pagination.
        
        Args:
            page: Page number (1-based)
            page_size: Number of results per page
            
        Returns:
            Dictionary containing results and pagination info
        """
        if not self.enabled:
            logger.debug("Memory system disabled, returning empty result")
            return {"results": [], "total": 0, "page": page, "page_size": page_size}
            
        try:
            # Calculate offset for pagination
            offset = (page - 1) * page_size
            
            # Query all strategy results with sorting by timestamp
            memories = self.client.search(
                query="type:strategy_results",
                user_id=MEM0_CONFIG["user_id"],
                limit=page_size,
                offset=offset,
                version="v2",
                output_format="v1.1",
                metadata_filter={
                    "type": {"eq": MEM0_CONFIG["metadata_types"]["strategy_results"]}
                },
                sort_by=[{"field": "timestamp", "order": "desc"}]  # Sort by timestamp descending
            )
            
            # Process results
            results = []
            for memory in memories:
                metadata = memory.get("metadata", {})
                content = memory.get("content", {})
                
                strategy_data = {
                    "id": memory.get("id"),
                    "regime": metadata.get("regime"),
                    "confidence": metadata.get("confidence"),
                    "risk_level": metadata.get("risk_level"),
                    "performance": {
                        "total_return": metadata.get("total_return", 0),
                        "total_pnl": metadata.get("total_pnl", 0),
                        "sortino_ratio": metadata.get("sortino_ratio", 0),
                        "win_rate": metadata.get("win_rate", 0),
                        "total_trades": metadata.get("total_trades", 0)
                    },
                    "score": metadata.get("score", 0),
                    "timestamp": metadata.get("timestamp"),
                    "parameters": content.get("strategy", {}).get("parameters", {})
                }
                results.append(strategy_data)
                
                # Ensure we don't exceed page_size
                if len(results) >= page_size:
                    break
            
            # Get total count (if available)
            total = len(results)  # This is just the current page count
            
            return {
                "results": results[:page_size],  # Ensure we don't return more than page_size
                "total": total,
                "page": page,
                "page_size": page_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get all strategies: {e}")
            logger.debug("Exception details:", exc_info=True)
            return {"results": [], "total": 0, "page": page, "page_size": page_size}

    def store_strategy(self, strategy: Union[StrategyContext, Dict[str, Any]]) -> bool:
        """Store a generated trading strategy in memory.
        
        Args:
            strategy: StrategyContext object or dictionary containing the strategy details
            
        Returns:
            bool: True if storage was successful
        """
        if not self.enabled:
            logger.debug("Memory system disabled, returning False")
            return False
            
        try:
            # Create a strategy dictionary that will work regardless of input type
            final_strategy_dict = {}
            
            # Handle the case when strategy might be a dspy.Prediction object
            if hasattr(strategy, '__dict__') and not isinstance(strategy, dict) and not isinstance(strategy, StrategyContext):
                # First, log what we're dealing with to help debug
                logger.debug(f"Converting object of type {type(strategy)} to dict")
                
                # Convert dspy.Prediction to dict by getting all attributes
                for key in dir(strategy):
                    # Skip private and special methods/attributes
                    if key.startswith('_') or callable(getattr(strategy, key)):
                        continue
                    
                    # Get the attribute value
                    value = getattr(strategy, key)
                    logger.debug(f"Found attribute {key} with value type {type(value)}")
                    
                    # Store in our dict
                    final_strategy_dict[key] = value
                
                # Add essential fields if missing
                if 'regime' not in final_strategy_dict:
                    final_strategy_dict['regime'] = 'UNKNOWN'
                
                if 'confidence' not in final_strategy_dict:
                    if hasattr(strategy, 'confidence') and getattr(strategy, 'confidence') is not None:
                        final_strategy_dict['confidence'] = float(getattr(strategy, 'confidence'))
                    else:
                        final_strategy_dict['confidence'] = 0.0
                
                if 'risk_level' not in final_strategy_dict:
                    final_strategy_dict['risk_level'] = 'moderate'
                
                # Convert parameters if it's a string
                if 'parameters' in final_strategy_dict and isinstance(final_strategy_dict['parameters'], str):
                    try:
                        final_strategy_dict['parameters'] = json.loads(final_strategy_dict['parameters'])
                    except json.JSONDecodeError:
                        final_strategy_dict['parameters'] = {'raw': final_strategy_dict['parameters']}
                elif 'parameters' not in final_strategy_dict:
                    # Create default parameters
                    final_strategy_dict['parameters'] = {}
                    
                    # Try to extract relevant info from other fields 
                    if hasattr(strategy, 'trade_signal'):
                        final_strategy_dict['parameters']['trade_signal'] = getattr(strategy, 'trade_signal')
                    
                    if hasattr(strategy, 'reasoning'):
                        final_strategy_dict['parameters']['reasoning'] = getattr(strategy, 'reasoning')
                    
                    # Add strategy-specific parameters that might be useful
                    for param_name in ['stop_loss', 'take_profit', 'position_size', 'indicators']:
                        if hasattr(strategy, param_name):
                            value = getattr(strategy, param_name)
                            # Handle different data types appropriately
                            if isinstance(value, str):
                                try:
                                    # Try to parse JSON strings
                                    final_strategy_dict['parameters'][param_name] = json.loads(value)
                                except json.JSONDecodeError:
                                    final_strategy_dict['parameters'][param_name] = value
                            else:
                                final_strategy_dict['parameters'][param_name] = value
            
            # Handle dictionary input 
            elif isinstance(strategy, dict):
                final_strategy_dict = strategy.copy()
                
                # Ensure required fields exist
                if 'regime' not in final_strategy_dict:
                    final_strategy_dict['regime'] = 'UNKNOWN'
                
                if 'confidence' not in final_strategy_dict:
                    final_strategy_dict['confidence'] = 0.0
                
                if 'risk_level' not in final_strategy_dict:
                    final_strategy_dict['risk_level'] = 'moderate'
                
                if 'parameters' not in final_strategy_dict:
                    final_strategy_dict['parameters'] = {}
                
                # Convert parameters if it's a string
                if isinstance(final_strategy_dict.get('parameters'), str):
                    try:
                        final_strategy_dict['parameters'] = json.loads(final_strategy_dict['parameters'])
                    except json.JSONDecodeError:
                        final_strategy_dict['parameters'] = {'raw': final_strategy_dict['parameters']}
            
            # Handle StrategyContext object
            elif isinstance(strategy, StrategyContext):
                final_strategy_dict = strategy.to_dict()
            
            # Create a StrategyContext object for consistent handling
            try:
                # Convert regime string to enum
                regime_value = final_strategy_dict.get('regime', 'UNKNOWN')
                if isinstance(regime_value, str):
                    try:
                        regime = MarketRegime(regime_value)
                    except ValueError:
                        regime = MarketRegime.UNKNOWN
                else:
                    regime = MarketRegime.UNKNOWN
                
                # Create StrategyContext
                strategy_context = StrategyContext(
                    regime=regime,
                    confidence=float(final_strategy_dict.get('confidence', 0.0)),
                    risk_level=str(final_strategy_dict.get('risk_level', 'unknown')),
                    parameters=final_strategy_dict.get('parameters', {}),
                    opportunity_score=float(final_strategy_dict.get('opportunity_score', 0.0))
                )
                
                # Convert back to dict for storage
                final_strategy_dict = strategy_context.to_dict()
            except Exception as e:
                logger.error(f"Error creating StrategyContext: {e}, using dictionary directly")
                # Ensure dict has required fields
                if 'regime' in final_strategy_dict and isinstance(final_strategy_dict['regime'], MarketRegime):
                    final_strategy_dict['regime'] = final_strategy_dict['regime'].value
            
            # Create a descriptive conversation about the strategy
            messages = [
                {
                    "role": "user",
                    "content": f"Store trading strategy:\n{json.dumps(final_strategy_dict, indent=2)}"
                },
                {
                    "role": "assistant", 
                    "content": json.dumps({
                        "strategy": final_strategy_dict,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }, indent=2)
                }
            ]
            
            # Debug what we're trying to store
            logger.debug(f"Storing strategy with keys: {list(final_strategy_dict.keys())}")
            
            # Add to memory with metadata
            try:
                strategy_content = {"strategy": final_strategy_dict}
                logger.debug(f"Final content type: {type(strategy_content)}")
                
                metadata = {
                    "type": MEM0_CONFIG["metadata_types"]["strategy"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "regime": str(final_strategy_dict.get('regime', 'UNKNOWN')),
                    "confidence": float(final_strategy_dict.get('confidence', 0.0)),
                    "risk_level": str(final_strategy_dict.get('risk_level', 'unknown'))
                }
                
                # Ensure all metadata values are strings or basic types
                for k, v in metadata.items():
                    if not isinstance(v, (str, int, float, bool)):
                        metadata[k] = str(v)
                
                response = self.client.add(
                    messages=messages,
                    user_id=MEM0_CONFIG["user_id"],
                    content=json.dumps(strategy_content),  # Force JSON serialization
                    metadata=metadata
                )
            except Exception as add_err:
                logger.error(f"Error in client.add: {add_err}")
                # Try without content as fallback
                try:
                    response = self.client.add(
                        messages=messages,
                        user_id=MEM0_CONFIG["user_id"],
                        metadata=metadata
                    )
                except Exception as fallback_err:
                    logger.error(f"Fallback also failed: {fallback_err}")
                    return False
            
            if response is not None:
                logger.info("Successfully stored strategy in memory")
                return True
            else:
                logger.warning("No valid response from memory storage")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store strategy: {e}")
            logger.debug("Exception details:", exc_info=True)
            return False
            
    def get_recent_performance(self, lookback_days: int = 7) -> Dict[str, Any]:
        """Get recent trading performance statistics.
        
        Args:
            lookback_days: Number of days to look back for performance data
            
        Returns:
            Dict containing aggregated performance metrics
        """
        if not self.enabled:
            logger.debug("Memory system disabled, returning empty performance data")
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "avg_return": 0.0,
                "avg_pnl": 0.0,
                "overall_win_rate": 0.0,
                "strategies_analyzed": 0,
                "lookback_days": lookback_days
            }
            
        try:
            # Calculate lookback timestamp
            lookback_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            
            # Query recent strategy results
            query = "Recent trading strategy results"
            memories = self.client.search(
                query=query,
                user_id=MEM0_CONFIG["user_id"],
                limit=100,  # Get more results for better statistics
                metadata_filter={
                    "type": MEM0_CONFIG["metadata_types"]["strategy_results"],
                    "timestamp": {"$gte": lookback_date.isoformat()}
                }
            )
            
            # Initialize aggregated metrics
            total_trades = 0
            total_pnl = 0.0
            total_return = 0.0
            win_count = 0
            strategies_analyzed = 0
            
            # Process each memory
            for memory in memories:
                metadata = memory.get("metadata", {})
                if metadata.get("type") == MEM0_CONFIG["metadata_types"]["strategy_results"]:
                    total_trades += metadata.get("total_trades", 0)
                    total_pnl += metadata.get("total_pnl", 0.0)
                    total_return += metadata.get("total_return", 0.0)
                    win_count += metadata.get("win_rate", 0.0) * metadata.get("total_trades", 0)
                    strategies_analyzed += 1
            
            # Calculate averages
            if strategies_analyzed > 0:
                avg_return = total_return / strategies_analyzed
                avg_pnl = total_pnl / strategies_analyzed
                overall_win_rate = win_count / total_trades if total_trades > 0 else 0.0
                
                return {
                    "total_trades": total_trades,
                    "total_pnl": total_pnl,
                    "avg_return": avg_return,
                    "avg_pnl": avg_pnl,
                    "overall_win_rate": overall_win_rate,
                    "strategies_analyzed": strategies_analyzed,
                    "lookback_days": lookback_days
                }
            
            # Return default values if no data found
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "avg_return": 0.0,
                "avg_pnl": 0.0,
                "overall_win_rate": 0.0,
                "strategies_analyzed": 0,
                "lookback_days": lookback_days
            }
                
        except Exception as e:
            logger.error(f"Failed to get recent performance: {e}")
            logger.debug("Exception details:", exc_info=True)
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "avg_return": 0.0,
                "avg_pnl": 0.0,
                "overall_win_rate": 0.0,
                "strategies_analyzed": 0,
                "lookback_days": lookback_days
            }

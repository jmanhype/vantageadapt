"""Memory manager for trading strategies."""

from typing import Dict, List, Optional, Any, Tuple
import os
import logging
from datetime import datetime
from mem0 import MemoryClient
<<<<<<< HEAD
from .types import MarketRegime, StrategyContext, BacktestResults
=======
from .models import MarketRegime, MarketContext as StrategyContext
>>>>>>> fb68cbd (feat: Implement memory system for strategy optimization)
import json

logger = logging.getLogger(__name__)

<<<<<<< HEAD
=======
# Type alias for backtest results
BacktestResults = Dict[str, Any]

>>>>>>> fb68cbd (feat: Implement memory system for strategy optimization)
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
<<<<<<< HEAD

=======
        
>>>>>>> fb68cbd (feat: Implement memory system for strategy optimization)
        try:
            self.client = MemoryClient(api_key=self.api_key)
            self.enabled = True
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            self.client = None

    def store_strategy_results(self, context: StrategyContext, results: BacktestResults) -> bool:
        """Store strategy results in memory.
        
        Args:
            context: The strategy context containing market regime and parameters
            results: The backtest results containing performance metrics
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        if not self.client or not self.enabled:
            logger.debug("Memory system disabled, skipping storage")
            return False

        try:
            # Convert context and results to a structured memory format
            memory_content = {
<<<<<<< HEAD
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
=======
                "market_regime": context.regime.value,
                "confidence": context.confidence,
                "risk_level": context.risk_level,
                "volatility_level": context.volatility_level,
                "trend_strength": context.trend_strength,
                "volume_profile": context.volume_profile,
                "key_levels": context.key_levels,
                "analysis": context.analysis,
                "performance": {
                    "total_return": float(results.get('total_return', 0)),
                    "total_pnl": float(results.get('total_pnl', 0)),
                    "sortino_ratio": float(results.get('sortino_ratio', 0)),
                    "win_rate": float(results.get('win_rate', 0)),
                    "total_trades": results.get('total_trades', 0)
>>>>>>> fb68cbd (feat: Implement memory system for strategy optimization)
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add memory with metadata
            response = self.client.add(
                messages=[{
                    "role": "system",
                    "content": json.dumps(memory_content),
                    "metadata": {
<<<<<<< HEAD
                        "type": "strategy_results",
                        "market_regime": context.market_regime.value,
                        "success": results.total_return > 0
=======
                    "type": "strategy_results",
                        "market_regime": context.regime.value,
                    "success": results.get('total_return', 0) > 0
>>>>>>> fb68cbd (feat: Implement memory system for strategy optimization)
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

    def store_successful_strategy(self, strategy_data: Dict[str, Any], metrics: Dict[str, float], market_context: Dict[str, Any]) -> bool:
        """Store a successful trading strategy with its performance metrics and market context.
        
        Args:
            strategy_data (Dict[str, Any]): Strategy configuration and parameters
            metrics (Dict[str, float]): Performance metrics from the strategy execution
            market_context (Dict[str, Any]): Market conditions when strategy was executed
            
        Returns:
            bool: Whether the strategy was successfully stored
        """
        try:
            # Calculate strategy score
            score = self._calculate_strategy_score(metrics)
            
            # Only store strategies that meet minimum performance threshold
            if score < 0.5:  # Configurable threshold
                return False
                
            memory_entry = {
                'strategy': strategy_data,
                'metrics': metrics,
                'market_context': market_context,
                'score': score,
                'timestamp': datetime.now().isoformat(),
                'regime': market_context.get('regime', 'UNKNOWN')
            }
            
            # Store in mem0.ai with appropriate metadata
<<<<<<< HEAD
            self.client.add_memory(
                user_id='trading_system',
                agent_id='strategy_optimizer',
                content=json.dumps(memory_entry),
                metadata={
                    'score': score,
                    'regime': market_context.get('regime', 'UNKNOWN'),
                    'total_return': metrics.get('total_return', 0),
                    'sortino_ratio': metrics.get('sortino_ratio', 0)
                }
=======
            self.client.add(
                messages=[{
                    "role": "system",
                    "content": json.dumps(memory_entry),
                    "metadata": {
                        'score': score,
                        'regime': market_context.get('regime', 'UNKNOWN'),
                        'total_return': metrics.get('total_return', 0),
                        'sortino_ratio': metrics.get('sortino_ratio', 0)
                    }
                }],
                user_id="trading_system",
                agent_id="strategy_optimizer"
>>>>>>> fb68cbd (feat: Implement memory system for strategy optimization)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing successful strategy: {str(e)}")
            return False
            
<<<<<<< HEAD
    def query_similar_strategies(self, market_regime: str = None, min_score: float = 0.5) -> List[Dict[str, Any]]:
        """Query for similar successful strategies based on market regime.
        
        Args:
            market_regime (str): Market regime to filter by
            min_score (float): Minimum strategy score to consider
            
        Returns:
            List[Dict[str, Any]]: List of similar successful strategies with their metrics
        """
        try:
            # Convert market regime enum to string if needed
            if hasattr(market_regime, 'value'):
                market_regime = market_regime.value
            
            # Query mem0.ai with market context as search criteria
            memories = self.client.get_all(
                user_id='trading_system',
                agent_id='strategy_optimizer',
                metadata_filter={
                    'score': {'$gte': min_score},
                    'regime': market_regime if market_regime else {'$exists': True}
                }
            )
            
            strategies = []
            for memory in memories:
                try:
                    content = memory.get('content', '{}')
                    if isinstance(content, str):
                        strategy_data = json.loads(content)
                    else:
                        strategy_data = content
                    strategies.append(strategy_data)
                except Exception as e:
                    logger.warning(f"Failed to parse memory content: {e}")
                    continue
                    
            # Sort by score
            strategies.sort(key=lambda x: x.get('score', 0), reverse=True)
            return strategies[:10]  # Return top 10 matches
=======
    def query_similar_strategies(self, query: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Query similar strategies from memory based on theme and market context.
        
        Args:
            query: Query parameters including theme and market context
            limit: Maximum number of strategies to return
            
        Returns:
            List of similar strategies found in memory
        """
        if not self.enabled:
            return []

        try:
            # Query memory for similar strategies
            response = self.client.search(
                user_id="trading_system",
                agent_id="strategy_optimizer",
                query=json.dumps(query),
                limit=limit
            )
            
            if response and isinstance(response, list):
                return [json.loads(memory.content) for memory in response]
            return []
>>>>>>> fb68cbd (feat: Implement memory system for strategy optimization)
            
        except Exception as e:
            logger.error(f"Error querying similar strategies: {str(e)}")
            return []
            
    def _calculate_strategy_score(self, metrics: Dict[str, float]) -> float:
        """Calculate an overall strategy score based on performance metrics.
        
        Args:
            metrics (Dict[str, float]): Strategy performance metrics
            
        Returns:
            float: Strategy score between 0 and 1
        """
        weights = {
            'total_return': 0.3,
            'sortino_ratio': 0.2,
            'win_rate': 0.2,
            'max_drawdown': 0.15,
            'profit_factor': 0.15
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            if metric == 'max_drawdown':
                # Convert drawdown to positive score
                value = max(0, 1 + value)
            score += value * weight
            
        return max(0, min(1, score))
        
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity score between two market contexts.
        
        Args:
            context1 (Dict[str, Any]): First market context
            context2 (Dict[str, Any]): Second market context
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Compare key metrics
            metrics = [
                'volatility_level',
                'trend_strength',
                'volume_profile',
                'risk_level'
            ]
            
            score = 0.0
            for metric in metrics:
                if metric in context1 and metric in context2:
                    if isinstance(context1[metric], (int, float)) and isinstance(context2[metric], (int, float)):
                        # Numerical comparison
                        diff = abs(context1[metric] - context2[metric])
                        score += 1.0 - min(1.0, diff)
                    else:
                        # String comparison
                        score += 1.0 if context1[metric] == context2[metric] else 0.0
                        
            # Normalize score
            return score / len(metrics)
            
        except Exception as e:
            logger.error(f"Error calculating context similarity: {str(e)}")
            return 0.0

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
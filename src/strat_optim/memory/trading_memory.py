"""Memory manager for trading strategies using mem0.ai."""

from typing import Dict, List, Optional, Any
import os
import logging
from datetime import datetime
import json
from mem0 import MemoryClient
from research.strategy.types import MarketRegime, StrategyContext, BacktestResults

logger = logging.getLogger(__name__)

class TradingMemoryManager:
    """Manages memory for trading strategies using mem0.ai."""

    def __init__(self, mem0_api_key: Optional[str] = None):
        """Initialize the trading memory manager.
        
        Args:
            mem0_api_key: mem0.ai API key for memory storage
        """
        self.enabled = False
        self.client = None
        
        if not mem0_api_key:
            logger.warning("Memory system disabled: Missing API key")
            return
            
        try:
            self.client = MemoryClient(api_key=mem0_api_key)
            self.enabled = True
            logger.info("Memory system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {str(e)}")
            self.client = None
            self.enabled = False

    def store_strategy_results(
        self,
        context: StrategyContext,
        results: BacktestResults,
        parameters: Dict[str, Any]
    ) -> None:
        """Store strategy results in memory.
        
        Args:
            context: Market context when strategy was executed
            results: Results from backtesting
            parameters: Strategy parameters used
        """
        if not self.enabled:
            logger.debug("Memory system disabled: Skipping store_strategy_results")
            return
            
        try:
            # Convert objects to dictionaries for JSON serialization
            context_dict = context.to_dict()
            results_dict = results.to_dict()
            
            # Create memory content
            content = {
                "market_regime": context_dict['market_regime'],
                "confidence": context_dict['confidence'],
                "risk_level": context_dict['risk_level'],
                "parameters": parameters,
                "performance": results_dict
            }
            
            # Create memory in mem0.ai format
            messages = [{
                "role": "system",
                "content": json.dumps(content),
                "metadata": {
                    "type": "strategy_results",
                    "timestamp": datetime.now().isoformat(),
                    "market_regime": context_dict['market_regime'],
                    "confidence": context_dict['confidence'],
                    "risk_level": context_dict['risk_level']
                }
            }]
            
            # Store memory
            self.client.add(
                messages=messages,
                user_id="trading_system",
                agent_id="strategy_optimizer"
            )
            logger.info(f"Stored strategy results for {context_dict['market_regime']}")
            
        except Exception as e:
            logger.error(f"Failed to store strategy results: {str(e)}")
            logger.debug(f"Attempted to store: {messages}")

    def query_similar_strategies(
        self,
        market_regime: MarketRegime,
        min_return: float = -1.0,
        min_sortino: float = 0.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query for similar strategies.
        
        Args:
            market_regime: Current market regime
            min_return: Minimum total return to consider
            min_sortino: Minimum Sortino ratio to consider
            limit: Maximum number of results to return
            
        Returns:
            List of similar strategies
        """
        if not self.enabled:
            logger.debug("Memory system disabled: Skipping query_similar_strategies")
            return []
            
        try:
            query = f"Find trading strategies for {market_regime.value} market regime"
            results = self.client.search(
                query=query,
                user_id="trading_system",
                agent_id="strategy_optimizer",
                limit=limit
            )
            
            # Log all results for debugging
            logger.debug(f"Found {len(results)} strategies before filtering")
            
            # Filter results based on performance criteria
            filtered_results = []
            for result in results:
                try:
                    content = json.loads(result.get("content", "{}"))
                    performance = content.get("performance", {})
                    
                    if (performance.get("total_return", -float("inf")) >= min_return and 
                        performance.get("sortino_ratio", -float("inf")) >= min_sortino):
                        filtered_results.append(result)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse result content: {result}")
                    continue
            
            logger.debug(f"Returning {len(filtered_results)} strategies after filtering")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to query similar strategies: {str(e)}")
            return []

    def get_optimal_parameters(
        self,
        market_regime: MarketRegime,
        current_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get optimal parameters based on historical performance.
        
        Args:
            market_regime: Current market regime
            current_params: Current strategy parameters
            
        Returns:
            Optimized parameters based on historical data
        """
        if not self.enabled:
            logger.debug("Memory system disabled: Using current parameters")
            return current_params
            
        similar_strategies = self.query_similar_strategies(
            market_regime=market_regime,
            min_return=0.0,  # Consider any positive return
            min_sortino=1.0  # Lower Sortino ratio requirement
        )
        
        if not similar_strategies:
            logger.info("No similar successful strategies found")
            return current_params
            
        # Calculate weighted average of parameters
        weighted_params = {}
        total_weight = 0
        
        for strategy in similar_strategies:
            try:
                content = json.loads(strategy.get("content", "{}"))
                performance = content.get("performance", {})
                params = content.get("parameters", {})
                
                # Use Sortino ratio as weight
                weight = performance.get("sortino_ratio", 0)
                if weight <= 0:
                    continue
                    
                total_weight += weight
                
                for param_name, param_value in params.items():
                    if param_name not in weighted_params:
                        weighted_params[param_name] = 0
                    weighted_params[param_name] += param_value * weight
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse strategy content: {strategy}")
                continue
        
        if total_weight > 0:
            optimized_params = {
                name: value / total_weight 
                for name, value in weighted_params.items()
            }
            logger.info("Generated optimized parameters from historical data")
            return optimized_params
        
        return current_params

    def reset(self) -> None:
        """Reset the memory store."""
        if not self.enabled:
            logger.debug("Memory system disabled: Skipping reset")
            return
            
        try:
            self.client.reset()
            logger.info("Reset trading memory store")
        except Exception as e:
            logger.error(f"Failed to reset memory store: {str(e)}") 
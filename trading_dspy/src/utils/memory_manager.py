"""Memory management utilities for the trading system."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from loguru import logger
import pandas as pd

class TradingMemoryManager:
    """Manager for storing and retrieving trading strategies."""

    def __init__(self, memory_dir: str):
        """Initialize memory manager.
        
        Args:
            memory_dir: Directory for storing memory files
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.strategies_file = self.memory_dir / 'strategies.json'
        self.strategies = self._load_strategies()
        
    def _load_strategies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load strategies from disk.
        
        Returns:
            Dictionary mapping market regimes to lists of strategies
        """
        try:
            if self.strategies_file.exists():
                with open(self.strategies_file, 'r') as f:
                    return json.load(f)
            return {}
            
        except Exception as e:
            logger.error(f"Error loading strategies: {str(e)}")
            return {}
            
    def _save_strategies(self) -> None:
        """Save strategies to disk."""
        try:
            with open(self.strategies_file, 'w') as f:
                json.dump(self.strategies, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving strategies: {str(e)}")
            
    def store_strategy_results(
        self,
        context: Dict[str, Any],
        results: Dict[str, Any],
        iteration: int
    ) -> None:
        """Store strategy results.
        
        Args:
            context: Strategy context (market regime, parameters)
            results: Strategy results (performance metrics)
            iteration: Iteration number
        """
        try:
            regime = context.get('market_regime', 'unknown')
            
            if regime not in self.strategies:
                self.strategies[regime] = []
                
            strategy_data = {
                'context': context,
                'results': results,
                'iteration': iteration,
                'timestamp': str(pd.Timestamp.now())
            }
            
            self.strategies[regime].append(strategy_data)
            self._save_strategies()
            
            logger.info(f"Stored strategy results for regime: {regime}")
            
        except Exception as e:
            logger.error(f"Error storing strategy results: {str(e)}")
            
    def query_similar_strategies(
        self,
        market_regime: str,
        n: int = 5
    ) -> List[Dict[str, Any]]:
        """Query similar strategies based on market regime.
        
        Args:
            market_regime: Market regime to query
            n: Number of strategies to return
            
        Returns:
            List of similar strategies
        """
        try:
            if market_regime not in self.strategies:
                return []
                
            # Sort by performance and return top n
            strategies = self.strategies[market_regime]
            sorted_strategies = sorted(
                strategies,
                key=lambda x: x['results'].get('metrics', {}).get('total_return', 0),
                reverse=True
            )
            
            return sorted_strategies[:n]
            
        except Exception as e:
            logger.error(f"Error querying similar strategies: {str(e)}")
            return []
            
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored strategies.
        
        Returns:
            Dictionary containing strategy statistics
        """
        try:
            stats = {
                'total_strategies': sum(len(s) for s in self.strategies.values()),
                'strategies_by_regime': {k: len(v) for k, v in self.strategies.items()},
                'best_performing': {}
            }
            
            # Find best performing strategy for each regime
            for regime, strategies in self.strategies.items():
                if not strategies:
                    continue
                    
                best_strategy = max(
                    strategies,
                    key=lambda x: x['results'].get('metrics', {}).get('total_return', 0)
                )
                
                stats['best_performing'][regime] = {
                    'total_return': best_strategy['results'].get('metrics', {}).get('total_return', 0),
                    'timestamp': best_strategy.get('timestamp'),
                    'iteration': best_strategy.get('iteration')
                }
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting strategy statistics: {str(e)}")
            return {} 
"""Strategic trading system with LLM-driven decision making."""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import json
import inspect
import importlib
import os

from .llm_interface import LLMInterface, MarketContext, StrategyInsight
from ..database import db
from ..database.models.trading import Strategy, Backtest
from prompts.prompt_manager import PromptManager
from .teachability import Teachability

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Container for trading decisions."""

    should_enter: bool
    should_exit: bool
    position_size: float
    entry_price: Optional[float]
    exit_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    reasoning: str
    volatility_threshold: float = 0.2
    default_position_size: float = 0.1

    def adjust_position_size(self, volatility: float) -> float:
        """Adjust position size based on volatility."""
        if volatility > self.volatility_threshold:
            return self.default_position_size / (volatility / self.volatility_threshold)
        return self.default_position_size


class StrategicTrader:
    """Strategic trading system with LLM-driven decision making."""

    def __init__(self):
        """Initialize strategic trader."""
        self.prompt_manager = PromptManager()
        self.llm = None
        self.market_context = None
        self.strategy_insights = None
        self.last_trade_time = None
        self.trade_cooldown = 300  # 5 minutes
        self.performance_history = []
        self.current_position = None
        self.teachability = None

    @classmethod
    async def create(cls) -> "StrategicTrader":
        """Create a new instance of StrategicTrader with initialized LLM interface.
        
        Returns:
            StrategicTrader: A new instance with initialized LLM interface.
        """
        instance = cls()
        instance.llm = await LLMInterface.create()
        return instance

    def add_teachability(self, teachability: Teachability) -> None:
        """Add teachability capability to the trader.
        
        Args:
            teachability: Teachability instance to use for learning
        """
        self.teachability = teachability
        logger.info("Added teachability capability to trader")

    def log_trade_metrics(self, metrics: Dict[str, float]) -> None:
        """Log detailed metrics for analysis and improvement."""
        logger.info(f"Trade metrics: {json.dumps(metrics, indent=2)}")

    async def should_exit_early(
        self, market_data: pd.DataFrame, current_position: TradingDecision
    ) -> bool:
        """Determine if we should exit a position early based on market conditions."""
        trend_indicator = market_data["trend"].iloc[-1]
        return trend_indicator == "bearish" and current_position.should_enter

    async def initialize(self):
        """Initialize LLM interface."""
        # No need to create LLM interface here as it's done in __init__
        pass

    async def analyze_market(self, market_data: pd.DataFrame) -> MarketContext:
        """Perform strategic market analysis."""
        if not self.llm:
            await self.initialize()

        # Ensure we have the correct column names
        if "dex_price" in market_data.columns and "price" not in market_data.columns:
            market_data = market_data.copy()
            market_data["price"] = market_data["dex_price"]

        self.market_context = await self.llm.analyze_market(market_data)
        return self.market_context

    async def generate_strategy(self, theme: str) -> Optional[StrategyInsight]:
        """Generate strategic trading insights using historical learning."""
        try:
            if not self.market_context:
                logger.error("Must analyze market first")
                return None

            # Get similar successful strategies
            similar_strategies = []
            if hasattr(self, 'memory_manager') and self.memory_manager:
                similar_strategies = self.memory_manager.query_similar_strategies(self.market_context.regime)
                if similar_strategies:
                    logger.info(f"Found {len(similar_strategies)} similar successful strategies to learn from")
                    # Log performance metrics of similar strategies
                    for strategy in similar_strategies:
                        perf = strategy.get('performance', {})
                        logger.info(f"Similar strategy performance: Return={perf.get('total_return', 0):.2%}, "
                                  f"Win Rate={perf.get('win_rate', 0):.2%}, "
                                  f"Sortino={perf.get('sortino_ratio', 0):.2f}")

            # Generate strategy with historical context
            self.strategy_insights = await self.llm.generate_strategy(
                theme=theme,
                market_context=self.market_context,
                similar_strategies=similar_strategies
            )

            if not self.strategy_insights:
                logger.error("Failed to generate strategy insights")
                return None

            # Process insights with teachability if enabled
            if self.teachability:
                logger.info("Processing strategy insights with teachability...")
                enhanced_insights = self.teachability.process_last_received_message(str(self.strategy_insights))
                logger.info("Strategy enhanced with learned patterns")

            # Track this strategy generation for performance comparison
            self._track_strategy_generation(self.strategy_insights)

            return self.strategy_insights

        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None

    def _track_strategy_generation(self, strategy: StrategyInsight) -> None:
        """Track strategy generation for performance analysis.
        
        Args:
            strategy: The generated strategy insights
        """
        if not hasattr(self, '_strategy_history'):
            self._strategy_history = []
        
        self._strategy_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'strategy': strategy,
            'market_regime': self.market_context.regime if self.market_context else None,
            'metrics': None  # Will be updated when performance results come in
        })

    async def generate_trading_rules(
        self,
        strategy_insights: StrategyInsight,
        market_context: MarketContext,
        performance_analysis: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """Generate trading rules and parameters.

        Args:
            strategy_insights: Strategy insights
            market_context: Market context
            performance_analysis: Optional performance analysis for strategy improvement

        Returns:
            Tuple of (conditions dict, parameters dict)
        """
        try:
            if not self.llm:
                await self.initialize()

            return await self.llm.generate_trading_rules(
                strategy_insights,
                market_context,
                performance_analysis=performance_analysis,
            )

        except Exception as e:
            logger.error(f"Error generating trading rules: {str(e)}")
            return {}, {}

    async def improve_strategy(
        self, metrics: Dict[str, float], trade_memory_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Improve strategy based on performance metrics and historical learning.
        
        Args:
            metrics: Trading performance metrics
            trade_memory_stats: Detailed trade statistics
            
        Returns:
            Dictionary containing improved strategy parameters and conditions
        """
        try:
            if not self.llm:
                await self.initialize()

            # Get historical performance data with enhanced context
            historical_data = await self._get_enhanced_historical_data()
            
            # Update strategy history with detailed performance tracking
            evolution_data = self._update_strategy_evolution(metrics, trade_memory_stats)
            
            # Analyze performance trends with machine learning insights
            performance_trends = await self._analyze_performance_trends(evolution_data)
            
            # Get performance analysis with historical context and evolution data
            analysis = await self.llm.analyze_performance(
                metrics=metrics,
                trade_memory_stats=trade_memory_stats,
                historical_performance=historical_data,
                performance_trends=performance_trends,
                evolution_data=evolution_data
            )
            
            # Process analysis with teachability
            if self.teachability:
                logger.info("Processing performance analysis with teachability...")
                analysis_text = self._prepare_teachability_text(
                    metrics, historical_data, performance_trends, evolution_data
                )
                self.teachability.process_last_received_message(analysis_text)

            # Generate improved trading rules with evolution context
            if self.market_context and self.strategy_insights:
                conditions, parameters = await self.llm.generate_trading_rules(
                    self.strategy_insights,
                    self.market_context,
                    performance_analysis=analysis,
                    evolution_context=evolution_data
                )
                
                # Store evolution step if successful
                if self._is_improvement(metrics, evolution_data):
                    await self._store_evolution_step(
                        parameters, conditions, metrics, evolution_data
                    )
                
                return {
                    "parameters": parameters,
                    "conditions": conditions,
                    "analysis": analysis,
                    "performance_trends": performance_trends,
                    "evolution_data": evolution_data
                }
            else:
                logger.error("Missing market context or strategy insights")
                return None

        except Exception as e:
            logger.error(f"Error improving strategy: {str(e)}")
            return None

    async def _get_enhanced_historical_data(self) -> List[Dict[str, Any]]:
        """Get enhanced historical performance data with learning context."""
        historical_data = []
        
        if hasattr(self, 'memory_manager') and self.memory_manager:
            similar_results = self.memory_manager.query_similar_strategies(
                self.market_context.regime,
                min_return=0.0,
                min_trades=10,
                max_results=10  # Increased for better learning
            )
            
            if similar_results:
                logger.info(f"Found {len(similar_results)} similar strategies for analysis")
                
                # Enhance historical data with success patterns
                for result in similar_results:
                    enhanced_result = await self._analyze_strategy_success_patterns(result)
                    if enhanced_result:
                        historical_data.append(enhanced_result)
        
        return historical_data

    def _update_strategy_evolution(
        self, metrics: Dict[str, float], trade_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update and track strategy evolution with detailed metrics.
        
        Args:
            metrics: Performance metrics
            trade_stats: Detailed trade statistics
            
        Returns:
            Dict containing evolution tracking data
        """
        current_timestamp = datetime.now()
        
        # Initialize evolution tracking if needed
        if not hasattr(self, '_evolution_history'):
            self._evolution_history = []
        
        # Calculate evolution metrics
        evolution_metrics = {
            'timestamp': current_timestamp,
            'performance': metrics,
            'trade_stats': trade_stats,
            'market_regime': self.market_context.regime.value if self.market_context else None,
            'improvement_metrics': self._calculate_improvement_metrics(metrics),
            'strategy_complexity': self._assess_strategy_complexity(),
            'adaptation_score': self._calculate_adaptation_score(metrics)
        }
        
        # Add to evolution history
        self._evolution_history.append(evolution_metrics)
        
        # Keep only recent history (last 10 iterations)
        if len(self._evolution_history) > 10:
            self._evolution_history = self._evolution_history[-10:]
        
        return {
            'current': evolution_metrics,
            'history': self._evolution_history,
            'trends': self._analyze_evolution_trends()
        }

    async def _analyze_performance_trends(
        self, evolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance trends with machine learning insights.
        
        Args:
            evolution_data: Strategy evolution data
            
        Returns:
            Dict containing trend analysis
        """
        if not evolution_data or not evolution_data.get('history'):
            return {}
        
        history = evolution_data['history']
        
        # Calculate trend metrics
        performance_trends = {
            'metric_trends': self._calculate_metric_trends(history),
            'regime_adaptation': self._analyze_regime_adaptation(history),
            'complexity_evolution': self._track_complexity_evolution(history),
            'learning_progress': self._assess_learning_progress(history)
        }
        
        return performance_trends

    def _prepare_teachability_text(
        self, metrics: Dict[str, float], historical_data: List[Dict[str, Any]],
        performance_trends: Dict[str, Any], evolution_data: Dict[str, Any]
    ) -> str:
        """Prepare comprehensive text for teachability processing."""
        return (
            f"Performance Analysis:\nMetrics: {json.dumps(metrics, indent=2)}\n"
            f"Historical Context: {len(historical_data)} similar strategies\n"
            f"Performance Trends: {json.dumps(performance_trends, indent=2)}\n"
            f"Evolution Data: {json.dumps(evolution_data, indent=2)}"
        )

    def _calculate_improvement_metrics(
        self, metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate detailed improvement metrics."""
        if not hasattr(self, '_last_metrics'):
            self._last_metrics = metrics
            return {}
        
        return {
            'return_improvement': metrics.get('total_return', 0) - self._last_metrics.get('total_return', 0),
            'sortino_improvement': metrics.get('sortino_ratio', 0) - self._last_metrics.get('sortino_ratio', 0),
            'win_rate_improvement': metrics.get('win_rate', 0) - self._last_metrics.get('win_rate', 0),
            'drawdown_improvement': metrics.get('max_drawdown', 0) - self._last_metrics.get('max_drawdown', 0)
        }

    def _assess_strategy_complexity(self) -> Dict[str, Any]:
        """Assess the complexity of current strategy."""
        if not hasattr(self, 'strategy_insights'):
            return {}
        
        return {
            'parameter_count': len(self.strategy_insights.parameters),
            'condition_count': len(self.strategy_insights.conditions.get('entry', [])) + 
                             len(self.strategy_insights.conditions.get('exit', [])),
            'indicator_count': self._count_unique_indicators(),
            'complexity_score': self._calculate_complexity_score()
        }

    def _calculate_adaptation_score(self, metrics: Dict[str, float]) -> float:
        """Calculate strategy adaptation score based on performance and market regime."""
        base_score = 0.0
        
        # Performance-based scoring
        if metrics.get('sortino_ratio', 0) > 2.0:
            base_score += 0.3
        if metrics.get('win_rate', 0) > 0.5:
            base_score += 0.2
        if abs(metrics.get('max_drawdown', 0)) < 0.2:
            base_score += 0.2
        
        # Regime-based adjustment
        if self.market_context and self.market_context.regime:
            regime_score = self._calculate_regime_compatibility()
            base_score *= (1 + regime_score)
        
        return min(1.0, base_score)

    def _analyze_evolution_trends(self) -> Dict[str, Any]:
        """Analyze trends in strategy evolution."""
        if not hasattr(self, '_evolution_history') or len(self._evolution_history) < 2:
            return {}
        
        history = self._evolution_history
        return {
            'performance_trend': self._calculate_trend([h['performance']['total_return'] 
                                                      for h in history]),
            'complexity_trend': self._calculate_trend([h.get('strategy_complexity', {})
                                                     .get('complexity_score', 0) 
                                                     for h in history]),
            'adaptation_trend': self._calculate_trend([h.get('adaptation_score', 0) 
                                                     for h in history])
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and strength."""
        if len(values) < 2:
            return 0.0
        
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        return sum(diffs) / len(diffs)

    async def _analyze_strategy_success_patterns(
        self, strategy_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze patterns in successful strategies."""
        try:
            # Extract key components
            parameters = strategy_result.get('parameters', {})
            performance = strategy_result.get('performance', {})
            
            # Analyze parameter relationships
            param_analysis = self._analyze_parameter_relationships(parameters, performance)
            
            # Identify success patterns
            success_patterns = self._identify_success_patterns(strategy_result)
            
            return {
                **strategy_result,
                'parameter_analysis': param_analysis,
                'success_patterns': success_patterns
            }
        except Exception as e:
            logger.error(f"Error analyzing strategy success patterns: {str(e)}")
            return None

    def _is_improvement(
        self, current_metrics: Dict[str, float], evolution_data: Dict[str, Any]
    ) -> bool:
        """Determine if current metrics show improvement."""
        if not evolution_data or not evolution_data.get('history'):
            return True
        
        history = evolution_data['history']
        if not history:
            return True
        
        last_metrics = history[-1]['performance']
        
        # Calculate weighted improvement score
        weights = {
            'total_return': 0.3,
            'sortino_ratio': 0.3,
            'win_rate': 0.2,
            'max_drawdown': 0.2
        }
        
        score = 0.0
        for metric, weight in weights.items():
            current_value = current_metrics.get(metric, 0)
            last_value = last_metrics.get(metric, 0)
            
            if metric == 'max_drawdown':
                # For drawdown, less negative is better
                score += weight * (1 if current_value > last_value else 0)
            else:
                score += weight * (1 if current_value > last_value else 0)
        
        return score > 0.6  # Improvement threshold

    async def _store_evolution_step(
        self, parameters: Dict[str, Any], conditions: Dict[str, Any],
        metrics: Dict[str, float], evolution_data: Dict[str, Any]
    ) -> None:
        """Store successful evolution step in memory."""
        try:
            if not hasattr(self, 'memory_manager'):
                return
            
            memory_entry = {
                "market_regime": self.market_context.regime.value,
                "parameters": parameters,
                "conditions": conditions,
                "performance": metrics,
                "evolution_data": evolution_data,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.memory_manager.add_memory(memory_entry)
            logger.info("Stored successful evolution step")
        except Exception as e:
            logger.error(f"Error storing evolution step: {str(e)}")

    async def analyze_performance(
        self, metrics: Dict[str, float], trade_memory_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze trading performance.

        Args:
            metrics: Trading performance metrics
            trade_memory_stats: Detailed trade statistics

        Returns:
            Dictionary containing performance analysis
        """
        try:
            if not self.llm:
                await self.initialize()

            return await self.llm.analyze_performance(metrics, trade_memory_stats)

        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return {
                "performance_summary": {
                    "overall_assessment": f"Analysis error: {str(e)}",
                    "key_strengths": [],
                    "key_weaknesses": [],
                }
            }

    async def save_strategy_results(
        self,
        theme: str,
        conditions: Dict[str, List[str]],
        parameters: Dict[str, Any],
        metrics: Dict[str, float],
        market_context: MarketContext,
        strategy_insights: StrategyInsight,
        performance_analysis: Dict[str, Any],
    ) -> None:
        """Save strategy results to database.

        Args:
            theme: Strategy theme
            conditions: Trading conditions
            parameters: Strategy parameters
            metrics: Performance metrics
            market_context: Market context
            strategy_insights: Strategy insights
            performance_analysis: Performance analysis
        """
        try:
            if not self.llm:
                await self.initialize()

            await self.llm.save_strategy_results(
                theme=theme,
                conditions=conditions,
                parameters=parameters,
                metrics=metrics,
                market_context=market_context,
                strategy_insights=strategy_insights,
                performance_analysis=performance_analysis,
            )

        except Exception as e:
            logger.error(f"Error saving strategy results: {str(e)}")
            return None

    async def apply_code_improvements(self, improvements: str) -> bool:
        """Apply code improvements suggested by GodelAgent.
        
        Args:
            improvements: String containing the improved code
            
        Returns:
            bool: True if improvements were successfully applied
        """
        try:
            # Backup current code
            current_file = inspect.getfile(self.__class__)
            backup_file = f"{current_file}.bak"
            
            # Create backup
            import shutil
            shutil.copy2(current_file, backup_file)
            
            # Write improved code
            with open(current_file, 'w') as f:
                f.write(improvements)
                
            # Reload the module
            module_name = self.__class__.__module__
            module = importlib.import_module(module_name)
            importlib.reload(module)
            
            logger.info("Successfully applied code improvements")
            return True
            
        except Exception as e:
            logger.error(f"Error applying code improvements: {str(e)}")
            
            # Restore from backup if it exists
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, current_file)
                logger.info("Restored from backup after failed improvement")
                
            return False

    def read_module_code(self) -> str:
        """Read the source code of the current module.
        
        Returns:
            str: Source code of the module
        """
        try:
            current_file = inspect.getfile(self.__class__)
            with open(current_file, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading module code: {str(e)}")
            return ""

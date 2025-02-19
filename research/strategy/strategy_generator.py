"""Strategic trading system with LLM-driven decision making."""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import json

from .llm_teachable import TeachableLLMInterface
from .llm_interface import MarketContext, StrategyInsight
from .memory_manager import TradingMemoryManager
from ..database import db
from ..database.models.trading import Strategy, Backtest
from prompts.prompt_manager import PromptManager

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
        self.memory_manager = None

    @classmethod
    async def create(cls) -> "StrategicTrader":
        """Create a new instance of StrategicTrader with initialized LLM interface.
        
        Returns:
            StrategicTrader: A new instance with initialized LLM interface.
        """
        instance = cls()
        instance.llm = await TeachableLLMInterface.create()
        instance.memory_manager = TradingMemoryManager()
        return instance

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
        """Generate strategic trading insights.
        
        Args:
            theme: Trading theme
            
        Returns:
            Optional[StrategyInsight]: Generated strategy insights
        """
        try:
            if not self.market_context:
                logger.error("No market context available")
                return None

            # Get similar strategies from memory
            similar_strategies = self.memory_manager.query_similar_strategies(self.market_context.regime)
            
            # Extract insights from similar strategies
            strategy_insights = None
            if similar_strategies:
                # Weight parameters by strategy scores
                total_score = sum(s["score"] for s in similar_strategies)
                if total_score > 0:
                    weighted_params = {}
                    for strategy in similar_strategies:
                        weight = strategy["score"] / total_score
                        params = strategy.get("parameters", {})
                        for key, value in params.items():
                            if isinstance(value, (int, float)):
                                weighted_params[key] = weighted_params.get(key, 0) + (value * weight)
                    
                    # Use weighted parameters as base for new strategy
                    strategy_insights = await self.llm.generate_strategy(
                        theme=theme,
                        market_context=self.market_context,
                        base_parameters=weighted_params
                    )
                    
                    logger.info(f"Generated strategy using {len(similar_strategies)} similar strategies as reference")
                else:
                    logger.info("Found similar strategies but total score is 0, generating fresh strategy")
            
            # If no useful similar strategies, generate fresh strategy
            if not strategy_insights:
                strategy_insights = await self.llm.generate_strategy(
                    theme=theme,
                    market_context=self.market_context
                )
                
            self.strategy_insights = strategy_insights
            return strategy_insights
            
        except Exception as e:
            logger.error(f"Failed to generate strategy: {str(e)}")
            return None

    async def generate_trading_rules(self, strategy: StrategyInsight, market_context: MarketContext) -> Dict[str, Any]:
        """Generate trading rules based on strategy insights and market context.
        
        Args:
            strategy: Strategy insights object
            market_context: Market context object
            
        Returns:
            Dictionary containing entry/exit conditions and parameters
        """
        try:
            # Generate rules using LLM
            rules = await self.llm.generate_trading_rules(strategy, market_context)
            if not rules:
                logger.error("Failed to generate trading rules")
                return {}
                
            conditions, parameters = rules
            
            # Ensure conditions is properly formatted for backtester
            if not isinstance(conditions, dict):
                logger.warning("Invalid conditions format from LLM, using defaults")
                conditions = {
                    'entry': ["(df_indicators['rsi'] < 30) & (df_indicators['price'] <= df_indicators['bb_lower'])"],
                    'exit': ["(df_indicators['rsi'] > 70) | (df_indicators['price'] >= df_indicators['bb_upper'])"]
                }
                
            # Validate and format conditions
            formatted_conditions = {
                'entry': [],
                'exit': []
            }
            
            # Process entry conditions
            for condition in conditions.get('entry', []):
                if isinstance(condition, str):
                    # Replace any df[] references with df_indicators[]
                    condition = condition.replace("df['", "df_indicators['").replace("data['", "df_indicators['")
                    formatted_conditions['entry'].append(condition)
                    
            # Process exit conditions
            for condition in conditions.get('exit', []):
                if isinstance(condition, str):
                    # Replace any df[] references with df_indicators[]
                    condition = condition.replace("df['", "df_indicators['").replace("data['", "df_indicators['")
                    formatted_conditions['exit'].append(condition)
                    
            # If no valid conditions found, use defaults
            if not formatted_conditions['entry'] or not formatted_conditions['exit']:
                logger.warning("No valid conditions found, using defaults")
                formatted_conditions = {
                    'entry': ["(df_indicators['rsi'] < 30) & (df_indicators['price'] <= df_indicators['bb_lower'])"],
                    'exit': ["(df_indicators['rsi'] > 70) | (df_indicators['price'] >= df_indicators['bb_upper'])"]
                }
                
            # Ensure parameters are properly formatted
            formatted_parameters = {
                'take_profit': float(parameters.get('take_profit', 0.05)),
                'stop_loss': float(parameters.get('stop_loss', 0.03)),
                'order_size': float(parameters.get('order_size', 0.1)),
                'max_orders': int(parameters.get('max_orders', 3)),
                'sl_window': int(parameters.get('sl_window', 400)),
                'post_buy_delay': int(parameters.get('post_buy_delay', 2)),
                'post_sell_delay': int(parameters.get('post_sell_delay', 5)),
                'enable_sl_mod': bool(parameters.get('enable_sl_mod', False)),
                'enable_tp_mod': bool(parameters.get('enable_tp_mod', False))
            }
            
            # Add conditions to parameters for backtester
            formatted_parameters['conditions'] = formatted_conditions
            
            return formatted_parameters
            
        except Exception as e:
            logger.error(f"Error generating trading rules: {str(e)}")
            logger.exception("Full traceback:")
            return {}

    async def improve_strategy(
        self, metrics: Dict[str, float], trade_memory_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Improve strategy based on performance metrics.

        Args:
            metrics: Trading performance metrics
            trade_memory_stats: Detailed trade statistics

        Returns:
            Dictionary containing improved strategy parameters and conditions
        """
        try:
            if not self.llm:
                await self.initialize()

            # Get performance analysis
            analysis = await self.llm.analyze_performance(metrics, trade_memory_stats)

            # Use the analysis to generate improved trading rules
            if self.market_context and self.strategy_insights:
                conditions, parameters = await self.llm.generate_trading_rules(
                    self.strategy_insights,
                    self.market_context,
                    performance_analysis=analysis,  # Pass analysis for context
                )
                return {
                    "parameters": parameters,
                    "conditions": conditions,
                    "analysis": analysis,
                }
            else:
                logger.error("Missing market context or strategy insights")
                return None

        except Exception as e:
            logger.error(f"Error improving strategy: {str(e)}")
            return None

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

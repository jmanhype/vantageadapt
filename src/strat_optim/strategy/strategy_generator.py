"""Strategic trading system with LLM-driven decision making."""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import json

from .llm_interface import LLMInterface, MarketContext, StrategyInsight
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
        self.llm = LLMInterface(self.prompt_manager)
        self.market_context = None
        self.strategy_insights = None
        self.last_trade_time = None
        self.trade_cooldown = 300  # 5 minutes
        self.performance_history = []
        self.current_position = None

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
        """Generate strategic trading insights."""
        try:
            if not self.market_context:
                logger.error("Must analyze market first")
                return None

            self.strategy_insights = await self.llm.generate_strategy(
                theme, self.market_context
            )

            if not self.strategy_insights:
                logger.error("Failed to generate strategy insights")
                return None

            return self.strategy_insights

        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None

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

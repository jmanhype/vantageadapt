"""Core signatures for DSPy modules."""

from typing import Dict, List, Optional, Any
from dspy import Signature, InputField, OutputField


class MarketAnalysisInput(Signature):
    """Input signature for market analysis module."""
    
    market_data: Dict[str, Any] = InputField(description="Raw market data as a dictionary")
    timeframe: str = InputField(description="Timeframe of the analysis")


class MarketAnalysisOutput(Signature):
    """Output signature for market analysis module."""
    
    market_context: Dict[str, Any] = OutputField(description="Market context including regime, confidence etc")
    analysis_text: str = OutputField(description="Detailed analysis explanation")
    risk_level: str = OutputField(description="Current market risk level")


class StrategyInput(Signature):
    """Input signature for strategy generation module."""
    
    market_context: Dict[str, Any] = InputField(description="Market context from analysis")
    theme: str = InputField(description="Trading strategy theme")
    base_parameters: Optional[Dict[str, Any]] = InputField(
        description="Optional base parameters for strategy", optional=True
    )


class StrategyOutput(Signature):
    """Output signature for strategy generation module."""
    
    reasoning: str = OutputField(description="Detailed reasoning for the strategy")
    trade_signal: str = OutputField(description="Generated trade signal (BUY/SELL/HOLD)")
    parameters: Dict[str, Any] = OutputField(description="Generated strategy parameters")
    confidence: float = OutputField(description="Confidence in the strategy (0-1)")


class TradingRulesInput(Signature):
    """Input signature for trading rules generation module."""
    
    strategy_insights: Dict[str, Any] = InputField(description="Strategy insights from generator")
    market_context: Dict[str, Any] = InputField(description="Current market context")
    performance_analysis: Optional[Dict[str, Any]] = InputField(
        description="Optional performance analysis for improvements", optional=True
    )


class TradingRulesOutput(Signature):
    """Output signature for trading rules generation module."""
    
    conditions: Dict[str, List[str]] = OutputField(description="Entry/exit conditions")
    parameters: Dict[str, Any] = OutputField(description="Risk management parameters")
    reasoning: str = OutputField(description="Explanation of the rules")


class BacktestInput(Signature):
    """Input signature for backtesting module."""
    
    trade_data: Dict[str, Any] = InputField(description="Historical trade data")
    parameters: Dict[str, Any] = InputField(description="Strategy parameters")
    conditions: Dict[str, List[str]] = InputField(description="Trading conditions")


class BacktestOutput(Signature):
    """Output signature for backtesting module."""
    
    metrics: Dict[str, float] = OutputField(description="Performance metrics")
    trade_details: Dict[str, Any] = OutputField(description="Detailed trade records")
    validation_passed: bool = OutputField(description="Whether strategy meets requirements")


class MemoryInput(Signature):
    """Input signature for memory module."""
    
    context: Dict[str, Any] = InputField(description="Strategy context")
    performance_metrics: Dict[str, float] = InputField(description="Performance metrics")
    iteration: int = InputField(description="Current iteration number")


class MemoryOutput(Signature):
    """Output signature for memory module."""
    
    stored: bool = OutputField(description="Whether storage was successful")
    similar_strategies: Optional[List[Dict[str, Any]]] = OutputField(
        description="Similar strategies if found", optional=True
    ) 
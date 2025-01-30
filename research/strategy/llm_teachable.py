"""LLM interface with teachability capabilities."""

from typing import Dict, List, Any, Optional, Tuple
import logging
from dotenv import load_dotenv
import os
import pandas as pd
from .llm_interface import LLMInterface, MarketContext, StrategyInsight
from .teachability import Teachability
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage
from autogen.agentchat.assistant_agent import ConversableAgent

logger = logging.getLogger(__name__)

class TeachableLLMInterface(LLMInterface):
    """LLM interface with teachability capabilities."""
    
    def __init__(self):
        """Initialize teachable LLM interface."""
        super().__init__()
        self.teachability = None
        
    @classmethod
    async def create(cls) -> 'TeachableLLMInterface':
        """Create and initialize a new TeachableLLMInterface instance."""
        instance = cls()
        await instance._initialize()
        return instance
        
    async def _initialize(self) -> None:
        """Initialize the teachable LLM interface."""
        try:
            # Initialize base LLM interface
            await super()._initialize()
            
            # Initialize teachability
            self.teachability = Teachability(
                agent_id="trading_agent",
                memory_client=None,  # Will use API key from env
                verbosity=1
            )
            
            # Create a base agent for teachability
            base_agent = ConversableAgent(
                name="trading_agent",
                system_message="""You are an expert trading agent that analyzes market conditions 
                and provides trading strategies. You have deep knowledge of technical analysis,
                market patterns, and risk management.""",
                llm_config={
                    "model": "gpt-4-1106-preview",
                    "temperature": 0,
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            )
            
            # Add teachability to the base agent
            self.teachability.add_to_agent(base_agent)
            logger.info("Teachability initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing teachable LLM interface: {str(e)}")
            raise
            
    async def analyze_market(self, market_data: pd.DataFrame) -> MarketContext:
        """Analyze market data with teachability enhancement."""
        try:
            # Get base market analysis
            market_context = await super().analyze_market(market_data)
            
            if not market_context:
                return None
                
            # Enhance with teachability
            if self.teachability and self.teachability.enabled:
                # Create market analysis message
                message = {
                    "role": "user",
                    "content": f"""Analyzing market regime {market_context.regime.value}:
                    - Confidence: {market_context.confidence}
                    - Volatility: {market_context.volatility_level}
                    - Risk Level: {market_context.risk_level}
                    - Analysis: {market_context.analysis}"""
                }
                
                # Process with teachability
                enhanced_content = self.teachability.process_last_received_message(message)
                
                # Update market context with enhanced insights
                if enhanced_content:
                    market_context.analysis["enhanced_insights"] = enhanced_content
                    
            return market_context
            
        except Exception as e:
            logger.error(f"Error in teachable market analysis: {str(e)}")
            return None
            
    async def generate_strategy(
        self, 
        theme: str, 
        market_context: MarketContext,
        base_parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[StrategyInsight]:
        """Generate strategy with teachability enhancement.
        
        Args:
            theme: Trading theme to focus on
            market_context: Current market context
            base_parameters: Optional parameters from similar successful strategies to use as base
            
        Returns:
            Optional[StrategyInsight]: Generated strategy insights
        """
        try:
            # Get base strategy insights
            strategy_insights = await super().generate_strategy(theme, market_context)
            
            if not strategy_insights:
                return None
                
            # Apply base parameters if provided
            if base_parameters:
                # Update strategy parameters with weighted values from similar strategies
                for key, value in base_parameters.items():
                    if hasattr(strategy_insights, key):
                        setattr(strategy_insights, key, value)
                        logger.info(f"Applied learned parameter {key}: {value}")
                
            # Enhance with teachability
            if self.teachability and self.teachability.enabled:
                # Create strategy message
                message = {
                    "role": "user",
                    "content": f"""Strategy for {theme} in {market_context.regime.value} market:
                    - Risk/Reward: {strategy_insights.risk_reward_target}
                    - Position Size: {strategy_insights.suggested_position_size}
                    - Trade Frequency: {strategy_insights.trade_frequency}
                    - Opportunity: {strategy_insights.opportunity_description}
                    - Base Parameters: {base_parameters if base_parameters else 'None'}"""
                }
                
                # Process with teachability
                enhanced_content = self.teachability.process_last_received_message(message)
                
                # Update strategy insights with enhanced information
                if enhanced_content:
                    strategy_insights.risk_management_notes.append(f"Enhanced Insights: {enhanced_content}")
                    
            return strategy_insights
            
        except Exception as e:
            logger.error(f"Error in teachable strategy generation: {str(e)}")
            return None 
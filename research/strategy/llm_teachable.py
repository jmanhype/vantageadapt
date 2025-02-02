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
import json

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

    async def generate_trading_rules(
        self,
        strategy_insights: StrategyInsight,
        market_context: MarketContext
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """Generate trading rules based on strategy insights and market context.
        
        Args:
            strategy_insights: Strategy insights object
            market_context: Market context object
            
        Returns:
            Tuple of (conditions dict, parameters dict)
        """
        try:
            # Create system message with explicit Python code requirements
            system_prompt = """You are an expert trading rules generator.

Your task is to generate specific trading rules that implement a given trading strategy.

IMPORTANT: You must respond with ONLY a JSON object in the following format, where conditions MUST be valid Python code:
{
    "conditions": {
        "entry": [
            "(df_indicators['rsi'] < 30) & (df_indicators['price'] <= df_indicators['bb_lower'])"
        ],
        "exit": [
            "(df_indicators['rsi'] > 70) | (df_indicators['price'] >= df_indicators['bb_upper'])"
        ]
    },
    "parameters": {
        "take_profit": 0.05,
        "stop_loss": 0.03,
        "order_size": 0.1,
        "max_orders": 3,
        "sl_window": 400,
        "post_buy_delay": 2,
        "post_sell_delay": 5,
        "enable_sl_mod": false,
        "enable_tp_mod": false
    }
}

Available indicators in df_indicators:
- price: Current price
- rsi: Relative Strength Index
- bb_upper: Bollinger Band Upper
- bb_lower: Bollinger Band Lower
- bb_mid: Bollinger Band Middle
- macd: MACD Line
- macd_signal: MACD Signal Line

Rules:
1. ALL conditions must be valid Python expressions using only the available indicators
2. Use only mathematical and logical operators (>, <, >=, <=, &, |)
3. NO natural language descriptions - only Python code
4. Each condition must evaluate to a boolean value
5. Parameters must be numeric values within reasonable ranges"""

            # Create user message with strategy and market context
            user_message = f"""Please generate specific trading rules as a JSON object based on the following strategy and market context:

Strategy:
{json.dumps(strategy_insights.to_dict(), indent=2)}

Market Context:
{json.dumps(market_context.to_dict(), indent=2)}

Remember:
1. Conditions must be valid Python code using available indicators
2. Use support/resistance levels from market context to inform indicator thresholds
3. Parameters should align with the strategy's risk/reward profile"""

            # Create messages for OpenAI chat completion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Use OpenAI chat completion directly
            llm = ChatOpenAI(
                model="gpt-4-1106-preview",
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}  # Force JSON response
            )
            
            response = await llm.agenerate([messages])
            if not response or not response.generations:
                logger.error("No response from LLM")
                return {}, {}
                
            try:
                response_text = response.generations[0][0].text
                logger.info(f"Raw LLM response: {response_text}")
                
                # Try to clean the response if needed
                cleaned_text = response_text.strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()
                
                logger.info(f"Cleaned response text: {cleaned_text}")
                
                try:
                    rules = json.loads(cleaned_text)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error: {str(e)}")
                    logger.error(f"Error at position {e.pos}: {cleaned_text[max(0, e.pos-20):min(len(cleaned_text), e.pos+20)]}")
                    return {}, {}
                
                conditions = rules.get('conditions', {})
                parameters = rules.get('parameters', {})
                
                # Validate conditions are Python code
                for condition_list in conditions.values():
                    for condition in condition_list:
                        if not isinstance(condition, str):
                            continue
                        # Check for natural language
                        if any(word in condition.lower() for word in ['touches', 'level', 'signs', 'change', 'confidence']):
                            logger.warning(f"Invalid condition format: {condition}")
                            return {}, {}
                        # Try to compile condition
                        try:
                            compile(condition, '<string>', 'eval')
                        except SyntaxError:
                            logger.warning(f"Invalid Python syntax in condition: {condition}")
                            return {}, {}
                            
                return conditions, parameters
                
            except Exception as e:
                logger.error(f"Error generating trading rules: {str(e)}")
                logger.exception("Full traceback:")
                return {}, {}
                
        except Exception as e:
            logger.error(f"Error in outer block: {str(e)}")
            logger.exception("Full traceback:")
            return {}, {}
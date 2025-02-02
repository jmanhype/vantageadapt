"""LLM interface for strategy generation."""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage

# Set up LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "kagnar"

from prompts.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime types with detailed characteristics."""
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"
    BREAKOUT = "BREAKOUT"
    BREAKDOWN = "BREAKDOWN"
    REVERSAL = "REVERSAL"
    UNKNOWN = "UNKNOWN"

@dataclass
class MarketContext:
    """Rich market context for strategic decisions."""
    regime: MarketRegime
    confidence: float
    volatility_level: float
    trend_strength: float
    volume_profile: str
    risk_level: str
    key_levels: Dict[str, List[float]]
    analysis: Dict[str, str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'volatility_level': self.volatility_level,
            'trend_strength': self.trend_strength,
            'volume_profile': self.volume_profile,
            'risk_level': self.risk_level,
            'key_levels': self.key_levels,
            'analysis': self.analysis
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketContext':
        """Create from dictionary with error handling."""
        try:
            regime = MarketRegime(data.get('regime', 'UNKNOWN'))
        except ValueError:
            regime = MarketRegime.UNKNOWN
            
        return cls(
            regime=regime,
            confidence=float(data.get('confidence', 0.0)),
            volatility_level=float(data.get('volatility_level', 0.0)),
            trend_strength=float(data.get('trend_strength', 0.0)),
            volume_profile=str(data.get('volume_profile', 'neutral')),
            risk_level=str(data.get('risk_level', 'medium')),
            key_levels=data.get('key_levels', {'support': [], 'resistance': []}),
            analysis=data.get('analysis', {
                'price_action': '',
                'volume_analysis': '',
                'momentum': '',
                'volatility': ''
            })
        )

@dataclass
class StrategyInsight:
    """Strategic trading insights and recommendations."""
    regime_change_probability: float
    suggested_position_size: float
    risk_reward_target: float
    entry_zones: List[Dict[str, float]]
    exit_zones: List[Dict[str, float]]
    stop_loss_zones: List[Dict[str, float]]
    trade_frequency: str
    position_sizing_advice: str
    risk_management_notes: List[str]
    opportunity_description: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime_change_probability': self.regime_change_probability,
            'suggested_position_size': self.suggested_position_size,
            'risk_reward_target': self.risk_reward_target,
            'entry_zones': self.entry_zones,
            'exit_zones': self.exit_zones,
            'stop_loss_zones': self.stop_loss_zones,
            'trade_frequency': self.trade_frequency,
            'position_sizing_advice': self.position_sizing_advice,
            'risk_management_notes': self.risk_management_notes,
            'opportunity_description': self.opportunity_description
        }

class LLMInterface:
    """Strategic trading advisor using LLM capabilities."""
    
    def __init__(self):
        """Initialize LLM interface."""
        self.client = None
        self.prompt_manager = PromptManager()
        self._current_strategy = None  # Store current strategy
        logger.info("Available prompts: %s", self.prompt_manager.get_all_prompt_keys())
        
    @classmethod
    async def create(cls) -> 'LLMInterface':
        """Create and initialize a new LLMInterface instance."""
        instance = cls()
        await instance._initialize()
        return instance

    async def _initialize(self) -> None:
        """Initialize the LLM interface."""
        try:
            # Initialize LangChain OpenAI client
            self.client = ChatOpenAI(
                model="gpt-4-1106-preview",
                temperature=0
            )
            logger.info("LLM interface initialized successfully")
            # Log loaded prompts
            for key in self.prompt_manager.get_all_prompt_keys():
                system = self.prompt_manager.get_prompt_content(key, 'system')
                user = self.prompt_manager.get_prompt_content(key, 'user_template')
                logger.info("Loaded prompt '%s':", key)
                logger.info("System prompt: %s", system[:100] + "..." if system else "None")
                logger.info("User template: %s", user[:100] + "..." if user else "None")
        except Exception as e:
            logger.error(f"Error initializing LLM interface: {str(e)}")
            raise
            
    def chat_completion(self, messages: List[Dict], model: str = "gpt-4-1106-preview", response_format: Dict = None) -> Any:
        """Send chat completion request using LangChain."""
        try:
            # Convert dict messages to LangChain message types
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Add JSON format instruction if response_format is set to json_object
                    if response_format and response_format.get("type") == "json_object":
                        msg["content"] = f"{msg['content']} Please provide your response in JSON format."
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
            
            # Make API request with automatic tracing
            response = self.client.invoke(
                lc_messages,
                response_format={"type": "json_object"} if response_format is None else response_format
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            return None

    def parse_json_response(self, response: Any) -> Dict:
        """Parse JSON response from LLM.
        
        Args:
            response: LLM response object
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            if not response:
                raise ValueError("Invalid response format")
                
            # LangChain response is already the content string
            content = response.content.strip()
            
            # Handle potential markdown code blocks
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
                
            # Clean up any remaining whitespace or newlines
            content = content.strip()
            
            # Remove comments and fix trailing commas
            lines = content.split('\n')
            clean_lines = []
            for line in lines:
                # Remove inline comments
                comment_idx = line.find('//')
                if comment_idx >= 0:
                    line = line[:comment_idx]
                # Remove trailing commas
                if line.rstrip().endswith(',}'):
                    line = line.replace(',}', '}')
                if line.rstrip().endswith(',]'):
                    line = line.replace(',]', ']')
                clean_lines.append(line)
            
            clean_json = '\n'.join(clean_lines)
            
            # Parse JSON
            return json.loads(clean_json)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            logger.error(f"Raw response: {content}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {}

    async def analyze_market(self, market_data: pd.DataFrame) -> MarketContext:
        """Analyze market data and return context."""
        try:
            # Calculate trend strength using price momentum
            price_changes = market_data['price'].pct_change()
            trend_strength = abs(price_changes.mean()) / price_changes.std() if price_changes.std() != 0 else 0.0
            trend_strength = min(max(trend_strength, 0.0), 1.0)  # Normalize to [0, 1]
            
            # Prepare market data summary
            summary = {
                'price': float(market_data['price'].iloc[-1]),
                'price_change': float(market_data['price'].pct_change().iloc[-1]),
                'volume': float(market_data['sol_volume'].iloc[-1]) if 'sol_volume' in market_data else 0.0,
                'volatility': float(market_data['price'].pct_change().std()),
                'trend_strength': float(trend_strength),
                'recent_high': float(market_data['price'].rolling(20).max().iloc[-1]),
                'recent_low': float(market_data['price'].rolling(20).min().iloc[-1])
            }
            
            # Get prompts from prompt manager
            system_prompt = self.prompt_manager.get_prompt_content('trading/market_analysis', 'system')
            user_prompt = self.prompt_manager.get_prompt_content('trading/market_analysis', 'user')
            
            if not system_prompt or not user_prompt:
                raise ValueError("Failed to load market analysis prompts")
            
            logger.info("Using market analysis prompts:")
            logger.info("System: %s", system_prompt[:200] + "...")
            logger.info("User template: %s", user_prompt[:200] + "...")
                
            # Format user prompt with market summary
            user_prompt = user_prompt.format(**summary)
            
            # Get LLM analysis
            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse response
            analysis = self.parse_json_response(response)
            if not analysis:
                logger.error("Failed to parse market analysis")
                return None
                
            logger.info(f"Raw LLM response: {response.content if response else 'None'}")
            logger.info(f"Parsed market analysis: {json.dumps(analysis, indent=2)}")
            
            # Validate required fields
            required_fields = ['regime', 'confidence', 'volatility_level', 'trend_strength', 'volume_profile', 'risk_level']
            missing_fields = [field for field in required_fields if field not in analysis]
            if missing_fields:
                logger.error(f"Missing required fields in market analysis: {missing_fields}")
                return None
            
            # Create MarketContext with default values for optional fields
            return MarketContext(
                regime=MarketRegime(analysis.get('regime', 'UNKNOWN')),
                confidence=float(analysis.get('confidence', 0.0)),
                volatility_level=float(analysis.get('volatility_level', 0.0)),
                trend_strength=float(analysis.get('trend_strength', 0.0)),
                volume_profile=str(analysis.get('volume_profile', 'neutral')),
                risk_level=str(analysis.get('risk_level', 'medium')),
                key_levels=analysis.get('key_levels', {'support': [], 'resistance': []}),
                analysis=analysis.get('analysis', {
                    'price_action': '',
                    'volume_analysis': '',
                    'momentum': '',
                    'volatility': ''
                })
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            return None

    async def generate_strategy(
        self, 
        theme: str, 
        market_context: MarketContext,
        base_parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[StrategyInsight]:
        """Generate trading strategy based on market context.
        
        Args:
            theme: Trading theme to focus on
            market_context: Current market context
            base_parameters: Optional parameters from similar successful strategies to use as base
            
        Returns:
            Optional[StrategyInsight]: Generated strategy insights
        """
        try:
            # Get prompts from prompt manager
            system_prompt = self.prompt_manager.get_prompt_content('trading/strategy_generation', 'system')
            user_template = self.prompt_manager.get_prompt_content('trading/strategy_generation', 'user_template')
            
            if not system_prompt or not user_template:
                logger.error("Missing required prompts for strategy generation")
                return None
                
            # Format user prompt with theme and market context
            user_prompt = user_template.format(
                theme=theme,
                market_context=json.dumps(market_context.to_dict(), indent=2),
                base_parameters=json.dumps(base_parameters, indent=2) if base_parameters else "null"
            )
            
            # Create messages for chat completion
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Get strategy response
            response = self.chat_completion(messages, response_format={"type": "json_object"})
            if not response:
                return None
                
            # Parse response
            strategy_data = self.parse_json_response(response)
            if not strategy_data:
                return None
                
            # Create StrategyInsight from response
            strategy = StrategyInsight(
                regime_change_probability=float(strategy_data.get('regime_change_probability', 0.0)),
                suggested_position_size=float(strategy_data.get('suggested_position_size', 0.0)),
                risk_reward_target=float(strategy_data.get('risk_reward_target', 0.0)),
                entry_zones=strategy_data.get('entry_zones', []),
                exit_zones=strategy_data.get('exit_zones', []),
                stop_loss_zones=strategy_data.get('stop_loss_zones', []),
                trade_frequency=str(strategy_data.get('trade_frequency', 'medium')),
                position_sizing_advice=str(strategy_data.get('position_sizing_advice', '')),
                risk_management_notes=strategy_data.get('risk_management_notes', []),
                opportunity_description=str(strategy_data.get('opportunity_description', ''))
            )
            
            # Apply base parameters if provided
            if base_parameters:
                for key, value in base_parameters.items():
                    if hasattr(strategy, key):
                        setattr(strategy, key, value)
                        logger.info(f"Applied base parameter {key}: {value}")
            
            self._current_strategy = strategy
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None

    async def get_current_strategy(self) -> Optional[StrategyInsight]:
        """Get the current strategy.
        
        Returns:
            The current strategy or None if no strategy is set
        """
        return self._current_strategy

    async def improve_strategy(self, trades_df: pd.DataFrame, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Improve strategy based on performance metrics and trade history.
        
        Args:
            trades_df: DataFrame containing trade history
            metrics: Dictionary of performance metrics
            
        Returns:
            Dictionary containing improved strategy parameters and conditions
        """
        try:
            # Prepare trade summary
            trade_summary = {
                'total_trades': len(trades_df),
                'win_rate': metrics.get('win_rate', 0.0),
                'profit_factor': metrics.get('profit_factor', 0.0),
                'avg_win': metrics.get('avg_win', 0.0),
                'avg_loss': metrics.get('avg_loss', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': metrics.get('sortino_ratio', 0.0)
            }
            
            # Get LLM analysis and improvements
            response = self.chat_completion(
                messages=[{
                    "role": "system",
                    "content": """You are an expert trading strategy optimizer. Analyze the performance metrics and suggest improvements.
You MUST respond with ONLY a JSON object containing improved strategy parameters and conditions in the following format (no other text):
{
    "analysis": {
        "strengths": ["string"],
        "weaknesses": ["string"],
        "opportunities": ["string"]
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
    },
    "conditions": {
        "entry": ["(df_indicators['rsi'] < 30)"],
        "exit": ["(df_indicators['rsi'] > 70)"]
    }
}"""
                }, {
                    "role": "user",
                    "content": f"Based on these performance metrics, suggest improvements to the strategy:\n\nTrade Summary:\n{json.dumps(trade_summary, indent=2)}"
                }],
                response_format={"type": "json_object"}
            )
            
            # Parse response
            improvements = self.parse_json_response(response)
            if not improvements:
                logger.error("Failed to parse strategy improvements")
                return {
                    'parameters': {
                        'take_profit': 0.05,
                        'stop_loss': 0.03,
                        'order_size': 0.1,
                        'max_orders': 3,
                        'sl_window': 400,
                        'post_buy_delay': 2,
                        'post_sell_delay': 5,
                        'enable_sl_mod': False,
                        'enable_tp_mod': False
                    },
                    'conditions': {
                        'entry': ["(df_indicators['rsi'] < 30)"],
                        'exit': ["(df_indicators['rsi'] > 70)"]
                    }
                }
                
            return improvements
            
        except Exception as e:
            logger.error(f"Error improving strategy: {str(e)}")
            return {
                'parameters': {
                    'take_profit': 0.05,
                    'stop_loss': 0.03,
                    'order_size': 0.1,
                    'max_orders': 3,
                    'sl_window': 400,
                    'post_buy_delay': 2,
                    'post_sell_delay': 5,
                    'enable_sl_mod': False,
                    'enable_tp_mod': False
                },
                'conditions': {
                    'entry': ["(df_indicators['rsi'] < 30)"],
                    'exit': ["(df_indicators['rsi'] > 70)"]
                }
            }

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
            # Load rules generation prompt
            prompt = self.prompt_manager.get_prompt_content('trading/rules_generation')
            if not prompt:
                logger.error("Rules generation prompt not found")
                return {}, {}
                
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

            # Generate rules
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = await self.llm.agenerate(messages)
            if not response or not response.generations:
                logger.error("No response from LLM")
                return {}, {}
                
            # Extract the response text
            response_text = response.generations[0].text
            if not response_text:
                logger.error("Empty response from LLM")
                return {}, {}
                
            try:
                rules = json.loads(response_text)
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
                
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                return {}, {}
                
        except Exception as e:
            logger.error(f"Error generating trading rules: {str(e)}")
            logger.exception("Full traceback:")
            return {}, {}

    async def analyze_performance(self, metrics: Dict[str, float], trade_memory_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading performance and provide insights.
        
        Args:
            metrics: Trading performance metrics
            trade_memory_stats: Detailed trade statistics
            
        Returns:
            Performance analysis dictionary
        """
        try:
            # Get prompts from prompt manager
            system_prompt = self.prompt_manager.get_prompt_content('evaluation/performance_analysis', 'system')
            user_prompt = self.prompt_manager.get_prompt_content('evaluation/performance_analysis', 'user_template')
            
            if not system_prompt or not user_prompt:
                raise ValueError("Failed to load performance analysis prompts")
                
            # Format user prompt with metrics and stats
            user_prompt = user_prompt.format(
                metrics=json.dumps(metrics, indent=2),
                trade_memory_stats=json.dumps(trade_memory_stats, indent=2)
            )
            
            # Get LLM analysis
            response = self.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            if not response:
                raise ValueError("Failed to get performance analysis from LLM")
                
            # Parse response
            analysis_data = self.parse_json_response(response)
            if not analysis_data:
                raise ValueError("Failed to parse performance analysis response")
                
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {str(e)}")
            return {
                "performance_summary": {
                    "overall_assessment": "Analysis failed",
                    "key_strengths": [],
                    "key_weaknesses": []
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
        performance_analysis: Dict[str, Any]
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
            # Create strategy description
            description = f"""Theme: {theme}
Market Context:
- Regime: {market_context.regime.value}
- Confidence: {market_context.confidence:.2f}
- Risk Level: {market_context.risk_level}
- Opportunity Score: {market_context.opportunity_score:.2f}

Strategy Insights:
- Risk/Reward Target: {strategy_insights.risk_reward_target:.2f}
- Position Sizing: {strategy_insights.position_sizing_advice}
- Trade Frequency: {strategy_insights.trade_frequency}
- Opportunity: {strategy_insights.opportunity_description}

Performance:
- Total Return: {metrics.get('total_return', 0.0):.2%}
- Win Rate: {metrics.get('win_rate', 0.0):.2%}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}
- Max Drawdown: {metrics.get('max_drawdown', 0.0):.2%}

Analysis:
{performance_analysis.get('overall_assessment', 'No assessment available')}"""

            # Save to database
            from ..database import db
            await db.save_strategy(
                theme=theme,
                description=description,
                conditions=conditions,
                parameters=parameters,
                metrics=metrics,
                market_context=market_context.to_dict(),
                strategy_insights=strategy_insights.to_dict(),
                performance_analysis=performance_analysis
            )
            
        except Exception as e:
            logger.error(f"Error saving strategy results: {str(e)}")
            return None
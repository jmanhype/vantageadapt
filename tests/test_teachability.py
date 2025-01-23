"""Test script for demonstrating the Teachability capability."""

import os
from typing import Dict, Any
import logging
from dotenv import load_dotenv
from autogen.agentchat.assistant_agent import ConversableAgent
from mem0 import MemoryClient
from src.strat_optim.memory.teachability import Teachability

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_trading_agent(llm_config: Dict[str, Any]) -> ConversableAgent:
    """Create a trading agent with teachability capability.
    
    Args:
        llm_config: Configuration for the language model
        
    Returns:
        ConversableAgent: The trading agent with teachability
    """
    # Create the base agent
    agent = ConversableAgent(
        name="trading_agent",
        system_message="""You are an expert trading agent that analyzes market conditions 
        and provides trading strategies. You have deep knowledge of technical analysis,
        market patterns, and risk management.""",
        llm_config=llm_config
    )
    
    # Initialize mem0 client
    memory_client = MemoryClient(
        api_key=os.getenv("MEM0_API_KEY")
    )
    
    # Create teachability capability
    teachability = Teachability(
        agent_id="trading_agent",
        memory_client=memory_client,
        verbosity=1,
        llm_config=llm_config
    )
    
    # Add teachability to agent
    teachability.add_to_agent(agent)
    
    return agent

def main():
    """Run the teachability demonstration."""
    # Set up LLM config
    llm_config = {
        "model": "gpt-4-turbo-preview",
        "temperature": 0.1,
        "max_tokens": 2000,
        "api_key": os.getenv("OPENAI_API_KEY")
    }
    
    # Create agent
    agent = create_trading_agent(llm_config)
    
    # Test messages
    test_messages = [
        """In a ranging market with low volatility (0.2), I found that using tight stop losses 
        (2-3% from entry) and taking profits at 1.5x the stop loss distance works well. This 
        strategy had a 65% win rate over 100 trades.""",
        
        """When trading breakouts in high volatility periods, it's crucial to wait for a 
        retest of the breakout level before entering. This approach reduced false breakouts 
        by 40% in my backtests.""",
        
        """For mean reversion strategies in crypto markets, I've observed that waiting for 
        3 consecutive red candles with decreasing volume often indicates a potential 
        reversal point."""
    ]
    
    # Process messages
    for i, message in enumerate(test_messages, 1):
        logger.info(f"\nProcessing message {i}...")
        
        # Create a user agent for sending messages
        user = ConversableAgent(
            name="user",
            llm_config=False  # No LLM needed for user
        )
        
        # Send message and get response
        response = agent.generate_reply(
            messages=[{"role": "user", "content": message}],
            sender=user
        )
        
        logger.info(f"Agent response: {response}")
        
        # Add some spacing between messages
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 
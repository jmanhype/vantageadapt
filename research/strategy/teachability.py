"""Teachability module for enhancing agent learning capabilities.

This module provides a capability that allows agents to learn and remember information
from conversations and apply it to future trading decisions.
"""

from typing import Dict, Optional, Union, List, Any
import os
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

from autogen.agentchat.assistant_agent import ConversableAgent
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from termcolor import colored
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from mem0 import MemoryClient

logger = logging.getLogger(__name__)

class Teachability(AgentCapability):
    """Capability that enables agents to learn and remember information from conversations."""
    
    DEFAULT_CONFIG = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 2000,
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        }
    }
    
    def __init__(
        self,
        agent_id: str,
        memory_client: Optional[MemoryClient] = None,
        api_version: str = "v1.1",
        verbosity: int = 0,
        reset_db: bool = False,
        recall_threshold: float = 0.7,
        max_num_retrievals: int = 5,
        llm_config: Optional[Dict] = None,
    ) -> None:
        """Initialize the Teachability class.

        Args:
            agent_id: Unique identifier for the agent
            memory_client: Memory client for storing and retrieving information
            api_version: API version for memory storage
            verbosity: Level of verbosity for logging
            reset_db: Whether to reset the memory database
            recall_threshold: Threshold for memory recall
            max_num_retrievals: Maximum number of memories to retrieve
            llm_config: Configuration for the language model
        """
        self.user_id = agent_id  # Store agent_id as user_id for memory operations
        self.memory = memory_client or MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        self.api_version = api_version
        self.verbosity = verbosity
        self.recall_threshold = recall_threshold
        self.max_num_retrievals = max_num_retrievals
        self.analyzer = None
        self.teachable_agent = None

        if reset_db:
            self.memory.reset()

        # Set default LLM config if none provided
        self.llm_config = llm_config or {
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 2000
        }

        # Ensure API key is set
        if isinstance(self.llm_config, dict) and "api_key" not in self.llm_config:
            self.llm_config["api_key"] = os.getenv("OPENAI_API_KEY")

    def add_to_agent(self, agent: ConversableAgent) -> None:
        """Add the teachability capability to an agent.
        
        Args:
            agent: The agent to add the capability to
        """
        self.teachable_agent = agent
        agent.register_hook(
            hookable_method="process_last_received_message",
            hook=self.process_last_received_message
        )

        if self.llm_config is None:
            self.llm_config = agent.llm_config
        assert self.llm_config, "Teachability requires a valid llm_config."

        self.analyzer = TextAnalyzerAgent(llm_config=self.llm_config)

        agent.update_system_message(
            agent.system_message +
            "\nYou've been given the special ability to remember trading patterns and insights from prior conversations."
        )

    def process_last_received_message(self, text: Union[str, Dict]) -> str:
        """Process the last received message and store relevant memories.

        Args:
            text: Text to process and potentially store as memory

        Returns:
            str: The processed message with expanded text
        """
        logger.debug("Processing message: %s", text)
        
        # Extract text content
        text_content = text["content"] if isinstance(text, dict) else text
        logger.debug("Extracted text content: %s", text_content)

        # Get existing memories
        memories = self.memory.get_all(user_id=self.user_id)
        logger.debug("Retrieved %d existing memories", len(memories) if memories else 0)

        # Consider storing new memories
        try:
            logger.debug("Considering memory storage for text: %s", text_content)
            self._consider_memo_storage(text_content)
            logger.debug("Memory storage consideration complete")
        except Exception as e:
            logger.error("Failed to store memories: %s", str(e))

        # Get updated memories
        updated_memories = self.memory.get_all(user_id=self.user_id)
        logger.debug("Retrieved %d memories after update", len(updated_memories) if updated_memories else 0)

        # Concatenate memory texts
        expanded_text = self._concatenate_memo_texts(updated_memories if updated_memories else [])
        logger.debug("Expanded text with memories: %s", expanded_text)

        return expanded_text

    def _consider_memo_storage(self, text: Union[str, Dict]) -> None:
        """Consider storing memory from text.

        Args:
            text: The text to consider storing, either as a string or a dictionary with a 'content' key.
        """
        logger = logging.getLogger(__name__)
        
        # Extract text content
        text_content = text["content"] if isinstance(text, dict) else text
        logger.debug("Considering memory storage for text: %s", text_content)
        
        # Check if text contains a trading task or strategy
        has_task = self._analyze(text_content, "Does this text contain a trading task or strategy? Answer yes or no.")
        logger.debug("Has trading task: %s", has_task)
        
        if has_task.lower() == "yes":
            try:
                # Extract trading advice
                advice = self._analyze(text_content, "What is the trading advice or strategy mentioned?")
                logger.debug("Extracted advice: %s", advice)
                
                # Extract specific task
                task = self._analyze(text_content, "What specific trading task or pattern is being discussed?")
                logger.debug("Extracted task: %s", task)
                
                # Generalize the task into a pattern
                pattern = self._analyze(task, "Convert this specific trading task into a more general pattern.")
                logger.debug("Generalized pattern: %s", pattern)
                
                # Store pattern-solution pair
                memory_content = {
                    "type": "pattern_solution",
                    "pattern": pattern,
                    "solution": advice,
                    "original_text": text_content,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Create memory in mem0ai format
                messages = [{
                    "role": "system",
                    "content": json.dumps(memory_content),
                    "metadata": {
                        "type": "pattern_solution",
                        "timestamp": datetime.now().isoformat(),
                        "agent_id": self.user_id
                    }
                }]
                
                logger.debug("Storing pattern-solution pair: %s", memory_content)
                self.memory.add(
                    messages=messages,
                    user_id=self.user_id
                )
                logger.debug("Successfully stored pattern-solution pair")
            except Exception as e:
                logger.error("Failed to store pattern-solution pair: %s", str(e))
        
        # Check if text contains general trading information
        has_info = self._analyze(text_content, "Does this text contain general trading information? Answer yes or no.")
        logger.debug("Has trading info: %s", has_info)
        
        if has_info.lower() == "yes":
            try:
                # Generate question
                question = self._analyze(text_content, "Generate a question that captures the key trading concept discussed.")
                logger.debug("Generated question: %s", question)
                
                # Extract answer
                answer = self._analyze(text_content, "What is the answer to this question based on the text?")
                logger.debug("Generated answer: %s", answer)
                
                # Store Q&A pair
                memory_content = {
                    "type": "qa_pair",
                    "question": question,
                    "answer": answer,
                    "original_text": text_content,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Create memory in mem0ai format
                messages = [{
                    "role": "system",
                    "content": json.dumps(memory_content),
                    "metadata": {
                        "type": "qa_pair",
                        "timestamp": datetime.now().isoformat(),
                        "agent_id": self.user_id
                    }
                }]
                
                logger.debug("Storing Q&A pair: %s", memory_content)
                self.memory.add(
                    messages=messages,
                    user_id=self.user_id
                )
                logger.debug("Successfully stored Q&A pair")
            except Exception as e:
                logger.error("Failed to store Q&A pair: %s", str(e))

    def _consider_memo_retrieval(self, comment: Union[Dict, str]) -> str:
        """Consider retrieving relevant memories for the comment.
        
        Args:
            comment: The comment to find relevant memories for
            
        Returns:
            The comment enhanced with relevant memories
        """
        if self.verbosity >= 1:
            print(colored("\nSearching for Relevant Trading Memories", "light_yellow"))
        memo_list = self._retrieve_relevant_memos(comment)

        response = self._analyze(
            comment,
            "Does the TEXT involve a trading task or decision? Answer with just yes or no.",
        )

        if "yes" in response.lower():
            if self.verbosity >= 1:
                print(colored("\nSearching for Similar Trading Patterns", "light_yellow"))
            task = self._analyze(
                comment,
                "Extract the core trading scenario or decision point.",
            )

            general_task = self._analyze(
                task,
                "Generalize this trading scenario into a pattern that might recur.",
            )

            memo_list.extend(self._retrieve_relevant_memos(general_task))

        memo_list = list(set(memo_list))
        return comment + self._concatenate_memo_texts(memo_list)

    def _retrieve_relevant_memos(self, input_text: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for the input text.
        
        Args:
            input_text: The text to find relevant memories for
            
        Returns:
            List of relevant memories
        """
        try:
            # Get all memories for our trading system
            memories = self.memory.get_all(
                user_id=self.user_id,
                limit=self.max_num_retrievals
            )
            
            # Filter memories based on relevance
            relevant_memories = []
            for memory in memories:
                try:
                    if isinstance(memory, dict):
                        if "content" in memory:
                            content = json.loads(memory["content"])
                            relevant_memories.append(content)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse memory: {e}")
                    continue

            if self.verbosity >= 1 and not relevant_memories:
                print(colored("\nNo Sufficiently Similar Memories Found", "light_yellow"))

            return relevant_memories[:self.max_num_retrievals]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def _concatenate_memo_texts(self, memo_list: List[Dict[str, Any]]) -> str:
        """Concatenate memory texts into a single string.
        
        Args:
            memo_list: List of memories to concatenate
            
        Returns:
            str: Concatenated memory texts
        """
        if not memo_list:
            return ""
            
        memo_texts = []
        for memo in memo_list:
            if isinstance(memo, dict):
                if "type" in memo and memo["type"] == "pattern_solution":
                    memo_texts.append(
                        f"\nRELEVANT PATTERN: {memo['pattern']}\n"
                        f"SOLUTION: {memo['solution']}"
                    )
                elif "type" in memo and memo["type"] == "qa_pair":
                    memo_texts.append(
                        f"\nRELEVANT Q&A:\n"
                        f"Q: {memo['question']}\n"
                        f"A: {memo['answer']}"
                    )
                
        return "\n".join(memo_texts)

    def _analyze(self, text: Union[Dict, str], prompt: str) -> str:
        """Analyze text using the analyzer agent.
        
        Args:
            text: The text to analyze
            prompt: The prompt to use for analysis
            
        Returns:
            str: The analysis result
        """
        try:
            if not self.analyzer:
                logger.info("Initializing TextAnalyzerAgent...")
                self.analyzer = TextAnalyzerAgent(
                    name="text_analyzer",
                    system_message="""You are a text analysis expert that helps extract and analyze information 
                    from trading-related text. You provide clear, concise answers to questions about the text.""",
                    llm_config=self.llm_config
                )
                logger.info("TextAnalyzerAgent initialized successfully")
                
            # Get the actual text content
            text_content = text.get("content", text) if isinstance(text, dict) else text
            
            # Log the analysis request
            logger.info("Analyzing text with prompt: %s", prompt)
            logger.info("Text content: %s", text_content)
            
            # Create a temporary agent for sending messages
            temp_agent = ConversableAgent(
                name="temp_agent",
                llm_config=False  # No LLM needed for this agent
            )
            
            # Get the response from the analyzer
            response = self.analyzer.generate_reply(
                messages=[
                    {
                        "role": "system",
                        "content": "You are analyzing trading-related text to extract specific information."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nTEXT: {text_content}"
                    }
                ],
                sender=temp_agent
            )
            
            # Log the response
            logger.info("Analyzer response: %s", response)
            
            return str(response).strip() if response is not None else ""
            
        except Exception as e:
            logger.error("Error in _analyze: %s", str(e), exc_info=True)
            raise

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize mem0ai client with configuration
    memory_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
    
    # Initialize teachability capability
    teachability = Teachability(
        agent_id="test_agent",
        memory_client=memory_client,
        verbosity=1
    )
    
    # Create a test agent
    agent = ConversableAgent(
        name="trading_agent",
        system_message="""You are an expert trading agent that analyzes market conditions 
        and provides trading strategies. You have deep knowledge of technical analysis,
        market patterns, and risk management.""",
        llm_config={
            "model": "gpt-4o",
            "temperature": 0.1,
            "max_tokens": 2000,
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    )
    
    # Add teachability to agent
    teachability.add_to_agent(agent)
    
    print("Teachability capability initialized successfully!")
    
    # Test messages containing different types of trading information
    test_messages = [
        {
            "role": "user",
            "content": """In a ranging market with low volatility (0.2), I found that using tight stop losses 
            (2-3% from entry) and taking profits at 1.5x the stop loss distance works well. This 
            strategy had a 65% win rate over 100 trades."""
        }
    ]
    
    # Process test messages
    for message in test_messages:
        print(f"\nProcessing message: {message['content']}")
        teachability.process_last_received_message(message) 
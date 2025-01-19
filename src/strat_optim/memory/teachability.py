"""Teachability module for enhancing agent learning capabilities.

This module provides a capability that allows agents to learn and remember information
from conversations and apply it to future trading decisions.
"""

from typing import Dict, Optional, Union
import os
from autogen.agentchat.assistant_agent import ConversableAgent
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from termcolor import colored
from .mem0 import Memory
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class TextAnalyzerAgent:
    """Agent for analyzing text using LLM.
    
    This agent is responsible for analyzing text using a language model,
    providing insights and extracting relevant information from the text.
    """
    
    def __init__(self, llm_config: Optional[Dict] = None) -> None:
        """Initialize the text analyzer agent.
        
        Args:
            llm_config: Configuration for the language model
        """
        self.llm_config = llm_config or {}
        
    def analyze(self, text: str, prompt: str, llm_config: Optional[Dict] = None) -> str:
        """Analyze text using the language model.
        
        Args:
            text: The text to analyze
            prompt: The prompt to use for analysis
            llm_config: Optional override for the language model configuration
            
        Returns:
            str: The analysis result
        """
        config = llm_config or self.llm_config
        
        # Create a temporary agent for analysis
        agent = ConversableAgent(
            name="analyzer",
            system_message="You are a helpful assistant that analyzes text based on specific prompts.",
            llm_config=config
        )
        
        # Construct the message for analysis
        message = f"{prompt}\n\nTEXT: {text}"
        
        # Get the response from the agent
        response = agent.generate_reply(
            messages=[{"role": "user", "content": message}],
            sender=ConversableAgent(name="user", llm_config=False)
        )
        
        # Log the response for debugging
        logger.debug("Analyzer response: %s", response)
        
        # Return the response, ensuring it's a string
        return str(response).strip() if response is not None else ""

class Teachability(AgentCapability):
    """Capability that enables agents to learn and remember information from conversations."""
    
    DEFAULT_CONFIG = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4-turbo-preview",
                "temperature": 0.1,
                "max_tokens": 2000,
                "api_key": os.getenv("OPENAI_API_KEY")
            }
        }
    }
    
    def __init__(
        self,
        agent_id: str,
        memory_client: Memory,
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
        self.memory = memory_client
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
            "model": "gpt-4-turbo-preview",
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
        logger.debug("Retrieved %d existing memories", len(memories))

        # Consider storing new memories
        try:
            logger.debug("Considering memory storage for text: %s", text_content)
            self._consider_memo_storage(text_content)
            logger.debug("Memory storage consideration complete")
        except Exception as e:
            logger.error("Failed to store memories: %s", str(e))

        # Get updated memories
        updated_memories = self.memory.get_all(user_id=self.user_id)
        logger.debug("Retrieved %d memories after update", len(updated_memories))

        # Concatenate memory texts
        expanded_text = self._concatenate_memo_texts(updated_memories)
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
                    "original_text": text_content
                }
                logger.debug("Storing pattern-solution pair: %s", memory_content)
                self.memory.add([memory_content], user_id=self.user_id)
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
                    "original_text": text_content
                }
                logger.debug("Storing Q&A pair: %s", memory_content)
                self.memory.add([memory_content], user_id=self.user_id)
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

    def _retrieve_relevant_memos(self, input_text: str) -> list:
        """Retrieve relevant memories for the input text.
        
        Args:
            input_text: The text to find relevant memories for
            
        Returns:
            List of relevant memories
        """
        search_results = self.memory.search(
            input_text,
            user_id=self.user_id,
            limit=self.max_num_retrievals
        )
        memo_list = [
            result["memory"]
            for result in search_results
            if result["score"] <= self.recall_threshold
        ]

        if self.verbosity >= 1 and not memo_list:
            print(colored("\nNo Sufficiently Similar Memories Found", "light_yellow"))

        return memo_list

    def _concatenate_memo_texts(self, memo_list: list) -> str:
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
            if isinstance(memo, dict) and "content" in memo:
                memo_texts.append(f"\nRELEVANT MEMORY: {memo['content']}")
                
        return "\n".join(memo_texts)

    def _analyze(self, text: Union[Dict, str], prompt: str) -> str:
        """Analyze text using the analyzer agent.
        
        Args:
            text: The text to analyze
            prompt: The prompt to use for analysis
            
        Returns:
            str: The analysis result
        """
        if not self.analyzer:
            self.analyzer = TextAnalyzerAgent(llm_config=self.llm_config)
            
        # Get the actual text content
        text_content = text.get("content", text) if isinstance(text, dict) else text
        
        # Log the analysis request
        logger.debug("Analyzing text with prompt: %s", prompt)
        logger.debug("Text content: %s", text_content)
        
        # Get the response from the analyzer
        response = self.analyzer.analyze(text_content, prompt)
        
        # Log the response
        logger.debug("Analyzer response: %s", response)
        
        return str(response).strip() if response is not None else "" 
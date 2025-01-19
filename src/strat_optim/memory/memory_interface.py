"""Memory interface for trading system using mem0ai.

This module provides a streamlined interface to the mem0ai memory system,
specifically designed for storing and retrieving trading-related information.
"""

from typing import Dict, List, Optional, Union, Any
import os
import logging
from datetime import datetime
from mem0.memory.main import Memory
from mem0.client.main import MemoryClient

logger = logging.getLogger(__name__)

class TradingMemory:
    """Interface for storing and retrieving trading-related memories using mem0ai."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        agent_id: str = "default_agent",
        user_id: str = "default_user"
    ) -> None:
        """Initialize the trading memory interface.

        Args:
            api_key: API key for mem0ai service
            openai_api_key: API key for OpenAI
            agent_id: Default agent ID for memory operations
            user_id: Default user ID for memory operations
        """
        # Set up API keys
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if api_key:
            os.environ["MEM0_API_KEY"] = api_key

        # Validate API key
        if not api_key and "MEM0_API_KEY" not in os.environ:
            raise ValueError("mem0ai API key is required")

        # Initialize client
        self.client = MemoryClient(api_key=api_key or os.environ["MEM0_API_KEY"])
        self.default_agent_id = agent_id
        self.default_user_id = user_id

    def add_memory(
        self,
        content: Union[str, Dict],
        memory_type: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Add a memory to storage.

        Args:
            content: Memory content to store
            memory_type: Type of memory (e.g., 'pattern', 'strategy', 'qa')
            user_id: Optional user ID (defaults to instance default)
            agent_id: Optional agent ID (defaults to instance default)
            metadata: Additional metadata to store

        Returns:
            List of stored memory records
        """
        try:
            # Prepare memory content
            if isinstance(content, dict):
                memory = content
            else:
                memory = {
                    "content": content,
                    "type": memory_type,
                    "timestamp": datetime.now().isoformat()
                }

            # Add metadata if provided
            if metadata:
                memory.update(metadata)

            # Store memory
            return self.client.add(
                messages=[memory],
                user_id=user_id or self.default_user_id,
                agent_id=agent_id or self.default_agent_id
            )
        except Exception as e:
            logger.error(f"Failed to add memory: {str(e)}")
            return []

    def get_all_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[Dict]:
        """Get all stored memories.

        Args:
            user_id: Optional user ID to filter memories
            agent_id: Optional agent ID to filter memories

        Returns:
            List of memory dictionaries
        """
        try:
            return self.client.get_all(
                user_id=user_id or self.default_user_id,
                agent_id=agent_id or self.default_agent_id
            )
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {str(e)}")
            return []

    def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """Search for relevant memories.

        Args:
            query: Search query string
            user_id: Optional user ID to filter memories
            agent_id: Optional agent ID to filter memories
            limit: Maximum number of results to return

        Returns:
            List of relevant memory dictionaries
        """
        try:
            return self.client.search(
                query=query,
                user_id=user_id or self.default_user_id,
                agent_id=agent_id or self.default_agent_id,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to search memories: {str(e)}")
            return []

    def reset(self) -> None:
        """Reset the memory storage."""
        try:
            self.client.reset()
            logger.debug("Successfully reset memory storage")
        except Exception as e:
            logger.error(f"Failed to reset memory storage: {str(e)}")
            raise 
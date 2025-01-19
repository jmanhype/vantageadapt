"""Memory system for storing and retrieving trading-related information.

This module provides a memory system that can store and retrieve various types of trading-related
information, including market patterns, trade outcomes, and strategy performance metrics.
"""

from typing import Dict, List, Optional, Union, Any, ClassVar
import chromadb
from chromadb.config import Settings
import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class Memory:
    """Memory class for storing and retrieving trading-related information."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "trading_memory",
        api_version: str = "v1.1",
        config: Optional[Dict] = None
    ) -> None:
        """Initialize the memory system.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            api_version: Version of the API for metadata
            config: Configuration dictionary for the memory system
        """
        self.config = config or {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "temperature": 0.1,
            "max_tokens": 2000
        }
        self.api_version = api_version
        
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
            
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

    @classmethod
    def from_config(cls, config: Dict) -> "Memory":
        """Create Memory instance from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Memory instance
        """
        return cls(
            persist_directory=None,
            collection_name="trading_memory",
            api_version="v1.1",
            config=config
        )
        
    def add(self, memories: List[Dict], user_id: str = "default", metadata: Optional[Dict] = None) -> None:
        """Add memories to storage.

        Args:
            memories: List of memory dictionaries to store
            user_id: ID of the user/agent storing the memory
            metadata: Additional metadata to store with the memories
        """
        if not memories:
            logger.debug("No memories to add")
            return

        documents = []
        ids = []
        metadatas = []

        try:
            for i, memory in enumerate(memories):
                # Convert content to string if it's a dictionary
                content = memory.get("content", "")
                if isinstance(content, dict):
                    content = json.dumps(content)

                # Generate unique ID based on memory type and timestamp
                memory_type = memory.get("type", "unknown")
                timestamp = datetime.now().isoformat()
                memory_id = f"{memory_type}_{timestamp}_{i}"

                # Prepare metadata
                memory_metadata = {
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "api_version": self.api_version,
                    "type": memory_type
                }
                if metadata:
                    memory_metadata.update(metadata)

                logger.debug("Prepared memory %d: content=%s, metadata=%s", i, content, memory_metadata)
                documents.append(content)
                ids.append(memory_id)
                metadatas.append(memory_metadata)

            if documents:
                self.collection.add(
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas
                )
                logger.debug("Successfully added %d memories", len(documents))
            else:
                logger.warning("No valid memories to add after processing")

        except Exception as e:
            logger.error("Failed to add memories: %s", str(e))
            raise
            
    def get_all(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get all stored memories.
        
        Args:
            user_id: Optional user ID to filter memories
            
        Returns:
            List of memory dictionaries
        """
        try:
            query = {"user_id": user_id} if user_id else None
            results = self.collection.get(
                where=query
            )
            
            memories = []
            for i in range(len(results["ids"])):
                memories.append({
                    "id": results["ids"][i],
                    "memory": {"content": results["documents"][i]},
                    "metadata": results["metadatas"][i]
                })
            return memories
            
        except Exception as e:
            logger.error("Failed to retrieve memories: %s", str(e))
            return []
            
    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """Search for relevant memories.
        
        Args:
            query: Search query string
            user_id: Optional user ID to filter memories
            limit: Maximum number of results to return
            
        Returns:
            List of relevant memory dictionaries
        """
        try:
            where = {"user_id": user_id} if user_id else None
            results = self.collection.query(
                query_texts=[query],
                where=where,
                n_results=limit
            )
            
            memories = []
            if results["ids"]:
                for i in range(len(results["ids"][0])):
                    memories.append({
                        "id": results["ids"][0][i],
                        "memory": {"content": results["documents"][0][i]},
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })
            return memories
            
        except Exception as e:
            logger.error("Failed to search memories: %s", str(e))
            return []
            
    def reset(self) -> None:
        """Reset the memory storage."""
        try:
            # Get all memory IDs
            results = self.collection.get()
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
            logger.debug("Successfully reset memory storage")
        except Exception as e:
            logger.error("Failed to reset memory storage: %s", str(e))
            raise 
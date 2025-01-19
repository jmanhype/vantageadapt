"""Memory system for storing and retrieving trading-related information.

This module provides a memory system that can store and retrieve various types of trading-related
information, including market patterns, trade outcomes, and strategy performance metrics.
"""

from typing import Dict, List, Optional, Union, Any
import chromadb
from chromadb.config import Settings
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Memory:
    """Memory system for storing and retrieving trading-related information.
    
    This class implements a vector-based memory system using ChromaDB for efficient
    storage and retrieval of trading-related information.
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = "./memory_store",
        collection_name: Optional[str] = "trading_memory",
    ) -> None:
        """Initialize the Memory system.
        
        Args:
            persist_directory: Directory where the memory database will be stored
            collection_name: Name of the collection in ChromaDB
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"Initialized memory system with collection: {collection_name}")
        
    def add(
        self,
        memories: List[Dict[str, str]],
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add memories to the storage system.
        
        Args:
            memories: List of memory dictionaries to store
            agent_id: Optional identifier for the agent storing the memory
            metadata: Optional metadata to store with the memories
        """
        if not memories:
            return
            
        documents = []
        ids = []
        metadatas = []
        
        for i, memory in enumerate(memories):
            memory_str = json.dumps(memory)
            timestamp = datetime.now().isoformat()
            
            doc_metadata = {
                "timestamp": timestamp,
                "type": "memory",
            }
            
            if agent_id:
                doc_metadata["agent_id"] = agent_id
                
            if metadata:
                doc_metadata.update(metadata)
                
            documents.append(memory_str)
            ids.append(f"mem_{timestamp}_{i}")
            metadatas.append(doc_metadata)
            
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        logger.debug(f"Added {len(memories)} memories to collection")
        
    def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories.
        
        Args:
            query: Search query string
            agent_id: Optional agent ID to filter results
            limit: Maximum number of results to return
            metadata_filter: Optional metadata filters
            
        Returns:
            List of relevant memories with their similarity scores
        """
        where = {}
        if agent_id:
            where["agent_id"] = agent_id
            
        if metadata_filter:
            where.update(metadata_filter)
            
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where or None
        )
        
        memories = []
        for i, doc in enumerate(results["documents"][0]):
            memories.append({
                "memory": json.loads(doc),
                "score": results["distances"][0][i] if "distances" in results else 0.0,
                "metadata": results["metadatas"][0][i]
            })
            
        return memories
        
    def get_all(
        self,
        agent_id: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve all stored memories.
        
        Args:
            agent_id: Optional agent ID to filter results
            metadata_filter: Optional metadata filters
            
        Returns:
            List of all memories matching the filters
        """
        where = {}
        if agent_id:
            where["agent_id"] = agent_id
            
        if metadata_filter:
            where.update(metadata_filter)
            
        results = self.collection.get(where=where or None)
        
        memories = []
        for i, doc in enumerate(results["documents"]):
            memories.append({
                "memory": json.loads(doc),
                "metadata": results["metadatas"][i]
            })
            
        return memories
        
    def reset(self) -> None:
        """Reset the memory system by deleting all stored memories."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)
        logger.info("Reset memory system - all memories cleared") 
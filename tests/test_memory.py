"""Tests for the mem0ai memory system."""

from typing import TYPE_CHECKING, Dict, List
import pytest
import os
from mem0.memory.main import Memory
from mem0.client.main import MemoryClient
from autogen.agentchat.assistant_agent import ConversableAgent
import json
import logging
import time

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from pytest_mock.plugin import MockerFixture

@pytest.fixture(autouse=True)
def setup_environment():
    """Set up environment variables for testing."""
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["MEM0_API_KEY"] = "m0-6cNnWWejrjhX1ndgOHiPZbJGUDGyZcLQrO5FE4Of"
    yield
    # Clean up after tests
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "MEM0_API_KEY" in os.environ:
        del os.environ["MEM0_API_KEY"]

@pytest.fixture
def memory_config() -> Dict:
    """Create a test configuration for Memory."""
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4-turbo-preview",
                "temperature": 0.1,
                "max_tokens": 2000,
                "api_key": os.environ["OPENAI_API_KEY"]
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": os.environ["OPENAI_API_KEY"]
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test_collection",
                "path": "/tmp/qdrant",
                "embedding_model_dims": 1536
            }
        },
        "version": "v1.1"
    }

@pytest.fixture
def memory() -> Memory:
    """Create a Memory instance for testing."""
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4-turbo-preview",
                "temperature": 0.1,
                "max_tokens": 2000,
                "api_key": os.environ["OPENAI_API_KEY"]
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": "/tmp/qdrant",
                "embedding_model_dims": 1536
            }
        },
        "api_version": "v1.1"
    }
    return Memory.from_config(config)

@pytest.fixture
def memory_client() -> MemoryClient:
    """Create a MemoryClient instance for testing."""
    return MemoryClient(api_key=os.environ["MEM0_API_KEY"])

def test_memory_initialization(memory: Memory) -> None:
    """Test initialization of Memory."""
    assert memory is not None
    assert memory.vector_store is not None

def test_memory_from_config(memory: Memory) -> None:
    """Test creating Memory from config."""
    assert memory is not None
    assert memory.vector_store is not None

def test_memory_add_and_retrieve(memory: Memory) -> None:
    """Test adding and retrieving memories."""
    test_memories = [
        {
            "role": "user",
            "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts.",
            "metadata": {
                "type": "test",
                "food": "vegetarian"
            }
        },
        {
            "role": "assistant",
            "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions.",
            "metadata": {
                "type": "test",
                "food": "vegetarian"
            }
        }
    ]
    result = memory.add(messages=test_memories, user_id="test_user")
    assert isinstance(result, list)
    
    all_memories = memory.get_all(user_id="test_user")
    assert isinstance(all_memories, list)
    assert len(all_memories) >= 1

def test_memory_search(memory: Memory) -> None:
    """Test memory search functionality."""
    test_memories = [
        {
            "role": "user",
            "content": "I'm planning a trip to Japan next month.",
            "metadata": {
                "type": "test",
                "category": "travel"
            }
        }
    ]

    result = memory.add(messages=test_memories, user_id="test_user")
    assert result is not None
    assert isinstance(result, list)

    search_results = memory.search("trip to Japan", user_id="test_user")
    assert isinstance(search_results, list)

def test_memory_reset(memory: Memory) -> None:
    """Test resetting memory."""
    test_memories = [
        {
            "role": "user",
            "content": "Test memory for reset",
            "metadata": {
                "type": "test"
            }
        }
    ]

    result = memory.add(messages=test_memories, user_id="test_user")
    assert result is not None
    assert isinstance(result, list)

    memory.reset()
    all_memories = memory.get_all(user_id="test_user")
    assert isinstance(all_memories, list)
    assert len(all_memories) == 0

def test_memory_client_initialization(memory_client: MemoryClient) -> None:
    """Test initialization of MemoryClient."""
    assert memory_client.api_key == os.environ["MEM0_API_KEY"]

def test_memory_client_add_and_retrieve(memory_client: MemoryClient) -> None:
    """Test adding and retrieving memories using MemoryClient."""
    test_memories = [
        {
            "role": "user",
            "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts.",
            "metadata": {
                "type": "test",
                "food": "vegetarian"
            }
        }
    ]
    result = memory_client.add(messages=test_memories, user_id="test_user", agent_id="test_agent")
    assert isinstance(result, list)
    
    # Wait for memories to be indexed
    time.sleep(2)
    
    all_memories = memory_client.get_all(user_id="test_user", agent_id="test_agent")
    assert isinstance(all_memories, list)
    assert len(all_memories) >= 1
    assert all(isinstance(m, dict) and "id" in m and "memory" in m and "agent_id" in m and "user_id" in m for m in all_memories)

def test_memory_client_search(memory_client: MemoryClient) -> None:
    """Test memory search functionality using MemoryClient."""
    test_memories = [
        {
            "role": "user",
            "content": "I'm planning a trip to Japan next month.",
            "metadata": {
                "type": "test",
                "category": "travel"
            }
        }
    ]

    result = memory_client.add(messages=test_memories, user_id="test_user", agent_id="test_agent")
    assert result is not None
    assert isinstance(result, list)

    # Wait for memories to be indexed
    time.sleep(2)

    search_results = memory_client.search("trip to Japan", user_id="test_user", agent_id="test_agent")
    assert isinstance(search_results, list)
    assert len(search_results) >= 0
    if len(search_results) > 0:
        assert all(isinstance(m, dict) and "id" in m and "memory" in m and "agent_id" in m and "user_id" in m for m in search_results) 
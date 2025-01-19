"""Tests for the memory system module."""

from typing import TYPE_CHECKING, Dict, List
import pytest
from src.strat_optim.memory import Memory, Teachability
from autogen.agentchat.assistant_agent import ConversableAgent

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from pytest_mock.plugin import MockerFixture

@pytest.fixture
def memory_client() -> Memory:
    """Create a Memory instance for testing.
    
    Returns:
        Memory: A configured Memory instance
    """
    return Memory(persist_directory="./test_memory", collection_name="test_collection")

@pytest.fixture
def teachable_agent(mocker: "MockerFixture") -> ConversableAgent:
    """Create a mock ConversableAgent for testing.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        ConversableAgent: A mocked ConversableAgent
    """
    agent = mocker.Mock(spec=ConversableAgent)
    agent.system_message = "Initial system message"
    agent.llm_config = {"model": "test-model"}
    return agent

def test_memory_initialization(memory_client: Memory) -> None:
    """Test Memory class initialization.
    
    Args:
        memory_client: Memory fixture
    """
    assert memory_client.persist_directory == "./test_memory"
    assert memory_client.collection_name == "test_collection"
    assert memory_client.collection is not None

def test_memory_add_and_retrieve(memory_client: Memory) -> None:
    """Test adding and retrieving memories.
    
    Args:
        memory_client: Memory fixture
    """
    test_memories = [
        {"role": "system", "content": "Test memory 1"},
        {"role": "system", "content": "Test memory 2"}
    ]
    
    memory_client.add(test_memories, agent_id="test_agent")
    
    # Test retrieval
    all_memories = memory_client.get_all(agent_id="test_agent")
    assert len(all_memories) == 2
    assert all_memories[0]["memory"]["content"] == "Test memory 1"
    assert all_memories[1]["memory"]["content"] == "Test memory 2"

def test_memory_search(memory_client: Memory) -> None:
    """Test memory search functionality.
    
    Args:
        memory_client: Memory fixture
    """
    test_memories = [
        {"role": "system", "content": "Pattern: Market breakout\nSolution: Enter long position"},
        {"role": "system", "content": "Pattern: Market consolidation\nSolution: Wait for breakout"}
    ]
    
    memory_client.add(test_memories, agent_id="test_agent")
    
    # Test search
    results = memory_client.search("market breakout", agent_id="test_agent", limit=1)
    assert len(results) == 1
    assert "breakout" in results[0]["memory"]["content"].lower()

def test_memory_reset(memory_client: Memory) -> None:
    """Test memory reset functionality.
    
    Args:
        memory_client: Memory fixture
    """
    test_memories = [{"role": "system", "content": "Test memory"}]
    memory_client.add(test_memories)
    
    memory_client.reset()
    
    all_memories = memory_client.get_all()
    assert len(all_memories) == 0

def test_teachability_initialization(teachable_agent: ConversableAgent) -> None:
    """Test Teachability class initialization.
    
    Args:
        teachable_agent: ConversableAgent fixture
    """
    teachability = Teachability(
        verbosity=1,
        agent_id="test_agent",
        llm_config={"model": "test-model"}
    )
    
    assert teachability.verbosity == 1
    assert teachability.agent_id == "test_agent"
    assert teachability.llm_config == {"model": "test-model"}

def test_teachability_add_to_agent(teachable_agent: ConversableAgent) -> None:
    """Test adding Teachability to an agent.
    
    Args:
        teachable_agent: ConversableAgent fixture
    """
    teachability = Teachability(agent_id="test_agent")
    teachability.add_to_agent(teachable_agent)
    
    assert teachability.teachable_agent == teachable_agent
    teachable_agent.register_hook.assert_called_once()
    teachable_agent.update_system_message.assert_called_once()

def test_teachability_process_message(
    mocker: "MockerFixture",
    teachable_agent: ConversableAgent
) -> None:
    """Test message processing in Teachability.
    
    Args:
        mocker: pytest-mock fixture
        teachable_agent: ConversableAgent fixture
    """
    teachability = Teachability(agent_id="test_agent")
    teachability.add_to_agent(teachable_agent)
    
    # Mock analyzer responses
    mocker.patch.object(
        teachability,
        '_analyze',
        side_effect=["yes", "Trading advice", "Buy signal", "Market trend"]
    )
    
    # Test message processing
    result = teachability.process_last_received_message("Test trading message")
    
    assert isinstance(result, str)
    assert teachability._analyze.call_count == 4 
"""Teachability module for enhancing agent learning capabilities.

This module provides a capability that allows agents to learn and remember information
from conversations and apply it to future trading decisions.
"""

from typing import Dict, Optional, Union
from autogen.agentchat.assistant_agent import ConversableAgent
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability
from autogen.agentchat.contrib.text_analyzer_agent import TextAnalyzerAgent
from termcolor import colored
from .mem0 import Memory

class Teachability(AgentCapability):
    """Capability that enables agents to learn and remember information from conversations."""
    
    def __init__(
        self,
        verbosity: Optional[int] = 0,
        reset_db: Optional[bool] = False,
        recall_threshold: Optional[float] = 1.5,
        max_num_retrievals: Optional[int] = 10,
        llm_config: Optional[Union[Dict, bool]] = None,
        agent_id: Optional[str] = None,
        memory_client: Optional[Memory] = None,
    ) -> None:
        """Initialize the Teachability capability.
        
        Args:
            verbosity: Level of verbosity for logging (0-2)
            reset_db: Whether to reset the memory database on initialization
            recall_threshold: Threshold for memory recall similarity
            max_num_retrievals: Maximum number of memories to retrieve
            llm_config: Configuration for the language model
            agent_id: Identifier for the agent
            memory_client: Optional pre-configured memory client
        """
        self.verbosity = verbosity
        self.recall_threshold = recall_threshold
        self.max_num_retrievals = max_num_retrievals
        self.llm_config = llm_config
        self.analyzer = None
        self.teachable_agent = None
        self.agent_id = agent_id
        self.memory = memory_client if memory_client else Memory()

        if reset_db:
            self.memory.reset()

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

    def process_last_received_message(self, text: Union[Dict, str]) -> Union[Dict, str]:
        """Process the last received message and enhance it with relevant memories.
        
        Args:
            text: The message to process
            
        Returns:
            The processed message, potentially enhanced with relevant memories
        """
        expanded_text = text
        if self.memory.get_all(agent_id=self.agent_id):
            expanded_text = self._consider_memo_retrieval(text)
        self._consider_memo_storage(text)
        return expanded_text

    def _consider_memo_storage(self, comment: Union[Dict, str]) -> None:
        """Consider storing the comment as a memory.
        
        Args:
            comment: The comment to potentially store
        """
        # Check for trading tasks or problems
        response = self._analyze(
            comment,
            "Does the TEXT contain a trading task, strategy, or problem to solve? Answer with just yes or no.",
        )

        if "yes" in response.lower():
            advice = self._analyze(
                comment,
                "Extract any trading advice, patterns, or insights from the TEXT that could be useful for similar situations. If none present, respond with 'none'.",
            )

            if "none" not in advice.lower():
                task = self._analyze(
                    comment,
                    "Extract the core trading task or problem from the TEXT. Focus on the essential elements.",
                )

                general_task = self._analyze(
                    task,
                    "Generalize this trading scenario into a broader pattern or situation type that might recur.",
                )

                if self.verbosity >= 1:
                    print(colored("\nStoring Trading Pattern-Solution Pair", "light_yellow"))
                self.memory.add(
                    [{
                        "role": "system",
                        "content": f"Pattern: {general_task}\nSolution: {advice}"
                    }],
                    agent_id=self.agent_id
                )

        # Check for general trading information
        response = self._analyze(
            comment,
            "Does the TEXT contain valuable trading information worth remembering? Answer with just yes or no.",
        )

        if "yes" in response.lower():
            question = self._analyze(
                comment,
                "How would a trader ask for this information in the future? Frame it as a question.",
            )

            answer = self._analyze(
                comment,
                "Extract the key trading information that answers this question.",
            )

            if self.verbosity >= 1:
                print(colored("\nStoring Trading Q&A Pair", "light_yellow"))
            self.memory.add(
                [{
                    "role": "system",
                    "content": f"Question: {question}\nAnswer: {answer}"
                }],
                agent_id=self.agent_id
            )

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
            agent_id=self.agent_id,
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
        """Concatenate memories into a single text.
        
        Args:
            memo_list: List of memories to concatenate
            
        Returns:
            Concatenated memory text
        """
        memo_texts = ""
        if memo_list:
            info = "\n# Relevant Trading Memories\n"
            for memo in memo_list:
                info += f"- {memo}\n"
            if self.verbosity >= 1:
                print(colored(f"\nEnhancing with Trading Memories:\n{info}\n", "light_yellow"))
            memo_texts += "\n" + info
        return memo_texts

    def _analyze(
        self,
        text_to_analyze: Union[Dict, str],
        analysis_instructions: Union[Dict, str]
    ) -> str:
        """Analyze text using the text analyzer agent.
        
        Args:
            text_to_analyze: Text to analyze
            analysis_instructions: Instructions for analysis
            
        Returns:
            Analysis result
        """
        self.analyzer.reset()
        self.teachable_agent.send(
            recipient=self.analyzer,
            message=text_to_analyze,
            request_reply=False,
            silent=(self.verbosity < 2)
        )
        self.teachable_agent.send(
            recipient=self.analyzer,
            message=analysis_instructions,
            request_reply=True,
            silent=(self.verbosity < 2)
        )
        return self.teachable_agent.last_message(self.analyzer)["content"] 
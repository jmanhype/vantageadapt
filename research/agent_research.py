#!/usr/bin/env python3
"""Agent Research Tool for comprehensive topic analysis.

This module provides a tool for agents to conduct comprehensive research on any topic
using the VectorShift API. It supports multi-threaded execution, dependency management,
and markdown output generation.
"""
import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import logging

from research.config.settings import (
    VECTORSHIFT_API_KEY,
    DEFAULT_OUTPUT_DIR,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    MAX_WORKERS,
    API_BASE_URL,
    VECTORSHIFT_CHATBOT_ID
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchQuery:
    """Research query configuration.
    
    Attributes:
        topic: The main topic to research
        focus_areas: Optional list of specific areas to focus on
        output_dir: Directory to save research results
        api_key: VectorShift API key
        max_retries: Maximum number of API request retries
        timeout: API request timeout in seconds
    """
    topic: str
    focus_areas: Optional[List[str]] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    api_key: str = VECTORSHIFT_API_KEY
    max_retries: int = MAX_RETRIES
    timeout: int = REQUEST_TIMEOUT

@dataclass
class ResearchTask:
    """Represents a research task.
    
    Attributes:
        id: Unique identifier for the task
        query: The research query to execute
        dependencies: Set of task IDs this task depends on
        focus_area: Optional specific focus area for the task
        result: Task execution result
        completed: Whether the task has been completed
    """
    id: str
    query: str
    dependencies: Set[str]
    focus_area: Optional[str] = None
    result: Any = None
    completed: bool = False

class AgentResearchTool:
    """Tool for agents to conduct comprehensive research on any topic."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the research tool.
        
        Args:
            api_key: Optional VectorShift API key. If not provided, uses the one from config.
        """
        self.api_key = api_key or VECTORSHIFT_API_KEY
        self.headers = {"Api-Key": self.api_key}
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def research_topic(
        self, 
        topic: str, 
        focus_areas: Optional[List[str]] = None, 
        output_dir: str = DEFAULT_OUTPUT_DIR
    ) -> Dict[str, Any]:
        """Research a topic comprehensively with optional focus areas.
        
        Args:
            topic: The main topic to research
            focus_areas: Optional list of specific areas to focus on
            output_dir: Directory to save research results
            
        Returns:
            Dict containing research results and metadata
        """
        query = ResearchQuery(topic=topic, focus_areas=focus_areas, output_dir=output_dir)
        tasks = self._create_research_tasks(query)
        results = self._execute_tasks(tasks)
        return self._format_results(query, results)

    def _create_research_tasks(self, query: ResearchQuery) -> List[ResearchTask]:
        """Create research tasks for the topic.
        
        Args:
            query: The research query configuration
            
        Returns:
            List of research tasks to execute
        """
        tasks = [
            ResearchTask(
                id="overview",
                query=f"""Provide a comprehensive overview of {query.topic}:
1. What are its key characteristics?
2. What are its main components?
3. What are its capabilities?
4. What are its limitations?""",
                dependencies=set()
            ),
            ResearchTask(
                id="technical_details",
                query=f"""Explain the technical details and implementation of {query.topic}:
1. What is the architecture?
2. What are the key methods?
3. What are the components?
4. How do they interact?""",
                dependencies=set()
            )
        ]
        
        if query.focus_areas:
            for i, area in enumerate(query.focus_areas):
                tasks.append(ResearchTask(
                    id=f"focus_{i+1}",
                    query=f"""Deep dive into {area} for {query.topic}:
1. What are the specifics?
2. How does it work?
3. What are best practices?
4. What are common issues?""",
                    dependencies={"overview"},
                    focus_area=area
                ))
        
        tasks.append(ResearchTask(
            id="synthesis",
            query=f"""Synthesize all findings about {query.topic}:
1. What are the key insights?
2. What patterns emerged?
3. What are the implications?
4. What recommendations can be made?""",
            dependencies=set(task.id for task in tasks)
        ))
        
        return tasks

    def _execute_task(self, task: ResearchTask, context: Dict[str, Any]) -> Any:
        """Execute a single research task.
        
        Args:
            task: The research task to execute
            context: Context from previous task results
            
        Returns:
            Task execution results
        """
        start_time = datetime.now()
        retries = 0
        last_error = None

        while retries < MAX_RETRIES:
            try:
                # Prepare API request
                url = f"{API_BASE_URL}/chatbots/run"
                data = {
                    "input": task.query,
                    "chatbot_id": VECTORSHIFT_CHATBOT_ID,
                    "context": context.get(task.id, "")
                }

                # Make API request
                response = requests.post(
                    url, 
                    headers=self.headers, 
                    json=data,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()
                result = response.json()

                # Extract content from response
                content = result.get("output", result.get("response", ""))
                execution_time = (datetime.now() - start_time).total_seconds()

                return {
                    "content": content,
                    "time": execution_time,
                    "focus_area": task.focus_area
                }

            except Exception as e:
                last_error = str(e)
                retries += 1
                if retries == MAX_RETRIES:
                    break

        return {"error": f"Task failed after {retries} retries. Last error: {last_error}"}

    def _execute_tasks(self, tasks: List[ResearchTask]) -> Dict[str, Any]:
        """Execute research tasks in dependency order.
        
        Args:
            tasks: List of research tasks to execute
            
        Returns:
            Dict containing execution results and metadata
        """
        try:
            # Build task graph
            graph = nx.DiGraph()
            task_map = {task.id: task for task in tasks}
            for task in tasks:
                graph.add_node(task.id)
                for dep in task.dependencies:
                    graph.add_edge(dep, task.id)

            # Check for cycles
            if not nx.is_directed_acyclic_graph(graph):
                raise ValueError("Task dependencies contain cycles")

            # Execute tasks in order
            context = {}
            results = {}
            start_time = datetime.now()

            for task_id in nx.topological_sort(graph):
                task = task_map[task_id]
                result = self._execute_task(task, context)
                task.completed = True
                task.result = result
                context[task.id] = result.get("content", "")
                results[task_id] = result

            total_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": True,
                "results": results,
                "total_time": total_time
            }

        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _format_results(self, query: ResearchQuery, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format research results into a structured output.
        
        Args:
            query: The research query configuration
            execution_results: Results from task execution
            
        Returns:
            Dict containing formatted results and metadata
        """
        if not execution_results.get("success"):
            return execution_results

        # Format findings
        findings = {}
        for task_id, result in execution_results["results"].items():
            findings[task_id] = {
                "content": result.get("content", ""),
                "time": result.get("time", 0),
                "focus_area": result.get("focus_area")
            }

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(query.output_dir, query.topic.lower().replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(
            output_dir,
            f"{query.topic.lower().replace(' ', '_')}_{timestamp}.md"
        )

        # Generate markdown content
        markdown_content = self._generate_markdown(query, findings)

        # Save markdown file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        return {
            "success": True,
            "findings": findings,
            "total_time": execution_results["total_time"],
            "output_file": output_file
        }

    def _generate_markdown(self, query: ResearchQuery, findings: Dict[str, Any]) -> str:
        """Generate markdown content from research findings.
        
        Args:
            query: The research query configuration
            findings: Research findings to format
            
        Returns:
            Formatted markdown content
        """
        markdown_content = f"""# {query.topic}
Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
        for task_id, finding in findings.items():
            markdown_content += f"## {task_id.replace('_', ' ').title()}\n\n"
            
            # Add Query section
            markdown_content += "### Query\n"
            if task_id == "overview":
                markdown_content += f"Provide a comprehensive overview of {query.topic}:\n"
                markdown_content += "1. What are its key characteristics?\n"
                markdown_content += "2. What are its main components?\n"
                markdown_content += "3. What are its capabilities?\n"
                markdown_content += "4. What are its limitations?\n\n"
            elif task_id == "technical_details":
                markdown_content += f"Explain the technical details and implementation of {query.topic}:\n"
                markdown_content += "1. What is the architecture?\n"
                markdown_content += "2. What are the key methods?\n"
                markdown_content += "3. What are the components?\n"
                markdown_content += "4. How do they interact?\n\n"
            elif task_id.startswith("focus_"):
                focus_area = finding.get("focus_area", "")
                markdown_content += f"Deep dive into {focus_area} for {query.topic}:\n"
                markdown_content += "1. What are the specifics?\n"
                markdown_content += "2. How does it work?\n"
                markdown_content += "3. What are best practices?\n"
                markdown_content += "4. What are common issues?\n\n"
            elif task_id == "synthesis":
                markdown_content += f"Synthesize all findings about {query.topic}:\n"
                markdown_content += "1. What are the key insights?\n"
                markdown_content += "2. What patterns emerged?\n"
                markdown_content += "3. What are the implications?\n"
                markdown_content += "4. What recommendations can be made?\n\n"
            
            # Add Response section
            markdown_content += "### Response\n"
            markdown_content += finding["content"]
            markdown_content += f"\nTime: {finding['time']:.2f} seconds\n\n"
            markdown_content += "---\n"

        return markdown_content 
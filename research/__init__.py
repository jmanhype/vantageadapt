"""Research package for comprehensive topic analysis.

This package provides tools and utilities for conducting research using the
VectorShift API, with support for multi-threaded execution and dependency
management.
"""

from research.agent_research import AgentResearchTool, ResearchQuery, ResearchTask

__all__ = ['AgentResearchTool', 'ResearchQuery', 'ResearchTask']
__version__ = '0.1.0' 
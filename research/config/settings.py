"""Configuration settings for the research tool.

This module contains configuration settings and environment variables for the research tool,
including settings for OpenAI, Mem0, DSPy, and VectorShift APIs.

Note: API keys should be stored in a .env file and not committed to version control.
Example .env file:
    OPENAI_API_KEY=your_openai_key
    MEM0_API_KEY=your_mem0_key
    VECTORSHIFT_API_KEY=your_vectorshift_key
"""
from typing import Dict, Any, Optional
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),  # Required: set in .env file
    "model": "gpt-4-turbo-preview"
}

# Mem0 Configuration
MEM0_CONFIG: Dict[str, Any] = {
    "api_key": os.getenv("MEM0_API_KEY", "m0-6xEdQezZE9VWDC7K0PUE17YRCn7jVycc2VECDtAt"),
    "user_id": str(uuid.uuid4()),  # Generate unique user ID
    "agent_id": "socratic_agent_v1",
    "api_version": "v1.1",  # Updated API version
    "rate_limit": {
        "max_retries": 3,
        "retry_delay": 60,  # Delay in seconds between retries
        "rate_limit_delay": 3600  # 1 hour delay when rate limit is hit
    },
    "index_delay": 5  # Wait 5 seconds for indexing after add operations
}

# DSPy Configuration
DSPY_CONFIG: Dict[str, Any] = {
    "lm": OPENAI_CONFIG["model"],
    "temperature": 0.7
}

# VectorShift Configuration
VECTORSHIFT_API_KEY: str = os.getenv(
    "VECTORSHIFT_API_KEY", 
    "sk_Ppsrv888tbSWvYn2B5afQFdakXfULIUd9n5HQtXAbjzJsFVh"
)
VECTORSHIFT_CHATBOT_ID: str = "6730406f16af217977dc1d4d"
API_BASE_URL: str = "https://api.vectorshift.ai/api"

# Research Configuration
DEFAULT_OUTPUT_DIR: str = "research_results"
MAX_RETRIES: int = 3
REQUEST_TIMEOUT: int = 30
MAX_WORKERS: int = 4  # Restored to original value

# Validate required API keys
def validate_api_keys() -> None:
    """Validate that all required API keys are set."""
    missing_keys = []
    
    if not OPENAI_CONFIG["api_key"]:
        missing_keys.append("OPENAI_API_KEY")
    if not MEM0_CONFIG["api_key"]:
        missing_keys.append("MEM0_API_KEY")
    if not VECTORSHIFT_API_KEY:
        missing_keys.append("VECTORSHIFT_API_KEY")
        
    if missing_keys:
        raise ValueError(
            f"Missing required API keys: {', '.join(missing_keys)}. "
            "Please set them in your .env file."
        )
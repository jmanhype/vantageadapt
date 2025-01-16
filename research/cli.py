#!/usr/bin/env python3
"""Command-line interface for the Agent Research Tool.

This module provides a command-line interface for conducting research using the
AgentResearchTool.
"""
import sys
import argparse
import logging
from typing import Optional, List

from research.agent_research import AgentResearchTool
from research.config.settings import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Agent Research Tool for comprehensive topic analysis"
    )
    
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic to research"
    )
    parser.add_argument(
        "--focus-areas",
        help="Focus areas for analysis (comma-separated)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--api-key",
        help="Vector Shift API key (default: from config)"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for the agent research tool CLI."""
    args = parse_args()
    
    # Initialize tool
    tool = AgentResearchTool(api_key=args.api_key)
    
    try:
        # Parse focus areas
        focus_areas: Optional[List[str]] = (
            args.focus_areas.split(",") if args.focus_areas else None
        )
        
        # Run research
        results = tool.research_topic(args.topic, focus_areas, args.output_dir)
        
        # Print results
        print(f"\n{args.topic}")
        print("-" * 50)
        
        if results["success"]:
            for task_id, finding in results["findings"].items():
                print(f"\n{task_id.upper()}:")
                if task_id.startswith("focus_"):
                    print(f"Focus Area: {finding['focus_area']}")
                print(finding["content"])
                print(f"Time: {finding['time']:.2f} seconds")
            print(f"\nTotal Time: {results['total_time']:.2f} seconds")
            print(f"\nResults saved to: {results['output_file']}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nResearch interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Research failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
"""Simple script to run the FastAPI server."""

import uvicorn
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    """Run the FastAPI server."""
    try:
        logger.info("Starting server...")
        uvicorn.run(
            "research.api.control_panel:app",
            host="0.0.0.0",
            port=8002,
            reload=True,
            log_level="debug"
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
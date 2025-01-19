import os
from mem0.memory.main import Memory
from mem0.client.main import MemoryClient

def main():
    # Set up OpenAI API key
    os.environ["OPENAI_API_KEY"] = "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"
    
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
    
    memory = Memory.from_config(config)
    print(f"Memory initialized: {memory}")
    print(f"Memory module: {Memory.__module__}")
    print(f"MemoryClient module: {MemoryClient.__module__}")

if __name__ == "__main__":
    main() 
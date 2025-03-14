#!/usr/bin/env python3

"""
Script to test the MiPro optimization with a small example.
"""

import os
import json
import dspy
from loguru import logger
from dotenv import load_dotenv

from src.utils.mipro_optimizer import MiProWrapper

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger.add("logs/test_mipro.log", rotation="1 day")

def main():
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        return
    
    # Initialize DSPy
    lm = dspy.LM("gpt-4-turbo-preview", api_key=api_key)
    dspy.configure(lm=lm)
    
    # Create a simple module for testing
    class SimpleClassifier(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prompt = "Classify the sentiment of the following text as positive or negative."
            
        def forward(self, text):
            """Classify text sentiment."""
            # Use dspy.Predict which will have access to the configured LM
            prediction = dspy.Predict("text -> sentiment")(text=text)
            sentiment = prediction.sentiment.lower()
            return {
                "sentiment": sentiment,
                "confidence": 0.9 if "positive" in sentiment or "negative" in sentiment else 0.5
            }
    
    # Create a MiPro optimizer
    mipro = MiProWrapper(
        use_v2=True,
        max_bootstrapped_demos=3,
        num_candidate_programs=3,
        temperature=0.7
    )
    
    # Create some simple examples
    examples = [
        {"text": "I love this product", "outputs": {"sentiment": "positive", "confidence": 0.9}},
        {"text": "This is terrible", "outputs": {"sentiment": "negative", "confidence": 0.9}},
        {"text": "Not bad at all", "outputs": {"sentiment": "positive", "confidence": 0.7}},
        {"text": "Could be better", "outputs": {"sentiment": "negative", "confidence": 0.6}}
    ]
    
    # Create a metric function
    def sentiment_metric(gold, pred, trace=None):
        """Metric function with correct signature for DSPy 2.6.12.
        
        Args:
            gold: Gold standard example
            pred: Prediction from the model
            trace: Optional trace information (added in newer DSPy)
            
        Returns:
            Score between 0.0 and 1.0
        """
        return 1.0 if gold.get('sentiment') == pred.get('sentiment', '') else 0.0
    
    # Initialize the module
    classifier = SimpleClassifier()
    
    # Test the preparation of examples
    try:
        logger.info("Testing example preparation")
        prepared_examples = mipro.prepare_examples(examples)
        logger.info(f"Successfully prepared {len(prepared_examples)} examples")
        
        # Test each example has inputs properly set
        for i, ex in enumerate(prepared_examples):
            inputs = ex.inputs()
            logger.info(f"Example {i+1} inputs: {list(inputs.keys()) if inputs else 'None'}")
            
        # Test optimization
        logger.info("Testing optimization")
        optimized_module = mipro.optimize(
            module=classifier,
            examples=examples,
            prompt_name="sentiment",
            metric_fn=sentiment_metric
        )
        
        if optimized_module:
            logger.info("Optimization successful!")
            logger.info(f"Original prompt: {classifier.prompt}")
            logger.info(f"Optimized prompt: {optimized_module.prompt}")
        else:
            logger.error("Optimization failed")
    
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    main()
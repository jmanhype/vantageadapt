"""Direct optimization of market analysis prompts."""

import os
import json
import dspy
import time
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger.add("logs/optimize_market_analysis.log", rotation="1 day")

# Load prompt and examples
prompt_path = Path("prompts/market_analysis.txt")
examples_path = Path("prompts/examples.json")

# Make sure the optimized prompts directory exists
optimized_dir = Path("prompts/optimized")
optimized_dir.mkdir(exist_ok=True)

# Load the original prompt
with open(prompt_path, "r") as f:
    original_prompt = f.read()
    
print(f"Original prompt loaded ({len(original_prompt)} chars)")

# Load examples
with open(examples_path, "r") as f:
    all_examples = json.load(f)
    market_examples = all_examples.get("market_analysis", [])
    
print(f"Loaded {len(market_examples)} market analysis examples")

# Setup DSPy - Use API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: No OpenAI API key found in environment.")
    exit(1)

# Configure DSPy with the API key - using correct DSPy API
lm = dspy.LM(model="gpt-4-turbo-preview", api_key=api_key)
dspy.configure(lm=lm)

# Define a simplified module for market analysis
class MarketAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = original_prompt
        
    def forward(self, market_data, timeframe, prompt):
        """Analyze market data and determine regime."""
        # Create filled template string
        filled_prompt = f"""
You are a market analyst tasked with analyzing financial market data.

Timeframe: {timeframe}
Market Data: {market_data}

Analyze the market conditions and provide your analysis in a clean, valid JSON format with the following structure:
{{
  "regime": "TRENDING_BULLISH | TRENDING_BEARISH | RANGING_HIGH_VOL | RANGING_LOW_VOL | UNKNOWN",
  "confidence": 0.0 to 1.0,
  "risk_level": "low | moderate | high",
  "analysis": "Detailed analysis of current market conditions"
}}
"""
        
        # Call the language model and get response
        response = self.lm(filled_prompt)
        
        # Try to extract the JSON part from the response
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response
                
            # Parse the JSON
            result = json.loads(json_str)
            
            # Return the result dictionary
            return {
                "regime": result.get("regime", "UNKNOWN"),
                "confidence": result.get("confidence", 0.0),
                "risk_level": result.get("risk_level", "moderate"),
                "analysis": result.get("analysis", "No analysis provided")
            }
        except Exception as e:
            print(f"Error processing response: {e}")
            # Return default values on error
            return {
                "regime": "UNKNOWN",
                "confidence": 0.0,
                "risk_level": "moderate",
                "analysis": "Error processing market analysis"
            }

# Create DSPy examples - Using with_inputs/with_outputs per DSPy API requirements
dspy_examples = []
success_count = 0
error_count = 0

for ex in market_examples:
    try:
        outputs = ex.get('outputs', {})
        
        # Ensure proper types for numerical values
        try:
            confidence = float(outputs.get('confidence', 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
            print(f"Warning: Invalid confidence value, using default 0.7")
        
        # Create empty example first
        example = dspy.Example()
        
        # Set input attributes directly
        example.market_data = {"summary": "Market data summary..."}
        example.timeframe = ex.get('timeframe', '1h')
        example.prompt = ex.get('prompt', '')
        
        # Mark which ones are inputs
        example = example.with_inputs('market_data', 'timeframe', 'prompt')
        
        # Set output attributes directly
        example.regime = outputs.get('regime', 'RANGING_LOW_VOL')
        example.confidence = confidence
        example.risk_level = outputs.get('risk_level', 'moderate')
        example.analysis = outputs.get('analysis', 'Market analysis details...')
        
        # Verify the example has at least some input fields set
        input_fields = example.inputs()
        if len(input_fields) == 0:
            print(f"Warning: Example has no input fields, skipping")
            error_count += 1
            continue
            
        dspy_examples.append(example)
        success_count += 1
        
    except Exception as e:
        print(f"Error creating example: {e}")
        error_count += 1

print(f"Created {len(dspy_examples)} DSPy examples ({success_count} successful, {error_count} errors)")

# Split examples into train/validation sets
if len(dspy_examples) <= 3:
    trainset = dspy_examples[:1]
    valset = dspy_examples[1:]
else:
    split_idx = max(1, int(len(dspy_examples) * 0.7))
    trainset = dspy_examples[:split_idx]
    valset = dspy_examples[split_idx:]

# Define a metric function for evaluation that works with dictionaries
def market_analysis_metric(gold_dict, pred_dict):
    """Metric function for market analysis optimization with dictionaries."""
    # Both gold and pred should already be dictionaries with our fields
    
    # Check regime match
    regime_match = 1 if gold_dict.get('regime') == pred_dict.get('regime', '') else 0
    
    # Check risk level match
    risk_match = 1 if gold_dict.get('risk_level') == pred_dict.get('risk_level', '') else 0
    
    # Check if analysis contains key indicators
    analysis = pred_dict.get('analysis', '')
    mentions_sma = 1 if 'SMA' in analysis or 'moving average' in analysis.lower() else 0
    mentions_volatility = 1 if 'volatil' in analysis.lower() else 0
    mentions_support = 1 if 'support' in analysis.lower() or 'resistance' in analysis.lower() else 0
    
    # Calculate score (weights add up to 1.0)
    score = (regime_match * 0.5) + (risk_match * 0.3) + (mentions_sma * 0.1) + (mentions_volatility * 0.05) + (mentions_support * 0.05)
    
    # Print detailed information about the comparison
    print(f"Metric: Regime match={regime_match}, Risk match={risk_match}, Score={score:.2f}")
    
    return score

# Set up the optimization using a simpler optimizer
print("Starting optimization...")
start_time = time.time()

# Create basic optimizer using a simpler approach
print("Using direct prompt enhancement with examples...")

# Create a basic module with examples
analyzer = MarketAnalyzer()

# Create an optimized prompt directly
optimized_prompt = analyzer.prompt

# Add few-shot examples directly to the prompt
optimized_prompt += "\n\n# Example Market Analyses\n\n"

for i, ex in enumerate(market_examples, 1):
    outputs = ex.get('outputs', {})
    # Add example to the prompt
    optimized_prompt += f"""
Example {i}:
Timeframe: {ex.get('timeframe', '1h')}
Market Context: [Market data summary]

Analysis:
```json
{{
  "regime": "{outputs.get('regime', 'UNKNOWN')}",
  "confidence": {outputs.get('confidence', 0.0)},
  "risk_level": "{outputs.get('risk_level', 'moderate')}",
  "analysis": "{outputs.get('analysis', 'Market analysis details...')}"
}}
```

"""

# Create module with optimized prompt
optimizer_module = analyzer
optimizer_module.prompt = optimized_prompt

# Simulate optimization
optimized_module = optimizer_module

# Save the optimized prompt
try:
    # Save the optimized prompt to file
    with open(optimized_dir / "market_analysis.txt", "w") as f:
        f.write(optimized_prompt)
    print(f"Saved optimized prompt ({len(optimized_prompt)} chars)")
    
    elapsed_time = time.time() - start_time
    print(f"Optimization completed in {elapsed_time:.2f} seconds")
    
except Exception as e:
    print(f"Error saving optimized prompt: {str(e)}")

# Verify and display the optimized prompt
try:
    optimized_path = optimized_dir / "market_analysis.txt"
    if optimized_path.exists():
        with open(optimized_path, "r") as f:
            opt_prompt = f.read()
        print(f"\nOptimized prompt (first 300 chars):\n{opt_prompt[:300]}...\n")
    else:
        print("No optimized prompt file found")
except Exception as e:
    print(f"Error reading optimized prompt: {str(e)}")
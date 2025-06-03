# Ensemble Optimization in DSPy

## Overview

The Ensemble optimizer in DSPy creates powerful aggregated models by combining multiple programs into a single unified program. Unlike standard optimization techniques that modify individual prompts, Ensemble optimization leverages the diversity of multiple model outputs to produce more robust and accurate predictions.

## How Ensemble Optimization Works

Ensemble optimization functions by taking a collection of DSPy programs (which could be optimized by different techniques like MiPro or BootstrapFewShot) and combining them through a reduction function (e.g., majority voting). When executed, the ensemble either:

1. Uses all provided programs to generate outputs
2. Randomly samples a subset of programs (if configured with a `size` parameter)

The outputs from these multiple models are then reduced to a single output using the specified reduction function.

### Core Components

- **Programs Collection**: Multiple DSPy modules/programs to be ensembled
- **Reduction Function**: A method to combine multiple outputs (typically majority voting)
- **Sampling Strategy**: Optional configuration to use only a subset of programs per execution

## Implementation Example

```python
import dspy
from dspy.teleprompt import Ensemble

# 1. Create your base DSPy module
class Classifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("text -> sentiment")

    def forward(self, text):
        return self.predict(text=text)

# 2. Create multiple instances with different optimizations
# (This could be different models optimized via MiPro, etc.)
base_classifiers = [Classifier() for _ in range(5)]

# 3. Create ensemble optimizer with majority voting
ensemble_teleprompter = Ensemble(
    reduce_fn=dspy.majority,  # Use majority voting to combine results
    size=3                   # Randomly select 3 models per execution
)

# 4. Compile the ensemble into a single program
ensemble_classifier = ensemble_teleprompter.compile(base_classifiers)

# 5. Use the ensembled program like any other DSPy module
result = ensemble_classifier("I absolutely loved this movie!")
print(result.sentiment)  # Output will be the majority vote
```

## Majority Voting in DSPy

The most common reduction function for ensembles is `dspy.majority`, which implements a basic voting mechanism:

```python
from dspy.predict import aggregation

# Create multiple agents
agents = [Agent() for _ in range(5)]

# Get predictions from each agent
completions = [agent(question=question) for agent in agents]

# Combine using majority voting
consensus = aggregation.majority(completions)
```

The majority function works by:

1. Extracting completions from each prediction
2. Determining which output field to use for voting
3. Normalizing the values to handle minor text variations
4. Counting occurrences of each value
5. Returning the most frequent value as the final prediction

## Advantages Over Single Models

Ensemble optimization offers several benefits compared to individual model optimization:

1. **Increased Robustness**: Less susceptible to individual model errors or biases
2. **Improved Accuracy**: Often achieves better results than any single component model
3. **Uncertainty Handling**: Better captures the range of possible answers
4. **Error Correction**: Can correct for occasional errors in individual models

## When to Use Ensemble Optimization

Ensemble optimization is particularly valuable when:

- You need high reliability for critical applications
- Individual models exhibit different strengths and weaknesses
- You want to balance between different optimization techniques
- You need improved confidence measures for predictions
- Your task involves complex reasoning where diverse perspectives help

## Implementation in Trading Systems

For trading applications, ensemble optimization can be used to:

1. **Combine Market Analysis**: Aggregate multiple market analyses with different focuses
2. **Strategy Consensus**: Implement a voting system across multiple strategy generators
3. **Risk Assessment**: Blend risk evaluations from different perspectives
4. **Volatility Prediction**: Improve accuracy by combining different volatility models

### Trading Example

```python
class MarketEnsemble(dspy.Module):
    def __init__(self, num_analysts=5):
        super().__init__()
        # Create multiple market analysts
        self.analysts = [MarketAnalyzer(prompt_manager) for _ in range(num_analysts)]
        
    def forward(self, market_data, timeframe):
        # Get predictions from each analyst
        analyses = [analyst(market_data=market_data, timeframe=timeframe) 
                   for analyst in self.analysts]
        
        # Combine the regime predictions using majority voting
        regimes = [a['market_context']['regime'] for a in analyses]
        confidence_values = [a['market_context']['confidence'] for a in analyses]
        
        # Get the majority regime
        from collections import Counter
        regime_counter = Counter(regimes)
        majority_regime = regime_counter.most_common(1)[0][0]
        
        # Calculate average confidence for the majority regime
        majority_indices = [i for i, r in enumerate(regimes) if r == majority_regime]
        avg_confidence = sum(confidence_values[i] for i in majority_indices) / len(majority_indices)
        
        return {
            "market_context": {
                "regime": majority_regime,
                "confidence": avg_confidence,
                "risk_level": "moderate"  # Could also use majority voting here
            },
            "analysis_text": "Ensemble analysis based on majority vote."
        }
```

## Advanced Configuration

### Custom Reduction Functions

While majority voting is the default, you can implement custom reduction functions:

```python
def weighted_ensemble(completions, weights=None):
    """Custom reduction that weights different models."""
    if weights is None:
        weights = [1.0] * len(completions)
    
    # Implementation details...
    
    return final_prediction
```

### Configuring Ensemble Size

You can control how many models are used in each prediction:

```python
# Use all models
full_ensemble = Ensemble(reduce_fn=dspy.majority)

# Use 3 randomly selected models per prediction
partial_ensemble = Ensemble(reduce_fn=dspy.majority, size=3)
```

## Serialization and Persistence

When saving ensemble models, each component program needs to be saved separately:

```python
# Save ensemble model
for i, program in enumerate(ensemble_classifier.programs):
    program.save(f"ensemble_component_{i}")

# Load ensemble model
loaded_programs = []
for i in range(5):  # Assuming 5 components
    loaded_programs.append(dspy.Module.load(f"ensemble_component_{i}"))
    
# Recreate ensemble
loaded_ensemble = Ensemble(reduce_fn=dspy.majority).compile(loaded_programs)
```

## Limitations and Considerations

1. **Increased Computation**: Requires running multiple models for each prediction
2. **Consistent Interfaces**: All component models must have matching signatures
3. **Diversity Requirement**: Works best when component models exhibit different behaviors
4. **Serialization Complexity**: Requires special handling for saving/loading

## References

- [DSPy Ensemble Documentation](https://dspy.ai/deep-dive/optimizers/Ensemble)
- [Majority Voting in DSPy](https://medium.com/@JacekWo/majority-voting-07de046af3dc)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
# MiPro Optimization System - Setup & Usage

This document provides instructions for setting up and using the Mixed Prompt Optimization (MiPro) system within the trading DSPy project.

## Overview

The MiPro system optimizes prompts for language models to improve the quality and consistency of responses. Three key components can be optimized:

1. **Market Analysis** - Optimizes market condition analysis prompts
2. **Strategy Generator** - Optimizes trading strategy creation prompts
3. **Trading Rules** - Optimizes rules generation for implementing strategies

## Setup

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- OpenAI API key
- Mem0 API key (optional, for memory management)

### Environment Setup

1. **Clone the repository and install dependencies**:
   ```bash
   git clone <repository-url>
   cd trading_dspy
   poetry install
   ```

2. **Configure environment variables**:
   Create or modify the `.env` file in the project root with:
   ```
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Memory System (Optional)
   MEM0_API_KEY=your_mem0_api_key_here
   MEM0_ORG_ID=your_org_id
   MEM0_PROJECT_ID=your_project_id
   MEM0_USER_ID=your_user_id
   
   # Storage
   CHROMADB_PERSIST_DIRECTORY=./trading_memories
   ```

## Usage

### Example Collection

Before optimization, you need to collect examples that demonstrate good model outputs for given inputs.

1. **Generate initial examples**:
   ```bash
   python generate_examples.py
   ```
   
   This creates synthetic examples with market data and corresponding analysis, strategy, and trading rules.

2. **Run the system to collect more examples**:
   ```bash
   python main.py
   ```
   
   As the system runs, it automatically collects examples from successful interactions.

3. **Check example status**:
   ```bash
   python -c "
   from src.utils.prompt_manager import PromptManager
   from src.modules.prompt_optimizer import PromptOptimizer
   
   # Initialize managers
   prompt_manager = PromptManager('prompts')
   prompt_optimizer = PromptOptimizer(prompt_manager)
   
   # Check optimization status
   status = prompt_optimizer.check_optimization_status()
   for name, info in status.items():
       print(f'{name}: {info[\"example_count\"]} examples, optimized={info[\"optimized\"]}')
   "
   ```

### Prompt Optimization

#### Optimizing Market Analysis Prompts

```bash
python optimize_market_analysis.py
```

This script:
1. Loads collected market analysis examples
2. Prepares them for DSPy optimization
3. Enhances the prompt through few-shot learning
4. Saves the optimized prompt to `prompts/optimized/market_analysis.txt`

#### Testing Optimization

```bash
python test_mipro.py
```

This runs a simple sentiment analysis example to verify the MiPro optimization system is working correctly.

### Using Optimized Prompts

Once optimized, the system automatically uses the optimized prompts. You can run:

```bash
python main.py
```

The `PromptManager` will load optimized prompts if available, improving the quality of generated analyses, strategies, and trading rules.

## Advanced Usage

### Manual Example Creation

You can manually add examples to improve optimization results:

```python
from src.utils.prompt_manager import PromptManager

# Initialize manager
prompt_manager = PromptManager('prompts')

# Add example
example = {
    'market_data': {...},  # Simplified market data
    'timeframe': '1h',
    'outputs': {
        'regime': 'TRENDING_BULLISH',
        'confidence': 0.8,
        'risk_level': 'moderate',
        'analysis': 'Detailed analysis text...'
    }
}

prompt_manager.add_example('market_analysis', example)
```

### Customizing Optimization Parameters

You can adjust optimization parameters in `src/utils/mipro_optimizer.py`:

```python
# Create a MiPro optimizer with custom parameters
mipro = MiProWrapper(
    use_v2=True,                 # Use MIPROv2 (recommended)
    max_bootstrapped_demos=5,    # Maximum number of bootstrapped demonstrations
    num_candidate_programs=8,    # Number of candidate programs to generate
    temperature=0.7              # Temperature for generation
)
```

## Troubleshooting

### Common Issues

1. **DSPy Example Creation Errors**
   - Ensure example formats match expected schema
   - Check that input fields match module signatures

2. **JSON Serialization Errors**
   - Complex objects (DataFrames, timestamps) should be simplified
   - Use the `_make_serializable()` method to convert non-serializable types

3. **Optimization Doesn't Improve Results**
   - Ensure enough diverse examples (at least 4-5)
   - Check that examples demonstrate high-quality outputs

### Debugging

For detailed logging during optimization:

```bash
# Set logging level to DEBUG in your code
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use environment variable
LOGLEVEL=DEBUG python optimize_market_analysis.py
```

## Future Improvements

- Add optimization for other modules
- Implement automated optimization scheduling
- Add support for different LLM providers
- Create web interface for example management
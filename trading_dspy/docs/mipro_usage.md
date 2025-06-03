# MiPro Optimization System - Setup & Usage

This document provides instructions for setting up and using the Mixed Prompt Optimization (MiPro) system within the trading DSPy project.

## Overview

The MiPro system optimizes prompts for language models to improve the quality and consistency of responses. Three key components can be optimized:

1. **Market Analysis** - Optimizes market condition analysis prompts
2. **Strategy Generator** - Optimizes trading strategy creation prompts
3. **Trading Rules** - Optimizes rules generation for implementing strategies

## How MiPro Works

MiPRO (Multiprompt Instruction PRoposal Optimizer) is a sophisticated optimization algorithm in DSPy designed to tune prompts for better language model performance. MiPROv2, the latest version, can optimize both instructions and few-shot examples jointly, functioning in either:

- **Few-shot mode**: Optimizes both instructions and example demonstrations
- **Zero-shot mode**: Optimizes only instructions without examples

MiPRO works in three key stages:

1. **Bootstrapping Stage**:
   - Takes your program and runs it multiple times across different inputs
   - Collects traces of input/output behavior for each module
   - Filters traces to keep only those that appear in trajectories with high scores according to your metric

2. **Grounded Proposal Stage**:
   - Analyzes your DSPy program's code, your data, and traces from program execution
   - Drafts multiple potential instructions for every prompt in your program
   - Uses context from the program and data to ground these instructions in the specific task

3. **Discrete Search Stage**:
   - Samples mini-batches from your training set
   - Proposes combinations of instructions and traces for constructing each prompt
   - Evaluates candidate programs on mini-batches
   - Updates a surrogate model to improve proposals over time
   - Uses Bayesian Optimization to find optimal combinations

## Setup

### Prerequisites

- Python 3.8+ (3.10 recommended)
- Poetry (for dependency management)
- GitHub account and personal access token (for VectorBTPro access)
- OpenAI API key
- Mem0 API key (optional, for memory management)
- Git LFS (for handling large files)

### Detailed Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd trading_dspy
   ```

2. **Configure GitHub token for VectorBTPro access**:
   
   VectorBTPro is a key dependency that requires authenticated access:
   
   ```bash
   # Configure your GitHub token for package access
   poetry config http-basic.vectorbt __token__ your_github_token_here
   
   # Alternative method using environment variable
   export GITHUB_TOKEN=your_github_token_here
   ```
   
   You may need to request access to the VectorBTPro repository from the project maintainers.

3. **Install dependencies using Poetry**:
   ```bash
   # Install all dependencies including development packages
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

4. **Install system-level dependencies**:
   
   Some packages may require additional system libraries:
   
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential libssl-dev libffi-dev python3-dev
   
   # macOS
   brew install openssl
   ```

5. **Install Git LFS for large files**:
   ```bash
   # Install Git LFS
   git lfs install
   
   # Pull LFS objects
   git lfs pull
   ```

6. **Configure environment variables**:
   Create or modify the `.env` file in the project root with:
   ```
   # OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # VectorBT Configuration
   GITHUB_TOKEN=your_github_token_here
   
   # Memory System (Optional but recommended)
   MEM0_API_KEY=your_mem0_api_key_here
   MEM0_ORG_ID=your_org_id
   MEM0_PROJECT_ID=your_project_id
   MEM0_USER_ID=your_user_id
   
   # Storage
   CHROMADB_PERSIST_DIRECTORY=./trading_memories
   ```

7. **Verify installation**:
   ```bash
   # Run a simple test to verify everything is set up correctly
   python -c "import dspy; import vectorbtpro as vbt; print('DSPy version:', dspy.__version__); print('VectorBTPro version:', vbt.__version__)"
   ```

### Dependency Details

The project relies on several key packages:

1. **DSPy** - Framework for programming language models
   - Version: 2.6.12+
   - Used for: Prompt optimization and LLM interactions

2. **VectorBTPro** - Advanced vectorized backtesting library
   - Version: 1.0.0+
   - Used for: Trading strategy backtesting
   - Requires: GitHub token for installation

3. **Mem0** - Memory management for language models
   - Used for: Storing and retrieving example contexts
   - Requires: Mem0 API key

4. **Dependencies table**:

   | Package       | Version      | Purpose                   | Special Requirements                     |
   |---------------|--------------|---------------------------|------------------------------------------|
   | dspy          | 2.6.12+      | LLM Programming           | OpenAI API key                           |
   | vectorbtpro   | 1.0.0+       | Backtesting               | GitHub token, libta-lib                  |
   | pandas        | 2.0.0+       | Data manipulation         | numpy                                    |
   | loguru        | 0.7.0+       | Logging                   | None                                     |
   | python-dotenv | 1.0.0+       | Environment management    | None                                     |
   | mem0          | 0.1.0+       | Memory management         | Mem0 API key                             |
   | chromadb      | 0.4.0+       | Vector database           | None                                     |

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

### Advanced MiPro Configuration

For more control over the MiPro optimization process, you can configure these advanced parameters:

```python
from dspy.teleprompt import MIPROv2

# Initialize optimizer with advanced parameters
teleprompter = MIPROv2(
    metric=your_metric_function,
    num_candidates=7,            # Number of candidate instructions to generate
    init_temperature=0.5,        # Temperature for generation diversity
    max_bootstrapped_demos=3,    # Maximum bootstrapped examples to include
    max_labeled_demos=4,         # Maximum labeled examples to include
    teacher_settings={           # Settings for bootstrapping model
        "lm": more_powerful_lm   # Use a more powerful model for bootstrapping
    },
    metric_threshold=0.6,        # Only keep examples above this score
    verbose=True,                # Show detailed logs
    track_stats=True             # Log optimization statistics
)

# Compile with advanced parameters
optimized_program = teleprompter.compile(
    your_program,
    trainset=your_training_data,
    valset=your_validation_data, # Optional separate validation set
    num_trials=15,               # Number of optimization trials
    minibatch=True,              # Use minibatch evaluation (faster)
    minibatch_size=25,           # Size of evaluation minibatches
    minibatch_full_eval_steps=10,# Full evaluation frequency
    program_aware_proposer=True, # Use program code for instruction proposal
    data_aware_proposer=True,    # Use data summary for instruction proposal
    tip_aware_proposer=True,     # Use tips for instruction proposal
    fewshot_aware_proposer=True  # Use few-shot examples for instruction proposal
)
```

#### Key Parameters Explained

**Initialization Parameters:**
- **metric**: Function that evaluates how good the outputs are
- **auto**: Set to "light", "medium", or "heavy" for automatic configuration
- **num_candidates**: More candidates = more options to search but slower
- **max_bootstrapped_demos**: More demos = better examples but longer prompts
- **init_temperature**: Higher = more creative but potentially less focused
- **metric_threshold**: Only keep examples above this performance score

**Compile Parameters:**
- **num_trials**: More trials = better optimization but longer runtime
- **minibatch**: Evaluate on smaller batches for efficiency (recommended)
- **minibatch_size**: Larger batches = more stable but slower evaluation
- **program_aware_proposer**: Use program insight for better instructions
- **data_aware_proposer**: Use dataset insight for better instructions

## Troubleshooting

### Common Issues

1. **VectorBTPro Installation Problems**
   
   VectorBTPro is a private package that requires GitHub authentication:
   
   ```bash
   # Error: Failed to authenticate with repository
   
   # Solution 1: Configure the token directly with Poetry
   poetry config http-basic.vectorbt __token__ your_github_token_here
   
   # Solution 2: Check your token has correct permissions
   # Make sure your GitHub token has 'read:packages' permission
   
   # Solution 3: Try installing with pip directly
   pip install vectorbtpro --extra-index-url https://__token__:${GITHUB_TOKEN}@pypi.fury.io/vectorbt/
   ```
   
   If you encounter issues with TA-Lib:
   
   ```bash
   # For macOS
   brew install ta-lib
   
   # For Ubuntu
   wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   tar -xzf ta-lib-0.4.0-src.tar.gz
   cd ta-lib/
   ./configure --prefix=/usr
   make
   sudo make install
   ```

2. **DSPy Example Creation Errors**
   - Ensure example formats match expected schema
   - Check that input fields match module signatures
   - Verify DSPy version compatibility (we require 2.6.12+)
   
   ```bash
   # Check your DSPy version
   python -c "import dspy; print(dspy.__version__)"
   
   # Update DSPy if needed
   pip install -U dspy
   ```

3. **JSON Serialization Errors**
   - Complex objects (DataFrames, timestamps) should be simplified
   - Use the `_make_serializable()` method to convert non-serializable types
   
   ```python
   # Example of handling complex objects
   from src.modules.prompt_optimizer import PromptOptimizer
   
   # Initialize optimizer
   optimizer = PromptOptimizer(prompt_manager)
   
   # Make complex data serializable
   serializable_data = optimizer._make_serializable(complex_data)
   ```

4. **Optimization Doesn't Improve Results**
   - Ensure enough diverse examples (at least 4-5)
   - Check that examples demonstrate high-quality outputs
   - Try adjusting optimization parameters
   
   ```python
   # Adjust optimization parameters
   mipro = MiProWrapper(
       use_v2=True,
       max_bootstrapped_demos=5,  # Try increasing this value
       num_candidate_programs=10, # Try increasing this value
       temperature=0.8            # Try adjusting temperature
   )
   ```

5. **Duplicate Example Detection Problems**
   - Check if examples are being skipped as duplicates in logs (`Skipping duplicate example for prompt...`)
   - Review the `add_example` method in `src.utils.prompt_manager` 
   - Consider modifying the duplicate detection criteria to be less strict
   - Ensure market data preprocessing provides sufficient variation between runs
   
   ```python
   # Example of checking duplicate detection
   from src.utils.prompt_manager import PromptManager
   
   prompt_manager = PromptManager('prompts')
   
   # Print current example comparison function
   print(prompt_manager._examples_are_similar.__code__)
   
   # Modify similarity threshold if needed
   prompt_manager.similarity_threshold = 0.8  # Make less strict (default is usually higher)
   ```

6. **Market Analysis Optimization Issues**
   - Ensure data preprocessing provides different time windows for each iteration
   - Verify the prepared data summary changes between optimization runs
   - Check that the prompt formatting correctly incorporates new examples
   - Monitor prompt length variations to confirm changes are being applied
   - Ensure different market regimes are being represented in the examples
   
   ```python
   # Debug preprocessor output
   from src.utils.data_preprocessor import preprocess_market_data
   
   # Force different time windows by using different offset parameters
   data1 = preprocess_market_data(raw_data, start_offset=0, window_size=100)
   data2 = preprocess_market_data(raw_data, start_offset=50, window_size=100)
   
   # Verify they're actually different
   import pandas as pd
   print(f"Data sets identical: {pd.DataFrame.equals(data1, data2)}")
   ```

7. **Memory System Connection Issues**
   
   If you encounter Mem0 connection problems:
   
   ```bash
   # Check your Mem0 API key is correct
   echo $MEM0_API_KEY
   
   # Ensure your organization and project IDs are valid
   echo $MEM0_ORG_ID
   echo $MEM0_PROJECT_ID
   
   # Try running with the --debug flag
   python main.py --debug
   ```

### Debugging

For detailed logging during optimization:

```bash
# Set logging level to DEBUG in your code
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use environment variable
LOGLEVEL=DEBUG python optimize_market_analysis.py
```

### Performance Optimization

If you're processing large datasets:

```bash
# Reduce memory usage for DataFrame operations
from src.utils.data_preprocessor import preprocess_market_data

# Use the low_memory option
data = preprocess_market_data(raw_data, low_memory=True)

# Limit data points for initial testing
data = preprocess_market_data(raw_data, max_points=10000)
```

## Future Improvements

- Add optimization for other modules
- Implement automated optimization scheduling
- Add support for different LLM providers
- Create web interface for example management
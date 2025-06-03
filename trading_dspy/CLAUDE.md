# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands
```bash
# Setup environment
poetry install

# Main entry points (different execution modes)
python main.py           # Full pipeline with MiPro optimization (GPT-4o-mini, 5 iterations)
python main_optimized.py # Multi-token optimization for cost efficiency (60% cost savings)
python main_simple.py    # Simplified single-iteration testing (GPT-3.5-turbo)
python main_fixed.py     # Fixed version with enum handling improvements

# Optimization scripts
python optimize_market_analysis.py     # Market analysis prompt optimization
python optimize_trading_rules_only.py  # Trading rules optimization only

# Backtesting scripts
python run_backtest.py          # Standard backtesting
python streamlined_backtest.py  # Streamlined backtest workflow
python direct_backtest.py       # Direct backtesting without pipeline

# Run tests
pytest                    # all tests
pytest test_*.py         # specific test pattern  
pytest test_mipro.py     # MiPro optimization tests
pytest test_optimization*.py  # Strategy optimization tests
pytest test_pipeline.py  # End-to-end pipeline tests
pytest -v                # verbose output

# Code quality
black .                  # format code (100 char line length)
isort .                  # sort imports  
mypy .                   # type checking
```

## Architecture Overview

This is a sophisticated DSPy-powered algorithmic trading system that combines market analysis, strategy generation, and backtesting with continuous prompt optimization.

### Core Pipeline Flow
The main pipeline (src/pipeline.py) orchestrates:
1. **Market Analysis** - Regime classification and detailed market analysis
2. **Strategy Generation** - Memory-based strategy creation with historical performance tracking  
3. **Trading Rules Generation** - Entry/exit conditions with parameter optimization
4. **Backtesting** - Multi-asset performance validation with risk metrics

### Key Modules
- **src/modules/**: Core DSPy modules for each pipeline stage
  - `market_analysis.py` - Market condition analysis with Chain-of-Thought reasoning
  - `market_regime.py` & `market_regime_enhanced.py` - Market regime classification
  - `strategy_generator.py` - Dynamic strategy creation with memory integration
  - `trading_rules.py` - Automated trading rules with technical indicators
  - `backtester.py` - Multi-asset backtesting with VectorBTPro integration
  - `prompt_optimizer.py` - MiPro optimization with component scoring
  - `policy_optimizer.py` - Strategy parameter optimization
- **src/utils/**: Support utilities
  - `memory_manager.py` - Mem0ai integration for persistent strategy memory
  - `prompt_manager.py` - Prompt template management and optimization
  - `data_preprocessor.py` - Market data preprocessing with technical indicators
  - `mipro_optimizer.py` - Advanced MiPro optimization wrapper
  - `types.py` - Core type definitions (MarketRegime, StrategyContext, BacktestResults)

### Memory & Optimization System
- **Mem0ai Integration**: Persistent strategy memory across sessions with performance tracking
- **MiPro Optimization**: Three-stage process (Bootstrapping → Grounded Proposal → Discrete Search)
- **Component Optimization**: Separate optimization for market analysis, strategy generation, and trading rules
- **Plateau Breaking**: Automatic optimization restart when performance stagnates
- **Example Collection**: Automatic gathering of successful execution traces for prompt improvement

### Data Flow & Processing
```
Raw Market Data → Data Preprocessor → Market Analysis → Strategy Generation → Trading Rules → Backtesting → Memory Storage
                                   ↓
                              MiPro Optimization ← Example Collection ← Performance Feedback
```
- Market data preprocessed with technical indicators (SMA, RSI, volatility, volume analysis)
- Strategy parameters stored as StrategyContext objects with serializable parameters
- Backtest results tracked with comprehensive metrics (total_return, win_rate, sortino_ratio, drawdown)
- Performance feedback drives continuous prompt optimization through MiPro

## Environment Setup
Required environment variables in `.env`:
```bash
OPENAI_API_KEY=your_openai_key
MEM0_API_KEY=your_mem0_key  
GITHUB_TOKEN=your_github_token  # Required for VectorBTPro access
```

### Special Dependencies
- **VectorBTPro**: Private package requiring GitHub token authentication
- **DSPy 2.3.3+**: Core LLM programming framework
- **Python 3.10+**: Required for modern type hints and performance features

## Performance Targets
Current system targets:
- Total Return: 100%+ (current: ~3%)
- Total Trades: 1000+ (current: 4)
- Sortino Ratio: >15 (current: ~13.6)
- Assets Traded: 100+ (current: 1)

## Common Issues & Debugging

### MarketRegime Enum Errors
- **Issue**: `'"RANGING_LOW_VOL"' is not a valid MarketRegime` - extra quotes in regime values
- **Fix**: Use `src/utils/enum_fix.py` or `main_fixed.py` which handles enum parsing properly
- **Root Cause**: LLM responses sometimes include extra quotation marks in JSON output

### VectorBTPro Installation
- **Issue**: Package not found during `poetry install`
- **Fix**: Install manually with GitHub token: 
  ```bash
  pip install --index-url https://username:token@pypi.fury.io/vectorbt/ vectorbtpro
  ```
- **Note**: Requires valid GitHub token with package access

### Memory & Optimization
- **MiPro plateau detection**: System automatically restarts optimization when no improvement
- **Memory storage**: Check `mem0_track/` for optimization history and failure logs
- **Component scoring**: Progressive metrics track individual component performance

## Code Style Guidelines
- **Imports**: stdlib → third-party → project modules, grouped with blank lines
- **Types**: Use type hints for all functions (parameters and returns)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Line length**: 100 characters maximum
- **Error handling**: Use try/except with specific exceptions, log errors with loguru
- **Logging**: Use loguru with contextual information and structured data
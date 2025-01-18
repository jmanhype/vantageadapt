# VantageAdapt

A Python-based trading strategy optimization and backtesting framework that uses adaptive algorithms and LLMs to generate and optimize trading strategies.

## Features

- Advanced backtesting engine with support for multiple assets
- Strategy optimization using LLM-based approaches with structured generation
- Real-time market analysis and regime detection
- Self-improving code through Gödel machine principles
- Performance analytics with vectorized operations
- Database integration for storing results and performance metrics
- Comprehensive trade tracking and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/jmanhype/vantageadapt.git
cd vantageadapt

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
vantageadapt/
├── src/                    # Source code
│   └── strat_optim/       # Main package
│       ├── strategy/      # Strategy generation and optimization
│       └── database/      # Database models and connections
├── tests/                  # Test files
├── docs/                   # Documentation
│   └── SYSTEM_DESIGN.md   # Detailed system architecture
├── research/              # Research and analysis
│   └── strategy/         # Strategy research implementations
└── prompts/              # LLM prompts for strategy generation
```

## Usage

### Data Preparation

The system expects trade data in a pickled dictionary format where each key is an asset symbol and each value is a pandas DataFrame containing:
- `dex_price`: Price data
- `timestamp`: Time information
- Volume and other optional metrics

Example data structure:
```python
{
    "BTC-USD": pd.DataFrame(...),
    "ETH-USD": pd.DataFrame(...),
    ...
}
```

### Running Strategy Optimization

```bash
# Basic usage with breakout trading theme
python main.py --theme "breakout trading" --data "path/to/data.pkl"

# Mean reversion strategy
python main.py --theme "mean reversion" --data "path/to/data.pkl"

# Trend following with specific data
python main.py --theme "trend following" --data "path/to/data.pkl"
```

### Trading Themes

The system supports various trading themes including:
- Breakout trading
- Mean reversion
- Trend following
- Range trading
- Momentum trading
- Volatility breakout

Each theme influences how the LLM generates strategies and trading rules.

### Understanding Output

The system will:
1. Load and analyze market data
2. Detect current market regime
3. Generate trading strategy based on theme
4. Optimize parameters through backtesting
5. Output:
   - Strategy performance metrics
   - Optimized parameters
   - Trading rules
   - Market analysis

Example output:
```
INFO: Loading trade data from path/to/data.pkl
INFO: Successfully loaded 65 assets
INFO: Market regime: RANGING_LOW_VOL (confidence: 0.85)
INFO: Generating strategy for breakout trading...
INFO: Optimizing parameters...
INFO: Best configuration found:
- Total Return: 2.45
- Sharpe Ratio: 1.87
- Win Rate: 0.62
```

### Key Components

1. **Market Analysis**
   - Real-time regime detection
   - Volatility and trend analysis
   - Risk level assessment

2. **Strategy Generation**
   - LLM-based strategy creation
   - Structured parameter generation
   - Dynamic rule adaptation

3. **Performance Optimization**
   - Parameter tuning through backtesting
   - Self-improving code generation
   - Performance metric tracking

## Configuration

- Environment variables can be set in `.env` file
- Trading parameters can be adjusted in `config/trading_params.yaml`
- Strategy generation prompts are in `prompts/trading/`

For detailed system architecture and components, see [System Design Documentation](docs/SYSTEM_DESIGN.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
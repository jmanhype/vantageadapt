# Trading DSPy System

A sophisticated algorithmic trading system built with DSPy that combines market analysis, strategy generation, and backtesting capabilities. The system uses advanced prompt optimization and memory management to continuously improve its trading strategies.

## 🌟 Features

- **Market Analysis**

  - Real-time market regime classification
  - Enhanced market analysis with multiple indicators
  - Risk level assessment and confidence scoring
- **Strategy Generation**

  - Dynamic strategy creation based on market conditions
  - Memory-based strategy optimization
  - Historical performance tracking
  - Confidence-based filtering
- **Trading Rules**

  - Automated trading rules generation
  - Parameter optimization
  - Entry/exit condition validation
  - Risk management integration
- **Backtesting**

  - Multi-asset backtesting support
  - Performance metrics calculation
  - Strategy validation
  - Risk-adjusted returns analysis
- **Prompt Optimization**

  - MiPro optimization for improved prompts
  - Example collection and management
  - Continuous learning from successful strategies
  - Bootstrap few-shot learning

## 📊 Current Performance

- Total Return: ~3% (Target: 100%+)
- Win Rate: 100%
- Total Trades: 4 (Target: 1000+)
- Sortino Ratio: ~13.6 (Target: >15)
- Assets Traded: 1 (Target: 100+)

## 🛠 Tech Stack

- **Core**: Python, DSPy
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: GPT-4 Turbo
- **Storage**: mem0ai for strategy memory
- **Optimization**: MiPro, Optuna
- **Logging**: Loguru

## 🚀 Getting Started

### Prerequisites

```bash
python -v >= 3.8
```

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd trading_dspy
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```
OPENAI_API_KEY=your_openai_key
MEM0_API_KEY=your_mem0_key
```

### Running the System

1. Start the trading pipeline:

```bash
python main.py
```

2. Run backtesting:

```bash
python optimize_market_analysis.py
```

## 📁 Project Structure

```
trading_dspy/
├── src/
│   ├── modules/
│   │   ├── market_analysis.py
│   │   ├── market_regime.py
│   │   ├── strategy_generator.py
│   │   ├── trading_rules.py
│   │   ├── backtester.py
│   │   └── prompt_optimizer.py
│   ├── utils/
│   │   ├── data_preprocessor.py
│   │   ├── memory_manager.py
│   │   ├── prompt_manager.py
│   │   └── types.py
│   ├── prompts/
│   ├── models/
│   └── pipeline.py
├── tests/
├── docs/
└── logs/
```

## 🔄 Pipeline Flow

1. **Market Analysis**

   - Regime classification
   - Detailed market analysis
   - Risk assessment
2. **Strategy Generation**

   - Query similar strategies
   - Generate new strategy
   - Validate strategy
3. **Trading Rules Generation**

   - Create entry/exit conditions
   - Set parameters
   - Validate rules
4. **Backtesting**

   - Apply trading rules
   - Calculate performance metrics
   - Store results

## 📈 Scaling Strategy

Current focus areas for improvement:

1. **Expand Asset Coverage**

   - Implement multi-asset tracking
   - Create asset selection module
   - Add sector-specific strategies
2. **Increase Trade Frequency**

   - Modify entry conditions
   - Add timeframe diversity
   - Optimize parameters
3. **Improve DSPy Pipeline**

   - Add more seed examples
   - Increase bootstrap rounds
   - Implement KNN few-shot learning
4. **Advanced Optimizations**

   - Fix memory system
   - Implement Bayesian optimization
   - Add vector store integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [DSPy](https://github.com/stanfordnlp/dspy)
- [mem0](https://mem0.ai)

## Context Priming

Read README.md, CLAUDE.md, ai_docs/*, and run git ls-files to understand this codebase.

## 📫 Support

For support, please open an issue in the repository or contact the maintainers.

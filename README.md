# VantageAdapt

A Python-based trading strategy optimization and backtesting framework that uses adaptive algorithms to optimize trading strategies.

## Features

- Advanced backtesting engine with support for multiple assets
- Strategy optimization using machine learning and LLM-based approaches
- Real-time market analysis and regime detection
- Trade pattern recognition and signal generation
- Performance analytics and visualization
- Database integration for storing results
- API for strategy deployment
- Web-based control panel for strategy management

## Installation

### Basic Installation

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

### Docker Installation

```bash
# Build and run using Docker Compose
docker-compose up -d

# For testing environment
docker-compose -f docker-compose.test.yml up -d

# For simple deployment without frontend
docker-compose -f docker-compose.simple.yml up -d
```

## Project Structure

```
vantageadapt/
├── src/                    # Source code
│   └── strat_optim/       # Main package
├── tests/                  # Test files
├── docs/                   # Documentation
├── config/                # Configuration files
├── research/             # Research and analysis
├── frontend/             # Web-based control panel
│   └── control-panel/    # React/TypeScript frontend
└── prompts/              # LLM prompts for strategy generation
```

## Usage

### Running Backtests

```bash
# Run backtest with specific theme and data
python main.py --theme "breakout trading" --data "path/to/data.pkl"

# Run with custom parameters
python main.py --theme "mean reversion" --data "path/to/data.pkl" --risk-level low
```

### Using the Control Panel

1. Start the frontend development server:
```bash
cd frontend/control-panel
npm install
npm run dev
```

2. Access the control panel at `http://localhost:5173`

### API Endpoints

The API server provides the following endpoints:

- `POST /api/strategy/generate` - Generate new trading strategy
- `GET /api/strategy/list` - List all strategies
- `POST /api/backtest/run` - Run backtest with specified parameters
- `GET /api/backtest/results` - Get backtest results

## Configuration

- Environment variables can be set in `.env` file
- Trading parameters can be adjusted in `config/trading_params.yaml`
- Strategy generation prompts are in `prompts/trading/`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
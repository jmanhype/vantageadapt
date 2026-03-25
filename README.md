# VantageAdapt

LLM-driven trading strategy optimization and backtesting framework in Python.

## What It Does

Takes historical trade data (pickled DataFrames keyed by asset symbol), detects market regimes, uses an LLM to generate trading strategies for a given theme, then optimizes parameters through backtesting.

## Architecture

| Directory | Purpose |
|---|---|
| `src/strat_optim/` | Core strategy generation, optimization, and database models |
| `trading_dspy/` | DSPy-based pipeline: market analysis, regime detection, strategy generation, backtesting |
| `config/prompts/` | YAML/Markdown prompts for LLM strategy generation |
| `frontend/control-panel/` | React/TypeScript control panel (Vite + Tailwind) |
| `config/grafana/` | Grafana dashboards and datasource provisioning |
| `tests/` | Unit tests for backtesting gaps, LLM timeouts, regime transitions, memory corruption |

## Supported Trading Themes

Breakout, mean reversion, trend following, range, momentum, volatility breakout.

## Requirements

- Python 3.10+
- Node.js (for the control panel frontend)
- PostgreSQL (for result storage; Grafana connects to it)
- Docker and Docker Compose (optional, for full stack)
- API keys for your LLM provider (configured in `.env`)

## Setup

```bash
git clone https://github.com/jmanhype/vantageadapt.git
cd vantageadapt
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add API keys
```

Docker alternative:

```bash
docker compose up -d
```

## Usage

```bash
python main.py --theme "breakout trading" --data path/to/data.pkl
```

Input data format: a pickled dict where keys are asset symbols and values are DataFrames with `dex_price` and `timestamp` columns.

## Status

Experimental. The system generates strategies and runs backtests, but there is no paper trading or live execution. The Godel machine self-improvement loop referenced in code is aspirational. The frontend control panel is partially built.

## License

MIT

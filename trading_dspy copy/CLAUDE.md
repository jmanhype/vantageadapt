# Trading DSPy Project Guidelines

## Build & Run Commands
```bash
# Setup environment
poetry install

# Run main application
python main.py

# Run backtest
python run_backtest.py

# Run tests
pytest               # all tests
pytest tests/test_backtester.py  # single test file
pytest tests/test_backtester.py::test_function_name  # specific test
pytest tests/ -v     # verbose output

# Code quality
black .              # format code
isort .              # sort imports
mypy .               # type checking
```

## Code Style Guidelines
- **Imports**: stdlib → third-party → project modules, grouped with blank lines
- **Types**: Use type hints for all functions (parameters and returns)
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Docstrings**: Google-style docstrings with Args/Returns sections
- **Line length**: 100 characters maximum
- **Error handling**: Use try/except with specific exceptions, log errors
- **Logging**: Use loguru with contextual information
- **Module structure**: Clear separation of concerns between components
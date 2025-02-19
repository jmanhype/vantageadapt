"""Script to run backtesting optimization with real data."""

import logging
from src.modules.backtester import run_parameter_optimization, load_trade_data
from loguru import logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the backtester optimization with real data."""
    logger.info("Starting backtesting optimization")
    
    # Load trade data
    data_path = "/Users/speed/StratOptimv4/big_optimize_1016.pkl"
    trade_data = load_trade_data(data_path)
    
    if not trade_data:
        logger.error("Failed to load trade data")
        return
    
    # Define trading conditions based on price action and technical indicators
    conditions = {
        'entry': [
            "price > sma_20",  # Price above 20-period SMA
            "rsi < 70",        # RSI not overbought
            "macd.macd > macd.signal"  # MACD crossover
        ],
        'exit': [
            "price < sma_20",  # Price below 20-period SMA
            "rsi > 30",        # RSI not oversold
            "macd.macd < macd.signal"  # MACD crossunder
        ]
    }
    
    # Run optimization
    results = run_parameter_optimization(trade_data=trade_data, conditions=conditions)
    
    if results:
        logger.info("Optimization completed successfully")
        logger.info("Summary of results:")
        logger.info(f"Total return: {results['backtest_results']['total_return']:.4f}")
        logger.info(f"Total trades: {results['backtest_results']['total_trades']}")
        logger.info(f"Win rate: {results['backtest_results']['win_rate']:.2%}")
        logger.info(f"Sortino ratio: {results['backtest_results']['sortino_ratio']:.4f}")
        
        # Print optimal parameters
        logger.info("\nOptimal parameters:")
        for param, value in results['parameters'].items():
            if not param.startswith('_'):  # Skip internal parameters
                logger.info(f"{param}: {value}")
    else:
        logger.error("Optimization failed")

if __name__ == "__main__":
    main() 
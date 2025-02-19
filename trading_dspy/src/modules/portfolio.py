"""Portfolio management module for backtesting and optimization."""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger
import optuna
from .trading_rules import apply_trading_rules

class Portfolio:
    """Portfolio class for managing multiple assets and optimization."""
    
    def __init__(self) -> None:
        """Initialize an empty portfolio."""
        self.assets: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Any] = {}
        
    def add_asset(self, symbol: str, data: pd.DataFrame) -> None:
        """Add an asset to the portfolio.
        
        Args:
            symbol: The asset symbol
            data: DataFrame containing the asset's price data
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data for {symbol} must be a pandas DataFrame")
        self.assets[symbol] = data
        
    def optimize_parameters(
        self,
        conditions: Optional[Dict[str, List[str]]] = None,
        parameter_ranges: Optional[Dict[str, List[Any]]] = None,
        n_trials: int = 100,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Optimize trading parameters across all assets.
        
        Args:
            conditions: Dictionary of entry and exit conditions
            parameter_ranges: Dictionary of parameter ranges to optimize
            n_trials: Number of optimization trials
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing optimization results
        """
        if not self.assets:
            raise ValueError("No assets in portfolio")
            
        def objective(trial):
            params = {
                "take_profit": trial.suggest_float("take_profit", 0.01, 0.15),
                "stop_loss": trial.suggest_float("stop_loss", 0.01, 0.15),
                "sl_window": trial.suggest_int("sl_window", 100, 1000),
                "max_orders": trial.suggest_int("max_orders", 1, 5),
                "order_size": trial.suggest_float("order_size", 0.001, 0.01),
                "post_buy_delay": trial.suggest_int("post_buy_delay", 1, 10),
                "post_sell_delay": trial.suggest_int("post_sell_delay", 1, 10),
                "macd_signal_fast": trial.suggest_int("macd_signal_fast", 100, 1000),
                "macd_signal_slow": trial.suggest_int("macd_signal_slow", 200, 2000),
                "macd_signal_signal": trial.suggest_int("macd_signal_signal", 50, 500),
            }
            
            total_return = 0
            n_assets = 0
            
            for symbol, data in self.assets.items():
                try:
                    # Split data into train/test
                    split_idx = int(len(data) * (1 - test_size))
                    train_data = data.iloc[:split_idx]
                    
                    # Apply trading rules with current parameters
                    results = apply_trading_rules(
                        train_data,
                        conditions=conditions,
                        **params
                    )
                    
                    if results is not None:
                        total_return += results.get("total_return", 0)
                        n_assets += 1
                        
                except Exception as e:
                    logger.warning(f"Error optimizing {symbol}: {str(e)}")
                    continue
                    
            return -total_return / max(1, n_assets)  # Negative for minimization
            
        # Create and run optimization study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = -study.best_value  # Convert back to positive
        
        # Test best parameters on test set
        test_results = {}
        for symbol, data in self.assets.items():
            try:
                # Use test portion of data
                split_idx = int(len(data) * (1 - test_size))
                test_data = data.iloc[split_idx:]
                
                results = apply_trading_rules(
                    test_data,
                    conditions=conditions,
                    **best_params
                )
                
                if results is not None:
                    test_results[symbol] = results
                    
            except Exception as e:
                logger.warning(f"Error testing {symbol}: {str(e)}")
                continue
                
        # Compile final results
        final_results = {
            "best_parameters": best_params,
            "train_performance": best_value,
            "test_results": test_results,
            "optimization_history": [
                {
                    "trial": trial.number,
                    "value": -trial.value,
                    "params": trial.params
                }
                for trial in study.trials
            ]
        }
        
        self.results = final_results
        return final_results 
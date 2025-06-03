"""
Hybrid Trading System: Combining DSPy with Real ML
This is where we bring it all together - DSPy for flexibility, ML for learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import json
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from .pipeline import TradingPipeline
from .ml_trading_engine import MLTradingModel, TradingSystemOrchestrator, TradeSignal
from .regime_strategy_optimizer import RegimeStrategyOptimizer, RegimeIdentifier
from .utils.data_preprocessor import DataPreprocessor
from .utils.memory_manager import TradingMemoryManager
from .utils.types import MarketRegime, StrategyContext, BacktestResults


class HybridTradingSystem:
    """
    The ultimate trading system that combines:
    1. DSPy for creative strategy generation and analysis
    2. ML models for actual learning from data
    3. Regime-based optimization for adaptive strategies
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        logger.info("Initializing Hybrid Trading System")
        
        # Initialize components
        self.pipeline = TradingPipeline(
            api_key=api_key,
            model=model,
            use_enhanced_regime=True,
            use_prompt_optimization=True
        )
        
        self.ml_orchestrator = TradingSystemOrchestrator()
        self.regime_optimizer = RegimeStrategyOptimizer()
        self.data_preprocessor = DataPreprocessor(use_all_features=True)
        
        # Track performance
        self.performance_history = []
        self.current_regime = None
        self.ml_trained = False
        
    def train_on_historical_data(self, historical_df: pd.DataFrame):
        """Train all ML components on historical data"""
        logger.info("Training hybrid system on historical data")
        
        # Preprocess data
        preprocessed = self.data_preprocessor.add_features(historical_df)
        
        # Train ML models
        logger.info("Training ML trading models")
        self.ml_orchestrator.train_system(preprocessed)
        
        # Train regime identifier and optimizer
        logger.info("Training regime identification and optimization")
        # Create dummy trades for now (in real system, use actual historical trades)
        dummy_trades = self._create_dummy_trades(preprocessed)
        self.regime_optimizer.train_on_historical_data(preprocessed, dummy_trades)
        
        self.ml_trained = True
        logger.info("Hybrid system training completed")
        
    def _create_dummy_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy trades for regime training (placeholder)"""
        # In a real system, this would use actual historical trades
        trades = []
        for i in range(100):
            idx = np.random.randint(0, len(df) - 1)
            trades.append({
                'timestamp': df.index[idx],
                'entry_price': df.iloc[idx]['close'],
                'exit_price': df.iloc[idx + 1]['close'],
                'pnl': np.random.randn() * 10,
                'return_pct': np.random.randn() * 0.05,
                'success': np.random.random() > 0.5,
                'holding_period_hours': np.random.randint(1, 24)
            })
        return pd.DataFrame(trades)
        
    async def generate_hybrid_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal using both DSPy and ML"""
        if not self.ml_trained:
            logger.warning("ML models not trained, using DSPy only")
            return await self._generate_dspy_signal(market_data)
            
        # Get current data
        current_df = market_data['raw_data']
        
        # 1. Identify current regime
        regime, confidence = self.regime_optimizer.regime_identifier.identify_regime(current_df)
        self.current_regime = regime
        logger.info(f"Current market regime: {regime} (confidence: {confidence:.2%})")
        
        # 2. Get ML prediction
        ml_signal = self.ml_orchestrator.generate_signal(current_df)
        logger.info(f"ML Signal: {ml_signal.action} with {ml_signal.probability:.2%} confidence")
        
        # 3. Get regime-optimized strategy
        regime_strategy = self.regime_optimizer.get_strategy_for_regime(regime, confidence)
        logger.info(f"Regime strategy confidence: {regime_strategy.confidence:.2%}")
        
        # 4. Run DSPy pipeline for additional insights
        dspy_results = await self._generate_dspy_signal(market_data)
        
        # 5. Combine all signals
        combined_signal = self._combine_signals(
            ml_signal=ml_signal,
            regime_strategy=regime_strategy,
            dspy_results=dspy_results,
            regime_confidence=confidence
        )
        
        return combined_signal
        
    async def _generate_dspy_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signal using DSPy pipeline"""
        # Run single iteration for speed
        results = self.pipeline.run(market_data, num_iterations=1)
        
        if results and 'iterations' in results and results['iterations']:
            return results['iterations'][0]
        return {}
        
    def _combine_signals(
        self,
        ml_signal: TradeSignal,
        regime_strategy: Any,
        dspy_results: Dict[str, Any],
        regime_confidence: float
    ) -> Dict[str, Any]:
        """Intelligently combine signals from all sources"""
        
        # Extract DSPy recommendations
        dspy_action = 'HOLD'
        dspy_confidence = 0.0
        
        if dspy_results and 'strategy' in dspy_results:
            dspy_strategy = dspy_results['strategy']
            if isinstance(dspy_strategy, dict):
                dspy_action = dspy_strategy.get('signal', 'HOLD')
                dspy_confidence = dspy_strategy.get('confidence', 0.0)
            
        # Weight signals based on confidence and past performance
        ml_weight = 0.6  # ML gets highest weight because it learns
        regime_weight = 0.3 * regime_confidence  # Regime weight scales with confidence
        dspy_weight = 0.1  # DSPy provides creative insights
        
        # Normalize weights
        total_weight = ml_weight + regime_weight + dspy_weight
        ml_weight /= total_weight
        regime_weight /= total_weight
        dspy_weight /= total_weight
        
        # Calculate combined confidence
        combined_confidence = (
            ml_signal.probability * ml_weight +
            regime_strategy.confidence * regime_weight +
            dspy_confidence * dspy_weight
        )
        
        # Determine final action
        if ml_signal.action == 'BUY' and combined_confidence > 0.6:
            final_action = 'BUY'
        else:
            final_action = 'HOLD'
            
        # Combine parameters
        combined_params = {
            'position_size': ml_signal.position_size,
            'stop_loss': ml_signal.stop_loss,
            'take_profit': ml_signal.take_profit,
            'regime': self.current_regime,
            'ml_confidence': ml_signal.probability,
            'regime_confidence': regime_confidence,
            'dspy_confidence': dspy_confidence,
            'combined_confidence': combined_confidence
        }
        
        # Combine reasoning
        reasoning = [
            f"Hybrid signal combining ML ({ml_weight:.1%}), Regime ({regime_weight:.1%}), and DSPy ({dspy_weight:.1%})",
            f"Current regime: {self.current_regime}",
            *ml_signal.reasoning,
            f"Regime strategy: {len(regime_strategy.entry_conditions)} entry conditions",
            f"Combined confidence: {combined_confidence:.2%}"
        ]
        
        if dspy_results and 'strategy' in dspy_results:
            reasoning.append(f"DSPy insight: {dspy_results['strategy'].get('reasoning', 'N/A')[:100]}...")
            
        return {
            'action': final_action,
            'confidence': combined_confidence,
            'parameters': combined_params,
            'reasoning': reasoning,
            'components': {
                'ml_signal': {
                    'action': ml_signal.action,
                    'confidence': ml_signal.probability,
                    'predicted_return': ml_signal.predicted_return
                },
                'regime_strategy': {
                    'regime': self.current_regime,
                    'confidence': regime_strategy.confidence,
                    'historical_win_rate': regime_strategy.historical_performance.get('win_rate', 0)
                },
                'dspy_signal': {
                    'action': dspy_action,
                    'confidence': dspy_confidence
                }
            }
        }
        
    def update_with_trade_result(self, trade_result: Dict[str, Any]):
        """Update all components with trade results for continuous learning"""
        logger.info(f"Updating system with trade result: PnL={trade_result.get('pnl', 0):.2f}")
        
        # Update ML model
        self.ml_orchestrator.update_performance(trade_result)
        
        # Update regime optimizer
        trade_result['regime'] = self.current_regime
        self.regime_optimizer.update_strategy_performance(trade_result)
        
        # Update DSPy memory
        if 'strategy_context' in trade_result:
            self.pipeline.memory_manager.store_strategy_results(
                context=trade_result['strategy_context'],
                results=trade_result,
                iteration=len(self.performance_history)
            )
            
        # Track performance
        self.performance_history.append(trade_result)
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        ml_report = self.ml_orchestrator.get_performance_report()
        
        # Calculate hybrid metrics
        total_trades = len(self.performance_history)
        if total_trades > 0:
            winning_trades = sum(1 for t in self.performance_history if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in self.performance_history)
            win_rate = winning_trades / total_trades
        else:
            winning_trades = 0
            total_pnl = 0
            win_rate = 0
            
        return {
            'hybrid_performance': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
            },
            'ml_performance': ml_report,
            'regime_performance': {
                regime: self.regime_optimizer.performance_tracker.get_regime_performance(regime)
                for regime in self.regime_optimizer.regime_identifier.regime_names.values()
                if self.regime_optimizer.performance_tracker.get_regime_performance(regime)
            },
            'components_used': {
                'ml_trained': self.ml_trained,
                'regimes_identified': len(self.regime_optimizer.regime_strategies),
                'dspy_examples_collected': self.pipeline.prompt_manager.get_example_count()
            }
        }
        
    def save_models(self, directory: str):
        """Save all trained models"""
        path = Path(directory)
        path.mkdir(exist_ok=True)
        
        # Save ML model
        self.ml_orchestrator.ml_model.save(path / "ml_model.pkl")
        
        # Save regime data
        regime_data = {
            'regime_strategies': {
                k: {
                    'entry_conditions': v.entry_conditions,
                    'exit_conditions': v.exit_conditions,
                    'parameters': v.parameters,
                    'historical_performance': v.historical_performance,
                    'confidence': v.confidence
                }
                for k, v in self.regime_optimizer.regime_strategies.items()
            }
        }
        
        with open(path / "regime_strategies.json", 'w') as f:
            json.dump(regime_data, f, indent=2, default=str)
            
        logger.info(f"Models saved to {directory}")
        
    def load_models(self, directory: str):
        """Load trained models"""
        path = Path(directory)
        
        # Load ML model
        self.ml_orchestrator.ml_model.load(path / "ml_model.pkl")
        self.ml_trained = True
        
        # Load regime strategies
        with open(path / "regime_strategies.json", 'r') as f:
            regime_data = json.load(f)
            
        # Reconstruct regime strategies
        for regime, data in regime_data['regime_strategies'].items():
            self.regime_optimizer.regime_strategies[regime] = type('RegimeStrategy', (), data)
            
        logger.info(f"Models loaded from {directory}")
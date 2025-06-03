"""Integrated main script combining Darwin Gödel Machine with DSPy pipeline."""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
import time
import warnings
import numpy as np
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings("ignore")

from loguru import logger

# Disable litellm caching to avoid annotation errors
os.environ["LITELLM_CACHING"] = "False"
os.environ["LITELLM_CACHE"] = "False"

from src.pipeline import TradingPipeline
from src.utils.data_preprocessor import preprocess_market_data
from src.utils.enum_fix import fix_market_context
from darwin_godel_evolution import RealDataGenome, RealDataBacktester, RealDataDGMEvolver

# Load environment variables
load_dotenv()

def load_pickle_data(file_path: str) -> Dict[str, Any]:
    """Load data from a pickle file."""
    try:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Data loaded successfully. Found {len(data)} tokens.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def patch_pipeline():
    """Patch the pipeline to fix enum issues."""
    import src.pipeline
    import src.modules.market_analysis
    from src.utils.enum_fix import safe_market_regime, fix_market_context
    
    # Patch the generate_strategy method
    original_generate_strategy = src.pipeline.TradingPipeline.generate_strategy
    
    def patched_generate_strategy(self, market_context, recent_performance):
        # Fix market context before using it
        fixed_context = fix_market_context(market_context.copy())
        return original_generate_strategy(self, fixed_context, recent_performance)
    
    src.pipeline.TradingPipeline.generate_strategy = patched_generate_strategy
    
    # Patch the analyze_market method
    original_analyze_market = src.pipeline.TradingPipeline.analyze_market
    
    def patched_analyze_market(self, market_data, timeframe="1min"):
        result = original_analyze_market(self, market_data, timeframe)
        if result and isinstance(result, dict):
            return fix_market_context(result)
        return result
    
    src.pipeline.TradingPipeline.analyze_market = patched_analyze_market
    
    # Patch MarketRegime creation in pipeline
    original_init = src.utils.types.StrategyContext.__init__
    
    def patched_init(self, regime, confidence, risk_level, parameters, opportunity_score=0.0):
        # Ensure regime is properly converted
        from src.utils.enum_fix import safe_market_regime
        regime = safe_market_regime(regime)
        original_init(self, regime, confidence, risk_level, parameters, opportunity_score)
    
    src.utils.types.StrategyContext.__init__ = patched_init

class DGMEnhancedPipeline(TradingPipeline):
    """Trading pipeline enhanced with Darwin Gödel Machine optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dgm_genomes = []
        self.best_genome = None
        logger.info("Initialized DGM-Enhanced Trading Pipeline")
    
    def load_evolved_strategy(self, strategy_file: str = "evolved_strategies/real_data_genome_bfa435c8.json"):
        """Load the best evolved strategy from DGM."""
        try:
            with open(strategy_file, 'r') as f:
                genome_data = json.load(f)
            
            # Create genome from loaded data
            self.best_genome = RealDataGenome(**genome_data)
            logger.info(f"Loaded evolved strategy with fitness: {self.best_genome.fitness}")
            logger.info(f"  Signal Method: {self.best_genome.signal_method}")
            logger.info(f"  Use ML Signals: {self.best_genome.use_ml_signals}")
            logger.info(f"  Edge Threshold: {self.best_genome.edge_threshold_bp} bp")
            
        except Exception as e:
            logger.warning(f"Could not load evolved strategy: {e}")
            # Use default genome
            self.best_genome = RealDataGenome()
    
    def enhance_trading_rules_with_dgm(self, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance trading rules using DGM-evolved parameters."""
        if not self.best_genome:
            return rules
        
        # Apply DGM parameters to rules
        enhanced_rules = rules.copy()
        
        # Check if rules have the expected structure
        if "conditions" in rules:
            # New format from trading rules
            if "entry" not in enhanced_rules["conditions"]:
                enhanced_rules["conditions"]["entry"] = []
            if "exit" not in enhanced_rules["conditions"]:
                enhanced_rules["conditions"]["exit"] = []
                
            # Add DGM-based entry conditions
            if self.best_genome.signal_method == "breakout":
                enhanced_rules["conditions"]["entry"].append("price > bb.upper")
            elif self.best_genome.signal_method == "momentum":
                enhanced_rules["conditions"]["entry"].append("macd.macd > macd.signal * 1.05")
            elif self.best_genome.signal_method == "mean_reversion":
                enhanced_rules["conditions"]["entry"].append("price < bb.lower")
            
            # Add DGM risk management
            enhanced_rules["parameters"] = enhanced_rules.get("parameters", {})
            enhanced_rules["parameters"]["stop_loss"] = self.best_genome.stop_loss_bp / 10000
            enhanced_rules["parameters"]["take_profit"] = self.best_genome.take_profit_bp / 10000
            enhanced_rules["parameters"]["position_size"] = self.best_genome.position_size_pct
            enhanced_rules["parameters"]["max_positions"] = self.best_genome.max_positions
            
        else:
            # Old format - create proper structure
            enhanced_rules = {
                "conditions": {
                    "entry": rules.get("entry_conditions", []),
                    "exit": rules.get("exit_conditions", [])
                },
                "parameters": rules.get("parameters", {}),
                "indicators": rules.get("indicators", [])
            }
            
            # Add DGM enhancements
            if self.best_genome.signal_method == "breakout":
                enhanced_rules["conditions"]["entry"].append("price > bb.upper")
            elif self.best_genome.signal_method == "momentum":
                enhanced_rules["conditions"]["entry"].append("macd.macd > macd.signal * 1.05")
            
            enhanced_rules["parameters"]["stop_loss"] = self.best_genome.stop_loss_bp / 10000
            enhanced_rules["parameters"]["take_profit"] = self.best_genome.take_profit_bp / 10000
            enhanced_rules["parameters"]["position_size"] = self.best_genome.position_size_pct
        
        # Advanced features from DGM
        if self.best_genome.use_trailing_stop:
            enhanced_rules["parameters"]["trailing_stop"] = {
                "enabled": True,
                "percentage": self.best_genome.trailing_stop_pct
            }
        
        if self.best_genome.use_ml_signals:
            enhanced_rules["conditions"]["entry"].append("ml_signal > 0.7")
        
        if self.best_genome.use_order_imbalance:
            enhanced_rules["conditions"]["entry"].append("order_imbalance > 0.6")
        
        logger.info("Enhanced trading rules with DGM parameters")
        return enhanced_rules
    
    def evolve_parameters_for_market(self, market_data: Any, generations: int = 10):
        """Run quick DGM evolution for current market conditions."""
        logger.info("Running DGM evolution for current market...")
        
        evolver = RealDataDGMEvolver(market_data)
        evolver.initialize_population(size=10)  # Smaller population for speed
        
        # Quick evolution
        for gen in range(generations):
            fitness_scores = []
            
            for genome in evolver.population:
                # Simulate backtest (simplified for speed)
                results = evolver.backtester.run_backtest(genome)
                genome.fitness = results['fitness']
                fitness_scores.append(results['fitness'])
            
            # Update best
            if max(fitness_scores) > (self.best_genome.fitness if self.best_genome else -float('inf')):
                idx = fitness_scores.index(max(fitness_scores))
                self.best_genome = evolver.population[idx]
                logger.info(f"New best genome found! Fitness: {self.best_genome.fitness:.3f}")
            
            # Create next generation
            evolver.population = evolver.create_next_generation()
        
        logger.info(f"Evolution complete. Best fitness: {self.best_genome.fitness:.3f}")
    
    def generate_trading_rules(self, strategy_context: Dict[str, Any], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading rules enhanced with DGM parameters."""
        # First get base rules from DSPy
        base_rules = super().generate_trading_rules(strategy_context, market_context)
        
        # Enhance with DGM
        enhanced_rules = self.enhance_trading_rules_with_dgm(base_rules)
        
        return enhanced_rules

def main():
    # Apply patches
    patch_pipeline()
    
    # Configure logging
    logger.add("logs/trading_dgm_integrated.log", rotation="1 day")
    logger.info("Starting DGM-Integrated Trading Strategy Pipeline")
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    Path("evolved_strategies").mkdir(exist_ok=True)
    
    # Load market data
    trade_data = load_pickle_data("/Users/speed/StratOptimv4/big_optimize_1016.pkl")
    
    # Initialize DGM-enhanced pipeline
    pipeline = DGMEnhancedPipeline(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",  # Use efficient model
        performance_thresholds={
            'min_return': 0.05,
            'min_trades': 5,
            'max_drawdown': 0.25
        },
        use_enhanced_regime=False,  # Disable to reduce complexity
        use_prompt_optimization=False  # Disable to avoid issues
    )
    
    # Load the best evolved strategy from previous DGM run
    pipeline.load_evolved_strategy()
    
    try:
        start_time = time.time()
        
        # Test on more tokens for better results
        test_tokens = ["$MICHI", "POPCAT", "BILLY", "MOTHER", "RETARDIO", "GOAT", "FWOG", "MOODENG"]
        all_results = []
        total_pnl = 0
        total_trades = 0
        total_wins = 0
        
        for i, token in enumerate(test_tokens):
            if token not in trade_data:
                logger.warning(f"Token {token} not found in data")
                continue
                
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing token {i+1}/{len(test_tokens)}: {token}")
            logger.info(f"{'='*50}")
            
            try:
                # Preprocess data
                preprocessed_data = preprocess_market_data(trade_data[token])
                
                # Run quick evolution for this specific token (optional)
                if i == 0:  # Only evolve on first token for speed
                    pipeline.evolve_parameters_for_market(preprocessed_data, generations=5)
                
                # Run pipeline with DGM-enhanced rules
                results = pipeline.run(
                    market_data=preprocessed_data,
                    num_iterations=3,  # Fewer iterations since we have DGM optimization
                    timeframe="1h"
                )
                
                if results and results.get("iterations"):
                    token_result = {
                        'token': token,
                        'best_performance': None,
                        'dgm_parameters': {
                            'signal_method': pipeline.best_genome.signal_method,
                            'edge_threshold_bp': pipeline.best_genome.edge_threshold_bp,
                            'position_size_pct': pipeline.best_genome.position_size_pct,
                            'use_ml_signals': pipeline.best_genome.use_ml_signals,
                            'use_trailing_stop': pipeline.best_genome.use_trailing_stop
                        }
                    }
                    
                    best_pnl = -float('inf')
                    
                    for iteration in results["iterations"]:
                        performance = iteration.get("performance", {})
                        backtest_results = performance.get("backtest_results", {})
                        
                        if backtest_results:
                            pnl = backtest_results.get("total_pnl", 0.0)
                            if pnl > best_pnl:
                                best_pnl = pnl
                                token_result['best_performance'] = {
                                    'total_pnl': pnl,
                                    'total_return': backtest_results.get("total_return", 0.0),
                                    'win_rate': backtest_results.get("win_rate", 0.0),
                                    'total_trades': backtest_results.get("total_trades", 0),
                                    'sortino_ratio': backtest_results.get("sortino_ratio", 0.0),
                                    'sharpe_ratio': backtest_results.get("metrics", {}).get("sharpe_ratio", 0.0)
                                }
                    
                    if token_result['best_performance']:
                        all_results.append(token_result)
                        total_pnl += token_result['best_performance']['total_pnl']
                        total_trades += token_result['best_performance']['total_trades']
                        total_wins += int(token_result['best_performance']['total_trades'] * 
                                         token_result['best_performance']['win_rate'])
                        
                        logger.info(f"\nBest performance for {token}:")
                        logger.info(f"  P&L: ${token_result['best_performance']['total_pnl']:.2f}")
                        logger.info(f"  Win Rate: {token_result['best_performance']['win_rate']:.2%}")
                        logger.info(f"  Trades: {token_result['best_performance']['total_trades']}")
                        logger.info(f"  Sortino: {token_result['best_performance']['sortino_ratio']:.2f}")
                    
            except Exception as e:
                logger.error(f"Error processing token {token}: {str(e)}")
                continue
        
        # Save final results
        execution_time = time.time() - start_time
        overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
        
        final_results = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model': 'gpt-4o-mini + DGM',
            'approach': 'DGM-Integrated Trading DSPy',
            'tokens_analyzed': len(all_results),
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'overall_win_rate': overall_win_rate,
            'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
            'token_results': all_results,
            'dgm_config': {
                'fitness': pipeline.best_genome.fitness,
                'signal_method': pipeline.best_genome.signal_method,
                'edge_threshold_bp': pipeline.best_genome.edge_threshold_bp,
                'use_ml_signals': pipeline.best_genome.use_ml_signals,
                'use_trailing_stop': pipeline.best_genome.use_trailing_stop,
                'use_order_imbalance': pipeline.best_genome.use_order_imbalance,
                'use_price_impact': pipeline.best_genome.use_price_impact
            },
            'execution_time': execution_time
        }
        
        output_file = f"results/dgm_integrated_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("DGM-INTEGRATED TRADING - FINAL RESULTS")
        logger.info("="*70)
        logger.info(f"Tokens Analyzed: {len(all_results)}")
        logger.info(f"Total P&L: ${total_pnl:,.2f}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Overall Win Rate: {overall_win_rate:.2%}")
        logger.info(f"Avg P&L per Trade: ${total_pnl / total_trades if total_trades > 0 else 0:.2f}")
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        
        # DGM Parameters Used
        logger.info("\n" + "="*70)
        logger.info("DGM PARAMETERS APPLIED")
        logger.info("="*70)
        logger.info(f"Signal Method: {pipeline.best_genome.signal_method}")
        logger.info(f"Edge Threshold: {pipeline.best_genome.edge_threshold_bp:.2f} bp")
        logger.info(f"Position Size: {pipeline.best_genome.position_size_pct:.1%}")
        logger.info(f"ML Signals: {pipeline.best_genome.use_ml_signals}")
        logger.info(f"Trailing Stop: {pipeline.best_genome.use_trailing_stop}")
        logger.info(f"Order Imbalance: {pipeline.best_genome.use_order_imbalance}")
        logger.info(f"HVF (High Volume Filter): DISABLED as requested")
        
        logger.info(f"\nResults saved to {output_file}")
        
        # Display token-by-token breakdown
        logger.info("\n" + "="*70)
        logger.info("TOKEN-BY-TOKEN BREAKDOWN")
        logger.info("="*70)
        for result in all_results:
            perf = result['best_performance']
            logger.info(f"{result['token']:10} | P&L: ${perf['total_pnl']:8.2f} | "
                       f"Trades: {perf['total_trades']:3} | "
                       f"Win Rate: {perf['win_rate']:6.2%} | "
                       f"Sortino: {perf['sortino_ratio']:6.2f}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.exception("Full traceback:")
        return

if __name__ == "__main__":
    main()
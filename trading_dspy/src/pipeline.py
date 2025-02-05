"""Main DSPy pipeline for trading strategy generation and optimization."""

from typing import Dict, Any, Optional, List
import asyncio
from pathlib import Path
import pandas as pd
from loguru import logger
import dspy
import time

from .modules.market_analysis import MarketAnalyzer, MarketRegimeClassifier
from .modules.strategy_generator import StrategyGenerator
from .modules.trading_rules import TradingRulesGenerator
from .modules.backtester import Backtester
from .utils.prompt_manager import PromptManager
from .utils.memory_manager import TradingMemoryManager


class TradingPipeline:
    """Main DSPy pipeline for trading strategy generation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        memory_dir: str = "memory",
        prompts_dir: str = "prompts",
        performance_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize trading pipeline.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier
            memory_dir: Directory for memory storage
            prompts_dir: Directory for prompts
            performance_thresholds: Optional performance thresholds
        """
        logger.info("Initializing TradingPipeline with model: {}", model)
        
        # Initialize DSPy with OpenAI
        logger.info("Configuring DSPy")
        lm = dspy.LM(model, api_key=api_key)
        dspy.configure(lm=lm)
        
        # Initialize managers
        logger.info("Initializing managers")
        self.prompt_manager = PromptManager(prompts_dir)
        self.memory_manager = TradingMemoryManager(memory_dir)
        
        # Initialize modules
        logger.info("Initializing pipeline modules")
        self.market_analyzer = MarketAnalyzer(self.prompt_manager)
        self.regime_classifier = MarketRegimeClassifier()
        self.strategy_generator = StrategyGenerator(self.prompt_manager, self.memory_manager)
        self.trading_rules_generator = TradingRulesGenerator(self.prompt_manager)
        self.backtester = Backtester(performance_thresholds)
        
        logger.info("Trading pipeline initialized successfully")

    def run(
        self,
        market_data: Dict[str, Any],
        timeframe: str,
        trading_theme: str,
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Run the trading pipeline.
        
        Args:
            market_data: Dictionary containing market data
            timeframe: Timeframe for analysis
            trading_theme: Theme for strategy generation
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary containing final results
        """
        try:
            pipeline_start_time = time.time()
            logger.info("Starting trading pipeline execution")
            logger.info("Market data summary: {} data points from {} to {}",
                       len(market_data['prices']),
                       market_data.get('dates', ['unknown'])[0],
                       market_data.get('dates', ['unknown'])[-1])
            
            best_result = None
            best_metrics = None
            
            for iteration in range(max_iterations):
                iteration_start_time = time.time()
                logger.info("\nStarting iteration {} of {}", iteration + 1, max_iterations)
                
                # 1. Market Analysis
                logger.info("Step 1: Market Analysis")
                market_context = self.analyze_market(market_data, timeframe)
                if not market_context:
                    logger.warning("Market analysis failed, skipping iteration")
                    continue
                
                logger.info("Market regime: {}", market_context.get('regime'))
                logger.info("Risk level: {}", market_context.get('risk_level'))
                
                # 2. Strategy Generation
                logger.info("Step 2: Strategy Generation")
                strategy = self.generate_strategy(market_context, trading_theme)
                if not strategy:
                    logger.warning("Strategy generation failed, skipping iteration")
                    continue
                
                logger.info("Generated strategy with confidence: {:.2f}", strategy.get('confidence', 0.0))
                
                # 3. Trading Rules Generation
                logger.info("Step 3: Trading Rules Generation")
                # Ensure strategy has parameters
                if 'parameters' not in strategy:
                    strategy['parameters'] = {}
                    
                # Set default parameters if not provided
                default_params = {
                    'stop_loss': 0.02,
                    'take_profit': 0.04,
                    'position_size': 0.1
                }
                
                # Update strategy parameters with defaults if missing
                for param, default_value in default_params.items():
                    if param not in strategy['parameters']:
                        strategy['parameters'][param] = default_value
                    # Ensure parameters are floats
                    else:
                        try:
                            strategy['parameters'][param] = float(strategy['parameters'][param])
                        except (ValueError, TypeError):
                            strategy['parameters'][param] = default_value
                
                # Create strategy insights with parameters
                strategy_insights = {
                    'reasoning': strategy.get('reasoning', ''),
                    'trade_signal': strategy.get('trade_signal', ''),
                    'parameters': strategy['parameters'],
                    'confidence': strategy.get('confidence', 0.0),
                    'entry_conditions': strategy.get('entry_conditions', []) if isinstance(strategy.get('entry_conditions'), list) else strategy.get('parameters', {}).get('entry_conditions', []),
                    'exit_conditions': strategy.get('exit_conditions', []) if isinstance(strategy.get('exit_conditions'), list) else strategy.get('parameters', {}).get('exit_conditions', [])
                }
                
                # Log parameters for debugging
                logger.debug("Strategy parameters: {}", strategy['parameters'])
                logger.debug("Entry conditions: {}", strategy_insights['entry_conditions'])
                logger.debug("Exit conditions: {}", strategy_insights['exit_conditions'])
                
                rules = self.generate_trading_rules(strategy_insights, market_context)
                if not rules:
                    logger.warning("Trading rules generation failed, skipping iteration")
                    continue
                
                # 4. Backtesting
                logger.info("Step 4: Backtesting")
                result = self.backtest(
                    market_data,
                    rules['parameters'],
                    rules['conditions']
                )
                
                if not result:
                    logger.warning("Backtesting failed, skipping iteration")
                    continue
                
                metrics = result.get('metrics', {})
                
                # Track best result
                if (best_metrics is None or 
                    metrics.get('total_return', 0) > best_metrics.get('total_return', 0)):
                    best_metrics = metrics
                    best_result = {
                        'strategy': strategy,
                        'rules': rules,
                        'performance': metrics
                    }
                    
                    # Store successful strategy
                    logger.info("New best strategy found")
                    logger.info("Performance metrics:")
                    for metric, value in metrics.items():
                        logger.info("  {}: {:.4f}", metric, value)
                    
                    self.memory_manager.store_strategy_results(
                        context={
                            'market_regime': market_context.get('regime'),
                            'parameters': rules['parameters']
                        },
                        results=result,
                        iteration=iteration
                    )
                    logger.info("New best strategy found with return: {:.4f}", metrics.get('total_return') or 0.0)
                
                # Check if performance meets requirements
                if result.get('validation_passed'):
                    logger.info("Strategy meets performance requirements!")
                    break
                
                iteration_duration = time.time() - iteration_start_time
                logger.info("Iteration {} completed in {:.2f} seconds", iteration + 1, iteration_duration)
            
            if best_result is None:
                logger.warning("No successful strategy found")
                return {'error': 'No valid strategy found'}
                
            pipeline_duration = time.time() - pipeline_start_time
            logger.info("\nStrategy generation completed in {:.2f} seconds", pipeline_duration)
            logger.info("Best metrics:")
            for metric, value in best_metrics.items():
                logger.info("  {}: {:.4f}", metric, value)
            
            return best_result

        except Exception as e:
            logger.error("Error in pipeline execution: {}", str(e))
            logger.exception("Full traceback:")
            return {'error': str(e)}

    def analyze_market(self, market_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        """Run market analysis.
        
        Args:
            market_data: Market data
            timeframe: Analysis timeframe
            
        Returns:
            Market context dictionary
        """
        try:
            start_time = time.time()
            logger.info("Starting market analysis phase")
            
            # Get initial regime classification
            logger.info("Running regime classification")
            regime = self.regime_classifier.forward(
                market_data=market_data,
                timeframe=timeframe
            )
            
            # Get detailed analysis
            logger.info("Running detailed market analysis")
            analysis = self.market_analyzer.forward(
                market_data=market_data,
                timeframe=timeframe
            )
            
            # Combine results
            result = {
                **regime.get('market_context', {}),
                **analysis.get('market_context', {}),
                'analysis_text': analysis.get('analysis_text', ''),
                'risk_level': analysis.get('risk_level', 'unknown')
            }
            
            duration = time.time() - start_time
            logger.info("Market analysis completed in {:.2f} seconds", duration)
            return result

        except Exception as e:
            logger.error("Error in market analysis: {}", str(e))
            logger.exception("Full traceback:")
            return {}

    def generate_strategy(
        self,
        market_context: Dict[str, Any],
        trading_theme: str
    ) -> Dict[str, Any]:
        """Generate trading strategy.
        
        Args:
            market_context: Market context
            trading_theme: Strategy theme
            
        Returns:
            Strategy dictionary
        """
        try:
            start_time = time.time()
            logger.info("Starting strategy generation phase")
            
            # Query similar strategies from memory
            logger.info("Querying similar strategies from memory")
            similar_strategies = self.memory_manager.query_similar_strategies(
                market_context.get('regime', 'unknown')
            )
            logger.info("Found {} similar strategies", len(similar_strategies))
            
            # Generate strategy
            logger.info("Generating new strategy")
            strategy = self.strategy_generator.forward(
                market_context=market_context,
                theme=trading_theme,
                base_parameters=similar_strategies[0]['context'].get('parameters') if similar_strategies else None
            )
            
            # Validate strategy
            logger.info("Validating generated strategy")
            is_valid, reason = self.strategy_generator.validate_strategy(strategy)
            if not is_valid:
                logger.warning("Invalid strategy: {}", reason)
                return {}
            
            duration = time.time() - start_time
            logger.info("Strategy generation completed in {:.2f} seconds", duration)    
            return strategy

        except Exception as e:
            logger.error("Error in strategy generation: {}", str(e))
            logger.exception("Full traceback:")
            return {}

    def generate_trading_rules(
        self,
        strategy_insights: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading rules.
        
        Args:
            strategy_insights: Strategy insights
            market_context: Market context
            
        Returns:
            Trading rules dictionary
        """
        try:
            start_time = time.time()
            logger.info("Starting trading rules generation phase")
            
            rules = self.trading_rules_generator.forward(
                strategy_insights=strategy_insights,
                market_context=market_context
            )
            
            duration = time.time() - start_time
            logger.info("Trading rules generation completed in {:.2f} seconds", duration)
            return rules

        except Exception as e:
            logger.error("Error in trading rules generation: {}", str(e))
            logger.exception("Full traceback:")
            return {}

    def backtest(
        self,
        market_data: Dict[str, Any],
        parameters: Dict[str, Any],
        conditions: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Run backtest.
        
        Args:
            market_data: Market data
            parameters: Strategy parameters
            conditions: Trading conditions
            
        Returns:
            Backtest results
        """
        try:
            start_time = time.time()
            logger.info("Starting backtesting phase")
            logger.info("Testing {} entry conditions and {} exit conditions",
                       len(conditions.get('entry', [])),
                       len(conditions.get('exit', [])))
            
            # Extract trade data
            if isinstance(market_data, dict):
                if "raw_data" in market_data:
                    trade_data = market_data["raw_data"]
                else:
                    # Create trade data from market data components
                    trade_data = pd.DataFrame({
                        'timestamp': pd.date_range(start='2024-01-01', periods=len(market_data['prices']), freq='1min'),
                        'dex_price': market_data['prices'],
                        'sol_pool': market_data.get('volumes', [1] * len(market_data['prices'])),  # Default to 1 if not provided
                        'coin_pool': [1] * len(market_data['prices'])  # Default to 1
                    })
            else:
                trade_data = market_data
                
            # Ensure required columns exist
            required_columns = ['timestamp', 'dex_price', 'sol_pool', 'coin_pool']
            for col in required_columns:
                if col not in trade_data.columns:
                    if col == 'timestamp':
                        trade_data[col] = pd.date_range(start='2024-01-01', periods=len(trade_data), freq='1min')
                    elif col == 'dex_price' and 'close' in trade_data.columns:
                        trade_data[col] = trade_data['close']
                    else:
                        trade_data[col] = 1.0  # Default value for missing columns
            
            # Format trade data for backtester
            formatted_trade_data = {
                'default': trade_data
            }
            
            result = self.backtester.forward(
                trade_data=formatted_trade_data,
                parameters=parameters,
                conditions=conditions
            )
            
            duration = time.time() - start_time
            logger.info("Backtesting completed in {:.2f} seconds", duration)
            if result:
                logger.info("Backtest metrics:")
                for metric, value in result.get('metrics', {}).items():
                    logger.info("  {}: {:.4f}", metric, value)
            return result

        except Exception as e:
            logger.error("Error in backtesting: {}", str(e))
            logger.exception("Full traceback:")
            return {}

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Memory statistics dictionary
        """
        return self.memory_manager.get_strategy_statistics()


def run_backtest() -> None:
    """Run a sample backtest using the new Backtester integration."""
    # Sample dummy parameters (these should align with your strategy requirements)
    parameters = {
        'take_profit': 0.08,
        'stop_loss': 0.12,
        'sl_window': 400,
        'max_orders': 3,
        'order_size': 0.0025,
        'post_buy_delay': 2,
        'post_sell_delay': 5,
        'macd_signal_fast': 120,
        'macd_signal_slow': 260,
        'macd_signal_signal': 90,
        'min_macd_signal_threshold': 0,
        'max_macd_signal_threshold': 0,
        'enable_sl_mod': False,
        'enable_tp_mod': False,
    }

    # Sample dummy conditions
    conditions = {
        'entry': ["df_indicators['price'] > 100"],
        'exit': ["df_indicators['price'] < 90"]
    }

    # Create dummy trade data (with minimal required columns)
    trade_data = pd.DataFrame({
        'dex_price': [101, 102, 103, 98, 97, 96],
        'sol_pool': [1, 1, 1, 1, 1, 1],
        'coin_pool': [1, 1, 1, 1, 1, 1],
        'timestamp': pd.date_range(start='2022-01-01', periods=6, freq='T')
    })

    # Instantiate the Backtester with a performance threshold
    bt = Backtester({'min_return': 0.01})

    # Run the backtest
    result = bt.forward(trade_data, parameters, conditions)
    print('Backtester integration result:', result)


# Optionally, you can call run_backtest() here or from the main execution path
if __name__ == '__main__':
    # Existing pipeline execution, if any
    # run_pipeline()  # Uncomment if run_pipeline() exists
    run_backtest() 
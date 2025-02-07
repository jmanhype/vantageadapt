"""Main DSPy pipeline for trading strategy generation and optimization."""

from typing import Dict, Any, Optional, List
import asyncio
from pathlib import Path
import pandas as pd
from loguru import logger
import dspy
import time
import os
from dotenv import load_dotenv

from .modules.market_analysis import MarketAnalyzer
from .modules.market_regime import MarketRegimeClassifier
from .modules.strategy_generator import StrategyGenerator
from .modules.trading_rules import TradingRulesGenerator
from .modules.backtester import Backtester
from .utils.prompt_manager import PromptManager
from .utils.memory_manager import TradingMemoryManager
from .utils.types import StrategyContext, MarketRegime, BacktestResults


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
        # Load environment variables
        load_dotenv()
        
        logger.info("Initializing TradingPipeline with model: {}", model)
        
        # Initialize DSPy with OpenAI
        logger.info("Configuring DSPy")
        lm = dspy.LM(model, api_key=api_key)
        dspy.configure(lm=lm)
        
        # Initialize managers
        logger.info("Initializing managers")
        self.prompt_manager = PromptManager(prompts_dir)
        # Use MEM0_API_KEY from environment for memory manager
        self.memory_manager = TradingMemoryManager(api_key=os.getenv("MEM0_API_KEY"))
        
        # Set default performance thresholds if not provided
        if performance_thresholds is None:
            performance_thresholds = {
                'min_return': 0.10,
                'min_trades': 10,
                'max_drawdown': 0.20
            }
        
        # Initialize modules
        logger.info("Initializing pipeline modules")
        self.market_analyzer = MarketAnalyzer(self.prompt_manager)
        self.regime_classifier = MarketRegimeClassifier()
        self.strategy_generator = StrategyGenerator(self.prompt_manager, self.memory_manager)
        self.trading_rules_generator = TradingRulesGenerator(self.prompt_manager)
        self.backtester = Backtester(performance_thresholds=performance_thresholds)
        
        # Store configuration
        self.model = model
        self.performance_thresholds = performance_thresholds
        
        logger.info("Trading pipeline initialized successfully")

    def run(self, market_data: Dict[str, Any], num_iterations: int = 5, timeframe: str = "1min") -> Dict[str, Any]:
        """Run the trading pipeline.
        
        Args:
            market_data: Market data dictionary containing preprocessed data
            num_iterations: Number of iterations to run
            timeframe: Timeframe for analysis (default: "1min")
            
        Returns:
            Dictionary containing pipeline results
        """
        try:
            logger.info("Starting trading pipeline execution")
            raw_data = market_data.get('raw_data')
            if raw_data is not None:
                logger.info("Market data summary: {} data points from {} to {}", 
                           len(raw_data),
                           raw_data.index[0],
                           raw_data.index[-1])
            
            # Query memory for recent performance
            recent_performance = self.memory_manager.get_recent_performance()
            if recent_performance:
                logger.info("Current system performance:")
                logger.info("  Total Strategies: {}", recent_performance.get("strategies_analyzed", 0))
                logger.info("  Total Trades: {}", recent_performance.get("total_trades", 0))
                logger.info("  Overall Win Rate: {:.2f}", recent_performance.get("overall_win_rate", 0))
            
            results = []
            for i in range(num_iterations):
                logger.info("\nStarting iteration {} of {}", i + 1, num_iterations)
                
                # Step 1: Market Analysis
                logger.info("Step 1: Market Analysis")
                market_context = self.analyze_market(market_data, timeframe)
                logger.info("Market regime: {}", market_context.get("regime", "UNKNOWN"))
                logger.info("Risk level: {}", market_context.get("risk_level", "unknown"))
                
                # Step 2: Strategy Generation
                logger.info("Step 2: Strategy Generation")
                strategy_insights = self.generate_strategy(market_context, recent_performance)
                logger.info("Generated strategy with confidence: {:.2f}", strategy_insights.get("confidence", 0))
                
                # Step 3: Trading Rules Generation
                logger.info("Step 3: Trading Rules Generation")
                rules = self.generate_trading_rules(strategy_insights, market_context)
                
                # Debug log the strategy parameters and conditions
                if "parameters" in strategy_insights:
                    logger.debug("Strategy parameters: {}", strategy_insights["parameters"])
                if "entry_conditions" in strategy_insights:
                    logger.debug("Entry conditions: {}", strategy_insights["entry_conditions"])
                if "exit_conditions" in strategy_insights:
                    logger.debug("Exit conditions: {}", strategy_insights["exit_conditions"])
                
                # Step 4: Backtesting
                logger.info("Step 4: Backtesting")
                if rules and "conditions" in rules:
                    backtest_results = self.backtest(
                        market_data,
                        strategy_insights.get("parameters", {}),
                        rules["conditions"]
                    )
                    
                    if backtest_results:
                        results.append({
                            "market_context": market_context,
                            "strategy": strategy_insights,
                            "rules": rules,
                            "performance": backtest_results
                        })
                
                # Update memory with results
                if results:
                    # Store the last iteration's results
                    last_result = results[-1]
                    self.memory_manager.store_strategy_results(
                        context=last_result["market_context"],
                        results=last_result["performance"],
                        iteration=len(results)
                    )
            
            return {"iterations": results} if results else {}
            
        except Exception as e:
            logger.error("Error in pipeline execution: {}", str(e))
            logger.exception("Full traceback:")
            return {}

    def analyze_market(self, market_data: Dict[str, Any], timeframe: str = "1min") -> Dict[str, Any]:
        """Run market analysis.
        
        Args:
            market_data: Market data dictionary containing preprocessed data
            timeframe: Timeframe for analysis (default: "1min")
            
        Returns:
            Market context dictionary
        """
        try:
            start_time = time.time()
            logger.info("Starting market analysis phase")
            
            # Get initial regime classification
            logger.info("Running regime classification")
            regime = self.regime_classifier.forward(
                market_data=market_data,  # Pass the preprocessed dictionary
                timeframe=timeframe
            )
            
            # Get detailed analysis
            logger.info("Running detailed market analysis")
            analysis = self.market_analyzer.forward(
                market_data=market_data,  # Pass the preprocessed dictionary
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
        recent_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading strategy.
        
        Args:
            market_context: Market context
            recent_performance: Recent performance metrics
            
        Returns:
            Strategy insights dictionary
        """
        try:
            start_time = time.time()
            logger.info("Starting strategy generation phase")
            
            # Query similar strategies from memory with pagination
            logger.info("Querying similar strategies from memory")
            page = 1
            page_size = 5
            similar_strategies = self.memory_manager.query_similar_strategies(
                market_context.get('regime', 'unknown'),
                page=page,
                page_size=page_size
            )
            
            total_strategies = len(similar_strategies)
            logger.info(f"Found {total_strategies} similar strategies on page {page}")
            
            # If we have strategies, log their performance
            if similar_strategies:
                logger.info("Top performing similar strategies:")
                for i, strat in enumerate(similar_strategies[:3], 1):
                    logger.info(
                        f"{i}. Return: {strat['performance']['total_return']:.2f}, "
                        f"Win Rate: {strat['performance']['win_rate']:.2f}, "
                        f"Score: {strat['score']:.2f}"
                    )
            
            # Get recent performance metrics
            recent_performance = self.memory_manager.get_recent_performance(lookback_days=7)
            logger.info("Recent strategy performance (7 days):")
            logger.info(f"  Total Trades: {recent_performance['total_trades']}")
            logger.info(f"  Avg Return: {recent_performance['avg_return']:.2f}")
            logger.info(f"  Win Rate: {recent_performance['overall_win_rate']:.2f}")
            
            # Generate strategy
            logger.info("Generating new strategy")
            strategy = self.strategy_generator.forward(
                market_context=market_context,
                theme="default",
                base_parameters=similar_strategies[0]['parameters'] if similar_strategies else None
            )
            
            # Create StrategyContext object
            if strategy:
                # Extract parameters for StrategyContext
                parameters = {
                    **strategy.get('parameters', {}),
                    'entry_conditions': strategy.get('entry_conditions', []),
                    'exit_conditions': strategy.get('exit_conditions', []),
                    'indicators': strategy.get('indicators', []),
                    'strategy_type': strategy.get('strategy_type', 'default')
                }
                
                strategy_context = StrategyContext(
                    regime=MarketRegime(market_context.get('regime', 'UNKNOWN')),
                    confidence=strategy.get('confidence', 0.0),
                    risk_level=market_context.get('risk_level', 'unknown'),
                    parameters=parameters,
                    opportunity_score=0.0
                )
                
                # Store the strategy context
                self.memory_manager.store_strategy(strategy_context)
            
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
            
            # Restructure the conditions if needed
            if rules and ('entry_conditions' in rules or 'exit_conditions' in rules):
                rules['conditions'] = {
                    'entry': rules.get('entry_conditions', []),
                    'exit': rules.get('exit_conditions', [])
                }
            
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
        """Run backtesting.
        
        Args:
            market_data: Market data dictionary containing preprocessed data
            parameters: Strategy parameters
            conditions: Dictionary containing entry/exit conditions
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Ensure conditions are in the correct format
            if not isinstance(conditions, dict) or 'entry' not in conditions or 'exit' not in conditions:
                logger.error("Invalid conditions format")
                return {}
            
            # Use the raw DataFrame directly
            df = market_data['raw_data'].copy()
            
            # Run backtesting with the prepared DataFrame
            raw_results = self.backtester.forward(
                trade_data={'$MICHI': df},  # Wrap in dict as backtester expects multiple assets
                parameters=parameters,
                conditions=conditions
            )
            
            # Convert raw results to BacktestResults object
            if raw_results and 'backtest_results' in raw_results:
                br = raw_results['backtest_results']
                metrics = br.get('metrics', {})
                
                # Extract stats from the first (and only) asset
                stats_df = metrics.get('per_asset_stats', {})
                total_return = sum(stats_df.get('total_return', {}).values())
                
                results = BacktestResults(
                    total_return=float(total_return),
                    total_pnl=float(br.get('total_pnl', 0.0)),
                    sortino_ratio=float(metrics.get('trade_memory_stats', {}).get('sortino_ratio', 0.0)),
                    win_rate=float(metrics.get('trade_memory_stats', {}).get('win_rate', 0.0)),
                    total_trades=int(metrics.get('trade_memory_stats', {}).get('total_trades', 0)),
                    trades=br.get('trades', {}),
                    metrics=metrics
                )
                return {'backtest_results': results.to_dict()}
            
            return raw_results

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
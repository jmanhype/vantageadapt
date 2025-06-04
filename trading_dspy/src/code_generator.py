#!/usr/bin/env python3
"""
Autonomous Code Generator - LLM writes new trading logic
Implements Kagan's vision: "The LLM would write the trading logic"
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import ast
import json
from loguru import logger
import dspy
from pathlib import Path
import subprocess
import tempfile

from src.utils.types import BacktestResults, MarketRegime, StrategyContext
from src.utils.memory_manager import MemoryManager


class TradingLogicGeneration(dspy.Signature):
    """Generate new trading logic based on performance analysis."""
    
    performance_analysis = dspy.InputField(desc="Analysis of current trading performance")
    failed_strategies = dspy.InputField(desc="Examples of strategies that failed")
    successful_patterns = dspy.InputField(desc="Patterns that led to successful trades")
    market_conditions = dspy.InputField(desc="Current and expected market conditions")
    
    trading_logic_code = dspy.OutputField(desc="Python code for new trading strategy")
    logic_explanation = dspy.OutputField(desc="Explanation of why this logic should work")
    expected_improvements = dspy.OutputField(desc="Expected performance improvements")


class IndicatorCodeGeneration(dspy.Signature):
    """Generate code for new technical indicators."""
    
    market_inefficiency = dspy.InputField(desc="Identified market inefficiency to exploit")
    existing_indicators = dspy.InputField(desc="List of currently used indicators")
    performance_gaps = dspy.InputField(desc="Areas where current indicators fail")
    
    indicator_code = dspy.OutputField(desc="Python code for new technical indicator")
    indicator_usage = dspy.OutputField(desc="How to use this indicator in trading")
    mathematical_basis = dspy.OutputField(desc="Mathematical explanation of the indicator")


class RiskManagementCodeGeneration(dspy.Signature):
    """Generate code for new risk management strategies."""
    
    risk_analysis = dspy.InputField(desc="Analysis of current risk exposure and failures")
    drawdown_patterns = dspy.InputField(desc="Patterns that led to significant drawdowns")
    volatility_regimes = dspy.InputField(desc="Different volatility regimes and their characteristics")
    
    risk_management_code = dspy.OutputField(desc="Python code for risk management logic")
    risk_parameters = dspy.OutputField(desc="Recommended risk parameters and limits")
    adaptation_rules = dspy.OutputField(desc="Rules for adapting risk based on market conditions")


class CodeGenerator(dspy.Module):
    """
    Implements Kagan's vision of LLM writing trading logic autonomously.
    This is the system that generates new code based on performance.
    """
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        super().__init__()
        
        # Code generation modules
        self.trading_logic_generator = dspy.ChainOfThought(TradingLogicGeneration)
        self.indicator_generator = dspy.ChainOfThought(IndicatorCodeGeneration)
        self.risk_generator = dspy.ChainOfThought(RiskManagementCodeGeneration)
        
        # Memory and tracking
        self.memory_manager = memory_manager or MemoryManager()
        
        # Generated code history
        self.generated_strategies = []
        self.successful_implementations = []
        self.failed_implementations = []
        
        # Code templates directory
        self.templates_dir = Path("src/generated_strategies")
        self.templates_dir.mkdir(exist_ok=True)
        
        logger.info("🤖 Code Generator initialized - Autonomous trading logic creation enabled")
    
    def generate_trading_strategy(self,
                                 performance_analysis: Dict[str, Any],
                                 market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate complete trading strategy code based on performance analysis.
        
        This implements Kagan's core requirement: "The LLM would write the trading logic"
        """
        logger.info("🔧 Generating new trading strategy code")
        
        # Analyze what strategies have failed
        failed_strategies = self._analyze_failed_strategies()
        
        # Identify successful patterns
        successful_patterns = self._identify_successful_patterns()
        
        try:
            # Generate new trading logic
            generation_result = self.trading_logic_generator(
                performance_analysis=json.dumps(performance_analysis, indent=2),
                failed_strategies=json.dumps(failed_strategies, indent=2),
                successful_patterns=json.dumps(successful_patterns, indent=2),
                market_conditions=json.dumps(market_conditions, indent=2)
            )
            
            # Parse and validate the generated code
            strategy_code = self._parse_and_validate_code(generation_result.trading_logic_code)
            
            # Create strategy package
            strategy_package = {
                'strategy_id': f"autogen_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'generated_at': datetime.now().isoformat(),
                'code': strategy_code,
                'explanation': generation_result.logic_explanation,
                'expected_improvements': generation_result.expected_improvements,
                'performance_context': performance_analysis,
                'market_context': market_conditions
            }
            
            # Save generated strategy
            self._save_generated_strategy(strategy_package)
            
            # Test the strategy syntax
            if self._test_strategy_syntax(strategy_code):
                logger.info("✅ Generated strategy code is syntactically valid")
                self.generated_strategies.append(strategy_package)
            else:
                logger.warning("⚠️ Generated strategy has syntax errors, attempting to fix...")
                strategy_code = self._fix_code_syntax(strategy_code)
                strategy_package['code'] = strategy_code
            
            return strategy_package
            
        except Exception as e:
            logger.error(f"Error generating trading strategy: {e}")
            return self._generate_fallback_strategy(performance_analysis)
    
    def generate_custom_indicator(self,
                                 market_inefficiency: str,
                                 current_indicators: List[str]) -> Dict[str, Any]:
        """
        Generate code for a new technical indicator to exploit market inefficiency.
        """
        logger.info(f"🔍 Generating custom indicator for: {market_inefficiency}")
        
        # Analyze performance gaps
        performance_gaps = self._analyze_indicator_gaps(current_indicators)
        
        try:
            generation_result = self.indicator_generator(
                market_inefficiency=market_inefficiency,
                existing_indicators=json.dumps(current_indicators),
                performance_gaps=json.dumps(performance_gaps)
            )
            
            # Parse and validate indicator code
            indicator_code = self._parse_and_validate_code(generation_result.indicator_code)
            
            indicator_package = {
                'indicator_id': f"indicator_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'name': f"Custom_{market_inefficiency.replace(' ', '_')}",
                'code': indicator_code,
                'usage': generation_result.indicator_usage,
                'mathematical_basis': generation_result.mathematical_basis,
                'target_inefficiency': market_inefficiency
            }
            
            # Test the indicator
            if self._test_indicator_logic(indicator_code):
                logger.info("✅ Generated indicator is valid")
                self._save_indicator(indicator_package)
            
            return indicator_package
            
        except Exception as e:
            logger.error(f"Error generating indicator: {e}")
            return self._generate_fallback_indicator(market_inefficiency)
    
    def generate_risk_management_system(self,
                                      risk_analysis: Dict[str, Any],
                                      historical_drawdowns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate new risk management code based on historical failures.
        """
        logger.info("🛡️ Generating enhanced risk management system")
        
        # Analyze drawdown patterns
        drawdown_patterns = self._analyze_drawdown_patterns(historical_drawdowns)
        
        # Identify volatility regimes
        volatility_regimes = self._identify_volatility_regimes()
        
        try:
            generation_result = self.risk_generator(
                risk_analysis=json.dumps(risk_analysis, indent=2),
                drawdown_patterns=json.dumps(drawdown_patterns, indent=2),
                volatility_regimes=json.dumps(volatility_regimes, indent=2)
            )
            
            # Parse risk management code
            risk_code = self._parse_and_validate_code(generation_result.risk_management_code)
            
            risk_package = {
                'risk_system_id': f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'code': risk_code,
                'parameters': self._parse_risk_parameters(generation_result.risk_parameters),
                'adaptation_rules': generation_result.adaptation_rules,
                'target_max_drawdown': 0.15,  # 15% max
                'generated_from': risk_analysis
            }
            
            # Save risk management system
            self._save_risk_system(risk_package)
            
            return risk_package
            
        except Exception as e:
            logger.error(f"Error generating risk management: {e}")
            return self._generate_fallback_risk_system()
    
    def evolve_existing_strategy(self,
                               current_strategy: StrategyContext,
                               performance_metrics: BacktestResults,
                               improvement_targets: Dict[str, float]) -> Dict[str, Any]:
        """
        Evolve an existing strategy by generating modifications.
        """
        logger.info("🧬 Evolving existing strategy through code generation")
        
        # Generate evolutionary modifications
        modifications = []
        
        # 1. Improve entry conditions if win rate is low
        if performance_metrics.win_rate < improvement_targets.get('min_win_rate', 0.5):
            entry_mod = self._generate_entry_improvements(current_strategy)
            modifications.append(entry_mod)
        
        # 2. Improve exit conditions if profits are low
        if performance_metrics.total_return < improvement_targets.get('min_return', 0.1):
            exit_mod = self._generate_exit_improvements(current_strategy)
            modifications.append(exit_mod)
        
        # 3. Add new indicators if needed
        if performance_metrics.total_trades < improvement_targets.get('min_trades', 100):
            indicator_mod = self._generate_indicator_additions(current_strategy)
            modifications.append(indicator_mod)
        
        # Combine modifications into evolved strategy
        evolved_strategy = self._combine_modifications(current_strategy, modifications)
        
        return {
            'evolved_strategy_id': f"evolved_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'base_strategy': current_strategy.strategy_id,
            'modifications': modifications,
            'code': evolved_strategy,
            'expected_improvements': self._estimate_improvements(modifications),
            'generation_method': 'evolution'
        }
    
    def _parse_and_validate_code(self, code_text: str) -> str:
        """Parse and validate generated Python code."""
        # Clean up code text
        code_text = code_text.strip()
        
        # Remove markdown code blocks if present
        if code_text.startswith('```python'):
            code_text = code_text[9:]
        if code_text.startswith('```'):
            code_text = code_text[3:]
        if code_text.endswith('```'):
            code_text = code_text[:-3]
        
        # Basic syntax validation
        try:
            ast.parse(code_text)
            return code_text
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {e}")
            # Attempt to fix common issues
            return self._fix_code_syntax(code_text)
    
    def _fix_code_syntax(self, code: str) -> str:
        """Attempt to fix common syntax errors in generated code."""
        # Fix common issues
        fixes = [
            # Add missing colons
            (r'def\s+(\w+)\s*\([^)]*\)\s*$', r'def \1():'),
            (r'if\s+([^:]+)$', r'if \1:'),
            (r'for\s+([^:]+)$', r'for \1:'),
            # Fix indentation (simplified)
            (r'^\s*def', 'def'),
            (r'^\s*class', 'class'),
        ]
        
        import re
        for pattern, replacement in fixes:
            code = re.sub(pattern, replacement, code, flags=re.MULTILINE)
        
        return code
    
    def _test_strategy_syntax(self, code: str) -> bool:
        """Test if strategy code is syntactically valid."""
        try:
            compile(code, '<generated>', 'exec')
            return True
        except:
            return False
    
    def _save_generated_strategy(self, strategy_package: Dict[str, Any]):
        """Save generated strategy to file."""
        strategy_file = self.templates_dir / f"{strategy_package['strategy_id']}.py"
        
        # Create full module code
        module_code = f'''#!/usr/bin/env python3
"""
Auto-generated Trading Strategy
Generated: {strategy_package['generated_at']}
Explanation: {strategy_package['explanation']}
Expected Improvements: {strategy_package['expected_improvements']}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

{strategy_package['code']}

# Strategy metadata
STRATEGY_INFO = {{
    'id': '{strategy_package['strategy_id']}',
    'generated_at': '{strategy_package['generated_at']}',
    'performance_context': {strategy_package['performance_context']},
    'market_context': {strategy_package['market_context']}
}}
'''
        
        with open(strategy_file, 'w') as f:
            f.write(module_code)
        
        logger.info(f"💾 Saved generated strategy: {strategy_file}")
        
        # Also save metadata
        metadata_file = self.templates_dir / f"{strategy_package['strategy_id']}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(strategy_package, f, indent=2)
    
    def _generate_fallback_strategy(self, performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simple fallback strategy when LLM fails."""
        # Basic momentum strategy as fallback
        fallback_code = '''
def generate_trading_signal(market_data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback momentum-based trading strategy."""
    
    # Calculate simple momentum
    short_window = params.get('short_window', 10)
    long_window = params.get('long_window', 30)
    
    short_ma = market_data['close'].rolling(short_window).mean()
    long_ma = market_data['close'].rolling(long_window).mean()
    
    # Generate signal
    signal = 0
    if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
        signal = 1  # Buy
    elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
        signal = -1  # Sell
    
    return {
        'signal': signal,
        'confidence': abs(short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1],
        'indicators': {
            'short_ma': short_ma.iloc[-1],
            'long_ma': long_ma.iloc[-1]
        }
    }
'''
        
        return {
            'strategy_id': f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'code': fallback_code,
            'explanation': 'Fallback momentum strategy due to generation failure',
            'expected_improvements': 'Basic trend following',
            'is_fallback': True
        }
    
    def _analyze_failed_strategies(self) -> List[Dict[str, Any]]:
        """Analyze patterns in failed strategy implementations."""
        # Get failed strategies from memory
        failed_data = []
        
        if self.memory_manager:
            failed_memories = self.memory_manager.search_memories(
                "failed_strategy", limit=10
            )
            
            for memory in failed_memories:
                failed_data.append({
                    'strategy_type': memory.get('metadata', {}).get('strategy_type'),
                    'failure_reason': memory.get('metadata', {}).get('failure_reason'),
                    'performance': memory.get('metadata', {}).get('performance', {})
                })
        
        # Add recent failures
        failed_data.extend(self.failed_implementations[-5:])
        
        return failed_data
    
    def _identify_successful_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns from successful strategies."""
        successful_patterns = []
        
        # Get successful strategies from memory
        if self.memory_manager:
            success_memories = self.memory_manager.search_memories(
                "successful_strategy", limit=10
            )
            
            for memory in success_memories:
                successful_patterns.append({
                    'pattern_type': memory.get('metadata', {}).get('pattern_type'),
                    'performance_metrics': memory.get('metadata', {}).get('performance'),
                    'key_features': memory.get('metadata', {}).get('key_features', [])
                })
        
        return successful_patterns
    
    def deploy_generated_strategy(self, strategy_package: Dict[str, Any]) -> bool:
        """
        Deploy a generated strategy for live testing.
        This is where Kagan's vision becomes real - the LLM's code runs autonomously.
        """
        logger.info(f"🚀 Deploying generated strategy: {strategy_package['strategy_id']}")
        
        try:
            # Validate strategy package
            if not self._validate_strategy_package(strategy_package):
                logger.error("Strategy package validation failed")
                return False
            
            # Create deployment configuration
            deployment_config = {
                'strategy_id': strategy_package['strategy_id'],
                'code_path': f"{self.templates_dir}/{strategy_package['strategy_id']}.py",
                'test_duration_hours': 24,  # Test for 24 hours
                'risk_limit': 0.02,  # 2% max risk per trade
                'position_limit': 0.1,  # 10% max position size
                'stop_monitoring_threshold': -0.05  # Stop if -5% loss
            }
            
            # Deploy to test environment
            success = self._deploy_to_test_env(deployment_config)
            
            if success:
                logger.info("✅ Strategy deployed successfully")
                self.successful_implementations.append(strategy_package)
                
                # Track deployment in memory
                if self.memory_manager:
                    self.memory_manager.add_memory(
                        f"Deployed strategy {strategy_package['strategy_id']}",
                        metadata={
                            'strategy_id': strategy_package['strategy_id'],
                            'deployment_time': datetime.now().isoformat(),
                            'expected_improvements': strategy_package['expected_improvements']
                        }
                    )
            else:
                logger.error("❌ Strategy deployment failed")
                self.failed_implementations.append(strategy_package)
            
            return success
            
        except Exception as e:
            logger.error(f"Error deploying strategy: {e}")
            return False
    
    def _deploy_to_test_env(self, config: Dict[str, Any]) -> bool:
        """Deploy strategy to test environment."""
        # In a real implementation, this would:
        # 1. Copy code to execution environment
        # 2. Start monitoring process
        # 3. Connect to trading system
        # 4. Begin paper trading
        
        # For now, simulate deployment
        logger.info(f"Simulating deployment of {config['strategy_id']}")
        return True  # Simulate success
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about code generation performance."""
        return {
            'total_generated': len(self.generated_strategies),
            'successful_deployments': len(self.successful_implementations),
            'failed_deployments': len(self.failed_implementations),
            'success_rate': len(self.successful_implementations) / max(len(self.generated_strategies), 1),
            'last_generation': self.generated_strategies[-1]['generated_at'] if self.generated_strategies else None,
            'active_strategies': self._count_active_strategies()
        }
    
    def _count_active_strategies(self) -> int:
        """Count currently active auto-generated strategies."""
        # In real implementation, would check actual deployments
        return len([s for s in self.successful_implementations 
                   if datetime.fromisoformat(s['generated_at']) > 
                   datetime.now().replace(hour=0, minute=0, second=0)])


async def main():
    """Test the Code Generator with sample scenarios."""
    logger.info("🤖 Testing Autonomous Code Generator")
    
    # Initialize generator
    generator = CodeGenerator()
    
    # Sample performance analysis
    performance_analysis = {
        'current_return': 0.05,
        'win_rate': 0.45,
        'issues': ['Low win rate', 'Poor entry timing', 'Exits too early'],
        'successful_trades': 450,
        'failed_trades': 550
    }
    
    # Sample market conditions
    market_conditions = {
        'regime': 'TRENDING_BULLISH',
        'volatility': 'medium',
        'trend_strength': 0.7
    }
    
    # Generate new trading strategy
    logger.info("Generating new trading strategy...")
    strategy = generator.generate_trading_strategy(performance_analysis, market_conditions)
    
    logger.info(f"Generated strategy: {strategy['strategy_id']}")
    logger.info(f"Explanation: {strategy['explanation']}")
    logger.info(f"Expected improvements: {strategy['expected_improvements']}")
    
    # Generate custom indicator
    logger.info("\nGenerating custom indicator...")
    indicator = generator.generate_custom_indicator(
        "Volume-price divergence in trending markets",
        ["RSI", "MACD", "SMA"]
    )
    
    logger.info(f"Generated indicator: {indicator['name']}")
    logger.info(f"Usage: {indicator['usage']}")
    
    # Test deployment
    logger.info("\nTesting strategy deployment...")
    deployed = generator.deploy_generated_strategy(strategy)
    logger.info(f"Deployment {'successful' if deployed else 'failed'}")
    
    # Show statistics
    stats = generator.get_generation_statistics()
    logger.info(f"\nGeneration Statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
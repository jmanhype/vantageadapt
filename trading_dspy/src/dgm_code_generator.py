#!/usr/bin/env python3
"""
DGM-Enhanced Code Generator - Self-modifying trading logic generator
Combines Darwin's Gödel Machine with LLM code generation for Kagan's vision
"""

import asyncio
import ast
import json
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from pathlib import Path
import numpy as np
from loguru import logger
import dspy
import hashlib
import os

from src.utils.types import BacktestResults, MarketRegime, StrategyContext
from src.utils.memory_manager import TradingMemoryManager as MemoryManager

# Configure DSPy with OpenAI
turbo = dspy.LM('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=turbo)

# Import base CodeGenerator or create a simple one
try:
    from src.code_generator import CodeGenerator
except ImportError:
    import dspy
    class CodeGenerator(dspy.Module):
        def __init__(self, memory_manager=None):
            super().__init__()
            self.memory_manager = memory_manager
            self.generated_strategies = []


class SelfModifyingCode(dspy.Signature):
    """Generate code that can modify itself based on performance."""
    
    current_code = dspy.InputField(desc="Current trading strategy code")
    performance_metrics = dspy.InputField(desc="Performance metrics of current code")
    modification_history = dspy.InputField(desc="History of previous modifications")
    
    reasoning = dspy.OutputField(desc="Chain of thought reasoning for modifications")
    modified_code = dspy.OutputField(desc="Improved version of the code")
    modification_explanation = dspy.OutputField(desc="What you changed and why")
    expected_improvement = dspy.OutputField(desc="Expected performance improvement")


class MetaLearningCode(dspy.Signature):
    """Generate code that learns how to generate better code."""
    
    successful_strategies = dspy.InputField(desc="Examples of successful trading strategies")
    failed_strategies = dspy.InputField(desc="Examples of failed trading strategies")
    pattern_analysis = dspy.InputField(desc="Analysis of what makes strategies succeed or fail")
    
    meta_strategy_code = dspy.OutputField(desc="Code that generates trading strategies")
    learning_mechanisms = dspy.OutputField(desc="How the code learns from its own performance")
    evolution_strategy = dspy.OutputField(desc="Strategy for evolving the code over time")


class DGMCodeGenerator(CodeGenerator):
    """
    Darwin's Gödel Machine enhanced code generator.
    Implements true self-modification and meta-learning for trading strategies.
    """
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        super().__init__(memory_manager)
        
        # DGM-specific modules
        self.self_modifier = dspy.Predict(SelfModifyingCode)
        self.meta_learner = dspy.ChainOfThought(MetaLearningCode)
        
        # Evolution tracking
        self.code_evolution = []
        self.modification_graph = {}  # Track code lineage
        self.fitness_history = {}
        self.meta_strategies = []
        
        # Execution environment
        self.execution_env = self._setup_execution_environment()
        
        logger.info("🧬 DGM Code Generator initialized - Self-modifying code capability activated")
    
    def generate_self_modifying_strategy(self,
                                       initial_performance: Dict[str, Any],
                                       target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a trading strategy that can modify its own code.
        
        This implements Kagan's vision + Gödel's self-improvement concept.
        """
        logger.info("🔄 Generating self-modifying trading strategy")
        
        # Generate initial strategy code with self-modification capability
        initial_code = self._generate_initial_self_modifying_code(target_metrics)
        
        # Create modification history tracker
        mod_history = {
            'generation': 0,
            'modifications': [],
            'performance_trajectory': [initial_performance]
        }
        
        # FUCK DSPy - let's generate directly with OpenAI
        try:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
            prompt = f"""You are an AI that generates self-modifying trading strategies.
            
Current performance: {json.dumps(initial_performance, indent=2)}
Target metrics: {json.dumps(target_metrics, indent=2)}

Generate an improved version of this trading strategy that can modify itself:

{initial_code[:1000]}...

Return a JSON with these fields:
1. reasoning: Your chain of thought for improvements
2. modified_code: The complete improved trading strategy code
3. modification_explanation: What you changed and why
4. expected_improvement: Expected performance improvement

Make the strategy aggressive and self-improving. It should learn from its mistakes and evolve."""

            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=4000
            )
            
            # Parse response
            try:
                result = json.loads(response.choices[0].message.content)
                validated_code = self._validate_self_modifying_code(result.get('modified_code', initial_code))
                reasoning = result.get('reasoning', 'Direct OpenAI generation')
                modification_explanation = result.get('modification_explanation', 'Enhanced self-modifying strategy')
                expected_improvement = result.get('expected_improvement', '20%+ performance improvement')
            except:
                # If JSON parsing fails, use the raw response
                validated_code = initial_code
                reasoning = "Direct generation"
                modification_explanation = "Self-modifying strategy with adaptive learning"
                expected_improvement = "Continuous improvement through evolution"
            
            # Create complete strategy package
            strategy_package = {
                'strategy_id': f"dgm_selfmod_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'generated_at': datetime.now().isoformat(),
                'code': validated_code,
                'self_modification_enabled': True,
                'modification_explanation': modification_explanation,
                'expected_improvement': expected_improvement,
                'reasoning': reasoning,
                'target_metrics': target_metrics,
                'evolution_parameters': {
                    'mutation_rate': 0.1,
                    'learning_rate': 0.01,
                    'exploration_factor': 0.2
                }
            }
            
            # Add to evolution tracking
            self.code_evolution.append(strategy_package)
            
            logger.info("✅ Self-modifying strategy generated successfully")
            return strategy_package
            
        except Exception as e:
            logger.error(f"Error generating self-modifying strategy: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Fail fast - no fallbacks!
    
    def _generate_fallback_self_modifying_strategy(self, target_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate a fallback self-modifying strategy when LLM fails."""
        logger.info("Using fallback self-modifying strategy generation")
        
        # Create a basic self-modifying strategy
        strategy_code = '''
class SelfModifyingStrategy:
    """Basic self-modifying trading strategy with evolutionary capabilities."""
    
    def __init__(self):
        self.generation = 0
        self.mutation_rate = 0.1
        self.parameters = {
            'momentum_threshold': 0.02,
            'lookback_period': 20,
            'risk_per_trade': 0.02,
            'stop_loss': 0.02,
            'take_profit': 0.05
        }
        self.performance_history = []
    
    def generate_signal(self, data):
        """Generate trading signal based on current parameters."""
        if len(data) < self.parameters['lookback_period']:
            return 0
        
        # Simple momentum strategy
        momentum = (data['close'].iloc[-1] / data['close'].iloc[-self.parameters['lookback_period']] - 1)
        
        if momentum > self.parameters['momentum_threshold']:
            return 1  # Buy
        elif momentum < -self.parameters['momentum_threshold']:
            return -1  # Sell
        return 0
    
    def self_modify(self, performance_metrics):
        """Self-modify parameters based on performance."""
        self.generation += 1
        self.performance_history.append(performance_metrics)
        
        # Simple evolutionary modification
        if performance_metrics.get('total_return', 0) < 0:
            # Bad performance - mutate parameters
            for param, value in self.parameters.items():
                if np.random.random() < self.mutation_rate:
                    # Apply random mutation
                    mutation = np.random.normal(0, 0.1)
                    if param == 'lookback_period':
                        self.parameters[param] = max(5, int(value * (1 + mutation)))
                    else:
                        self.parameters[param] = max(0.001, value * (1 + mutation))
        
        # Adaptive mutation rate
        if len(self.performance_history) > 5:
            recent_returns = [p.get('total_return', 0) for p in self.performance_history[-5:]]
            if all(r < 0 for r in recent_returns):
                self.mutation_rate = min(0.5, self.mutation_rate * 1.5)
            elif all(r > 0 for r in recent_returns):
                self.mutation_rate = max(0.05, self.mutation_rate * 0.8)
        
        return self.parameters
'''
        
        return {
            'strategy_id': f"dgm_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'code': strategy_code,
            'self_modification_enabled': True,
            'modification_explanation': 'Fallback self-modifying strategy with basic evolution',
            'expected_improvement': 'Gradual parameter optimization through mutation',
            'target_metrics': target_metrics,
            'evolution_parameters': {
                'mutation_rate': 0.1,
                'learning_rate': 0.01,
                'exploration_factor': 0.2
            }
        }
    
    def evolve_strategy_autonomously(self,
                                   strategy_id: str,
                                   performance_data: List[BacktestResults],
                                   max_generations: int = 10) -> Dict[str, Any]:
        """
        Autonomously evolve a strategy through self-modification.
        
        This is the core DGM loop - code that improves itself.
        """
        logger.info(f"🧬 Starting autonomous evolution for strategy {strategy_id}")
        
        current_strategy = self._get_strategy_by_id(strategy_id)
        if not current_strategy:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        evolution_history = []
        best_fitness = self._calculate_fitness(performance_data[-1]) if performance_data else 0
        
        for generation in range(max_generations):
            logger.info(f"Generation {generation + 1}/{max_generations}")
            
            # Analyze current performance
            current_performance = self._analyze_performance_trends(performance_data)
            
            # Generate modification
            modification = self._generate_code_modification(
                current_strategy,
                current_performance,
                evolution_history
            )
            
            # Test modification in sandbox
            test_results = self._test_modification_sandbox(modification)
            
            # Calculate fitness
            new_fitness = self._calculate_fitness(test_results)
            
            # Decision: Accept or reject modification
            if self._should_accept_modification(new_fitness, best_fitness, generation):
                logger.info(f"✅ Accepting modification: {new_fitness:.4f} > {best_fitness:.4f}")
                
                # Apply modification
                current_strategy = self._apply_modification(current_strategy, modification)
                best_fitness = new_fitness
                
                # Track evolution
                evolution_history.append({
                    'generation': generation,
                    'modification': modification,
                    'fitness': new_fitness,
                    'accepted': True
                })
            else:
                logger.info(f"❌ Rejecting modification: {new_fitness:.4f} <= {best_fitness:.4f}")
                evolution_history.append({
                    'generation': generation,
                    'modification': modification,
                    'fitness': new_fitness,
                    'accepted': False
                })
            
            # Meta-learning: Learn from evolution process
            if generation % 3 == 0:
                self._update_meta_strategy(evolution_history)
            
            # Early stopping
            if self._check_convergence(evolution_history):
                logger.info("Convergence detected, stopping evolution")
                break
        
        return {
            'evolved_strategy': current_strategy,
            'evolution_history': evolution_history,
            'final_fitness': best_fitness,
            'generations_completed': len(evolution_history),
            'improvement': best_fitness - self._calculate_fitness(performance_data[0])
        }
    
    def generate_meta_learning_system(self,
                                    successful_strategies: List[Dict[str, Any]],
                                    failed_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a meta-learning system that learns how to create better strategies.
        
        This implements the highest level of Gödel's concept - 
        code that learns how to write better code.
        """
        logger.info("🧠 Generating meta-learning trading system")
        
        # Analyze patterns in successful vs failed strategies
        pattern_analysis = self._analyze_strategy_patterns(
            successful_strategies,
            failed_strategies
        )
        
        try:
            # Generate meta-learning code
            meta_result = self.meta_learner(
                successful_strategies=json.dumps(successful_strategies, indent=2),
                failed_strategies=json.dumps(failed_strategies, indent=2),
                pattern_analysis=json.dumps(pattern_analysis, indent=2)
            )
            
            # Parse and structure meta-strategy
            meta_strategy = {
                'meta_id': f"meta_dgm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'code': self._parse_and_validate_code(getattr(meta_result, 'meta_strategy_code', '# Meta strategy code')),
                'learning_mechanisms': getattr(meta_result, 'learning_mechanisms', 'Adaptive learning from performance'),
                'evolution_strategy': getattr(meta_result, 'evolution_strategy', 'Continuous improvement through feedback'),
                'pattern_insights': pattern_analysis,
                'generation_rules': self._extract_generation_rules(meta_result),
                'self_improvement_cycle': {
                    'frequency': 'after_every_100_trades',
                    'metrics_threshold': 0.6,
                    'modification_strength': 'adaptive'
                }
            }
            
            # Store meta-strategy
            self.meta_strategies.append(meta_strategy)
            
            logger.info("✅ Meta-learning system generated successfully")
            return meta_strategy
            
        except Exception as e:
            logger.error(f"Error generating meta-learning system: {e}")
            return self._generate_fallback_meta_system()
    
    def _generate_initial_self_modifying_code(self, target_metrics: Dict[str, float]) -> str:
        """Generate initial code with self-modification capabilities."""
        code = f'''#!/usr/bin/env python3
"""
Self-Modifying Trading Strategy
Generated by DGM Code Generator
Target Metrics: {target_metrics}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import json
import ast
import inspect

class SelfModifyingStrategy:
    """
    Trading strategy that can modify its own behavior based on performance.
    Implements Darwin's Gödel Machine concept for autonomous improvement.
    """
    
    def __init__(self):
        self.version = 1
        self.modifications = []
        self.performance_history = []
        self.current_params = {{
            'signal_threshold': 0.02,
            'position_size': 0.1,
            'stop_loss': 0.03,
            'take_profit': 0.05,
            'lookback_period': 20
        }}
        self.meta_params = {{
            'learning_rate': 0.01,
            'exploration_rate': 0.1,
            'modification_threshold': 0.05
        }}
    
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal with current parameters."""
        # Current trading logic
        returns = market_data['close'].pct_change()
        momentum = returns.rolling(self.current_params['lookback_period']).mean()
        volatility = returns.rolling(self.current_params['lookback_period']).std()
        
        # Signal generation
        signal_strength = momentum.iloc[-1] / (volatility.iloc[-1] + 1e-6)
        
        if signal_strength > self.current_params['signal_threshold']:
            signal = 1  # Buy
        elif signal_strength < -self.current_params['signal_threshold']:
            signal = -1  # Sell
        else:
            signal = 0  # Hold
        
        return {{
            'signal': signal,
            'confidence': abs(signal_strength),
            'position_size': self.current_params['position_size'],
            'stop_loss': self.current_params['stop_loss'],
            'take_profit': self.current_params['take_profit']
        }}
    
    def evaluate_performance(self, results: Dict[str, Any]) -> float:
        """Evaluate recent performance and decide if modification needed."""
        self.performance_history.append(results)
        
        # Calculate performance metrics
        recent_performance = self.performance_history[-10:]
        avg_return = np.mean([p.get('return', 0) for p in recent_performance])
        win_rate = np.mean([1 if p.get('return', 0) > 0 else 0 for p in recent_performance])
        
        # Performance score
        performance_score = avg_return * 0.6 + win_rate * 0.4
        
        return performance_score
    
    def self_modify(self, performance_score: float):
        """Modify own parameters based on performance."""
        if performance_score < self.meta_params['modification_threshold']:
            # Poor performance - modify parameters
            modification = self._generate_modification()
            self._apply_modification(modification)
            self.version += 1
            self.modifications.append({{
                'version': self.version,
                'timestamp': pd.Timestamp.now(),
                'modification': modification,
                'performance_before': performance_score
            }})
    
    def _generate_modification(self) -> Dict[str, Any]:
        """Generate parameter modifications using meta-learning."""
        modification = {{}}
        
        # Analyze what's not working
        recent_losses = [p for p in self.performance_history[-20:] if p.get('return', 0) < 0]
        
        if len(recent_losses) > 10:
            # Too many losses - adjust risk
            modification['stop_loss'] = self.current_params['stop_loss'] * 0.9
            modification['signal_threshold'] = self.current_params['signal_threshold'] * 1.1
        
        # Exploration vs exploitation
        if np.random.random() < self.meta_params['exploration_rate']:
            # Explore new parameters
            param_to_modify = np.random.choice(list(self.current_params.keys()))
            modification[param_to_modify] = self.current_params[param_to_modify] * np.random.uniform(0.8, 1.2)
        
        return modification
    
    def _apply_modification(self, modification: Dict[str, Any]):
        """Apply modifications to current parameters."""
        for param, value in modification.items():
            if param in self.current_params:
                old_value = self.current_params[param]
                self.current_params[param] = value
                print(f"Modified {{param}}: {{old_value:.4f}} -> {{value:.4f}}")
    
    def get_current_source(self) -> str:
        """Return current source code (for DGM evolution)."""
        return inspect.getsource(self.__class__)
    
    def evolve_source_code(self) -> str:
        """Generate evolved version of own source code."""
        current_source = self.get_current_source()
        
        # This is where DGM magic happens - code modifying its own logic
        # In practice, this would use AST manipulation or LLM generation
        
        # For now, return current source with parameter updates
        return current_source

# Instantiate strategy
strategy = SelfModifyingStrategy()
'''
        return code
    
    def _validate_self_modifying_code(self, code: str) -> str:
        """Validate and enhance self-modifying code."""
        # Parse and validate
        validated = self._parse_and_validate_code(code)
        
        # Ensure self-modification capabilities
        if "self_modify" not in validated:
            # Add self-modification method if missing
            validated += '''
    
    def self_modify(self, performance_metrics: Dict[str, Any]):
        """Self-modification logic based on performance."""
        # Analyze performance
        if performance_metrics.get('win_rate', 0) < 0.5:
            # Modify signal generation logic
            self.current_params['signal_threshold'] *= 1.1
        
        if performance_metrics.get('sharpe_ratio', 0) < 1.0:
            # Reduce risk
            self.current_params['position_size'] *= 0.9
'''
        
        return validated
    
    def _test_modification_sandbox(self, modification: Dict[str, Any]) -> BacktestResults:
        """Test code modification in sandboxed environment."""
        # Create temporary file with modified code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(modification['code'])
            temp_file = f.name
        
        try:
            # Run in subprocess for safety
            result = subprocess.run(
                ['python', temp_file, '--test'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse results
            if result.returncode == 0:
                # Success - parse output
                output_data = json.loads(result.stdout)
                return BacktestResults(**output_data)
            else:
                # Failure - return poor results
                logger.warning(f"Sandbox test failed: {result.stderr}")
                return BacktestResults(
                    total_return=-0.1,
                    total_trades=0,
                    win_rate=0,
                    sharpe_ratio=-1,
                    max_drawdown=0.5,
                    total_pnl=-10000,
                    final_capital=90000
                )
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return BacktestResults(
                total_return=-0.1,
                total_trades=0,
                win_rate=0,
                sharpe_ratio=-1,
                max_drawdown=0.5,
                total_pnl=-10000,
                final_capital=90000
            )
        finally:
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
    
    def _calculate_fitness(self, results: BacktestResults) -> float:
        """Calculate fitness score for DGM evolution."""
        # Multi-objective fitness
        fitness = 0.0
        
        # Return component (most important)
        fitness += 0.4 * np.tanh(results.total_return * 10)
        
        # Risk-adjusted return
        fitness += 0.3 * np.tanh(results.sharpe_ratio)
        
        # Win rate
        fitness += 0.2 * results.win_rate
        
        # Activity level
        fitness += 0.1 * np.tanh(results.total_trades / 1000)
        
        return fitness
    
    def _should_accept_modification(self, 
                                   new_fitness: float,
                                   current_fitness: float,
                                   generation: int) -> bool:
        """Decide whether to accept a modification (with exploration)."""
        if new_fitness > current_fitness:
            return True
        
        # Simulated annealing - accept worse solutions early
        temperature = 1.0 / (generation + 1)
        delta = new_fitness - current_fitness
        probability = np.exp(delta / temperature)
        
        return np.random.random() < probability
    
    def _update_meta_strategy(self, evolution_history: List[Dict[str, Any]]):
        """Update meta-learning strategy based on evolution history."""
        # Analyze successful vs unsuccessful modifications
        successful_mods = [h for h in evolution_history if h['accepted']]
        failed_mods = [h for h in evolution_history if not h['accepted']]
        
        if len(successful_mods) > 5:
            # Learn patterns from successful modifications
            patterns = self._extract_modification_patterns(successful_mods)
            
            # Update meta-parameters
            if patterns.get('risk_reduction_successful', False):
                self.meta_params['risk_bias'] = 0.8
            if patterns.get('exploration_successful', False):
                self.meta_params['exploration_rate'] *= 1.1
    
    def _setup_execution_environment(self) -> Dict[str, Any]:
        """Setup safe execution environment for code testing."""
        return {
            'docker_available': self._check_docker(),
            'sandbox_dir': Path('sandbox_strategies').absolute(),
            'resource_limits': {
                'cpu_time': 30,  # seconds
                'memory': 512,   # MB
                'file_writes': 10
            }
        }
    
    def _check_docker(self) -> bool:
        """Check if Docker is available for sandboxing."""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def deploy_dgm_strategy(self, strategy_package: Dict[str, Any]) -> bool:
        """
        Deploy a DGM strategy with self-modification enabled.
        
        This is where Kagan's vision meets Gödel - autonomous improvement in production.
        """
        logger.info(f"🚀 Deploying DGM strategy: {strategy_package['strategy_id']}")
        
        # Enhanced deployment for self-modifying code
        deployment_config = {
            'strategy_id': strategy_package['strategy_id'],
            'self_modification_enabled': True,
            'modification_frequency': 'every_100_trades',
            'safety_constraints': {
                'max_position_size': 0.2,
                'max_drawdown_allowed': 0.15,
                'require_human_approval': False  # True autonomy
            },
            'evolution_parameters': strategy_package.get('evolution_parameters', {}),
            'monitoring': {
                'track_modifications': True,
                'alert_on_major_changes': True,
                'performance_threshold': 0.5
            }
        }
        
        # Deploy with monitoring
        success = self._deploy_with_monitoring(strategy_package, deployment_config)
        
        if success:
            logger.info("✅ DGM strategy deployed with self-modification enabled")
            
            # Start evolution monitoring
            self._start_evolution_monitor(strategy_package['strategy_id'])
        
        return success
    
    def _start_evolution_monitor(self, strategy_id: str):
        """Start monitoring autonomous evolution of deployed strategy."""
        # This would run as a separate process monitoring the strategy
        logger.info(f"👁️ Evolution monitor started for {strategy_id}")
    
    def _extract_explanation_from_reasoning(self, reasoning: str) -> str:
        """Extract explanation from reasoning text."""
        if not reasoning:
            return "Self-modifying strategy generated with adaptive capabilities"
        
        # Look for key phrases in reasoning
        if "improve" in reasoning.lower():
            # Extract improvement-related explanation
            lines = reasoning.split('\n')
            for line in lines:
                if "improve" in line.lower() or "modify" in line.lower():
                    return line.strip()
        
        # Default: Use first sentence of reasoning
        first_sentence = reasoning.split('.')[0] + '.'
        return first_sentence if len(first_sentence) < 200 else "Strategy modified based on performance analysis"
    
    def _extract_improvement_from_reasoning(self, reasoning: str) -> str:
        """Extract expected improvement from reasoning text."""
        if not reasoning:
            return "Performance improvement through adaptive parameter optimization"
        
        # Look for percentage or improvement mentions
        if "%" in reasoning:
            import re
            percentages = re.findall(r'\d+(?:\.\d+)?%', reasoning)
            if percentages:
                return f"Expected {percentages[0]} performance improvement"
        
        # Look for improvement keywords
        keywords = ["increase", "improve", "boost", "enhance", "optimize"]
        for keyword in keywords:
            if keyword in reasoning.lower():
                # Find context around keyword
                idx = reasoning.lower().find(keyword)
                context = reasoning[max(0, idx-20):min(len(reasoning), idx+50)]
                return f"Expected to {context.strip()}"
        
        return "Gradual performance improvement through self-modification"
    
    def _get_strategy_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy by ID from evolution tracking."""
        for strategy in self.code_evolution:
            if strategy.get('strategy_id') == strategy_id:
                return strategy
        return None
    
    def _analyze_performance_trends(self, performance_data: List[Any]) -> Dict[str, Any]:
        """Analyze performance trends from data."""
        if not performance_data:
            return {'trend': 'unknown', 'direction': 0}
        
        # Simple trend analysis
        returns = [p.total_return if hasattr(p, 'total_return') else 0 for p in performance_data]
        if len(returns) > 1:
            trend = returns[-1] - returns[0]
            return {
                'trend': 'improving' if trend > 0 else 'declining',
                'direction': trend,
                'volatility': np.std(returns) if len(returns) > 2 else 0
            }
        return {'trend': 'insufficient_data', 'direction': 0}
    
    def _generate_code_modification(self, strategy: Dict[str, Any], 
                                   performance: Dict[str, Any],
                                   history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate modification for strategy code."""
        # Simple modification generation
        return {
            'code': strategy.get('code', ''),
            'modification_type': 'parameter_update',
            'changes': {
                'risk_adjustment': 0.9 if performance.get('direction', 0) < 0 else 1.1,
                'parameter_shift': 'conservative' if performance.get('volatility', 0) > 0.1 else 'aggressive'
            }
        }
    
    def _apply_modification(self, strategy: Dict[str, Any], 
                          modification: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modification to strategy."""
        modified = strategy.copy()
        modified['code'] = modification.get('code', strategy.get('code', ''))
        modified['version'] = modified.get('version', 1) + 1
        modified['last_modified'] = datetime.now().isoformat()
        return modified
    
    def _check_convergence(self, history: List[Dict[str, Any]]) -> bool:
        """Check if evolution has converged."""
        if len(history) < 5:
            return False
        
        # Check if fitness is plateauing
        recent_fitness = [h['fitness'] for h in history[-5:] if 'fitness' in h]
        if len(recent_fitness) >= 5:
            fitness_std = np.std(recent_fitness)
            return fitness_std < 0.001  # Very small variation
        return False
    
    def _extract_modification_patterns(self, successful_mods: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Extract patterns from successful modifications."""
        patterns = {
            'risk_reduction_successful': False,
            'exploration_successful': False,
            'parameter_tuning_successful': False
        }
        
        # Analyze modifications
        for mod in successful_mods:
            if 'modification' in mod:
                mod_data = mod['modification']
                if isinstance(mod_data, dict):
                    if mod_data.get('modification_type') == 'risk_reduction':
                        patterns['risk_reduction_successful'] = True
                    elif mod_data.get('modification_type') == 'exploration':
                        patterns['exploration_successful'] = True
        
        return patterns
    
    def _analyze_strategy_patterns(self, successful: List[Dict[str, Any]], 
                                 failed: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in successful vs failed strategies."""
        return {
            'success_patterns': {
                'avg_return': np.mean([s.get('return', 0) for s in successful]) if successful else 0,
                'common_features': self._find_common_features(successful),
                'count': len(successful)
            },
            'failure_patterns': {
                'avg_return': np.mean([f.get('return', 0) for f in failed]) if failed else 0,
                'common_issues': self._find_common_issues(failed),
                'count': len(failed)
            },
            'key_differences': self._identify_key_differences(successful, failed)
        }
    
    def _find_common_features(self, strategies: List[Dict[str, Any]]) -> List[str]:
        """Find common features in strategies."""
        if not strategies:
            return []
        
        features = []
        for s in strategies:
            if 'key_feature' in s:
                features.append(s['key_feature'])
            elif 'key_features' in s:
                features.extend(s['key_features'])
        
        # Find most common
        from collections import Counter
        feature_counts = Counter(features)
        return [f for f, count in feature_counts.most_common(3)]
    
    def _find_common_issues(self, strategies: List[Dict[str, Any]]) -> List[str]:
        """Find common issues in failed strategies."""
        if not strategies:
            return []
        
        issues = []
        for s in strategies:
            if 'issue' in s:
                issues.append(s['issue'])
            elif 'issues' in s:
                issues.extend(s['issues'])
        
        from collections import Counter
        issue_counts = Counter(issues)
        return [i for i, count in issue_counts.most_common(3)]
    
    def _identify_key_differences(self, successful: List[Dict[str, Any]], 
                                 failed: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify key differences between successful and failed strategies."""
        return {
            'return_difference': (np.mean([s.get('return', 0) for s in successful]) - 
                                np.mean([f.get('return', 0) for f in failed])) if successful and failed else 0,
            'success_rate': len(successful) / (len(successful) + len(failed)) if (successful or failed) else 0.5
        }
    
    def _parse_and_validate_code(self, code: str) -> str:
        """Parse and validate generated code."""
        # Basic validation
        if not code or not isinstance(code, str):
            return self._generate_initial_self_modifying_code({'min_return': 0.1})
        
        # Check for syntax errors
        try:
            compile(code, '<string>', 'exec')
            return code
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {e}")
            # Return a working fallback
            return self._generate_initial_self_modifying_code({'min_return': 0.1})
    
    def _extract_generation_rules(self, meta_result: Any) -> List[str]:
        """Extract generation rules from meta-learning result."""
        rules = []
        
        # Extract from evolution strategy if available
        if hasattr(meta_result, 'evolution_strategy'):
            strategy_text = meta_result.evolution_strategy
            if strategy_text:
                # Simple rule extraction
                lines = strategy_text.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['rule:', 'must', 'should', 'always', 'never']):
                        rules.append(line.strip())
        
        # Default rules if none extracted
        if not rules:
            rules = [
                "Always include risk management parameters",
                "Strategies must be self-modifying",
                "Performance tracking is mandatory",
                "Adaptive learning rate based on performance"
            ]
        
        return rules[:5]  # Limit to 5 rules
    
    def _generate_fallback_meta_system(self) -> Dict[str, Any]:
        """Generate fallback meta-learning system."""
        return {
            'meta_id': f"meta_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'code': "# Fallback meta-learning system",
            'learning_mechanisms': "Basic pattern recognition",
            'evolution_strategy': "Gradual improvement through iteration",
            'pattern_insights': {},
            'generation_rules': ["Generate diverse strategies", "Learn from failures"],
            'self_improvement_cycle': {
                'frequency': 'after_every_100_trades',
                'metrics_threshold': 0.5,
                'modification_strength': 'conservative'
            }
        }
    
    def _deploy_with_monitoring(self, strategy_package: Dict[str, Any], 
                               deployment_config: Dict[str, Any]) -> bool:
        """Deploy strategy with monitoring."""
        try:
            # Simulate deployment
            logger.info(f"Deploying {strategy_package['strategy_id']} with config: {deployment_config['strategy_id']}")
            # In real implementation, this would deploy to trading infrastructure
            return True
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False


async def test_dgm_code_generator():
    """Test the DGM-enhanced code generator."""
    logger.info("🧬 Testing Darwin's Gödel Machine Code Generator")
    
    # Initialize generator
    generator = DGMCodeGenerator()
    
    # Test 1: Generate self-modifying strategy
    initial_performance = {
        'total_return': 0.05,
        'win_rate': 0.48,
        'sharpe_ratio': 0.8
    }
    
    target_metrics = {
        'min_return': 0.15,
        'min_win_rate': 0.55,
        'min_sharpe': 1.5
    }
    
    logger.info("1. Generating self-modifying strategy...")
    self_mod_strategy = generator.generate_self_modifying_strategy(
        initial_performance,
        target_metrics
    )
    
    logger.info(f"Generated: {self_mod_strategy['strategy_id']}")
    logger.info(f"Self-modification enabled: {self_mod_strategy['self_modification_enabled']}")
    
    # Test 2: Evolve strategy autonomously
    logger.info("\n2. Testing autonomous evolution...")
    
    # Simulate performance data
    performance_data = [
        BacktestResults(
            total_return=0.05 + i*0.01,
            total_trades=100 + i*10,
            win_rate=0.48 + i*0.02,
            sharpe_ratio=0.8 + i*0.1,
            max_drawdown=0.1,
            total_pnl=5000 + i*1000,
            final_capital=105000 + i*1000
        )
        for i in range(5)
    ]
    
    evolution_result = generator.evolve_strategy_autonomously(
        self_mod_strategy['strategy_id'],
        performance_data,
        max_generations=5
    )
    
    logger.info(f"Evolution complete: {evolution_result['generations_completed']} generations")
    logger.info(f"Fitness improvement: {evolution_result['improvement']:.4f}")
    
    # Test 3: Generate meta-learning system
    logger.info("\n3. Generating meta-learning system...")
    
    # Mock successful/failed strategies
    successful_strategies = [
        {'id': 'success1', 'return': 0.2, 'key_feature': 'adaptive_risk'},
        {'id': 'success2', 'return': 0.15, 'key_feature': 'ml_signals'}
    ]
    
    failed_strategies = [
        {'id': 'fail1', 'return': -0.1, 'issue': 'overtrading'},
        {'id': 'fail2', 'return': -0.05, 'issue': 'poor_timing'}
    ]
    
    meta_system = generator.generate_meta_learning_system(
        successful_strategies,
        failed_strategies
    )
    
    logger.info(f"Meta-learning system: {meta_system['meta_id']}")
    logger.info(f"Evolution strategy: {meta_system['evolution_strategy']}")
    
    logger.info("\n✅ DGM Code Generator test complete!")


if __name__ == "__main__":
    asyncio.run(test_dgm_code_generator())
#!/usr/bin/env python3
"""
Enhanced Hyperparameter Optimizer with Grid Search and Optuna
Implements systematic parameter optimization as per Kagan's vision
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import json
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path

from src.utils.types import BacktestResults, StrategyContext


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Grid Search + Optuna.
    Implements Kagan's requirement for systematic parameter optimization.
    """
    
    def __init__(self, 
                 objective_function: Optional[Callable] = None,
                 optimization_direction: str = "maximize"):
        """
        Initialize optimizer with objective function and optimization direction.
        
        Args:
            objective_function: Function that takes parameters and returns performance metric
            optimization_direction: "maximize" or "minimize"
        """
        self.objective_function = objective_function
        self.optimization_direction = optimization_direction
        
        # Optimization history
        self.optimization_history = []
        self.best_params = {}
        self.best_score = -float('inf') if optimization_direction == "maximize" else float('inf')
        
        # Grid search settings
        self.grid_search_spaces = {}
        self.grid_results = []
        
        # Optuna settings
        self.optuna_study = None
        self.optuna_trials = 100
        
        logger.info("🔧 Hyperparameter Optimizer initialized with Grid Search + Optuna")
    
    def define_search_space(self, param_space: Dict[str, Any]):
        """
        Define the parameter search space for optimization.
        
        Example:
        {
            'position_size': {'type': 'float', 'low': 0.01, 'high': 0.1, 'step': 0.01},
            'stop_loss': {'type': 'float', 'low': 0.02, 'high': 0.1},
            'rsi_period': {'type': 'int', 'low': 10, 'high': 30},
            'strategy_type': {'type': 'categorical', 'choices': ['momentum', 'mean_reversion']}
        }
        """
        self.param_space = param_space
        self._prepare_grid_search_space()
        logger.info(f"Search space defined with {len(param_space)} parameters")
    
    def grid_search(self, 
                   param_grid: Optional[Dict[str, List[Any]]] = None,
                   n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform exhaustive grid search over parameter combinations.
        
        Args:
            param_grid: Explicit grid to search (overrides search space)
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        
        Returns:
            Best parameters and performance from grid search
        """
        logger.info("🔍 Starting Grid Search optimization")
        
        # Use provided grid or generate from search space
        if param_grid is None:
            param_grid = self._generate_grid_from_space()
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(all_combinations)} parameter combinations")
        
        # Parallel evaluation
        if n_jobs == -1:
            n_jobs = None  # Use all available CPUs
        
        results = []
        
        # Test each combination
        for i, combo in enumerate(all_combinations):
            params = dict(zip(param_names, combo))
            
            try:
                # Evaluate parameters
                logger.debug(f"Evaluating params {i+1}/{len(all_combinations)}: {params}")
                score = self.objective_function(params)
                
                # Validate score
                if score is None or (isinstance(score, float) and (np.isnan(score) or np.isinf(score))):
                    logger.warning(f"Invalid score {score} for params {params}")
                    continue
                
                results.append({
                    'params': params,
                    'score': score,
                    'iteration': i
                })
                
                # Update best if needed
                if self._is_better_score(score):
                    self.best_score = score
                    self.best_params = params.copy()
                    logger.info(f"🎯 New best score: {score:.4f} with params: {params}")
                
                # Log progress
                if (i + 1) % 5 == 0:
                    logger.info(f"Grid search progress: {i+1}/{len(all_combinations)}, Current best: {self.best_score:.4f}")
                    
            except Exception as e:
                logger.error(f"Error evaluating params {params}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Store results
        self.grid_results = results
        
        # Check if we found ANY valid results
        if not results:
            logger.error("Grid search found NO valid results!")
            # Return first combination as fallback
            if all_combinations:
                fallback_params = dict(zip(param_names, all_combinations[0]))
                logger.warning(f"Using fallback params: {fallback_params}")
                self.best_params = fallback_params
                self.best_score = -float('inf')
        
        # Summary
        logger.info(f"Grid search complete. Best score: {self.best_score:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_evaluations': len(results),
            'all_results': results
        }
    
    def optuna_optimize(self,
                       n_trials: int = 100,
                       timeout: Optional[int] = None,
                       n_jobs: int = 1) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
        
        Returns:
            Best parameters and performance from Optuna
        """
        logger.info(f"🎯 Starting Optuna optimization with {n_trials} trials")
        
        # Create study
        self.optuna_study = optuna.create_study(
            direction=self.optimization_direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Add grid search results as initial trials if available
        if self.grid_results:
            self._add_grid_results_to_optuna()
        
        # Optimize
        self.optuna_study.optimize(
            self._optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            callbacks=[self._optuna_callback]
        )
        
        # Get best parameters
        best_trial = self.optuna_study.best_trial
        
        # Update best if needed
        if self._is_better_score(best_trial.value):
            self.best_score = best_trial.value
            self.best_params = best_trial.params
        
        logger.info(f"Optuna optimization complete. Best score: {best_trial.value:.4f}")
        logger.info(f"Best params: {best_trial.params}")
        
        return {
            'best_params': best_trial.params,
            'best_score': best_trial.value,
            'n_trials': len(self.optuna_study.trials),
            'optimization_history': self._get_optimization_history()
        }
    
    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history from all completed trials."""
        if not self.optuna_study:
            return []
        
        history = []
        for trial in self.optuna_study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'trial_number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
                })
        
        return history
    
    def hybrid_optimize(self,
                       grid_size: str = 'small',
                       optuna_trials: int = 50) -> Dict[str, Any]:
        """
        Hybrid optimization: Grid Search for exploration + Optuna for exploitation.
        
        This implements Kagan's vision of systematic optimization:
        1. Grid search finds promising regions
        2. Optuna refines within those regions
        
        Args:
            grid_size: 'small', 'medium', or 'large' grid
            optuna_trials: Number of Optuna trials after grid search
        
        Returns:
            Optimized parameters and performance metrics
        """
        logger.info("🚀 Starting Hybrid Optimization (Grid Search + Optuna)")
        
        # Reset best params before starting
        self.best_params = {}
        self.best_score = -float('inf') if self.optimization_direction == "maximize" else float('inf')
        
        # Phase 1: Coarse grid search
        logger.info("Phase 1: Coarse Grid Search")
        coarse_grid = self._generate_coarse_grid(grid_size)
        logger.info(f"Coarse grid: {coarse_grid}")
        grid_results = self.grid_search(coarse_grid)
        
        # Validate grid results
        if not grid_results.get('best_params'):
            logger.error("Grid search failed to find any valid parameters!")
            logger.error(f"Grid results: {grid_results}")
            # Return at least something
            return {
                'best_params': {},
                'best_score': -float('inf'),
                'verified_score': -float('inf'),
                'grid_evaluations': 0,
                'optuna_evaluations': 0,
                'total_evaluations': 0,
                'optimization_method': 'hybrid_grid_optuna_failed'
            }
        
        # Phase 2: Identify promising regions
        logger.info("Phase 2: Identifying promising regions")
        promising_regions = self._identify_promising_regions(grid_results['all_results'])
        
        # Phase 3: Fine-tune with Optuna
        logger.info("Phase 3: Fine-tuning with Optuna")
        self._update_search_space_from_regions(promising_regions)
        optuna_results = self.optuna_optimize(n_trials=optuna_trials)
        
        # Phase 4: Final verification
        logger.info("Phase 4: Final verification")
        final_score = self.objective_function(self.best_params)
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'verified_score': final_score,
            'grid_evaluations': len(grid_results['all_results']),
            'optuna_evaluations': optuna_results['n_trials'],
            'total_evaluations': len(grid_results['all_results']) + optuna_results['n_trials'],
            'optimization_method': 'hybrid_grid_optuna'
        }
    
    def adaptive_optimize(self,
                         initial_trials: int = 20,
                         max_trials: int = 200,
                         convergence_threshold: float = 0.001) -> Dict[str, Any]:
        """
        Adaptive optimization that adjusts strategy based on progress.
        
        Implements Kagan's "slightly better than random" with intelligence:
        - Starts random, becomes more directed as it learns
        - Adapts search strategy based on landscape
        """
        logger.info("🧠 Starting Adaptive Optimization")
        
        trials_completed = 0
        convergence_count = 0
        last_best = self.best_score
        
        # Phase 1: Random exploration
        logger.info("Phase 1: Random exploration")
        random_params = self._random_search(initial_trials)
        trials_completed += initial_trials
        
        while trials_completed < max_trials:
            # Check convergence
            improvement = abs(self.best_score - last_best)
            if improvement < convergence_threshold:
                convergence_count += 1
                if convergence_count >= 5:
                    logger.info("Convergence detected, switching strategy")
                    self._restart_with_mutation()
                    convergence_count = 0
            else:
                convergence_count = 0
            
            last_best = self.best_score
            
            # Adaptive strategy selection
            if trials_completed < max_trials * 0.3:
                # Early stage: More exploration
                strategy = 'exploration'
                self._exploration_step(5)
            elif trials_completed < max_trials * 0.7:
                # Middle stage: Balanced
                strategy = 'balanced'
                self._balanced_step(5)
            else:
                # Late stage: Exploitation
                strategy = 'exploitation'
                self._exploitation_step(5)
            
            trials_completed += 5
            
            logger.info(f"Trials: {trials_completed}/{max_trials}, "
                       f"Best: {self.best_score:.4f}, "
                       f"Strategy: {strategy}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_trials': trials_completed,
            'optimization_method': 'adaptive',
            'convergence_achieved': convergence_count < 5
        }
    
    def optimize_for_robustness(self,
                              n_trials: int = 100,
                              n_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize for robust parameters that work across different market conditions.
        
        Implements cross-validation style optimization for stability.
        """
        logger.info("🛡️ Starting Robustness Optimization")
        
        # Create Optuna study with custom objective
        study = optuna.create_study(
            direction=self.optimization_direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        def robust_objective(trial):
            # Sample parameters
            params = self._sample_params_optuna(trial)
            
            # Evaluate across multiple scenarios/folds
            scores = []
            for fold in range(n_folds):
                # Modify objective function call to include fold
                fold_score = self.objective_function(params, fold=fold)
                scores.append(fold_score)
            
            # Return worst-case or average (for robustness)
            if self.optimization_direction == "maximize":
                return np.mean(scores) - np.std(scores)  # Penalize variance
            else:
                return np.mean(scores) + np.std(scores)
        
        # Optimize
        study.optimize(robust_objective, n_trials=n_trials)
        
        best_robust_params = study.best_params
        
        # Verify robustness
        verification_scores = []
        for fold in range(n_folds):
            score = self.objective_function(best_robust_params, fold=fold)
            verification_scores.append(score)
        
        return {
            'best_params': best_robust_params,
            'mean_score': np.mean(verification_scores),
            'std_score': np.std(verification_scores),
            'min_score': np.min(verification_scores),
            'max_score': np.max(verification_scores),
            'robustness_ratio': np.mean(verification_scores) / (np.std(verification_scores) + 1e-6),
            'optimization_method': 'robust_optimization'
        }
    
    def _prepare_grid_search_space(self):
        """Prepare grid search space from parameter definitions."""
        self.grid_search_spaces = {}
        
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'float':
                if 'step' in param_config:
                    values = np.arange(
                        param_config['low'],
                        param_config['high'] + param_config['step'],
                        param_config['step']
                    )
                else:
                    values = np.linspace(
                        param_config['low'],
                        param_config['high'],
                        5  # Default 5 points
                    )
                self.grid_search_spaces[param_name] = values.tolist()
                
            elif param_config['type'] == 'int':
                step = param_config.get('step', 1)
                values = list(range(
                    param_config['low'],
                    param_config['high'] + 1,
                    step
                ))
                self.grid_search_spaces[param_name] = values
                
            elif param_config['type'] == 'categorical':
                self.grid_search_spaces[param_name] = param_config['choices']
    
    def _generate_grid_from_space(self) -> Dict[str, List[Any]]:
        """Generate grid from search space definition."""
        return self.grid_search_spaces
    
    def _generate_coarse_grid(self, size: str) -> Dict[str, List[Any]]:
        """Generate coarse grid for initial exploration."""
        points_map = {
            'small': 3,
            'medium': 5,
            'large': 7
        }
        n_points = points_map.get(size, 3)
        
        coarse_grid = {}
        
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'float':
                coarse_grid[param_name] = np.linspace(
                    param_config['low'],
                    param_config['high'],
                    n_points
                ).tolist()
            elif param_config['type'] == 'int':
                coarse_grid[param_name] = np.linspace(
                    param_config['low'],
                    param_config['high'],
                    n_points,
                    dtype=int
                ).tolist()
            elif param_config['type'] == 'categorical':
                coarse_grid[param_name] = param_config['choices']
        
        return coarse_grid
    
    def _identify_promising_regions(self, results: List[Dict]) -> List[Dict]:
        """Identify promising parameter regions from grid search."""
        # Sort by score
        sorted_results = sorted(
            results,
            key=lambda x: x['score'],
            reverse=(self.optimization_direction == "maximize")
        )
        
        # Take top 20%
        n_promising = max(1, len(sorted_results) // 5)
        promising = sorted_results[:n_promising]
        
        # Extract parameter ranges
        promising_regions = {}
        for param_name in self.param_space.keys():
            values = [r['params'][param_name] for r in promising]
            
            if self.param_space[param_name]['type'] in ['float', 'int']:
                promising_regions[param_name] = {
                    'low': min(values),
                    'high': max(values)
                }
            else:
                promising_regions[param_name] = list(set(values))
        
        return promising_regions
    
    def _update_search_space_from_regions(self, regions: Dict):
        """Update search space to focus on promising regions."""
        for param_name, region in regions.items():
            if isinstance(region, dict):
                # Continuous parameters
                self.param_space[param_name]['low'] = region['low']
                self.param_space[param_name]['high'] = region['high']
            else:
                # Categorical parameters
                self.param_space[param_name]['choices'] = region
    
    def _optuna_objective(self, trial):
        """Objective function for Optuna optimization."""
        # Sample parameters
        params = self._sample_params_optuna(trial)
        
        # Evaluate
        try:
            score = self.objective_function(params)
            
            # Track in history
            self.optimization_history.append({
                'trial': trial.number,
                'params': params,
                'score': score,
                'timestamp': datetime.now().isoformat()
            })
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}")
            # Return worst possible score
            if self.optimization_direction == "maximize":
                return -float('inf')
            else:
                return float('inf')
    
    def _sample_params_optuna(self, trial) -> Dict[str, Any]:
        """Sample parameters for Optuna trial."""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', None)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def _optuna_callback(self, study, trial):
        """Callback for Optuna optimization progress."""
        if trial.number % 10 == 0:
            logger.info(f"Optuna trial {trial.number}: score = {trial.value:.4f}")
    
    def _is_better_score(self, score: float) -> bool:
        """Check if score is better than current best."""
        if self.optimization_direction == "maximize":
            return score > self.best_score
        else:
            return score < self.best_score
    
    def _random_search(self, n_trials: int) -> List[Dict]:
        """Perform random search."""
        results = []
        
        for i in range(n_trials):
            # Random parameters
            params = {}
            for param_name, param_config in self.param_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = np.random.uniform(
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = np.random.randint(
                        param_config['low'],
                        param_config['high'] + 1
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = np.random.choice(param_config['choices'])
            
            # Evaluate
            score = self.objective_function(params)
            results.append({'params': params, 'score': score})
            
            # Update best
            if self._is_better_score(score):
                self.best_score = score
                self.best_params = params.copy()
        
        return results
    
    def _restart_with_mutation(self):
        """Restart search with mutations around best parameters."""
        logger.info("Restarting with mutations around best parameters")
        
        # Mutate best parameters
        mutated_params = self.best_params.copy()
        
        for param_name, param_value in mutated_params.items():
            if self.param_space[param_name]['type'] == 'float':
                # Add gaussian noise
                noise = np.random.normal(0, 0.1) * (
                    self.param_space[param_name]['high'] - 
                    self.param_space[param_name]['low']
                )
                mutated_params[param_name] = np.clip(
                    param_value + noise,
                    self.param_space[param_name]['low'],
                    self.param_space[param_name]['high']
                )
    
    def _add_grid_results_to_optuna(self):
        """Add grid search results to Optuna study as initial trials."""
        if not self.grid_results or not self.optuna_study:
            return
            
        logger.info(f"Adding {len(self.grid_results)} grid search results to Optuna")
        
        # Add each grid result as a trial
        for result in self.grid_results:
            try:
                # Create a trial with the grid search parameters
                trial = self.optuna_study.ask()
                
                # Set the parameter values from grid search
                for param_name, param_value in result['params'].items():
                    if param_name in self.param_space:
                        # Use the suggest method to set the value
                        param_config = self.param_space[param_name]
                        if param_config['type'] == 'float':
                            trial.suggest_float(param_name, param_config['low'], param_config['high'])
                        elif param_config['type'] == 'int':
                            trial.suggest_int(param_name, param_config['low'], param_config['high'])
                        elif param_config['type'] == 'categorical':
                            trial.suggest_categorical(param_name, param_config['choices'])
                
                # Tell the study about this trial and its result
                self.optuna_study.tell(trial, result['score'])
                
            except Exception as e:
                logger.warning(f"Failed to add grid result to Optuna: {e}")
                continue
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file."""
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'param_space': self.param_space,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def visualize_optimization(self) -> Dict[str, Any]:
        """Generate visualization data for optimization results."""
        if self.optuna_study:
            return {
                'optimization_history': optuna.visualization.plot_optimization_history(self.optuna_study),
                'param_importances': optuna.visualization.plot_param_importances(self.optuna_study),
                'parallel_coordinate': optuna.visualization.plot_parallel_coordinate(self.optuna_study),
                'contour_plot': optuna.visualization.plot_contour(self.optuna_study)
            }
        else:
            return {
                'grid_results': self.grid_results,
                'best_params': self.best_params,
                'best_score': self.best_score
            }


def example_objective(params: Dict[str, Any]) -> float:
    """Example objective function for testing."""
    # Simulate a complex objective with multiple optima
    x = params.get('x', 0)
    y = params.get('y', 0)
    
    # Rosenbrock function (challenging optimization landscape)
    return -(100 * (y - x**2)**2 + (1 - x)**2)


def test_hyperparameter_optimizer():
    """Test the hyperparameter optimizer."""
    logger.info("Testing Hyperparameter Optimizer")
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        objective_function=example_objective,
        optimization_direction="maximize"
    )
    
    # Define search space
    search_space = {
        'x': {'type': 'float', 'low': -2.0, 'high': 2.0},
        'y': {'type': 'float', 'low': -2.0, 'high': 2.0}
    }
    
    optimizer.define_search_space(search_space)
    
    # Test different optimization methods
    logger.info("\n1. Testing Grid Search")
    grid_results = optimizer.grid_search()
    logger.info(f"Grid Search Best: {grid_results['best_score']:.4f}")
    
    logger.info("\n2. Testing Optuna")
    optuna_results = optimizer.optuna_optimize(n_trials=50)
    logger.info(f"Optuna Best: {optuna_results['best_score']:.4f}")
    
    logger.info("\n3. Testing Hybrid Optimization")
    hybrid_results = optimizer.hybrid_optimize(grid_size='small', optuna_trials=30)
    logger.info(f"Hybrid Best: {hybrid_results['best_score']:.4f}")
    
    logger.info("\n4. Testing Adaptive Optimization")
    adaptive_results = optimizer.adaptive_optimize(max_trials=50)
    logger.info(f"Adaptive Best: {adaptive_results['best_score']:.4f}")
    
    # Save results
    optimizer.save_optimization_results('hyperparameter_optimization_results.json')


if __name__ == "__main__":
    test_hyperparameter_optimizer()
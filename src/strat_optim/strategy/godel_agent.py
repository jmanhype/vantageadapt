"""Gödel Agent for self-improving trading strategies."""

import os
import json
import logging
import ast
import black
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import importlib.util
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from research.strategy.models import MarketRegime
import yaml
import openai

logger = logging.getLogger(__name__)

@dataclass
class GridSearchResult:
    """Represents a grid search result with performance metrics."""
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    market_regime: str
    timestamp: datetime
    score: float

@dataclass
class ParameterVersion:
    """Represents a version of strategy parameters with performance metrics."""
    
    version_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    market_regime: str
    score: float
    timestamp: datetime
    parent_version: Optional[str] = None
    
    @property
    def key_metrics(self) -> Dict[str, float]:
        """Return key performance metrics."""
        return {
            'total_return': self.metrics.get('total_return', 0),
            'sortino_ratio': self.metrics.get('sortino_ratio', 0),
            'win_rate': self.metrics.get('win_rate', 0),
            'avg_trade_pnl': self.metrics.get('avg_trade_pnl', 0)
        }

@dataclass
class StrategyImprovement:
    """Represents a proposed strategy improvement."""
    description: str
    code_changes: List[Dict[str, Any]]
    expected_impact: str
    evaluation_score: float = 0.0
    feedback: List[str] = None

@dataclass
class StrategyEvaluation:
    """Evaluation results for a strategy configuration."""
    
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    market_regime: str
    score: float
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]

@dataclass
class ParameterHistory:
    """Class to track parameter performance history."""
    
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    market_regime: str
    timestamp: datetime
    success: bool
    evaluation: Optional[StrategyEvaluation] = None

class GodelAgent:
    """Agent for self-improving trading strategies."""

    def __init__(self, improvement_threshold: float = 0.1, max_iterations: int = 5,
                 backup_dir: str = "backups", prompt_dir: str = "prompts/trading",
                 max_parallel_tests: int = 4):
        """Initialize the agent."""
        self.improvement_threshold = improvement_threshold
        self.max_iterations = max_iterations
        self.backup_dir = backup_dir
        self.prompt_dir = prompt_dir
        self.max_parallel_tests = max_parallel_tests
        self.last_metrics = None
        self.best_metrics = None
        self.parameter_history: List[ParameterHistory] = []
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.grid_search_history: Dict[str, List[GridSearchResult]] = {}
        
        # Version control for parameters
        self.parameter_versions: List[ParameterVersion] = []
        self.current_version: Optional[str] = None
        self.best_version: Optional[str] = None
        
        # Create backup directory if it doesn't exist
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

    def _generate_version_id(self) -> str:
        """Generate a unique version ID based on timestamp."""
        return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def _add_parameter_version(self, parameters: Dict[str, Any], metrics: Dict[str, float],
                             market_regime: str) -> str:
        """Add a new parameter version and return its ID."""
        version_id = self._generate_version_id()
        score = self._calculate_strategy_score(metrics)
        
        version = ParameterVersion(
            version_id=version_id,
            parameters=parameters.copy(),
            metrics=metrics.copy(),
            market_regime=market_regime,
            score=score,
            timestamp=datetime.now(),
            parent_version=self.current_version
        )
        
        self.parameter_versions.append(version)
        self.current_version = version_id
        
        # Update best version if this is the best so far
        if not self.best_version or score > self._get_version_by_id(self.best_version).score:
            self.best_version = version_id
            
        return version_id
        
    def _get_version_by_id(self, version_id: str) -> Optional[ParameterVersion]:
        """Retrieve a parameter version by its ID."""
        for version in self.parameter_versions:
            if version.version_id == version_id:
                return version
        return None
        
    def _get_best_versions(self, market_regime: str, n: int = 5) -> List[ParameterVersion]:
        """Get the top N performing versions for a specific market regime."""
        regime_versions = [v for v in self.parameter_versions if v.market_regime == market_regime]
        return sorted(regime_versions, key=lambda x: x.score, reverse=True)[:n]
        
    def evaluate_strategy(self, metrics: Dict[str, float], parameters: Dict[str, Any],
                         market_regime: str) -> StrategyEvaluation:
        """Evaluate strategy performance and provide detailed analysis."""
        score = self._calculate_strategy_score(metrics)
        strengths = []
        weaknesses = []
        suggestions = []
        
        # Analyze metrics
        if metrics.get('total_return', 0) > 0:
            strengths.append("Positive total return")
        else:
            weaknesses.append("Negative total return")
            suggestions.append("Consider adjusting entry/exit thresholds")
            
        if metrics.get('sortino_ratio', 0) > 1.0:
            strengths.append("Good risk-adjusted returns")
        else:
            weaknesses.append("Poor risk-adjusted returns")
            suggestions.append("Review risk management parameters")
            
        if metrics.get('win_rate', 0) > 0.5:
            strengths.append("Above average win rate")
        else:
            weaknesses.append("Below average win rate")
            suggestions.append("Analyze trade entry conditions")
            
        if metrics.get('max_drawdown', 0) < -0.2:
            weaknesses.append("Large drawdown")
            suggestions.append("Implement stricter stop-loss rules")
            
        return StrategyEvaluation(
            parameters=parameters,
            metrics=metrics,
            market_regime=market_regime,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions
        )

    def _calculate_strategy_score(self, metrics: Dict[str, float]) -> float:
        """Calculate an overall strategy score based on multiple metrics."""
        weights = {
            'total_return': 0.3,
            'sortino_ratio': 0.2,
            'win_rate': 0.2,
            'max_drawdown': 0.15,
            'profit_factor': 0.15
        }
        
        score = 0.0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            if metric == 'max_drawdown':
                # Convert drawdown to positive score (less negative is better)
                value = max(0, 1 + value)
            score += value * weight
            
        return max(0, min(1, score))  # Normalize between 0 and 1

    def parallel_parameter_test(self, parameter_sets: List[Dict[str, Any]], 
                              market_regimes: Set[str]) -> List[ParameterHistory]:
        """Test multiple parameter sets across different market regimes in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_tests) as executor:
            futures = []
            for params in parameter_sets:
                for regime in market_regimes:
                    futures.append(
                        executor.submit(self._test_parameters, params, regime)
                    )
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel parameter test: {str(e)}")
                    
        return results

    def _test_parameters(self, parameters: Dict[str, Any], 
                        market_regime: str) -> Optional[ParameterHistory]:
        """Test a single parameter set in a specific market regime."""
        try:
            # This would be implemented by the strategy runner
            metrics = self._run_strategy_test(parameters, market_regime)
            
            if metrics:
                evaluation = self.evaluate_strategy(metrics, parameters, market_regime)
                return ParameterHistory(
                    parameters=parameters,
                    metrics=metrics,
                    market_regime=market_regime,
                    timestamp=datetime.now(),
                    success=evaluation.score > 0.6,
                    evaluation=evaluation
                )
        except Exception as e:
            logger.error(f"Error testing parameters: {str(e)}")
        return None

    def _run_strategy_test(self, parameters: Dict[str, Any], 
                          market_regime: str) -> Optional[Dict[str, float]]:
        """Placeholder for strategy testing implementation."""
        # This would be implemented by the concrete strategy runner
        return None

    def read_module_code(self, module_path: str) -> Optional[str]:
        """Read module code from file."""
        try:
            spec = importlib.util.spec_from_file_location("module", module_path)
            if spec and spec.origin:
                with open(spec.origin, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read module {module_path}: {str(e)}")
        return None 

    def track_performance(self, metrics: Dict[str, Any], parameters: Dict[str, Any], 
                         market_regime: str) -> bool:
        """Track performance metrics and determine if improvement occurred."""
        if not metrics:
            return False
            
        # Add new version
        version_id = self._add_parameter_version(parameters, metrics, market_regime)
        version = self._get_version_by_id(version_id)
        
        if not self.last_metrics:
            self.last_metrics = metrics
            self.best_metrics = metrics
            success = True
        else:
            # Calculate improvement using version scores
            last_version = self._get_version_by_id(self.current_version)
            improvement = (version.score - last_version.score) / max(0.1, abs(last_version.score))
            success = improvement >= self.improvement_threshold
            self.last_metrics = metrics

        # Record parameter history
        history_entry = ParameterHistory(
            parameters=parameters,
            metrics=metrics,
            market_regime=market_regime,
            timestamp=datetime.now(),
            success=success,
            evaluation=self.evaluate_strategy(metrics, parameters, market_regime)
        )
        self.parameter_history.append(history_entry)
        
        # Update success patterns if performance was good
        if self._meets_success_criteria(metrics):
            if market_regime not in self.success_patterns:
                self.success_patterns[market_regime] = []
            
            # Store parameters with their performance metrics
            success_entry = {
                'parameters': parameters,
                'metrics': metrics,
                'score': version.score
            }
            self.success_patterns[market_regime].append(success_entry)
            
            # Keep only top 5 performing parameter sets
            self.success_patterns[market_regime].sort(key=lambda x: x['score'], reverse=True)
            self.success_patterns[market_regime] = self.success_patterns[market_regime][:5]
            
        return success

    def _meets_success_criteria(self, metrics: Dict[str, float]) -> bool:
        """Determine if a set of metrics meets success criteria."""
        # Calculate weighted score instead of strict thresholds
        weights = {
            'total_return': 0.3,
            'sortino_ratio': 0.25,
            'win_rate': 0.25,
            'total_trades': 0.2
        }
        
        score = 0
        score += weights['total_return'] * (1 if metrics.get('total_return', 0) > 0 else 0)
        score += weights['sortino_ratio'] * (metrics.get('sortino_ratio', 0) / 2)  # Normalized to ~0-1
        score += weights['win_rate'] * (metrics.get('win_rate', 0))
        score += weights['total_trades'] * min(1.0, metrics.get('total_trades', 0) / 100)
        
        return score > 0.6  # Success if weighted score > 60%

    def get_successful_parameters(self, market_regime: str) -> Optional[Dict[str, Any]]:
        """Get successful parameters for a specific market regime."""
        best_versions = self._get_best_versions(market_regime, n=1)
        if best_versions:
            return best_versions[0].parameters
        return None

    def analyze_parameter_trends(self) -> Dict[str, Dict[str, Any]]:
        """Analyze trends in parameter performance."""
        if not self.parameter_history:
            return {}
            
        trends = {}
        for entry in self.parameter_history:
            if entry.success:
                for param_name, param_value in entry.parameters.items():
                    if isinstance(param_value, (int, float)):
                        if param_name not in trends:
                            trends[param_name] = {
                                'values': [],
                                'metrics': [],
                                'market_regimes': []
                            }
                        trends[param_name]['values'].append(param_value)
                        trends[param_name]['metrics'].append(entry.metrics)
                        trends[param_name]['market_regimes'].append(entry.market_regime)
                        
        # Calculate correlations and optimal ranges
        for param_name in trends:
            values = np.array(trends[param_name]['values'])
            returns = np.array([m.get('total_return', 0) for m in trends[param_name]['metrics']])
            
            if len(values) > 1:
                # Calculate correlation with returns
                correlation = np.corrcoef(values, returns)[0, 1]
                
                # Find optimal range based on top performing parameters
                sorted_indices = np.argsort(returns)[-5:]  # Top 5 performing values
                optimal_range = {
                    'min': float(np.min(values[sorted_indices])),
                    'max': float(np.max(values[sorted_indices])),
                    'correlation': float(correlation)
                }
                trends[param_name]['optimal_range'] = optimal_range
                
        return trends

    def suggest_parameter_improvements(self, current_params: Dict[str, Any], 
                                     market_regime: str) -> Optional[Dict[str, Any]]:
        """Suggest parameter improvements based on historical performance."""
        if not self.parameter_history:
            return None
            
        # Get successful parameters for the current market regime
        regime_params = self.get_successful_parameters(market_regime)
        if regime_params:
            return regime_params
            
        # Analyze parameter trends
        trends = self.analyze_parameter_trends()
        if not trends:
            return None
            
        # Suggest improvements based on successful parameter trends
        improvements = {}
        for param_name, values in trends.items():
            if param_name in current_params and values:
                # Calculate the trend direction using recent values
                recent_values = values[-min(5, len(values)):]
                if len(recent_values) > 1:
                    trend_direction = np.mean(np.diff(recent_values))
                    current_value = current_params[param_name]
                    
                    # Suggest adjustment based on trend
                    if isinstance(current_value, (int, float)):
                        adjustment = 0.1 * abs(current_value) * (1 if trend_direction > 0 else -1)
                        improvements[param_name] = current_value + adjustment
                        
        return improvements if improvements else None

    def propose_patch(self, module_code: str, metrics: Dict[str, Any], context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Generate code improvements based on performance metrics."""
        try:
            prompt = self._create_improvement_prompt(module_code, metrics, context)
            suggestions = self._get_llm_suggestions(prompt)
            
            if not suggestions or 'improvements' not in suggestions:
                return None
                
            valid_improvements = []
            for improvement in suggestions['improvements']:
                if self._validate_improvement_suggestion(improvement):
                    # Pre-validate each code change
                    temp_code = module_code
                    valid_changes = True
                    
                    for change in improvement['code_changes']:
                        if change['type'] == 'add':
                            temp_code = self._insert_code(temp_code, change['location'], change['code'])
                        elif change['type'] == 'modify':
                            temp_code = self._modify_code(temp_code, change['location'], change['code'])
                        elif change['type'] == 'delete':
                            temp_code = self._delete_code(temp_code, change['location'])
                            
                        # Verify the resulting code is still valid
                        try:
                            formatted_code = self.format_code(temp_code)
                            if not self.validate_code_changes(module_code, formatted_code):
                                valid_changes = False
                                break
                        except Exception:
                            valid_changes = False
                            break
                            
                    if valid_changes:
                        valid_improvements.append(improvement)
                        
            return valid_improvements if valid_improvements else None
            
        except Exception as e:
            logger.error(f"Error proposing patch: {str(e)}")
            return None

    def revert_last_change(self, module_path: str) -> bool:
        """Revert to last backup if improvement failed."""
        try:
            backup_path = os.path.join(self.backup_dir, f"{os.path.basename(module_path)}.bak")
            if os.path.exists(backup_path):
                with open(backup_path, 'r') as f:
                    backup_code = f.read()
                with open(module_path, 'w') as f:
                    f.write(backup_code)
                return True
        except Exception as e:
            logger.error(f"Error reverting changes: {str(e)}")
        return False

    def modify_prompts(self, context: Dict[str, Any]) -> bool:
        """Modify strategy generation prompts based on performance."""
        try:
            # Read current prompts
            strategy_prompt_path = os.path.join(self.prompt_dir, "strategy_generation.yaml")
            rules_prompt_path = os.path.join(self.prompt_dir, "rules_generation.yaml")
            
            with open(strategy_prompt_path, 'r') as f:
                strategy_prompt = yaml.safe_load(f)
            with open(rules_prompt_path, 'r') as f:
                rules_prompt = yaml.safe_load(f)
                
            # Create improvement prompt
            prompt = self._create_prompt_improvement_prompt(strategy_prompt, rules_prompt, context)
            
            # Get LLM suggestions
            response = self._get_llm_suggestions(prompt)
            
            if not response or 'prompt_improvements' not in response:
                return False
                
            # Apply improvements to prompts
            if 'strategy_prompt' in response['prompt_improvements']:
                strategy_prompt = self._apply_prompt_improvements(
                    strategy_prompt, 
                    response['prompt_improvements']['strategy_prompt']
                )
                with open(strategy_prompt_path, 'w') as f:
                    yaml.dump(strategy_prompt, f)
                    
            if 'rules_prompt' in response['prompt_improvements']:
                rules_prompt = self._apply_prompt_improvements(
                    rules_prompt,
                    response['prompt_improvements']['rules_prompt']
                )
                with open(rules_prompt_path, 'w') as f:
                    yaml.dump(rules_prompt, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error modifying prompts: {str(e)}")
            return False

    def _create_improvement_prompt(self, module_code: str, metrics: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create a prompt for requesting code improvements."""
        prompt = f"""You are an expert Python developer tasked with improving a trading strategy module.
The module's current performance metrics are:
{json.dumps(metrics, indent=2)}

The current market context is:
{json.dumps(context, indent=2)}

The code that needs improvement is:
{module_code}

Please suggest specific improvements to enhance the strategy's performance. For each improvement:
1. Maintain proper Python syntax and indentation
2. Preserve existing class and method structures
3. Use type hints for all function parameters and return values
4. Include detailed docstrings for new methods
5. Follow PEP 8 style guidelines
6. Ensure all code blocks are properly indented
7. Keep the existing module organization

Format your response as a JSON object with this structure:
{{
    "improvements": [
        {{
            "description": "Brief description of the improvement",
            "code_changes": [
                {{
                    "type": "add|modify|delete",
                    "location": "Class or method name/docstring where change should be applied",
                    "code": "The actual code to add/modify (with proper indentation)"
                }}
            ],
            "expected_impact": "How this change will improve performance"
        }}
    ]
}}

The code changes should be complete, properly indented Python code that can be directly inserted into the module."""

        return prompt

    def _create_prompt_improvement_prompt(self, strategy_prompt: Dict, rules_prompt: Dict, 
                                        context: Dict[str, Any]) -> str:
        """Create prompt for improving strategy generation prompts."""
        market_regime = context.get('market_regime', 'unknown')
        total_return = context.get('total_return', 0)
        win_rate = context.get('win_rate', 0)
        failures = context.get('failures', [])
        
        prompt = f"""As a trading strategy optimizer, analyze these prompts and suggest improvements:

Current Performance:
- Market Regime: {market_regime}
- Total Return: {total_return}
- Win Rate: {win_rate}
- Failures: {', '.join(failures)}

Current Strategy Prompt:
{yaml.dump(strategy_prompt)}

Current Rules Prompt:
{yaml.dump(rules_prompt)}

Suggest specific improvements that:
1. Better handle the current market regime
2. Address performance failures
3. Improve strategy adaptation
4. Enhance parameter optimization
5. Maintain prompt clarity and structure

Respond with a JSON object containing:
{{
    "prompt_improvements": {{
        "strategy_prompt": {{
            "changes": [
                {{
                    "type": "add|modify|delete",
                    "location": "string",
                    "content": "string"
                }}
            ]
        }},
        "rules_prompt": {{
            "changes": [
                {{
                    "type": "add|modify|delete",
                    "location": "string",
                    "content": "string"
                }}
            ]
        }}
    }}
}}"""
        return prompt

    def _get_llm_suggestions(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get improvement suggestions from LLM."""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are an expert code improvement assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            if not response.choices:
                return None
                
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error getting LLM suggestions: {str(e)}")
            return None

    def format_code(self, code: str) -> str:
        """Format code using black formatter and verify syntax."""
        try:
            # First verify the code is valid Python syntax
            ast.parse(code)
            
            # Format the code using black
            formatted_code = black.format_str(code, mode=black.FileMode())
            
            # Verify the formatted code is still valid Python
            ast.parse(formatted_code)
            
            return formatted_code
        except Exception as e:
            logger.error(f"Error formatting code: {str(e)}")
            return code

    def validate_code_changes(self, original_code: str, modified_code: str) -> bool:
        """Validate that code changes maintain proper structure."""
        try:
            # Parse both versions into AST
            original_ast = ast.parse(original_code)
            modified_ast = ast.parse(modified_code)
            
            # Get the original class and function names
            original_classes = {
                node.name: [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                for node in ast.walk(original_ast)
                if isinstance(node, ast.ClassDef)
            }
            
            # Verify all classes and methods still exist in modified code
            modified_classes = {
                node.name: [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                for node in ast.walk(modified_ast)
                if isinstance(node, ast.ClassDef)
            }
            
            # Check that all original classes and methods are preserved
            for class_name, methods in original_classes.items():
                if class_name not in modified_classes:
                    logger.error(f"Missing class: {class_name}")
                    return False
                for method in methods:
                    if method not in modified_classes[class_name]:
                        logger.error(f"Missing method: {class_name}.{method}")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating code changes: {str(e)}")
            return False

    def _apply_improvements(self, current_code: str, improvements: List[Dict[str, Any]]) -> str:
        """Apply code improvements with intelligent formatting."""
        new_code = current_code
        
        try:
            for improvement in improvements:
                temp_code = new_code
                for change in improvement['code_changes']:
                    if change['type'] == 'add':
                        temp_code = self._insert_code(temp_code, change['location'], change['code'])
                    elif change['type'] == 'modify':
                        temp_code = self._modify_code(temp_code, change['location'], change['code'])
                    elif change['type'] == 'delete':
                        temp_code = self._delete_code(temp_code, change['location'])
                
                # Format the modified code
                formatted_code = self.format_code(temp_code)
                
                # Validate the changes maintain code structure
                if self.validate_code_changes(current_code, formatted_code):
                    new_code = formatted_code
                else:
                    logger.warning(f"Skipping invalid improvement: {improvement['description']}")
                    continue
                    
            return new_code
            
        except Exception as e:
            logger.error(f"Error applying improvements: {str(e)}")
            return current_code

    def _insert_code(self, code: str, location: str, new_code: str) -> str:
        """Insert new code at specified location."""
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Find the insertion point
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if location in ast.get_docstring(node, clean=False) or location in node.name:
                        # Calculate proper indentation based on node's level
                        indent_level = len(location) - len(location.lstrip())
                        indented_code = '\n'.join(' ' * indent_level + line for line in new_code.split('\n'))
                        
                        # Insert the code at the appropriate position
                        lines = code.split('\n')
                        insert_point = node.lineno
                        lines.insert(insert_point, indented_code)
                        return '\n'.join(lines)
            
            return code
        except Exception as e:
            logger.error(f"Error inserting code: {str(e)}")
            return code

    def _modify_code(self, code: str, location: str, new_code: str) -> str:
        """Modify code at specified location."""
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Find the modification point
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if location in ast.get_docstring(node, clean=False) or location in node.name:
                        # Calculate proper indentation
                        indent_level = len(location) - len(location.lstrip())
                        indented_code = '\n'.join(' ' * indent_level + line for line in new_code.split('\n'))
                        
                        # Replace the old code with new code
                        lines = code.split('\n')
                        start_line = node.lineno - 1
                        end_line = node.end_lineno
                        lines[start_line:end_line] = indented_code.split('\n')
                        return '\n'.join(lines)
            
            return code
        except Exception as e:
            logger.error(f"Error modifying code: {str(e)}")
            return code

    def _delete_code(self, code: str, location: str) -> str:
        """Delete code at specified location."""
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Find the deletion point
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if location in ast.get_docstring(node, clean=False) or location in node.name:
                        # Remove the code block
                        lines = code.split('\n')
                        del lines[node.lineno - 1:node.end_lineno]
                    return '\n'.join(lines)
                    
            return code
        except Exception as e:
            logger.error(f"Error deleting code: {str(e)}")
            return code

    def _apply_prompt_improvements(self, current_prompt: Dict, improvements: Dict[str, Any]) -> Dict:
        """Apply improvements to prompts."""
        try:
            new_prompt = current_prompt.copy()
            for change in improvements['changes']:
                if change['type'] == 'add':
                    # Add new content at specified location
                    location = change['location'].split('.')
                    self._add_to_dict(new_prompt, location, change['content'])
                elif change['type'] == 'modify':
                    # Modify existing content
                    location = change['location'].split('.')
                    self._modify_dict(new_prompt, location, change['content'])
                elif change['type'] == 'delete':
                    # Delete specified content
                    location = change['location'].split('.')
                    self._delete_from_dict(new_prompt, location)
                    
            return new_prompt
                    
        except Exception as e:
            logger.error(f"Error applying prompt improvements: {str(e)}")
            return current_prompt

    def _add_to_dict(self, d: Dict, location: List[str], content: Any) -> None:
        """Add content to nested dictionary."""
        current = d
        for key in location[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[location[-1]] = content

    def _modify_dict(self, d: Dict, location: List[str], content: Any) -> None:
        """Modify content in nested dictionary."""
        current = d
        for key in location[:-1]:
            current = current[key]
        current[location[-1]] = content

    def _delete_from_dict(self, d: Dict, location: List[str]) -> None:
        """Delete content from nested dictionary."""
        current = d
        for key in location[:-1]:
            current = current[key]
        del current[location[-1]] 

    def _validate_improvement_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Validate that an improvement suggestion is properly structured."""
        try:
            # Check basic structure
            if not isinstance(suggestion.get('code_changes', []), list):
                return False
                
            for change in suggestion['code_changes']:
                # Verify required fields
                if not all(k in change for k in ['type', 'location', 'code']):
                    return False
                    
                # Verify change type
                if change['type'] not in ['add', 'modify', 'delete']:
                    return False
                    
                # For add/modify changes, verify code is valid Python
                if change['type'] in ['add', 'modify']:
                    try:
                        # Try to parse the code as valid Python
                        ast.parse(change['code'])
                        
                        # Check indentation consistency
                        lines = change['code'].split('\n')
                        if len(lines) > 1:
                            indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
                            if not all(i % 4 == 0 for i in indents):  # Check 4-space indentation
                                return False
                    except SyntaxError:
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error validating improvement suggestion: {str(e)}")
            return False 

    def optimize_strategy(self, module_code: str, metrics: Dict[str, Any], 
                         context: Dict[str, Any], max_attempts: int = 3) -> Optional[str]:
        """Optimize strategy through evaluator-optimizer loop."""
        best_improvement = None
        best_score = 0.0
        
        for attempt in range(max_attempts):
            # Generate improvements
            improvements = self.propose_improvements(module_code, metrics, context)
            if not improvements:
                continue
                
            # Evaluate each improvement
            for improvement in improvements:
                # Apply improvement temporarily
                temp_code = self._apply_single_improvement(module_code, improvement)
                
                # Evaluate the improvement
                evaluation = self._evaluate_improvement(temp_code, improvement, metrics, context)
                improvement.evaluation_score = evaluation.score
                improvement.feedback = evaluation.improvement_suggestions
                
                # Track best improvement
                if evaluation.score > best_score:
                    best_score = evaluation.score
                    best_improvement = improvement
                    
                # If we found a very good improvement, use it
                if evaluation.score > 0.8:
                    break
                    
            # If we found a good enough improvement, apply it
            if best_improvement and best_improvement.evaluation_score > 0.6:
                return self._apply_single_improvement(module_code, best_improvement)
                
            # Otherwise, use feedback to generate better improvements in next iteration
            context['previous_attempts'] = [
                {'score': imp.evaluation_score, 'feedback': imp.feedback}
                for imp in improvements
            ]
            
        return None

    def _evaluate_improvement(self, code: str, improvement: StrategyImprovement,
                            metrics: Dict[str, Any], context: Dict[str, Any]) -> StrategyEvaluation:
        """Evaluate a proposed strategy improvement."""
        try:
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(code, improvement, metrics, context)
            
            # Get evaluation from LLM
            evaluation = self._get_llm_evaluation(prompt)
            
            if evaluation:
                return StrategyEvaluation(
                    parameters=improvement.code_changes,
                    metrics=metrics,
                    market_regime=context.get('market_regime', 'unknown'),
                    score=evaluation.get('score', 0.0),
                    strengths=evaluation.get('strengths', []),
                    weaknesses=evaluation.get('weaknesses', []),
                    improvement_suggestions=evaluation.get('suggestions', [])
                )
        except Exception as e:
            logger.error(f"Error evaluating improvement: {str(e)}")
            
        return StrategyEvaluation(
            parameters={},
            metrics=metrics,
            market_regime=context.get('market_regime', 'unknown'),
            score=0.0,
            strengths=[],
            weaknesses=["Failed to evaluate"],
            improvement_suggestions=[]
        )

    def _create_evaluation_prompt(self, code: str, improvement: StrategyImprovement,
                                metrics: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create prompt for evaluating strategy improvements."""
        return f"""Evaluate this trading strategy improvement:

Original Metrics:
{json.dumps(metrics, indent=2)}

Market Context:
{json.dumps(context, indent=2)}

Proposed Improvement:
{improvement.description}

Modified Code:
{code}

Evaluate the improvement for:
1. Code correctness and maintainability
2. Expected performance impact
3. Risk management considerations
4. Market regime adaptability
5. Parameter optimization potential

Respond with a JSON object:
{{
    "score": float,  # 0.0 to 1.0
    "strengths": [str],
    "weaknesses": [str],
    "suggestions": [str]
}}"""

    def _get_llm_evaluation(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get evaluation from LLM."""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are an expert trading strategy evaluator."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            if not response.choices:
                return None
                
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error getting LLM evaluation: {str(e)}")
            return None

    def _apply_single_improvement(self, code: str, improvement: StrategyImprovement) -> str:
        """Apply a single improvement to the code."""
        try:
            new_code = code
            for change in improvement.code_changes:
                if change['type'] == 'add':
                    new_code = self._insert_code(new_code, change['location'], change['code'])
                elif change['type'] == 'modify':
                    new_code = self._modify_code(new_code, change['location'], change['code'])
                elif change['type'] == 'delete':
                    new_code = self._delete_code(new_code, change['location'])
                    
            return self.format_code(new_code)
        except Exception as e:
            logger.error(f"Error applying improvement: {str(e)}")
            return code 

    async def parallel_strategy_optimization(self, module_code: str, metrics: Dict[str, Any],
                                          context: Dict[str, Any], num_parallel: int = 3) -> Optional[str]:
        """Optimize strategy using parallel evaluation and aggregation."""
        try:
            # Generate multiple improvement proposals in parallel
            improvements = await self._generate_parallel_improvements(module_code, metrics, context, num_parallel)
            if not improvements:
                return None

            # Evaluate improvements in parallel
            evaluations = await self._evaluate_parallel_improvements(
                module_code, improvements, metrics, context
            )

            # Aggregate and select best improvement
            best_improvement = self._aggregate_improvements(improvements, evaluations)
            if best_improvement and best_improvement.evaluation_score > 0.6:
                return self._apply_single_improvement(module_code, best_improvement)

            return None

        except Exception as e:
            logger.error(f"Error in parallel strategy optimization: {str(e)}")
            return None

    async def _generate_parallel_improvements(
        self, module_code: str, metrics: Dict[str, Any], 
        context: Dict[str, Any], num_parallel: int
    ) -> List[StrategyImprovement]:
        """Generate multiple strategy improvements in parallel."""
        async def generate_single_improvement(variation: str) -> Optional[StrategyImprovement]:
            # Create a specialized prompt for each parallel attempt
            specialized_context = context.copy()
            specialized_context['optimization_focus'] = variation
            
            prompt = self._create_improvement_prompt(module_code, metrics, specialized_context)
            suggestions = await self._async_llm_suggestions(prompt)
            
            if suggestions and 'improvements' in suggestions:
                for improvement in suggestions['improvements']:
                    if self._validate_improvement_suggestion(improvement):
                        return StrategyImprovement(
                            description=improvement['description'],
                            code_changes=improvement['code_changes'],
                            expected_impact=improvement['expected_impact']
                        )
            return None

        # Define different optimization focuses
        variations = [
            'performance_optimization',
            'risk_management',
            'market_adaptation',
            'parameter_tuning',
            'strategy_robustness'
        ]

        # Generate improvements in parallel
        tasks = [
            generate_single_improvement(variation)
            for variation in variations[:num_parallel]
        ]
        
        improvements = await asyncio.gather(*tasks)
        return [imp for imp in improvements if imp is not None]

    async def _evaluate_parallel_improvements(
        self, module_code: str, improvements: List[StrategyImprovement],
        metrics: Dict[str, Any], context: Dict[str, Any]
    ) -> List[StrategyEvaluation]:
        """Evaluate multiple improvements in parallel."""
        async def evaluate_single_improvement(improvement: StrategyImprovement) -> StrategyEvaluation:
            temp_code = self._apply_single_improvement(module_code, improvement)
            return await self._async_evaluate_improvement(temp_code, improvement, metrics, context)

        tasks = [
            evaluate_single_improvement(improvement)
            for improvement in improvements
        ]
        
        return await asyncio.gather(*tasks)

    def _aggregate_improvements(
        self, improvements: List[StrategyImprovement],
        evaluations: List[StrategyEvaluation]
    ) -> Optional[StrategyImprovement]:
        """Aggregate and select the best improvement."""
        if not improvements or not evaluations:
            return None

        # Combine improvements with their evaluations
        for imp, eval in zip(improvements, evaluations):
            imp.evaluation_score = eval.score
            imp.feedback = eval.improvement_suggestions

        # Select the best improvement
        best_improvement = max(improvements, key=lambda x: x.evaluation_score)
        return best_improvement if best_improvement.evaluation_score > 0.6 else None

    async def _async_llm_suggestions(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Get suggestions from LLM asynchronously."""
        try:
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": "You are an expert code improvement assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            if not response.choices:
                return None
                
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error getting async LLM suggestions: {str(e)}")
            return None

    async def _async_evaluate_improvement(
        self, code: str, improvement: StrategyImprovement,
        metrics: Dict[str, Any], context: Dict[str, Any]
    ) -> StrategyEvaluation:
        """Evaluate improvement asynchronously."""
        try:
            prompt = self._create_evaluation_prompt(code, improvement, metrics, context)
            evaluation = await self._async_llm_suggestions(prompt)
            
            if evaluation:
                return StrategyEvaluation(
                    parameters=improvement.code_changes,
                    metrics=metrics,
                    market_regime=context.get('market_regime', 'unknown'),
                    score=evaluation.get('score', 0.0),
                    strengths=evaluation.get('strengths', []),
                    weaknesses=evaluation.get('weaknesses', []),
                    improvement_suggestions=evaluation.get('suggestions', [])
                )
        except Exception as e:
            logger.error(f"Error in async evaluation: {str(e)}")
            
        return StrategyEvaluation(
            parameters={},
            metrics=metrics,
            market_regime=context.get('market_regime', 'unknown'),
            score=0.0,
            strengths=[],
            weaknesses=["Failed to evaluate"],
            improvement_suggestions=[]
        ) 

    def track_grid_search_results(self, grid_results: Dict[str, Any], market_regime: str) -> None:
        """Track and learn from grid search results.
        
        Args:
            grid_results: Results from grid search optimization
            market_regime: Current market regime
        """
        try:
            if market_regime not in self.grid_search_history:
                self.grid_search_history[market_regime] = []

            # Extract metrics and parameters
            metrics = grid_results.get('metrics', {})
            parameters = grid_results.get('parameters', {})
            
            # Calculate score
            score = self._calculate_strategy_score(metrics)
            
            # Create and store result
            result = GridSearchResult(
                parameters=parameters,
                metrics=metrics,
                market_regime=market_regime,
                timestamp=datetime.now(),
                score=score
            )
            
            self.grid_search_history[market_regime].append(result)
            
            # Keep only top N results per regime
            self.grid_search_history[market_regime].sort(key=lambda x: x.score, reverse=True)
            self.grid_search_history[market_regime] = self.grid_search_history[market_regime][:10]
            
            # Update success patterns
            if score > 0.6:  # Only store truly successful results
                if market_regime not in self.success_patterns:
                    self.success_patterns[market_regime] = []
                
                self.success_patterns[market_regime].append({
                    'parameters': parameters,
                    'metrics': metrics,
                    'score': score,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep success patterns sorted and limited
                self.success_patterns[market_regime].sort(key=lambda x: x['score'], reverse=True)
                self.success_patterns[market_regime] = self.success_patterns[market_regime][:5]
                
            logger.info(f"Tracked grid search results for {market_regime} regime with score {score:.4f}")
            
        except Exception as e:
            logger.error(f"Error tracking grid search results: {str(e)}")

    def optimize_grid_ranges(self, market_regime: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Optimize grid search ranges based on learned patterns.
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary of optimized parameter ranges, or None if insufficient data
        """
        try:
            if market_regime not in self.grid_search_history:
                return None
                
            history = self.grid_search_history[market_regime]
            if not history:
                return None
                
            # Get top performing results
            top_results = sorted(history, key=lambda x: x.score, reverse=True)[:5]
            
            # Initialize ranges
            ranges = {}
            
            # Parameters to optimize
            params_to_optimize = [
                'take_profit', 'stop_loss', 
                'macd_signal_fast', 'macd_signal_slow', 'macd_signal_signal',
                'sl_window', 'order_size', 'max_orders'
            ]
            
            for param in params_to_optimize:
                values = [r.parameters.get(param) for r in top_results if param in r.parameters]
                if not values:
                    continue
                    
                values = [float(v) for v in values if v is not None]
                if not values:
                    continue
                
                min_val = min(values)
                max_val = max(values)
                
                # Add margin around successful values
                value_range = max_val - min_val
                margin = value_range * 0.2  # 20% margin
                
                # Ensure we don't go below 0 for certain parameters
                min_bound = max(0, min_val - margin)
                max_bound = max_val + margin
                
                ranges[param] = {
                    'min': float(min_bound),
                    'max': float(max_bound)
                }
            
            if not ranges:
                return None
                
            logger.info(f"Generated optimized ranges for {market_regime} regime: {json.dumps(ranges, indent=2)}")
            return ranges
            
        except Exception as e:
            logger.error(f"Error optimizing grid ranges: {str(e)}")
            return None

    def get_best_parameters(self, market_regime: str) -> Optional[Dict[str, Any]]:
        """Get the best performing parameters for a given market regime.
        
        Args:
            market_regime: Market regime to get parameters for
            
        Returns:
            Dictionary of best parameters, or None if no data available
        """
        try:
            if market_regime not in self.grid_search_history:
                return None
                
            history = self.grid_search_history[market_regime]
            if not history:
                return None
                
            # Get best result
            best_result = max(history, key=lambda x: x.score)
            return best_result.parameters
            
        except Exception as e:
            logger.error(f"Error getting best parameters: {str(e)}")
            return None 
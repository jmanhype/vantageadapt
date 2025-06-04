#!/usr/bin/env python3
"""
Strategic Analyzer - LLM that analyzes its own performance and makes strategic decisions
Implements Kagan's vision: "The LLM would then review the statistics... and make some sort of change"
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import dspy
from pathlib import Path
import json
import os

from src.utils.types import BacktestResults, MarketRegime, StrategyContext
from src.utils.memory_manager import TradingMemoryManager as MemoryManager

# Configure DSPy with OpenAI
import dspy

# Initialize OpenAI language model using the correct class
turbo = dspy.LM('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=turbo)

# Create a simple PerformanceTracker if not available
class PerformanceTracker:
    def __init__(self):
        pass


class StrategicInsight(dspy.Signature):
    """Analyze trading performance and generate strategic insights."""
    
    performance_data = dspy.InputField(desc="Trading performance metrics and statistics")
    market_conditions = dspy.InputField(desc="Market conditions during trading period")
    failure_patterns = dspy.InputField(desc="Identified patterns in losing trades")
    
    strategic_analysis = dspy.OutputField(desc="Deep analysis of why strategies succeeded or failed")
    improvement_hypothesis = dspy.OutputField(desc="Theories about how to improve performance")
    action_recommendations = dspy.OutputField(desc="Specific actions to implement")


class StrategyModification(dspy.Signature):
    """Generate modifications to trading strategy based on analysis."""
    
    current_strategy = dspy.InputField(desc="Current trading strategy parameters and logic")
    strategic_insights = dspy.InputField(desc="Analysis and recommendations from performance review")
    market_outlook = dspy.InputField(desc="Expected market conditions going forward")
    
    strategy_modifications = dspy.OutputField(desc="Specific changes to make to trading strategy")
    risk_adjustments = dspy.OutputField(desc="Updates to risk management parameters")
    new_indicators = dspy.OutputField(desc="New technical indicators or signals to incorporate")


class TradePatternAnalysis(dspy.Signature):
    """Analyze patterns in winning and losing trades."""
    
    winning_trades = dspy.InputField(desc="Details of profitable trades")
    losing_trades = dspy.InputField(desc="Details of unprofitable trades")
    market_data = dspy.InputField(desc="Market conditions during each trade")
    
    winning_patterns = dspy.OutputField(desc="Common patterns found in winning trades")
    losing_patterns = dspy.OutputField(desc="Common patterns found in losing trades")
    avoidance_rules = dspy.OutputField(desc="Rules to avoid identified failure patterns")


class StrategicAnalyzer(dspy.Module):
    """
    Implements Kagan's vision of LLM reviewing performance and making strategic changes.
    This is the "brain" that was missing - the system that thinks about its own performance.
    """
    
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        super().__init__()
        
        # Core analysis modules
        self.strategic_insight = dspy.ChainOfThought(StrategicInsight)
        self.strategy_modifier = dspy.ChainOfThought(StrategyModification)
        self.pattern_analyzer = dspy.ChainOfThought(TradePatternAnalysis)
        
        # Memory and tracking
        self.memory_manager = memory_manager or MemoryManager()
        self.performance_tracker = PerformanceTracker()
        
        # Analysis history
        self.analysis_history = []
        self.strategy_evolution = []
        
        logger.info("🧠 Strategic Analyzer initialized - Kagan's AI trading brain activated")
    
    def analyze_portfolio_performance(self, 
                                    backtest_results: BacktestResults,
                                    trade_history: pd.DataFrame,
                                    market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Kagan's requirement: "The LLM would then review the statistics that are outputted"
        
        This method performs deep analysis of trading performance to understand:
        - Why certain strategies worked or failed
        - What market conditions led to success/failure
        - How to improve future performance
        """
        logger.info("🔍 Analyzing portfolio performance with AI strategic thinking")
        
        # Prepare performance data for LLM analysis
        performance_summary = self._prepare_performance_summary(backtest_results, trade_history)
        
        # Analyze failure patterns (Kagan: "types of trades that led to worst/best trades")
        failure_analysis = self._analyze_failure_patterns(trade_history, market_conditions)
        
        # Generate strategic insights using LLM
        try:
            insights = self.strategic_insight(
                performance_data=json.dumps(performance_summary, indent=2),
                market_conditions=json.dumps(market_conditions, indent=2),
                failure_patterns=json.dumps(failure_analysis, indent=2)
            )
            
            # Parse and structure the insights
            strategic_analysis = {
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': performance_summary,
                'strategic_analysis': insights.strategic_analysis,
                'improvement_hypothesis': insights.improvement_hypothesis,
                'action_recommendations': insights.action_recommendations,
                'failure_patterns': failure_analysis
            }
            
            # Store analysis in memory for future reference
            self._store_strategic_analysis(strategic_analysis)
            
            logger.info(f"✅ Strategic analysis complete: {insights.improvement_hypothesis[:100]}...")
            
            return strategic_analysis
            
        except Exception as e:
            logger.error(f"Error in strategic analysis: {e}")
            raise  # Fail fast - no fallbacks!
    
    def generate_strategy_modifications(self,
                                      current_strategy: StrategyContext,
                                      strategic_insights: Dict[str, Any],
                                      market_outlook: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Kagan's requirement: "make some sort of change based on the output"
        
        This method generates specific modifications to trading strategy based on analysis.
        """
        logger.info("🔧 Generating strategy modifications based on AI analysis")
        
        # Prepare current strategy description
        strategy_description = self._describe_current_strategy(current_strategy)
        
        # Get market outlook
        market_forecast = self._prepare_market_outlook(market_outlook)
        
        try:
            modifications = self.strategy_modifier(
                current_strategy=strategy_description,
                strategic_insights=json.dumps(strategic_insights, indent=2),
                market_outlook=market_forecast
            )
            
            # Parse modifications into actionable changes
            strategy_changes = {
                'timestamp': datetime.now().isoformat(),
                'original_strategy': current_strategy.to_dict(),
                'modifications': self._parse_strategy_modifications(modifications.strategy_modifications),
                'risk_adjustments': self._parse_risk_adjustments(modifications.risk_adjustments),
                'new_indicators': self._parse_new_indicators(modifications.new_indicators),
                'reasoning': strategic_insights.get('improvement_hypothesis', '')
            }
            
            # Track strategy evolution
            self._track_strategy_evolution(strategy_changes)
            
            logger.info(f"📈 Strategy modifications generated: {len(strategy_changes['modifications'])} changes")
            
            return strategy_changes
            
        except Exception as e:
            logger.error(f"Error generating strategy modifications: {e}")
            raise  # Fail fast - no fallbacks!
    
    def analyze_trade_patterns(self, trade_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Kagan's specific requirement: "types of trades that happened that led to the worst and the best trades"
        
        This method performs deep pattern analysis to understand:
        - What patterns lead to winning trades
        - What patterns lead to losing trades
        - How to avoid failure patterns
        """
        logger.info("🔍 Analyzing trade patterns for winners vs losers")
        
        # Separate winning and losing trades
        winning_trades = trade_history[trade_history['pnl'] > 0]
        losing_trades = trade_history[trade_history['pnl'] <= 0]
        
        # Prepare detailed trade information
        winning_details = self._prepare_trade_details(winning_trades)
        losing_details = self._prepare_trade_details(losing_trades)
        
        # Get market data for each trade
        market_context = self._get_market_context_for_trades(trade_history)
        
        try:
            pattern_analysis = self.pattern_analyzer(
                winning_trades=json.dumps(winning_details, indent=2),
                losing_trades=json.dumps(losing_details, indent=2),
                market_data=json.dumps(market_context, indent=2)
            )
            
            # Structure pattern analysis results
            patterns = {
                'timestamp': datetime.now().isoformat(),
                'total_trades_analyzed': len(trade_history),
                'winning_patterns': self._parse_patterns(pattern_analysis.winning_patterns),
                'losing_patterns': self._parse_patterns(pattern_analysis.losing_patterns),
                'avoidance_rules': self._parse_avoidance_rules(pattern_analysis.avoidance_rules),
                'statistical_significance': self._calculate_pattern_significance(trade_history)
            }
            
            # Store patterns for future reference
            self._store_pattern_analysis(patterns)
            
            logger.info(f"✅ Pattern analysis complete: {len(patterns['winning_patterns'])} winning patterns, "
                       f"{len(patterns['losing_patterns'])} losing patterns identified")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            raise  # Fail fast - no fallbacks!
    
    def generate_strategic_recommendations(self, 
                                         performance_history: List[Dict[str, Any]],
                                         current_market_regime: MarketRegime) -> List[Dict[str, Any]]:
        """
        Generate high-level strategic recommendations based on comprehensive analysis.
        """
        logger.info("💡 Generating strategic recommendations")
        
        # Analyze performance trends
        performance_trends = self._analyze_performance_trends(performance_history)
        
        # Identify systematic issues
        systematic_issues = self._identify_systematic_issues(performance_history)
        
        # Generate recommendations
        recommendations = []
        
        # 1. Performance-based recommendations
        if performance_trends['return_trend'] < 0:
            recommendations.append({
                'type': 'strategy_overhaul',
                'priority': 'high',
                'recommendation': 'Consider fundamental strategy redesign due to negative return trend',
                'specific_actions': [
                    'Review and update entry/exit criteria',
                    'Reassess position sizing algorithm',
                    'Consider alternative technical indicators'
                ]
            })
        
        # 2. Risk-based recommendations
        if systematic_issues.get('excessive_drawdown', False):
            recommendations.append({
                'type': 'risk_management',
                'priority': 'critical',
                'recommendation': 'Implement stricter risk controls to reduce drawdown',
                'specific_actions': [
                    'Reduce maximum position size',
                    'Implement portfolio-level stop loss',
                    'Add volatility-based position scaling'
                ]
            })
        
        # 3. Market regime-based recommendations
        regime_recommendations = self._get_regime_specific_recommendations(current_market_regime)
        recommendations.extend(regime_recommendations)
        
        # 4. Pattern-based recommendations
        if self.analysis_history:
            pattern_recommendations = self._get_pattern_based_recommendations()
            recommendations.extend(pattern_recommendations)
        
        logger.info(f"📋 Generated {len(recommendations)} strategic recommendations")
        
        return recommendations
    
    def _prepare_performance_summary(self, 
                                   backtest_results: BacktestResults,
                                   trade_history: pd.DataFrame) -> Dict[str, Any]:
        """Prepare comprehensive performance summary for analysis."""
        
        # Calculate advanced metrics
        consecutive_losses = self._calculate_consecutive_losses(trade_history)
        time_in_drawdown = self._calculate_time_in_drawdown(trade_history)
        
        return {
            'total_return': backtest_results.total_return,
            'win_rate': backtest_results.win_rate,
            'sortino_ratio': backtest_results.sortino_ratio,
            'max_drawdown': backtest_results.metrics.get('max_drawdown', 0.0),
            'total_trades': backtest_results.total_trades,
            'avg_win': trade_history[trade_history['pnl'] > 0]['pnl'].mean() if len(trade_history) > 0 else 0,
            'avg_loss': trade_history[trade_history['pnl'] < 0]['pnl'].mean() if len(trade_history) > 0 else 0,
            'profit_factor': self._calculate_profit_factor(trade_history),
            'consecutive_losses': consecutive_losses,
            'time_in_drawdown': time_in_drawdown,
            'trades_by_hour': self._analyze_trades_by_hour(trade_history),
            'trades_by_asset': self._analyze_trades_by_asset(trade_history)
        }
    
    def _analyze_failure_patterns(self, 
                                trade_history: pd.DataFrame,
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement Kagan's specific requirement: 
        "the price went down consecutively for one hour. We bought five times and kept getting stopped out"
        """
        
        failure_patterns = {
            'consecutive_stop_outs': [],
            'repeated_entries_same_direction': [],
            'timing_failures': [],
            'market_condition_failures': []
        }
        
        # Analyze consecutive stop-outs
        stop_out_sequences = self._find_consecutive_stop_outs(trade_history)
        if stop_out_sequences:
            failure_patterns['consecutive_stop_outs'] = stop_out_sequences
        
        # Find repeated entries in same direction
        repeated_entries = self._find_repeated_entries(trade_history)
        if repeated_entries:
            failure_patterns['repeated_entries_same_direction'] = repeated_entries
        
        # Analyze timing failures
        timing_analysis = self._analyze_entry_exit_timing(trade_history)
        if timing_analysis['poor_timing_trades']:
            failure_patterns['timing_failures'] = timing_analysis['poor_timing_trades']
        
        # Market condition specific failures
        regime_failures = self._analyze_regime_specific_failures(trade_history, market_conditions)
        if regime_failures:
            failure_patterns['market_condition_failures'] = regime_failures
        
        return failure_patterns
    
    def _find_consecutive_stop_outs(self, trades: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find sequences where we got stopped out multiple times consecutively."""
        sequences = []
        current_sequence = []
        
        for i, trade in trades.iterrows():
            if trade.get('exit_reason') == 'stop_loss':
                current_sequence.append({
                    'timestamp': trade['entry_time'],
                    'asset': trade['asset'],
                    'loss': trade['pnl'],
                    'price_movement': trade.get('price_movement', 'unknown')
                })
            else:
                if len(current_sequence) >= 3:  # 3+ consecutive stop-outs
                    sequences.append({
                        'start_time': current_sequence[0]['timestamp'],
                        'end_time': current_sequence[-1]['timestamp'],
                        'count': len(current_sequence),
                        'total_loss': sum(t['loss'] for t in current_sequence),
                        'trades': current_sequence
                    })
                current_sequence = []
        
        return sequences
    
    def _describe_current_strategy(self, strategy: StrategyContext) -> str:
        """Create human-readable description of current strategy."""
        return f"""
        Current Trading Strategy:
        - Risk per trade: {strategy.parameters.get('risk_percentage', 2)}%
        - Position sizing: {strategy.parameters.get('position_sizing', 'fixed')}
        - Entry indicators: {strategy.parameters.get('entry_indicators', [])}
        - Exit indicators: {strategy.parameters.get('exit_indicators', [])}
        - Stop loss: {strategy.parameters.get('stop_loss_pct', 2)}%
        - Take profit: {strategy.parameters.get('take_profit_pct', 5)}%
        - Market regimes traded: {strategy.parameters.get('allowed_regimes', 'all')}
        """
    
    def _parse_strategy_modifications(self, modifications_text: str) -> List[Dict[str, Any]]:
        """Parse LLM-generated modifications into actionable changes."""
        modifications = []
        
        # Extract specific parameter changes
        lines = modifications_text.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                modifications.append({
                    'parameter': key.strip(),
                    'new_value': value.strip(),
                    'type': 'parameter_update'
                })
        
        return modifications
    
    def _track_strategy_evolution(self, changes: Dict[str, Any]):
        """Track how strategy evolves over time."""
        self.strategy_evolution.append(changes)
        
        # Keep only recent history
        if len(self.strategy_evolution) > 100:
            self.strategy_evolution = self.strategy_evolution[-100:]
        
        # Save to memory
        if self.memory_manager:
            self.memory_manager.store_analysis_result(
                "strategy_evolution",
                {
                    'evolution_history': self.strategy_evolution,
                    'total_modifications': len(self.strategy_evolution),
                    'last_update': datetime.now().isoformat()
                }
            )
    
    def _calculate_pattern_significance(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical significance of identified patterns."""
        if len(trades) < 30:
            return {'confidence': 'low', 'sample_size': len(trades)}
        
        return {
            'confidence': 'high' if len(trades) > 100 else 'medium',
            'sample_size': len(trades),
            'statistical_power': min(len(trades) / 100, 1.0)
        }
    
    def _fallback_analysis(self, performance: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': performance,
            'strategic_analysis': 'Automated analysis based on performance metrics',
            'improvement_hypothesis': self._generate_basic_hypothesis(performance),
            'action_recommendations': self._generate_basic_recommendations(performance),
            'failure_patterns': patterns
        }
    
    def _generate_basic_hypothesis(self, performance: Dict[str, Any]) -> str:
        """Generate basic improvement hypothesis based on metrics."""
        if performance['win_rate'] < 0.4:
            return "Low win rate suggests entry criteria may be too loose. Consider tightening entry conditions."
        elif performance['max_drawdown'] > 0.2:
            return "High drawdown indicates excessive risk. Consider reducing position sizes or tightening stops."
        elif performance['total_return'] < 0:
            return "Negative returns require strategy overhaul. Review all components systematically."
        else:
            return "Performance is acceptable but can be optimized. Focus on consistency improvements."
    
    def _calculate_consecutive_losses(self, trade_history: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses."""
        if len(trade_history) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for _, trade in trade_history.iterrows():
            if trade.get('pnl', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_time_in_drawdown(self, trade_history: pd.DataFrame) -> float:
        """Calculate percentage of time in drawdown."""
        if len(trade_history) == 0:
            return 0.0
        
        # Simplified calculation
        losing_trades = len(trade_history[trade_history.get('pnl', 0) < 0])
        return losing_trades / len(trade_history) if len(trade_history) > 0 else 0.0
    
    def _calculate_profit_factor(self, trade_history: pd.DataFrame) -> float:
        """Calculate profit factor."""
        if len(trade_history) == 0:
            return 1.0
            
        gross_profit = trade_history[trade_history['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trade_history[trade_history['pnl'] < 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def _analyze_trades_by_hour(self, trade_history: pd.DataFrame) -> Dict[int, float]:
        """Analyze performance by hour of day."""
        if len(trade_history) == 0:
            return {}
        
        # Group by hour and calculate average PnL
        hourly_perf = {}
        for hour in range(24):
            hourly_trades = trade_history[pd.to_datetime(trade_history.get('entry_time', '')).dt.hour == hour]
            if len(hourly_trades) > 0:
                hourly_perf[hour] = hourly_trades['pnl'].mean()
        
        return hourly_perf
    
    def _analyze_trades_by_asset(self, trade_history: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze performance by asset."""
        if len(trade_history) == 0:
            return {}
        
        asset_perf = {}
        for asset in trade_history['asset'].unique():
            asset_trades = trade_history[trade_history['asset'] == asset]
            asset_perf[asset] = {
                'total_pnl': asset_trades['pnl'].sum(),
                'avg_pnl': asset_trades['pnl'].mean(),
                'trade_count': len(asset_trades)
            }
        
        return asset_perf
    
    def _generate_basic_recommendations(self, performance: Dict[str, Any]) -> List[str]:
        """Generate basic recommendations based on performance."""
        recommendations = []
        
        if performance['win_rate'] < 0.4:
            recommendations.append("Improve entry signal quality")
        if performance['consecutive_losses'] > 5:
            recommendations.append("Implement maximum consecutive loss limit")
        if performance.get('profit_factor', 1) < 1.2:
            recommendations.append("Improve risk/reward ratio on trades")
            
        return recommendations
    
    def _store_strategic_analysis(self, analysis: Dict[str, Any]):
        """Store strategic analysis in memory."""
        self.analysis_history.append(analysis)
        if self.memory_manager:
            try:
                # Try the method that exists in TradingMemoryManager
                self.memory_manager.store_strategy_performance(
                    strategy_id="strategic_analysis",
                    performance=analysis
                )
            except:
                # Just store in memory
                pass
    
    def _prepare_trade_details(self, trades: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare detailed trade information for analysis."""
        details = []
        for _, trade in trades.iterrows():
            details.append({
                'timestamp': str(trade.get('entry_time', '')),
                'asset': trade.get('asset', ''),
                'pnl': float(trade.get('pnl', 0)),
                'exit_reason': trade.get('exit_reason', ''),
                'duration': trade.get('duration', 0)
            })
        return details
    
    def _get_market_context_for_trades(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Get market context for trades."""
        # Simplified market context
        return {
            'volatility': 'moderate',
            'trend': 'neutral',
            'volume': 'average'
        }
    
    def _parse_patterns(self, patterns_text: str) -> List[Dict[str, Any]]:
        """Parse pattern text into structured data."""
        patterns = []
        if patterns_text:
            for line in patterns_text.strip().split('\n'):
                if line.strip():
                    patterns.append({
                        'description': line.strip(),
                        'type': 'identified_pattern'
                    })
        return patterns
    
    def _parse_avoidance_rules(self, rules_text: str) -> List[str]:
        """Parse avoidance rules."""
        rules = []
        if rules_text:
            for line in rules_text.strip().split('\n'):
                if line.strip():
                    rules.append(line.strip())
        return rules
    
    def _store_pattern_analysis(self, patterns: Dict[str, Any]):
        """Store pattern analysis results."""
        if self.memory_manager:
            self.memory_manager.store_analysis_result("pattern_analysis", patterns)
    
    def _fallback_pattern_analysis(self, winning_trades: pd.DataFrame, losing_trades: pd.DataFrame) -> Dict[str, Any]:
        """Fallback pattern analysis."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_trades_analyzed': len(winning_trades) + len(losing_trades),
            'winning_patterns': [],
            'losing_patterns': [],
            'avoidance_rules': [],
            'statistical_significance': {'confidence': 'low'}
        }
    
    def _fallback_modifications(self, strategy: StrategyContext, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategy modifications."""
        return {
            'timestamp': datetime.now().isoformat(),
            'original_strategy': strategy.to_dict(),
            'modifications': [],
            'risk_adjustments': [],
            'new_indicators': [],
            'reasoning': 'Fallback modifications due to error'
        }
    
    def _prepare_market_outlook(self, regime: Optional[MarketRegime]) -> str:
        """Prepare market outlook description."""
        if regime:
            return f"Market regime: {regime.value}"
        return "Market regime: Unknown"
    
    def _parse_risk_adjustments(self, adjustments_text: str) -> List[Dict[str, Any]]:
        """Parse risk adjustments."""
        adjustments = []
        if adjustments_text:
            lines = adjustments_text.strip().split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    adjustments.append({
                        'parameter': key.strip(),
                        'adjustment': value.strip()
                    })
        return adjustments
    
    def _parse_new_indicators(self, indicators_text: str) -> List[str]:
        """Parse new indicators."""
        indicators = []
        if indicators_text:
            for line in indicators_text.strip().split('\n'):
                if line.strip():
                    indicators.append(line.strip())
        return indicators
    
    def _find_repeated_entries(self, trades: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find repeated entries in same direction."""
        repeated = []
        # Simplified implementation
        return repeated
    
    def _analyze_entry_exit_timing(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze entry and exit timing."""
        return {'poor_timing_trades': []}
    
    def _analyze_regime_specific_failures(self, trades: pd.DataFrame, market_conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze failures specific to market regimes."""
        return []
    
    def _analyze_performance_trends(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if not history:
            return {'return_trend': 0}
        
        # Simple trend calculation
        returns = [h.get('current_performance', {}).get('total_return', 0) for h in history]
        if len(returns) > 1:
            return {'return_trend': returns[-1] - returns[0]}
        return {'return_trend': 0}
    
    def _identify_systematic_issues(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify systematic issues in performance."""
        issues = {}
        if history:
            # Check for excessive drawdown
            max_dd = max(h.get('current_performance', {}).get('max_drawdown', 0) for h in history)
            if max_dd > 0.2:
                issues['excessive_drawdown'] = True
        return issues
    
    def _get_regime_specific_recommendations(self, regime: MarketRegime) -> List[Dict[str, Any]]:
        """Get recommendations specific to market regime."""
        recommendations = []
        if regime == MarketRegime.TRENDING_BULLISH:
            recommendations.append({
                'type': 'regime_optimization',
                'priority': 'medium',
                'recommendation': 'Optimize for trending bullish conditions',
                'specific_actions': ['Increase position sizes on breakouts']
            })
        return recommendations
    
    def _get_pattern_based_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations based on identified patterns."""
        return []


async def main():
    """Test the Strategic Analyzer with sample data."""
    logger.info("🧠 Testing Strategic Analyzer - Kagan's AI Brain")
    
    # Initialize analyzer
    analyzer = StrategicAnalyzer()
    
    # Create sample data
    sample_results = BacktestResults(
        total_return=0.15,
        total_pnl=15000,
        sortino_ratio=1.2,
        win_rate=0.54,
        total_trades=500,
        trades=[],
        metrics={}
    )
    
    # Create sample trade history
    trade_data = {
        'entry_time': pd.date_range('2024-01-01', periods=100, freq='H'),
        'asset': ['BTC'] * 50 + ['ETH'] * 50,
        'pnl': np.random.normal(50, 200, 100),
        'exit_reason': ['stop_loss'] * 20 + ['take_profit'] * 30 + ['signal'] * 50
    }
    trade_history = pd.DataFrame(trade_data)
    
    # Analyze performance
    market_conditions = {'regime': 'TRENDING_BULLISH', 'volatility': 'high'}
    analysis = analyzer.analyze_portfolio_performance(sample_results, trade_history, market_conditions)
    
    logger.info(f"Strategic Analysis: {analysis['strategic_analysis'][:200]}...")
    logger.info(f"Improvement Hypothesis: {analysis['improvement_hypothesis'][:200]}...")
    
    # Generate strategy modifications
    current_strategy = StrategyContext(
        strategy_id="test_strategy",
        parameters={'risk_percentage': 2, 'stop_loss_pct': 2}
    )
    
    modifications = analyzer.generate_strategy_modifications(
        current_strategy, 
        analysis,
        MarketRegime.TRENDING_BULLISH
    )
    
    logger.info(f"Generated {len(modifications['modifications'])} strategy modifications")
    
    # Analyze patterns
    patterns = analyzer.analyze_trade_patterns(trade_history)
    logger.info(f"Identified {len(patterns['winning_patterns'])} winning patterns")
    logger.info(f"Identified {len(patterns['losing_patterns'])} losing patterns")


if __name__ == "__main__":
    asyncio.run(main())
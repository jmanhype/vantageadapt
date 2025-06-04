#!/usr/bin/env python3
"""
Trade Pattern Analyzer - Deep analysis of winning vs losing trade patterns
Implements Kagan's requirement: "types of trades that led to worst/best trades"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import dspy
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os

from src.utils.types import BacktestResults, MarketRegime

# Configure DSPy with OpenAI
turbo = dspy.LM('gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=turbo)


class WinningPatternAnalysis(dspy.Signature):
    """Analyze patterns in highly profitable trades."""
    
    winning_trades = dspy.InputField(desc="Details of the most profitable trades")
    market_conditions = dspy.InputField(desc="Market conditions during winning trades")
    technical_indicators = dspy.InputField(desc="Technical indicator values at trade entry")
    
    entry_patterns = dspy.OutputField(desc="Common patterns at entry points of winning trades")
    holding_patterns = dspy.OutputField(desc="Patterns during the holding period")
    exit_patterns = dspy.OutputField(desc="Patterns that led to profitable exits")
    replication_strategy = dspy.OutputField(desc="How to replicate these winning patterns")


class LosingPatternAnalysis(dspy.Signature):
    """Analyze patterns in losing trades to avoid them."""
    
    losing_trades = dspy.InputField(desc="Details of trades that resulted in losses")
    market_conditions = dspy.InputField(desc="Market conditions during losing trades")
    failure_indicators = dspy.InputField(desc="Warning signs that preceded losses")
    
    entry_mistakes = dspy.OutputField(desc="Common mistakes at entry that led to losses")
    hold_too_long_patterns = dspy.OutputField(desc="Patterns where positions were held too long")
    stop_loss_patterns = dspy.OutputField(desc="Patterns that should trigger earlier exits")
    avoidance_rules = dspy.OutputField(desc="Specific rules to avoid these losing patterns")


class MarketRegimePatterns(dspy.Signature):
    """Analyze trade patterns specific to market regimes."""
    
    regime_trades = dspy.InputField(desc="Trades grouped by market regime")
    regime_transitions = dspy.InputField(desc="Performance during regime transitions")
    regime_characteristics = dspy.InputField(desc="Key characteristics of each regime")
    
    regime_strategies = dspy.OutputField(desc="Optimal strategies for each market regime")
    transition_strategies = dspy.OutputField(desc="How to trade regime transitions")
    regime_indicators = dspy.OutputField(desc="Best indicators for each regime")


class TradePatternAnalyzer(dspy.Module):
    """
    Deep pattern analysis to understand what makes trades successful or fail.
    This implements Kagan's specific focus on understanding trade patterns.
    """
    
    def __init__(self):
        super().__init__()
        
        # Pattern analysis modules
        self.winning_analyzer = dspy.ChainOfThought(WinningPatternAnalysis)
        self.losing_analyzer = dspy.ChainOfThought(LosingPatternAnalysis)
        self.regime_analyzer = dspy.ChainOfThought(MarketRegimePatterns)
        
        # Pattern storage
        self.identified_patterns = {
            'winning': [],
            'losing': [],
            'regime_specific': {},
            'time_based': {},
            'asset_specific': {}
        }
        
        # Statistical thresholds
        self.min_pattern_occurrences = 5
        self.significance_threshold = 0.95
        
        logger.info("📊 Trade Pattern Analyzer initialized")
    
    def analyze_all_patterns(self,
                           trade_history: pd.DataFrame,
                           market_data: pd.DataFrame,
                           min_pattern_size: int = 10) -> Dict[str, Any]:
        """
        Comprehensive pattern analysis across all dimensions.
        """
        logger.info(f"🔍 Analyzing patterns in {len(trade_history)} trades")
        
        # 1. Basic win/loss pattern analysis
        basic_patterns = self.analyze_win_loss_patterns(trade_history, market_data)
        
        # 2. Time-based patterns
        time_patterns = self.analyze_time_patterns(trade_history)
        
        # 3. Market regime patterns
        regime_patterns = self.analyze_regime_patterns(trade_history, market_data)
        
        # 4. Consecutive patterns (Kagan's specific interest)
        consecutive_patterns = self.analyze_consecutive_patterns(trade_history)
        
        # 5. Volume and liquidity patterns
        volume_patterns = self.analyze_volume_patterns(trade_history, market_data)
        
        # 6. Technical indicator patterns
        indicator_patterns = self.analyze_indicator_patterns(trade_history, market_data)
        
        # 7. Asset-specific patterns
        asset_patterns = self.analyze_asset_patterns(trade_history)
        
        # Combine all analyses
        comprehensive_analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_trades_analyzed': len(trade_history),
            'basic_patterns': basic_patterns,
            'time_patterns': time_patterns,
            'regime_patterns': regime_patterns,
            'consecutive_patterns': consecutive_patterns,
            'volume_patterns': volume_patterns,
            'indicator_patterns': indicator_patterns,
            'asset_patterns': asset_patterns,
            'actionable_insights': self._generate_actionable_insights(
                basic_patterns, time_patterns, regime_patterns, 
                consecutive_patterns, volume_patterns, indicator_patterns
            )
        }
        
        # Store patterns for future reference
        self._store_patterns(comprehensive_analysis)
        
        return comprehensive_analysis
    
    def analyze_win_loss_patterns(self,
                                trades: pd.DataFrame,
                                market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Deep dive into what makes trades win or lose.
        """
        logger.info("Analyzing win/loss patterns...")
        
        # Separate winners and losers
        winners = trades[trades['pnl'] > 0].copy()
        losers = trades[trades['pnl'] <= 0].copy()
        
        # Get top/bottom performers for deep analysis
        top_winners = winners.nlargest(min(20, len(winners)), 'pnl')
        worst_losers = losers.nsmallest(min(20, len(losers)), 'pnl')
        
        # Prepare data for LLM analysis
        winning_details = self._prepare_trade_details(top_winners, market_data)
        losing_details = self._prepare_trade_details(worst_losers, market_data)
        
        try:
            # Analyze winning patterns
            win_analysis = self.winning_analyzer(
                winning_trades=json.dumps(winning_details, indent=2),
                market_conditions=json.dumps(self._get_market_conditions(top_winners, market_data)),
                technical_indicators=json.dumps(self._get_technical_indicators(top_winners, market_data))
            )
            
            # Analyze losing patterns
            loss_analysis = self.losing_analyzer(
                losing_trades=json.dumps(losing_details, indent=2),
                market_conditions=json.dumps(self._get_market_conditions(worst_losers, market_data)),
                failure_indicators=json.dumps(self._get_failure_indicators(worst_losers, market_data))
            )
            
            return {
                'winning_patterns': {
                    'entry_patterns': win_analysis.entry_patterns,
                    'holding_patterns': win_analysis.holding_patterns,
                    'exit_patterns': win_analysis.exit_patterns,
                    'replication_strategy': win_analysis.replication_strategy,
                    'statistics': self._calculate_pattern_statistics(winners)
                },
                'losing_patterns': {
                    'entry_mistakes': loss_analysis.entry_mistakes,
                    'hold_too_long': loss_analysis.hold_too_long_patterns,
                    'stop_loss_patterns': loss_analysis.stop_loss_patterns,
                    'avoidance_rules': loss_analysis.avoidance_rules,
                    'statistics': self._calculate_pattern_statistics(losers)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in win/loss pattern analysis: {e}")
            return self._fallback_win_loss_analysis(winners, losers)
    
    def analyze_consecutive_patterns(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Kagan's specific interest: "price went down consecutively... bought five times"
        """
        logger.info("Analyzing consecutive trade patterns...")
        
        consecutive_wins = []
        consecutive_losses = []
        current_streak = []
        streak_type = None
        
        for idx, trade in trades.iterrows():
            is_win = trade['pnl'] > 0
            
            if streak_type is None:
                streak_type = 'win' if is_win else 'loss'
                current_streak = [trade]
            elif (streak_type == 'win' and is_win) or (streak_type == 'loss' and not is_win):
                current_streak.append(trade)
            else:
                # Streak broken
                if len(current_streak) >= 3:  # Minimum 3 consecutive
                    if streak_type == 'win':
                        consecutive_wins.append({
                            'count': len(current_streak),
                            'total_pnl': sum(t['pnl'] for t in current_streak),
                            'start_time': current_streak[0]['entry_time'],
                            'end_time': current_streak[-1]['exit_time'],
                            'assets': [t['asset'] for t in current_streak],
                            'pattern': self._identify_streak_pattern(current_streak)
                        })
                    else:
                        consecutive_losses.append({
                            'count': len(current_streak),
                            'total_loss': sum(t['pnl'] for t in current_streak),
                            'start_time': current_streak[0]['entry_time'],
                            'end_time': current_streak[-1]['exit_time'],
                            'assets': [t['asset'] for t in current_streak],
                            'pattern': self._identify_streak_pattern(current_streak),
                            'warning_signs': self._identify_warning_signs(current_streak)
                        })
                
                # Start new streak
                streak_type = 'win' if is_win else 'loss'
                current_streak = [trade]
        
        return {
            'winning_streaks': consecutive_wins,
            'losing_streaks': consecutive_losses,
            'max_winning_streak': max([s['count'] for s in consecutive_wins]) if consecutive_wins else 0,
            'max_losing_streak': max([s['count'] for s in consecutive_losses]) if consecutive_losses else 0,
            'streak_reversal_patterns': self._analyze_streak_reversals(consecutive_wins, consecutive_losses),
            'recommendations': self._generate_streak_recommendations(consecutive_losses)
        }
    
    def analyze_time_patterns(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns based on time of day, day of week, etc.
        """
        logger.info("Analyzing time-based patterns...")
        
        # Add time features
        trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
        trades['day_of_week'] = pd.to_datetime(trades['entry_time']).dt.dayofweek
        trades['day_of_month'] = pd.to_datetime(trades['entry_time']).dt.day
        
        # Analyze by hour
        hourly_performance = trades.groupby('hour').agg({
            'pnl': ['mean', 'sum', 'count'],
            'asset': 'count'
        })
        
        # Analyze by day of week
        daily_performance = trades.groupby('day_of_week').agg({
            'pnl': ['mean', 'sum', 'count'],
            'asset': 'count'
        })
        
        # Find best/worst times
        best_hours = hourly_performance['pnl']['mean'].nlargest(3).index.tolist()
        worst_hours = hourly_performance['pnl']['mean'].nsmallest(3).index.tolist()
        
        best_days = daily_performance['pnl']['mean'].nlargest(2).index.tolist()
        worst_days = daily_performance['pnl']['mean'].nsmallest(2).index.tolist()
        
        return {
            'best_trading_hours': best_hours,
            'worst_trading_hours': worst_hours,
            'best_trading_days': self._day_names(best_days),
            'worst_trading_days': self._day_names(worst_days),
            'hourly_win_rate': self._calculate_hourly_win_rate(trades),
            'session_patterns': self._analyze_session_patterns(trades),
            'time_recommendations': self._generate_time_recommendations(
                best_hours, worst_hours, best_days, worst_days
            )
        }
    
    def analyze_volume_patterns(self,
                              trades: pd.DataFrame,
                              market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how volume affects trade outcomes.
        """
        logger.info("Analyzing volume patterns...")
        
        volume_patterns = {
            'high_volume_success': [],
            'low_volume_failures': [],
            'volume_spikes': [],
            'volume_divergences': []
        }
        
        for idx, trade in trades.iterrows():
            # Get volume data around trade
            trade_time = pd.to_datetime(trade['entry_time'])
            volume_window = self._get_volume_window(market_data, trade_time, trade['asset'])
            
            if volume_window is not None and len(volume_window) > 0:
                avg_volume = volume_window['volume'].mean()
                trade_volume = volume_window['volume'].iloc[-1]
                volume_ratio = trade_volume / avg_volume if avg_volume > 0 else 1
                
                # Categorize based on volume
                if volume_ratio > 2 and trade['pnl'] > 0:
                    volume_patterns['high_volume_success'].append({
                        'trade_id': idx,
                        'volume_ratio': volume_ratio,
                        'pnl': trade['pnl'],
                        'asset': trade['asset']
                    })
                elif volume_ratio < 0.5 and trade['pnl'] < 0:
                    volume_patterns['low_volume_failures'].append({
                        'trade_id': idx,
                        'volume_ratio': volume_ratio,
                        'pnl': trade['pnl'],
                        'asset': trade['asset']
                    })
        
        return {
            'volume_impact': self._calculate_volume_impact(volume_patterns),
            'optimal_volume_range': self._find_optimal_volume_range(trades, market_data),
            'volume_patterns': volume_patterns,
            'volume_rules': self._generate_volume_rules(volume_patterns)
        }
    
    def analyze_indicator_patterns(self,
                                 trades: pd.DataFrame,
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze which technical indicator combinations work best.
        """
        logger.info("Analyzing technical indicator patterns...")
        
        indicator_performance = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        
        # Common indicator combinations
        indicator_combos = [
            ['rsi_oversold', 'macd_bullish'],
            ['rsi_overbought', 'macd_bearish'],
            ['sma_crossover', 'volume_spike'],
            ['bollinger_squeeze', 'rsi_divergence'],
            ['support_bounce', 'volume_confirmation']
        ]
        
        # Analyze each combination
        for combo in indicator_combos:
            combo_key = '_'.join(combo)
            
            # Find trades matching this indicator combination
            matching_trades = self._find_indicator_matches(trades, market_data, combo)
            
            for trade in matching_trades:
                if trade['pnl'] > 0:
                    indicator_performance[combo_key]['wins'] += 1
                else:
                    indicator_performance[combo_key]['losses'] += 1
                indicator_performance[combo_key]['total_pnl'] += trade['pnl']
        
        # Calculate success rates
        indicator_stats = {}
        for combo, perf in indicator_performance.items():
            total_trades = perf['wins'] + perf['losses']
            if total_trades > 0:
                indicator_stats[combo] = {
                    'win_rate': perf['wins'] / total_trades,
                    'avg_pnl': perf['total_pnl'] / total_trades,
                    'total_trades': total_trades,
                    'reliability': self._calculate_reliability(perf)
                }
        
        return {
            'best_indicators': self._rank_indicators(indicator_stats),
            'indicator_combinations': indicator_stats,
            'regime_specific_indicators': self._analyze_regime_indicators(trades, market_data),
            'indicator_recommendations': self._generate_indicator_recommendations(indicator_stats)
        }
    
    def analyze_asset_patterns(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify asset-specific trading patterns.
        """
        logger.info("Analyzing asset-specific patterns...")
        
        asset_patterns = {}
        
        for asset in trades['asset'].unique():
            asset_trades = trades[trades['asset'] == asset]
            
            if len(asset_trades) >= self.min_pattern_occurrences:
                asset_patterns[asset] = {
                    'total_trades': len(asset_trades),
                    'win_rate': len(asset_trades[asset_trades['pnl'] > 0]) / len(asset_trades),
                    'avg_pnl': asset_trades['pnl'].mean(),
                    'total_pnl': asset_trades['pnl'].sum(),
                    'best_entry_patterns': self._find_best_entries(asset_trades),
                    'optimal_holding_time': self._calculate_optimal_holding(asset_trades),
                    'volatility_profile': self._analyze_asset_volatility(asset_trades),
                    'correlation_patterns': self._find_correlation_patterns(asset_trades, trades)
                }
        
        return {
            'asset_rankings': self._rank_assets(asset_patterns),
            'asset_specific_strategies': self._generate_asset_strategies(asset_patterns),
            'diversification_recommendations': self._analyze_diversification(asset_patterns),
            'asset_patterns': asset_patterns
        }
    
    def find_hidden_patterns(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Use clustering to find hidden patterns not visible to simple analysis.
        """
        logger.info("🔍 Searching for hidden patterns using ML clustering...")
        
        # Prepare features for clustering
        features = self._prepare_clustering_features(trades)
        
        if len(features) < 10:
            logger.warning("Not enough data for clustering analysis")
            return {}
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Find optimal number of clusters
        optimal_k = self._find_optimal_clusters(scaled_features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Analyze each cluster
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_trades = trades[clusters == cluster_id]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_trades),
                'avg_pnl': cluster_trades['pnl'].mean(),
                'win_rate': len(cluster_trades[cluster_trades['pnl'] > 0]) / len(cluster_trades),
                'characteristics': self._describe_cluster(cluster_trades, features[clusters == cluster_id]),
                'trading_rules': self._generate_cluster_rules(cluster_trades)
            }
        
        return {
            'hidden_patterns': cluster_analysis,
            'pattern_significance': self._assess_pattern_significance(cluster_analysis),
            'implementation_strategy': self._create_pattern_strategy(cluster_analysis)
        }
    
    def _prepare_trade_details(self, trades: pd.DataFrame, market_data: pd.DataFrame) -> List[Dict]:
        """Prepare detailed trade information for analysis."""
        details = []
        
        for idx, trade in trades.iterrows():
            details.append({
                'trade_id': idx,
                'asset': trade['asset'],
                'entry_time': str(trade['entry_time']),
                'exit_time': str(trade['exit_time']),
                'pnl': float(trade['pnl']),
                'pnl_percentage': float(trade.get('pnl_pct', trade['pnl'] / 1000)),
                'holding_period': str(pd.to_datetime(trade['exit_time']) - pd.to_datetime(trade['entry_time'])),
                'entry_price': float(trade.get('entry_price', 0)),
                'exit_price': float(trade.get('exit_price', 0)),
                'position_size': float(trade.get('position_size', 0))
            })
        
        return details
    
    def _identify_streak_pattern(self, streak: List[Dict]) -> str:
        """Identify the pattern in a winning/losing streak."""
        if not streak:
            return "unknown"
        
        # Check if same asset
        assets = [t['asset'] for t in streak]
        if len(set(assets)) == 1:
            return f"single_asset_streak_{assets[0]}"
        
        # Check if same time period
        times = [pd.to_datetime(t['entry_time']).hour for t in streak]
        if len(set(times)) <= 2:
            return f"time_cluster_streak_hour_{times[0]}"
        
        # Check if increasing/decreasing position sizes
        if 'position_size' in streak[0]:
            sizes = [t['position_size'] for t in streak]
            if all(sizes[i] >= sizes[i-1] for i in range(1, len(sizes))):
                return "martingale_pattern"
            elif all(sizes[i] <= sizes[i-1] for i in range(1, len(sizes))):
                return "anti_martingale_pattern"
        
        return "mixed_pattern"
    
    def _generate_actionable_insights(self, *pattern_analyses) -> List[Dict[str, str]]:
        """Generate specific actionable insights from all pattern analyses."""
        insights = []
        
        # Extract key insights from each analysis
        for analysis in pattern_analyses:
            if isinstance(analysis, dict):
                # Add win/loss insights
                if 'winning_patterns' in analysis:
                    insights.append({
                        'type': 'opportunity',
                        'priority': 'high',
                        'insight': 'Winning pattern identified',
                        'action': analysis['winning_patterns'].get('replication_strategy', 'Replicate winning conditions')
                    })
                
                # Add time-based insights
                if 'best_trading_hours' in analysis:
                    insights.append({
                        'type': 'timing',
                        'priority': 'medium',
                        'insight': f"Best trading hours: {analysis['best_trading_hours']}",
                        'action': f"Focus trading activity during hours {analysis['best_trading_hours']}"
                    })
                
                # Add consecutive loss insights
                if 'losing_streaks' in analysis and analysis.get('max_losing_streak', 0) > 3:
                    insights.append({
                        'type': 'risk',
                        'priority': 'critical',
                        'insight': f"Max losing streak: {analysis['max_losing_streak']} trades",
                        'action': "Implement maximum consecutive loss limit of 3 trades"
                    })
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        insights.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return insights
    
    def _calculate_pattern_statistics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical measures for pattern reliability."""
        if len(trades) == 0:
            return {}
        
        return {
            'count': len(trades),
            'win_rate': len(trades[trades['pnl'] > 0]) / len(trades),
            'avg_pnl': float(trades['pnl'].mean()),
            'std_pnl': float(trades['pnl'].std()),
            'sharpe_ratio': float(trades['pnl'].mean() / trades['pnl'].std()) if trades['pnl'].std() > 0 else 0,
            'max_win': float(trades['pnl'].max()),
            'max_loss': float(trades['pnl'].min()),
            'profit_factor': abs(trades[trades['pnl'] > 0]['pnl'].sum() / trades[trades['pnl'] < 0]['pnl'].sum()) if len(trades[trades['pnl'] < 0]) > 0 else float('inf')
        }
    
    def _store_patterns(self, analysis: Dict[str, Any]):
        """Store identified patterns for future reference."""
        # Update pattern storage
        if 'basic_patterns' in analysis:
            self.identified_patterns['winning'].append(analysis['basic_patterns'].get('winning_patterns', {}))
            self.identified_patterns['losing'].append(analysis['basic_patterns'].get('losing_patterns', {}))
        
        if 'regime_patterns' in analysis:
            for regime, patterns in analysis['regime_patterns'].items():
                if regime not in self.identified_patterns['regime_specific']:
                    self.identified_patterns['regime_specific'][regime] = []
                self.identified_patterns['regime_specific'][regime].append(patterns)
        
        # Keep only recent patterns (last 100)
        for key in ['winning', 'losing']:
            if len(self.identified_patterns[key]) > 100:
                self.identified_patterns[key] = self.identified_patterns[key][-100:]
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all identified patterns."""
        return {
            'total_patterns_identified': sum(len(v) if isinstance(v, list) else len(v) 
                                           for v in self.identified_patterns.values()),
            'winning_patterns': len(self.identified_patterns['winning']),
            'losing_patterns': len(self.identified_patterns['losing']),
            'regime_patterns': len(self.identified_patterns['regime_specific']),
            'most_reliable_patterns': self._find_most_reliable_patterns(),
            'pattern_evolution': self._analyze_pattern_evolution()
        }
    
    def _find_most_reliable_patterns(self) -> List[Dict[str, Any]]:
        """Find patterns with highest reliability scores."""
        reliable_patterns = []
        
        # Check winning patterns
        for pattern in self.identified_patterns['winning'][-10:]:  # Last 10
            if isinstance(pattern, dict) and 'statistics' in pattern:
                stats = pattern['statistics']
                if stats.get('win_rate', 0) > 0.7 and stats.get('count', 0) > 10:
                    reliable_patterns.append({
                        'type': 'winning',
                        'pattern': pattern,
                        'reliability_score': stats['win_rate'] * min(stats['count'] / 50, 1.0)
                    })
        
        # Sort by reliability
        reliable_patterns.sort(key=lambda x: x['reliability_score'], reverse=True)
        
        return reliable_patterns[:5]  # Top 5


def test_pattern_analyzer():
    """Test the pattern analyzer with sample data."""
    analyzer = TradePatternAnalyzer()
    
    # Create sample trade data
    np.random.seed(42)
    n_trades = 200
    
    trade_data = pd.DataFrame({
        'entry_time': pd.date_range('2024-01-01', periods=n_trades, freq='H'),
        'exit_time': pd.date_range('2024-01-01 01:00:00', periods=n_trades, freq='H'),
        'asset': np.random.choice(['BTC', 'ETH', 'SOL'], n_trades),
        'pnl': np.random.normal(0, 100, n_trades),
        'entry_price': np.random.uniform(1000, 2000, n_trades),
        'exit_price': np.random.uniform(1000, 2000, n_trades),
        'position_size': np.random.uniform(0.1, 1.0, n_trades)
    })
    
    # Create sample market data
    market_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_trades*2, freq='30min'),
        'close': np.random.uniform(1000, 2000, n_trades*2),
        'volume': np.random.uniform(100, 1000, n_trades*2),
        'asset': 'BTC'
    })
    
    # Run analysis
    patterns = analyzer.analyze_all_patterns(trade_data, market_data)
    
    logger.info("Pattern Analysis Complete:")
    logger.info(f"Total trades analyzed: {patterns['total_trades_analyzed']}")
    logger.info(f"Actionable insights: {len(patterns['actionable_insights'])}")
    
    # Get summary
    summary = analyzer.get_pattern_summary()
    logger.info(f"Pattern Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    test_pattern_analyzer()
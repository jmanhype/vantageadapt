"""Trade analysis module for analyzing trading performance."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TradePattern:
    """Container for trade pattern analysis."""
    pattern_type: str
    frequency: int
    avg_return: float
    win_rate: float
    context: Dict[str, Any]
    description: str

@dataclass
class MarketContext:
    """Container for market context analysis."""
    volatility: float
    trend: str
    volume_profile: str
    price_level: float
    support_resistance: List[float]
    description: str

@dataclass
class TradeAnalysis:
    """Container for comprehensive trade analysis."""
    patterns: List[TradePattern]
    market_context: MarketContext
    entry_analysis: Dict[str, Any]
    exit_analysis: Dict[str, Any]
    risk_metrics: Dict[str, float]
    behavioral_metrics: Dict[str, Any]

class TradeAnalyzer:
    """Analyzes trading performance and patterns."""
    
    def __init__(self, trades_df: pd.DataFrame, price_data: pd.DataFrame):
        """Initialize trade analyzer.
        
        Args:
            trades_df: DataFrame containing trade records
            price_data: DataFrame containing price history
        """
        self.trades_df = trades_df.copy()
        self.price_data = price_data.copy()
        
        # Convert index to datetime if numeric
        if pd.api.types.is_numeric_dtype(self.trades_df.index):
            self.trades_df.index = pd.to_datetime(self.trades_df.index, unit='s')
        
        # Initialize analysis results
        self.patterns: List[TradePattern] = []
        self.market_contexts: Dict[datetime, MarketContext] = {}
        self.entry_exit_analysis: Dict[str, Any] = {}
        self.risk_metrics: Dict[str, float] = {}
        self.behavioral_metrics: Dict[str, Any] = {}

    def analyze_trade_patterns(self) -> List[TradePattern]:
        """Analyze trading patterns.
            
        Returns:
            List of identified trade patterns
        """
        patterns = []
        
        # Analyze consecutive wins/losses
        streak_data = self._analyze_streaks()
        if streak_data:
            patterns.append(TradePattern(
                pattern_type="win_loss_streaks",
                frequency=streak_data["frequency"],
                avg_return=streak_data["avg_return"],
                win_rate=streak_data["win_rate"],
                context=streak_data["context"],
                description=streak_data["description"]
            ))
            
        # Analyze time-based patterns
        time_patterns = self._analyze_time_patterns()
        patterns.extend(time_patterns)
        
        # Analyze size-based patterns
        size_patterns = self._analyze_size_patterns()
        patterns.extend(size_patterns)
        
        # Analyze market condition patterns
        market_patterns = self._analyze_market_condition_patterns()
        patterns.extend(market_patterns)
        
        return patterns

    def analyze_market_context(self) -> Dict[datetime, MarketContext]:
        """Analyze market context around trades.
        
        Returns:
            Dictionary mapping trade times to market context
        """
        contexts = {}
        
        for idx, trade in self.trades_df.iterrows():
            # Get price data around trade
            window = self._get_price_window(idx, minutes=60)
            
            if window is not None:
                # Calculate volatility
                volatility = window['price'].pct_change().std()
                
                # Determine trend
                trend = 'up' if window['price'].iloc[-1] > window['price'].iloc[0] else 'down'
                
                # Analyze volume profile
                volume_profile = self._analyze_volume_profile(window)
                
                # Find support/resistance levels
                support_resistance = self._find_support_resistance(window)
                
                # Create context object
                contexts[idx] = MarketContext(
                    volatility=volatility,
                    trend=trend,
                    volume_profile=volume_profile,
                    price_level=trade['price'],
                    support_resistance=support_resistance,
                    description=self._generate_context_description(
                        volatility, trend, volume_profile, support_resistance
                    )
                )
                
        return contexts

    def analyze_entries_exits(self) -> Dict[str, Any]:
        """Analyze entry and exit execution.
            
        Returns:
            Dictionary containing entry/exit analysis
        """
        analysis = {
            'entries': self._analyze_entries(),
            'exits': self._analyze_exits(),
            'timing_efficiency': self._analyze_timing_efficiency(),
            'missed_opportunities': self._analyze_missed_opportunities()
        }
        return analysis

    def analyze_risk_metrics(self) -> Dict[str, float]:
        """Calculate detailed risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        metrics = {
            'max_drawdown': self._calculate_max_drawdown(),
            'value_at_risk': self._calculate_var(),
            'expected_shortfall': self._calculate_expected_shortfall(),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(),
            'position_concentration': self._calculate_position_concentration()
        }
        return metrics

    def analyze_behavioral_patterns(self) -> Dict[str, Any]:
        """Analyze trading behavior patterns.
        
        Returns:
            Dictionary containing behavioral analysis
        """
        patterns = {
            'overtrading': self._detect_overtrading(),
            'hesitation': self._detect_hesitation(),
            'revenge_trading': self._detect_revenge_trading(),
            'position_sizing': self._analyze_position_sizing(),
            'risk_management': self._analyze_risk_management_adherence()
        }
        return patterns

    def get_comprehensive_analysis(self) -> TradeAnalysis:
        """Get comprehensive trade analysis.
        
        Returns:
            TradeAnalysis object containing all analysis results
        """
        # Run all analyses
        self.patterns = self.analyze_trade_patterns()
        self.market_contexts = self.analyze_market_context()
        self.entry_exit_analysis = self.analyze_entries_exits()
        self.risk_metrics = self.analyze_risk_metrics()
        self.behavioral_metrics = self.analyze_behavioral_patterns()
        
        # Get most recent market context
        latest_context = next(iter(sorted(self.market_contexts.items(), reverse=True)))[1]
        
        return TradeAnalysis(
            patterns=self.patterns,
            market_context=latest_context,
            entry_analysis=self.entry_exit_analysis['entries'],
            exit_analysis=self.entry_exit_analysis['exits'],
            risk_metrics=self.risk_metrics,
            behavioral_metrics=self.behavioral_metrics
        )

    def _analyze_streaks(self) -> Optional[Dict[str, Any]]:
        """Analyze winning and losing streaks."""
        if len(self.trades_df) < 2:
            return None
            
        # Calculate trade results
        self.trades_df['is_win'] = self.trades_df['pnl'] > 0
        
        # Find streaks
        streak_data = {
            'win_streaks': [],
            'loss_streaks': []
        }
        
        current_streak = 1
        current_type = self.trades_df['is_win'].iloc[0]
        
        for i in range(1, len(self.trades_df)):
            if self.trades_df['is_win'].iloc[i] == current_type:
                current_streak += 1
            else:
                if current_type:
                    streak_data['win_streaks'].append(current_streak)
                else:
                    streak_data['loss_streaks'].append(current_streak)
                current_streak = 1
                current_type = self.trades_df['is_win'].iloc[i]
                
        # Add final streak
        if current_type:
            streak_data['win_streaks'].append(current_streak)
        else:
            streak_data['loss_streaks'].append(current_streak)
            
        # Calculate streak statistics
        avg_win_streak = np.mean(streak_data['win_streaks']) if streak_data['win_streaks'] else 0
        avg_loss_streak = np.mean(streak_data['loss_streaks']) if streak_data['loss_streaks'] else 0
        max_win_streak = max(streak_data['win_streaks']) if streak_data['win_streaks'] else 0
        max_loss_streak = max(streak_data['loss_streaks']) if streak_data['loss_streaks'] else 0
        
        return {
            "frequency": len(streak_data['win_streaks']) + len(streak_data['loss_streaks']),
            "avg_return": self.trades_df['pnl'].mean(),
            "win_rate": len(streak_data['win_streaks']) / (len(streak_data['win_streaks']) + len(streak_data['loss_streaks'])),
            "context": {
                "avg_win_streak": avg_win_streak,
                "avg_loss_streak": avg_loss_streak,
                "max_win_streak": max_win_streak,
                "max_loss_streak": max_loss_streak
            },
            "description": f"Average win streak: {avg_win_streak:.1f}, Average loss streak: {avg_loss_streak:.1f}"
        }

    def _analyze_time_patterns(self) -> List[TradePattern]:
        """Analyze time-based trading patterns."""
        patterns = []
        
        # Add hour of day analysis
        self.trades_df['hour'] = self.trades_df.index.hour
        hourly_stats = self.trades_df.groupby('hour')['pnl'].agg(['mean', 'count', 'sum'])
        
        for hour, stats in hourly_stats.iterrows():
            if stats['count'] >= 5:  # Minimum sample size
                patterns.append(TradePattern(
                    pattern_type="hourly",
                    frequency=int(stats['count']),
                    avg_return=float(stats['mean']),
                    win_rate=float((self.trades_df[self.trades_df['hour'] == hour]['pnl'] > 0).mean()),
                    context={"hour": hour},
                    description=f"Hour {hour:02d}:00 pattern"
                ))
                
        # Add day of week analysis
        self.trades_df['day'] = self.trades_df.index.day_name()
        daily_stats = self.trades_df.groupby('day')['pnl'].agg(['mean', 'count', 'sum'])
        
        for day, stats in daily_stats.iterrows():
            if stats['count'] >= 5:
                patterns.append(TradePattern(
                    pattern_type="daily",
                    frequency=int(stats['count']),
                    avg_return=float(stats['mean']),
                    win_rate=float((self.trades_df[self.trades_df['day'] == day]['pnl'] > 0).mean()),
                    context={"day": day},
                    description=f"{day} pattern"
                ))
                
        return patterns

    def _analyze_size_patterns(self) -> List[TradePattern]:
        """Analyze position size-based patterns."""
        patterns = []
        
        # Analyze performance by position size quartiles
        self.trades_df['size_quartile'] = pd.qcut(self.trades_df['size'], 4, labels=['small', 'medium', 'large', 'xlarge'])
        size_stats = self.trades_df.groupby('size_quartile')['pnl'].agg(['mean', 'count', 'sum'])
        
        for size, stats in size_stats.iterrows():
            patterns.append(TradePattern(
                pattern_type="position_size",
                frequency=int(stats['count']),
                avg_return=float(stats['mean']),
                win_rate=float((self.trades_df[self.trades_df['size_quartile'] == size]['pnl'] > 0).mean()),
                context={"size_category": size},
                description=f"{size.capitalize()} position size pattern"
            ))
            
        return patterns

    def _analyze_market_condition_patterns(self) -> List[TradePattern]:
        """Analyze patterns based on market conditions."""
        patterns = []
        
        # Analyze volatility impact
        self.trades_df['volatility'] = self.trades_df['price'].pct_change().rolling(20).std()
        self.trades_df['volatility_quartile'] = pd.qcut(self.trades_df['volatility'], 4, labels=['low', 'medium', 'high', 'extreme'])
        
        vol_stats = self.trades_df.groupby('volatility_quartile')['pnl'].agg(['mean', 'count', 'sum'])
        
        for vol, stats in vol_stats.iterrows():
            if stats['count'] >= 5:
                patterns.append(TradePattern(
                    pattern_type="volatility",
                    frequency=int(stats['count']),
                    avg_return=float(stats['mean']),
                    win_rate=float((self.trades_df[self.trades_df['volatility_quartile'] == vol]['pnl'] > 0).mean()),
                    context={"volatility_level": vol},
                    description=f"{vol.capitalize()} volatility pattern"
                ))
                
        return patterns

    def _get_price_window(self, timestamp: datetime, minutes: int = 60) -> Optional[pd.DataFrame]:
        """Get price data window around timestamp."""
        try:
            start_idx = self.price_data.index.get_loc(timestamp - pd.Timedelta(minutes=minutes))
            end_idx = self.price_data.index.get_loc(timestamp + pd.Timedelta(minutes=minutes))
            return self.price_data.iloc[start_idx:end_idx+1]
        except KeyError:
            return None

    def _analyze_volume_profile(self, window: pd.DataFrame) -> str:
        """Analyze volume profile in price window."""
        if 'volume' not in window.columns:
            return 'unknown'
            
        avg_volume = window['volume'].mean()
        recent_volume = window['volume'].iloc[-5:].mean()
        
        if recent_volume > avg_volume * 1.5:
            return 'increasing'
        elif recent_volume < avg_volume * 0.5:
            return 'decreasing'
        else:
            return 'stable'

    def _find_support_resistance(self, window: pd.DataFrame) -> List[float]:
        """Find support and resistance levels."""
        levels = []
        
        # Simple method using price quartiles
        price_quartiles = window['price'].quantile([0.25, 0.75])
        levels.extend(price_quartiles.tolist())
        
        return levels

    def _generate_context_description(self, volatility: float, trend: str, volume_profile: str, support_resistance: List[float]) -> str:
        """Generate market context description."""
        return f"{trend.capitalize()} trend with {volume_profile} volume and {volatility:.2%} volatility"

    def _analyze_entries(self) -> Dict[str, Any]:
        """Analyze entry execution."""
        entry_analysis = {
            'timing': self._analyze_entry_timing(),
            'price_levels': self._analyze_entry_price_levels(),
            'success_rate': self._calculate_entry_success_rate()
        }
        return entry_analysis

    def _analyze_exits(self) -> Dict[str, Any]:
        """Analyze exit execution."""
        exit_analysis = {
            'timing': self._analyze_exit_timing(),
            'price_levels': self._analyze_exit_price_levels(),
            'efficiency': self._calculate_exit_efficiency()
        }
        return exit_analysis

    def _analyze_timing_efficiency(self) -> float:
        """Calculate timing efficiency score."""
        # Simple implementation - can be enhanced
        if len(self.trades_df) == 0:
            return 0.0
            
        profitable_trades = self.trades_df[self.trades_df['pnl'] > 0]
        return len(profitable_trades) / len(self.trades_df)

    def _analyze_missed_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze potential missed trading opportunities."""
        missed = []
        
        # Simple implementation - can be enhanced
        # Look for price movements that would have been profitable
        return missed

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.trades_df) == 0:
            return 0.0
            
        cumulative_returns = (1 + self.trades_df['pnl']).cumprod()
        rolling_max = cumulative_returns.expanding(min_periods=1).max()
        drawdowns = cumulative_returns / rolling_max - 1
        return float(drawdowns.min())

    def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        if len(self.trades_df) == 0:
            return 0.0
            
        return float(np.percentile(self.trades_df['pnl'], (1 - confidence) * 100))

    def _calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(self.trades_df) == 0:
            return 0.0
            
        var = self._calculate_var(confidence)
        return float(self.trades_df[self.trades_df['pnl'] <= var]['pnl'].mean())

    def _calculate_risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        if len(self.trades_df) == 0:
            return 0.0
            
        avg_win = self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean()
        avg_loss = abs(self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean())
        
        return float(avg_win / avg_loss if avg_loss != 0 else 0)

    def _calculate_position_concentration(self) -> float:
        """Calculate position concentration risk."""
        if len(self.trades_df) == 0:
            return 0.0
            
        # Use Herfindahl-Hirschman Index
        position_sizes = self.trades_df['size']
        total_size = position_sizes.sum()
        if total_size == 0:
            return 0.0
            
        market_shares = position_sizes / total_size
        hhi = (market_shares ** 2).sum()
        
        return float(hhi)

    def _detect_overtrading(self) -> Dict[str, Any]:
        """Detect potential overtrading behavior."""
        if len(self.trades_df) == 0:
            return {'detected': False, 'evidence': []}
            
        # Calculate trade frequency
        trade_frequency = len(self.trades_df) / (self.trades_df.index[-1] - self.trades_df.index[0]).days
        
        # Look for clusters of trades
        trade_gaps = self.trades_df.index[1:] - self.trades_df.index[:-1]
        short_gaps = trade_gaps[trade_gaps < pd.Timedelta(minutes=5)]
        
        evidence = []
        if trade_frequency > 10:  # More than 10 trades per day
            evidence.append(f"High trade frequency: {trade_frequency:.1f} trades/day")
        if len(short_gaps) > len(self.trades_df) * 0.3:  # More than 30% of trades are close together
            evidence.append(f"Frequent short-term trading: {len(short_gaps)} trades < 5min apart")
            
            return {
            'detected': len(evidence) > 0,
            'evidence': evidence
        }

    def _detect_hesitation(self) -> Dict[str, Any]:
        """Detect hesitation in trade execution."""
        if len(self.trades_df) == 0:
            return {'detected': False, 'evidence': []}
            
        evidence = []
        
        # Look for delayed entries after signals
        # This requires signal data which we don't have yet
        
        return {
            'detected': len(evidence) > 0,
            'evidence': evidence
        }

    def _detect_revenge_trading(self) -> Dict[str, Any]:
        """Detect potential revenge trading behavior."""
        if len(self.trades_df) == 0:
            return {'detected': False, 'evidence': []}
            
        evidence = []
        
        # Look for increased position sizes after losses
        self.trades_df['prev_pnl'] = self.trades_df['pnl'].shift(1)
        self.trades_df['size_change'] = self.trades_df['size'] / self.trades_df['size'].shift(1)
        
        revenge_trades = self.trades_df[
            (self.trades_df['prev_pnl'] < 0) & 
            (self.trades_df['size_change'] > 1.5)  # 50% size increase
        ]
        
        if len(revenge_trades) > 0:
            evidence.append(f"Found {len(revenge_trades)} potential revenge trades")
            
        return {
            'detected': len(evidence) > 0,
            'evidence': evidence
        }

    def _analyze_position_sizing(self) -> Dict[str, Any]:
        """Analyze position sizing behavior."""
        if len(self.trades_df) == 0:
            return {}
            
        analysis = {
            'avg_size': float(self.trades_df['size'].mean()),
            'size_volatility': float(self.trades_df['size'].std()),
            'size_trend': 'increasing' if self.trades_df['size'].corr(pd.Series(range(len(self.trades_df)))) > 0 else 'decreasing',
            'size_distribution': self.trades_df['size'].describe().to_dict()
        }
        return analysis

    def _analyze_risk_management_adherence(self) -> Dict[str, Any]:
        """Analyze adherence to risk management rules."""
        if len(self.trades_df) == 0:
            return {}
            
        analysis = {
            'max_position_size': float(self.trades_df['size'].max()),
            'avg_loss_size': float(abs(self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean())),
            'max_daily_loss': float(self.trades_df.resample('D')['pnl'].sum().min()),
            'risk_per_trade': float(self.trades_df['pnl'].std())
        }
        return analysis

    def _analyze_entry_timing(self) -> Dict[str, float]:
        """Analyze entry timing effectiveness."""
        if len(self.trades_df) == 0:
            return {}
            
        # Calculate entry timing metrics
        entry_timing = {
            'avg_entry_slippage': 0.0,  # Requires intended entry price
            'entry_timing_score': self._calculate_entry_timing_score()
        }
        return entry_timing

    def _analyze_entry_price_levels(self) -> Dict[str, Any]:
        """Analyze entry price levels."""
        if len(self.trades_df) == 0:
            return {}
            
        # Analyze entry prices relative to recent price history
        entry_levels = {
            'avg_entry_price': float(self.trades_df['price'].mean()),
            'entry_price_distribution': self.trades_df['price'].describe().to_dict()
        }
        return entry_levels

    def _calculate_entry_success_rate(self) -> float:
        """Calculate entry success rate."""
        if len(self.trades_df) == 0:
            return 0.0
            
        return float((self.trades_df['pnl'] > 0).mean())

    def _analyze_exit_timing(self) -> Dict[str, float]:
        """Analyze exit timing effectiveness."""
        if len(self.trades_df) == 0:
            return {}
            
        # Calculate exit timing metrics
        exit_timing = {
            'avg_exit_slippage': 0.0,  # Requires intended exit price
            'exit_timing_score': self._calculate_exit_timing_score()
        }
        return exit_timing

    def _analyze_exit_price_levels(self) -> Dict[str, Any]:
        """Analyze exit price levels."""
        if len(self.trades_df) == 0:
            return {}
            
        # Analyze exit prices relative to recent price history
        exit_levels = {
            'avg_exit_price': float(self.trades_df['price'].mean()),
            'exit_price_distribution': self.trades_df['price'].describe().to_dict()
        }
        return exit_levels

    def _calculate_exit_efficiency(self) -> float:
        """Calculate exit efficiency score."""
        if len(self.trades_df) == 0:
            return 0.0
            
        # Simple implementation - can be enhanced
        profitable_exits = self.trades_df[self.trades_df['pnl'] > 0]
        return float(len(profitable_exits) / len(self.trades_df))

    def _calculate_entry_timing_score(self) -> float:
        """Calculate entry timing effectiveness score."""
        if len(self.trades_df) == 0:
            return 0.0
            
        # Simple implementation - can be enhanced
        return float((self.trades_df['pnl'] > 0).mean())

    def _calculate_exit_timing_score(self) -> float:
        """Calculate exit timing effectiveness score."""
        if len(self.trades_df) == 0:
            return 0.0
            
        # Simple implementation - can be enhanced
        return float((self.trades_df['pnl'] > 0).mean())
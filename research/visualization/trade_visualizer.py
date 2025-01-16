"""Visualization module for trade analysis."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from ..analysis.trade_analyzer import TradeAnalysis, TradePattern, MarketContext

logger = logging.getLogger(__name__)

class TradeVisualizer:
    """Visualize trading results and performance metrics."""

    def __init__(self, price_data: pd.DataFrame, trades_df: pd.DataFrame):
        """Initialize trade visualizer.
        
        Args:
            price_data: DataFrame with price history
            trades_df: DataFrame with trade history
        """
        self.price_data = price_data.copy()
        self.trades_df = trades_df.copy()
        
        # Ensure index is datetime
        if pd.api.types.is_numeric_dtype(self.trades_df.index):
            self.trades_df.index = pd.to_datetime(self.trades_df.index, unit='s')
            
        # Determine price column
        if 'dex_price' in self.price_data.columns:
            self.price_col = 'dex_price'
        elif 'price' in self.price_data.columns:
            self.price_col = 'price'
        elif 'close' in self.price_data.columns:
            self.price_col = 'close'
        else:
            # If no standard price column found, use the first numeric column
            numeric_cols = self.price_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.price_col = numeric_cols[0]
            else:
                raise ValueError("No suitable price column found in price_data")
                
    def plot_trades(self, title: str = "Trade Analysis", filename: Optional[str] = None) -> None:
        """Plot comprehensive trade analysis.
        
        Args:
            title: Plot title
            filename: Optional file to save plot to
        """
        try:
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Price and Trades",
                    "Trade Size Distribution",
                    "Returns Distribution",
                    "Win/Loss Patterns",
                    "Trade Duration",
                    "Market Impact"
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot price and trades
            self._add_price_trades_subplot(fig, row=1, col=1)
            
            # Plot trade size distribution
            self._add_size_distribution_subplot(fig, row=1, col=2)
            
            # Plot returns distribution
            self._add_returns_distribution_subplot(fig, row=2, col=1)
            
            # Plot win/loss patterns
            self._add_win_loss_patterns_subplot(fig, row=2, col=2)
            
            # Plot trade duration
            self._add_trade_duration_subplot(fig, row=3, col=1)
            
            # Plot market impact
            self._add_market_impact_subplot(fig, row=3, col=2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=1200,
                showlegend=True,
                template="plotly_white"
            )
            
            # Save if filename provided
            if filename:
                fig.write_html(filename)
                
        except Exception as e:
            logger.error(f"Error plotting trades: {str(e)}")
            logger.exception("Full traceback:")
            
    def plot_returns(self, title: str = "Returns Analysis", filename: Optional[str] = None) -> None:
        """Plot comprehensive returns analysis.
        
        Args:
            title: Plot title
            filename: Optional file to save plot to
        """
        try:
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Cumulative Returns",
                    "Rolling Metrics",
                    "Drawdown Analysis",
                    "Return Attribution",
                    "Risk Metrics",
                    "Performance Breakdown"
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot cumulative returns
            self._add_cumulative_returns_subplot(fig, row=1, col=1)
            
            # Plot rolling metrics
            self._add_rolling_metrics_subplot(fig, row=1, col=2)
            
            # Plot drawdown analysis
            self._add_drawdown_analysis_subplot(fig, row=2, col=1)
            
            # Plot return attribution
            self._add_return_attribution_subplot(fig, row=2, col=2)
            
            # Plot risk metrics
            self._add_risk_metrics_subplot(fig, row=3, col=1)
            
            # Plot performance breakdown
            self._add_performance_breakdown_subplot(fig, row=3, col=2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=1200,
                showlegend=True,
                template="plotly_white"
            )
            
            # Save if filename provided
            if filename:
                fig.write_html(filename)
                
        except Exception as e:
            logger.error(f"Error plotting returns: {str(e)}")
            logger.exception("Full traceback:")
            
    def plot_market_context(self, market_context: MarketContext, title: str = "Market Context", filename: Optional[str] = None) -> None:
        """Plot market context analysis.
        
        Args:
            market_context: MarketContext object
            title: Plot title
            filename: Optional file to save plot to
        """
        try:
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Price Levels and S/R",
                    "Volatility Profile",
                    "Volume Analysis",
                    "Market Regime"
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot price levels and support/resistance
            self._add_price_levels_subplot(fig, market_context, row=1, col=1)
            
            # Plot volatility profile
            self._add_volatility_profile_subplot(fig, market_context, row=1, col=2)
            
            # Plot volume analysis
            self._add_volume_analysis_subplot(fig, market_context, row=2, col=1)
            
            # Plot market regime
            self._add_market_regime_subplot(fig, market_context, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                showlegend=True,
                template="plotly_white"
            )
            
            # Save if filename provided
            if filename:
                fig.write_html(filename)
                
        except Exception as e:
            logger.error(f"Error plotting market context: {str(e)}")
            logger.exception("Full traceback:")
            
    def plot_pattern_analysis(self, patterns: List[TradePattern], title: str = "Pattern Analysis", filename: Optional[str] = None) -> None:
        """Plot trade pattern analysis.
        
        Args:
            patterns: List of TradePattern objects
            title: Plot title
            filename: Optional file to save plot to
        """
        try:
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Pattern Distribution",
                    "Pattern Performance",
                    "Pattern Context",
                    "Pattern Timeline"
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Plot pattern distribution
            self._add_pattern_distribution_subplot(fig, patterns, row=1, col=1)
            
            # Plot pattern performance
            self._add_pattern_performance_subplot(fig, patterns, row=1, col=2)
            
            # Plot pattern context
            self._add_pattern_context_subplot(fig, patterns, row=2, col=1)
            
            # Plot pattern timeline
            self._add_pattern_timeline_subplot(fig, patterns, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title=title,
                height=800,
                showlegend=True,
                template="plotly_white"
            )
            
            # Save if filename provided
            if filename:
                fig.write_html(filename)
                
        except Exception as e:
            logger.error(f"Error plotting pattern analysis: {str(e)}")
            logger.exception("Full traceback:")
            
    def _add_price_trades_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add price and trades subplot."""
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=self.price_data.index,
                y=self.price_data[self.price_col],
                name="Price",
                line=dict(color='blue', width=1)
            ),
            row=row, col=col
        )
        
        # Add trade entries
        fig.add_trace(
            go.Scatter(
                x=self.trades_df.index,
                y=self.trades_df['entry_price'],
                mode='markers',
                name="Entries",
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color=['green' if pnl > 0 else 'red' for pnl in self.trades_df['pnl']],
                    line=dict(width=1, color='black')
                )
            ),
            row=row, col=col
        )
        
        # Add trade exits
        fig.add_trace(
            go.Scatter(
                x=self.trades_df.index,
                y=self.trades_df['exit_price'],
                mode='markers',
                name="Exits",
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color=['green' if pnl > 0 else 'red' for pnl in self.trades_df['pnl']],
                    line=dict(width=1, color='black')
                )
            ),
            row=row, col=col
        )
        
    def _add_size_distribution_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add trade size distribution subplot."""
        fig.add_trace(
            go.Histogram(
                x=self.trades_df['size'],
                name="Trade Size",
                nbinsx=20,
                marker_color='blue'
            ),
            row=row, col=col
        )
        
    def _add_returns_distribution_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add returns distribution subplot."""
        fig.add_trace(
            go.Histogram(
                x=self.trades_df['pnl'],
                name="Returns",
                nbinsx=20,
                marker_color='green'
            ),
            row=row, col=col
        )
        
    def _add_win_loss_patterns_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add win/loss patterns subplot."""
        # Calculate win/loss streaks
        self.trades_df['is_win'] = self.trades_df['pnl'] > 0
        streak_lengths = []
        current_streak = 1
        
        for i in range(1, len(self.trades_df)):
            if self.trades_df['is_win'].iloc[i] == self.trades_df['is_win'].iloc[i-1]:
                current_streak += 1
            else:
                streak_lengths.append(current_streak)
                current_streak = 1
                
        if current_streak > 1:
            streak_lengths.append(current_streak)
            
        fig.add_trace(
            go.Histogram(
                x=streak_lengths,
                name="Streak Length",
                nbinsx=10,
                marker_color='purple'
            ),
            row=row, col=col
        )
        
    def _add_trade_duration_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add trade duration subplot."""
        if 'duration' not in self.trades_df.columns:
            self.trades_df['duration'] = (
                self.trades_df.index - self.trades_df.index.shift(1)
            ).dt.total_seconds() / 3600  # Convert to hours
            
        fig.add_trace(
            go.Histogram(
                x=self.trades_df['duration'],
                name="Duration (hours)",
                nbinsx=20,
                marker_color='orange'
            ),
            row=row, col=col
        )
        
    def _add_market_impact_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add market impact subplot."""
        # Calculate price impact (simple implementation)
        if 'price_impact' not in self.trades_df.columns:
            self.trades_df['price_impact'] = (
                self.trades_df['exit_price'] - self.trades_df['entry_price']
            ) / self.trades_df['entry_price']
            
        fig.add_trace(
            go.Scatter(
                x=self.trades_df['size'],
                y=self.trades_df['price_impact'],
                mode='markers',
                name="Market Impact",
                marker=dict(
                    color='red',
                    size=8,
                    opacity=0.6
                )
            ),
            row=row, col=col
        )
        
    def _add_cumulative_returns_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add cumulative returns subplot."""
        cumulative_returns = (1 + self.trades_df['pnl']).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=self.trades_df.index,
                y=cumulative_returns,
                name="Cumulative Returns",
                line=dict(color='blue', width=2)
            ),
            row=row, col=col
        )
        
    def _add_rolling_metrics_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add rolling metrics subplot."""
        window = 20  # Rolling window size
        
        # Calculate rolling metrics
        rolling_return = self.trades_df['pnl'].rolling(window).mean()
        rolling_vol = self.trades_df['pnl'].rolling(window).std()
        rolling_sharpe = rolling_return / rolling_vol
        
        # Plot rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=self.trades_df.index,
                y=rolling_sharpe,
                name="Rolling Sharpe",
                line=dict(color='green', width=2)
            ),
            row=row, col=col
        )
        
    def _add_drawdown_analysis_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add drawdown analysis subplot."""
        cumulative_returns = (1 + self.trades_df['pnl']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        fig.add_trace(
            go.Scatter(
                x=self.trades_df.index,
                y=drawdowns,
                name="Drawdown",
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='red', width=1)
            ),
            row=row, col=col
        )
        
    def _add_return_attribution_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add return attribution subplot."""
        # Group returns by some attribute (e.g., hour of day)
        self.trades_df['hour'] = self.trades_df.index.hour
        hourly_returns = self.trades_df.groupby('hour')['pnl'].mean()
        
        fig.add_trace(
            go.Bar(
                x=hourly_returns.index,
                y=hourly_returns.values,
                name="Hourly Returns",
                marker_color='purple'
            ),
            row=row, col=col
        )
        
    def _add_risk_metrics_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add risk metrics subplot."""
        # Calculate various risk metrics
        window = 20
        rolling_vol = self.trades_df['pnl'].rolling(window).std()
        rolling_var = np.percentile(self.trades_df['pnl'].rolling(window), 5)
        
        fig.add_trace(
            go.Scatter(
                x=self.trades_df.index,
                y=rolling_vol,
                name="Rolling Vol",
                line=dict(color='orange', width=2)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.trades_df.index,
                y=rolling_var,
                name="Rolling VaR",
                line=dict(color='red', width=2)
            ),
            row=row, col=col
        )
        
    def _add_performance_breakdown_subplot(self, fig: go.Figure, row: int, col: int) -> None:
        """Add performance breakdown subplot."""
        # Calculate performance metrics by trade size quartile
        self.trades_df['size_quartile'] = pd.qcut(self.trades_df['size'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_stats = self.trades_df.groupby('size_quartile')['pnl'].agg(['mean', 'std', 'count'])
        
        fig.add_trace(
            go.Bar(
                x=quartile_stats.index,
                y=quartile_stats['mean'],
                name="Mean Return",
                marker_color='blue'
            ),
            row=row, col=col
        )
        
    def _add_price_levels_subplot(self, fig: go.Figure, market_context: MarketContext, row: int, col: int) -> None:
        """Add price levels and support/resistance subplot."""
        # Plot price
        fig.add_trace(
            go.Scatter(
                x=self.price_data.index,
                y=self.price_data['price'],
                name="Price",
                line=dict(color='blue', width=1)
            ),
            row=row, col=col
        )
        
        # Add support/resistance levels
        for level in market_context.support_resistance:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="gray",
                row=row,
                col=col
            )
            
    def _add_volatility_profile_subplot(self, fig: go.Figure, market_context: MarketContext, row: int, col: int) -> None:
        """Add volatility profile subplot."""
        # Calculate rolling volatility
        rolling_vol = self.price_data['price'].pct_change().rolling(window=20).std()
        
        fig.add_trace(
            go.Scatter(
                x=self.price_data.index,
                y=rolling_vol,
                name="Volatility",
                line=dict(color='orange', width=2)
            ),
            row=row, col=col
        )
        
        # Add current volatility level
        fig.add_hline(
            y=market_context.volatility,
            line_dash="dash",
            line_color="red",
            row=row,
            col=col
        )
        
    def _add_volume_analysis_subplot(self, fig: go.Figure, market_context: MarketContext, row: int, col: int) -> None:
        """Add volume analysis subplot."""
        if 'volume' in self.price_data.columns:
            # Plot volume
            fig.add_trace(
                go.Bar(
                    x=self.price_data.index,
                    y=self.price_data['volume'],
                    name="Volume",
                    marker_color='lightgray'
                ),
                row=row, col=col
            )
            
            # Add volume profile annotation
            fig.add_annotation(
                text=f"Volume Profile: {market_context.volume_profile}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.05,
                showarrow=False,
                row=row,
                col=col
            )
            
    def _add_market_regime_subplot(self, fig: go.Figure, market_context: MarketContext, row: int, col: int) -> None:
        """Add market regime subplot."""
        # Create a summary of market regime
        fig.add_annotation(
            text=(
                f"Trend: {market_context.trend}<br>"
                f"Volatility: {market_context.volatility:.2%}<br>"
                f"Volume Profile: {market_context.volume_profile}<br>"
                f"Description: {market_context.description}"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12),
            align="center",
            row=row,
            col=col
        )
        
    def _add_pattern_distribution_subplot(self, fig: go.Figure, patterns: List[TradePattern], row: int, col: int) -> None:
        """Add pattern distribution subplot."""
        pattern_types = [p.pattern_type for p in patterns]
        pattern_counts = pd.Series(pattern_types).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=pattern_counts.index,
                y=pattern_counts.values,
                name="Pattern Count",
                marker_color='blue'
            ),
            row=row, col=col
        )
        
    def _add_pattern_performance_subplot(self, fig: go.Figure, patterns: List[TradePattern], row: int, col: int) -> None:
        """Add pattern performance subplot."""
        pattern_df = pd.DataFrame([
            {
                'type': p.pattern_type,
                'return': p.avg_return,
                'win_rate': p.win_rate
            }
            for p in patterns
        ])
        
        fig.add_trace(
            go.Scatter(
                x=pattern_df['win_rate'],
                y=pattern_df['return'],
                mode='markers+text',
                text=pattern_df['type'],
                textposition="top center",
                name="Pattern Performance",
                marker=dict(
                    size=10,
                    color='blue'
                )
            ),
            row=row, col=col
        )
        
    def _add_pattern_context_subplot(self, fig: go.Figure, patterns: List[TradePattern], row: int, col: int) -> None:
        """Add pattern context subplot."""
        # Create a summary of pattern contexts
        context_summary = "<br>".join([
            f"{p.pattern_type}: {p.description}"
            for p in patterns[:5]  # Show top 5 patterns
        ])
        
        fig.add_annotation(
            text=context_summary,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12),
            align="left",
            row=row,
            col=col
        )
        
    def _add_pattern_timeline_subplot(self, fig: go.Figure, patterns: List[TradePattern], row: int, col: int) -> None:
        """Add pattern timeline subplot."""
        # Create timeline of pattern occurrences
        pattern_timeline = pd.DataFrame([
            {
                'time': self.trades_df.index[p.frequency - 1],
                'pattern': p.pattern_type,
                'return': p.avg_return
            }
            for p in patterns
        ])
        
        fig.add_trace(
            go.Scatter(
                x=pattern_timeline['time'],
                y=pattern_timeline['return'],
                mode='markers+text',
                text=pattern_timeline['pattern'],
                textposition="top center",
                name="Pattern Timeline",
                marker=dict(
                    size=10,
                    color='purple'
                )
            ),
            row=row, col=col
        )
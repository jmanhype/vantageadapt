"""
Real-time Trading Dashboard
Implements Kagan's vision: "visualizing a little bit of like being able to look at the changes in performance in the runs"
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
import sqlite3
from pathlib import Path


class TradingDashboard:
    """
    Kagan's Vision: Real-time monitoring and visualization of trading performance
    """
    
    def __init__(self, db_path: str = "strategy_performance.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Setup SQLite database for performance tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create performance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                system_name TEXT,
                total_pnl REAL,
                total_return REAL,
                total_trades INTEGER,
                win_rate REAL,
                assets_traded INTEGER,
                avg_return_per_trade REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                configuration TEXT
            )
        ''')
        
        # Create individual trade tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS individual_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                timestamp TEXT,
                asset TEXT,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                return_pct REAL,
                exit_reason TEXT,
                win INTEGER,
                FOREIGN KEY (run_id) REFERENCES performance_runs (id)
            )
        ''')
        
        # Create prompt optimization tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_optimization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                prompt_name TEXT,
                performance_score REAL,
                iteration INTEGER,
                prompt_content TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_performance_run(self, system_name: str, metrics: Dict[str, Any], 
                          trades: List[Dict[str, Any]] = None) -> int:
        """Log a complete performance run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert performance run
        cursor.execute('''
            INSERT INTO performance_runs 
            (timestamp, system_name, total_pnl, total_return, total_trades, win_rate, 
             assets_traded, avg_return_per_trade, sharpe_ratio, max_drawdown, configuration)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            system_name,
            metrics.get('total_pnl', 0),
            metrics.get('total_return', 0),
            metrics.get('total_trades', 0),
            metrics.get('win_rate', 0),
            metrics.get('assets_traded', 0),
            metrics.get('avg_return_per_trade', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown', 0),
            json.dumps(metrics.get('configuration', {}))
        ))
        
        run_id = cursor.lastrowid
        
        # Insert individual trades if provided
        if trades:
            for trade in trades:
                cursor.execute('''
                    INSERT INTO individual_trades
                    (run_id, timestamp, asset, entry_price, exit_price, pnl, 
                     return_pct, exit_reason, win)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    trade.get('timestamp', datetime.now().isoformat()),
                    trade.get('asset', ''),
                    trade.get('entry_price', 0),
                    trade.get('exit_price', 0),
                    trade.get('pnl', 0),
                    trade.get('return_pct', 0),
                    trade.get('exit_reason', ''),
                    1 if trade.get('win', False) else 0
                ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Logged performance run for {system_name} with ID {run_id}")
        return run_id
    
    def get_performance_history(self, system_name: str = None, 
                              days: int = 30) -> pd.DataFrame:
        """Get performance history from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM performance_runs 
            WHERE timestamp > ?
        '''
        params = [datetime.now() - timedelta(days=days)]
        
        if system_name:
            query += ' AND system_name = ?'
            params.append(system_name)
        
        query += ' ORDER BY timestamp'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def create_performance_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create performance trend chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Return Over Time', 'Win Rate Trend', 
                          'Total Trades', 'PnL Progression'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if df.empty:
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Group by system for multiple lines
        for system in df['system_name'].unique():
            system_df = df[df['system_name'] == system]
            
            # Total Return
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['total_return'],
                          mode='lines+markers', name=f'{system} Return',
                          line=dict(width=2)),
                row=1, col=1
            )
            
            # Win Rate
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['win_rate'],
                          mode='lines+markers', name=f'{system} Win Rate',
                          line=dict(width=2)),
                row=1, col=2
            )
            
            # Total Trades
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['total_trades'],
                          mode='lines+markers', name=f'{system} Trades',
                          line=dict(width=2)),
                row=2, col=1
            )
            
            # PnL
            fig.add_trace(
                go.Scatter(x=system_df['timestamp'], y=system_df['total_pnl'],
                          mode='lines+markers', name=f'{system} PnL',
                          line=dict(width=2)),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True, 
                         title_text="Trading System Performance Dashboard")
        return fig
    
    def create_kagan_benchmark_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create Kagan benchmark comparison chart"""
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(text="No data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Kagan benchmarks (adjusted)
        kagan_benchmarks = {
            'return_target': 10.0,  # 10%
            'trades_target': 100,   # 100 trades
            'assets_target': 10     # 10 assets
        }
        
        latest_data = df.groupby('system_name').last()
        
        systems = latest_data.index.tolist()
        returns = (latest_data['total_return'] * 100).tolist()  # Convert to %
        trades = latest_data['total_trades'].tolist()
        assets = latest_data['assets_traded'].tolist()
        
        # Create radar chart
        categories = ['Return (%)', 'Trades', 'Assets']
        
        for i, system in enumerate(systems):
            fig.add_trace(go.Scatterpolar(
                r=[returns[i], trades[i], assets[i]],
                theta=categories,
                fill='toself',
                name=system,
                line=dict(width=2)
            ))
        
        # Add benchmark line
        fig.add_trace(go.Scatterpolar(
            r=[kagan_benchmarks['return_target'], 
               kagan_benchmarks['trades_target'],
               kagan_benchmarks['assets_target']],
            theta=categories,
            fill='toself',
            name='Kagan Target',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(
                    kagan_benchmarks['return_target'] * 1.5,
                    kagan_benchmarks['trades_target'] * 1.5,
                    kagan_benchmarks['assets_target'] * 1.5
                )])
            ),
            showlegend=True,
            title="Kagan Benchmark Comparison"
        )
        
        return fig
    
    def create_trade_distribution_chart(self, run_id: int) -> go.Figure:
        """Create trade distribution analysis"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query('''
            SELECT * FROM individual_trades WHERE run_id = ?
        ''', conn, params=[run_id])
        
        conn.close()
        
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No trade data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PnL Distribution', 'Return % Distribution',
                          'Win/Loss by Asset', 'Exit Reasons'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # PnL Distribution
        fig.add_trace(
            go.Histogram(x=df['pnl'], name='PnL', nbinsx=20),
            row=1, col=1
        )
        
        # Return % Distribution
        fig.add_trace(
            go.Histogram(x=df['return_pct'] * 100, name='Return %', nbinsx=20),
            row=1, col=2
        )
        
        # Win/Loss by Asset
        asset_stats = df.groupby('asset')['win'].agg(['count', 'sum']).reset_index()
        asset_stats['win_rate'] = asset_stats['sum'] / asset_stats['count']
        
        fig.add_trace(
            go.Bar(x=asset_stats['asset'], y=asset_stats['win_rate'], 
                   name='Win Rate by Asset'),
            row=2, col=1
        )
        
        # Exit Reasons
        exit_reasons = df['exit_reason'].value_counts()
        fig.add_trace(
            go.Pie(labels=exit_reasons.index, values=exit_reasons.values,
                   name="Exit Reasons"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True,
                         title_text="Trade Analysis Dashboard")
        return fig


def run_dashboard():
    """Run the Streamlit dashboard"""
    st.set_page_config(
        page_title="Kagan Trading Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🚀 Kagan Autonomous Trading Dashboard")
    st.markdown("*Real-time monitoring of LLM-powered trading performance*")
    
    dashboard = TradingDashboard()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    days_to_show = st.sidebar.slider("Days of History", 1, 90, 30)
    system_filter = st.sidebar.selectbox(
        "System Filter", 
        ["All"] + list(dashboard.get_performance_history()['system_name'].unique())
        if not dashboard.get_performance_history().empty else ["All"]
    )
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Load data
    df = dashboard.get_performance_history(
        system_name=None if system_filter == "All" else system_filter,
        days=days_to_show
    )
    
    if not df.empty:
        latest_stats = df.groupby('system_name').last()
        
        # Key metrics
        with col1:
            st.metric("Total Systems", len(latest_stats))
        
        with col2:
            avg_return = latest_stats['total_return'].mean()
            st.metric("Avg Return", f"{avg_return:.2%}")
        
        with col3:
            total_trades = latest_stats['total_trades'].sum()
            st.metric("Total Trades", f"{total_trades:,}")
        
        with col4:
            avg_win_rate = latest_stats['win_rate'].mean()
            st.metric("Avg Win Rate", f"{avg_win_rate:.1%}")
        
        # Performance charts
        st.subheader("📊 Performance Trends")
        perf_chart = dashboard.create_performance_chart(df)
        st.plotly_chart(perf_chart, use_container_width=True)
        
        # Kagan benchmarks
        st.subheader("🎯 Kagan Benchmark Comparison")
        benchmark_chart = dashboard.create_kagan_benchmark_chart(df)
        st.plotly_chart(benchmark_chart, use_container_width=True)
        
        # Current status
        st.subheader("📈 Current System Status")
        
        for system in latest_stats.index:
            with st.expander(f"System: {system}"):
                system_data = latest_stats.loc[system]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PnL", f"${system_data['total_pnl']:,.2f}")
                    st.metric("Return", f"{system_data['total_return']:.2%}")
                
                with col2:
                    st.metric("Trades", f"{system_data['total_trades']:,}")
                    st.metric("Win Rate", f"{system_data['win_rate']:.1%}")
                
                with col3:
                    st.metric("Assets", f"{system_data['assets_traded']:,}")
                    st.metric("Sharpe", f"{system_data['sharpe_ratio']:.2f}")
                
                # Kagan benchmark status
                st.markdown("**Kagan Benchmark Status:**")
                return_pass = "✅" if system_data['total_return'] >= 0.10 else "❌"
                trades_pass = "✅" if system_data['total_trades'] >= 100 else "❌"
                assets_pass = "✅" if system_data['assets_traded'] >= 10 else "❌"
                
                st.markdown(f"- Return ≥10%: {return_pass} ({system_data['total_return']:.1%})")
                st.markdown(f"- Trades ≥100: {trades_pass} ({system_data['total_trades']})")
                st.markdown(f"- Assets ≥10: {assets_pass} ({system_data['assets_traded']})")
    
    else:
        st.info("No performance data available. Run a trading system to populate the dashboard.")
        st.markdown("### 🎯 Kagan's Vision Implementation Status:")
        st.markdown("- ✅ Real-time dashboard interface")
        st.markdown("- ✅ Performance tracking database")
        st.markdown("- ✅ Kagan benchmark monitoring")
        st.markdown("- ⏳ Waiting for system data...")


# Convenience function for integration
def log_system_performance(system_name: str, metrics: Dict[str, Any], 
                         trades: List[Dict[str, Any]] = None):
    """Convenience function to log performance from trading systems"""
    dashboard = TradingDashboard()
    return dashboard.log_performance_run(system_name, metrics, trades)


if __name__ == "__main__":
    run_dashboard()
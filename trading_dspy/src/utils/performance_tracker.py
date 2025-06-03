"""Performance tracking utilities for the trading system."""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger


class PerformanceTracker:
    """Track and analyze trading performance metrics."""
    
    def __init__(self):
        self.trades = []
        self.balance_history = []
        self.initial_balance = 10000  # $10k starting balance
        self.current_balance = self.initial_balance
        
    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade."""
        trade['timestamp'] = datetime.now()
        self.trades.append(trade)
        
        # Update balance
        pnl = trade.get('pnl', 0)
        self.current_balance += pnl
        self.balance_history.append({
            'timestamp': trade['timestamp'],
            'balance': self.current_balance,
            'pnl': pnl
        })
        
    def get_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
        # Basic metrics
        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('pnl', 0) <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl', 0) for t in self.trades)
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Average win/loss
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            returns = [t.get('return_pct', 0) for t in self.trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from balance history."""
        if not self.balance_history:
            return 0
            
        balances = [b['balance'] for b in self.balance_history]
        peak = balances[0]
        max_dd = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
        
    def get_summary(self) -> str:
        """Get a formatted summary of performance."""
        metrics = self.get_metrics()
        
        summary = f"""
Performance Summary:
==================
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.2%}
Total PnL: ${metrics['total_pnl']:.2f}
Total Return: {metrics['total_return']:.2%}
Avg Win: ${metrics['avg_win']:.2f}
Avg Loss: ${metrics['avg_loss']:.2f}
Profit Factor: {metrics['profit_factor']:.2f}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2%}
"""
        return summary
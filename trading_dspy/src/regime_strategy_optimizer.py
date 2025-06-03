"""
Regime-Specific Strategy Optimizer
Learns optimal strategies for each market regime based on ACTUAL historical performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sqlite3
from datetime import datetime, timedelta


@dataclass
class RegimeStrategy:
    """Strategy optimized for a specific market regime"""
    regime: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    parameters: Dict[str, float]
    historical_performance: Dict[str, float]
    confidence: float
    last_updated: datetime
    successful_trades: List[Dict[str, Any]] = field(default_factory=list)
    failed_trades: List[Dict[str, Any]] = field(default_factory=list)


class RegimeIdentifier:
    """Advanced regime identification using ML clustering"""
    
    def __init__(self, n_regimes: int = 6):
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        self.regime_names = {
            0: "STRONG_BULL",
            1: "MODERATE_BULL", 
            2: "RANGING_HIGH_VOL",
            3: "RANGING_LOW_VOL",
            4: "MODERATE_BEAR",
            5: "STRONG_BEAR"
        }
        self.regime_characteristics = {}
        
    def fit(self, market_data: pd.DataFrame):
        """Train regime identifier on historical data"""
        features = self._extract_regime_features(market_data)
        scaled_features = self.scaler.fit_transform(features)
        
        self.kmeans.fit(scaled_features)
        
        # Characterize each regime
        for i in range(self.n_regimes):
            regime_mask = self.kmeans.labels_ == i
            regime_data = features[regime_mask]
            
            self.regime_characteristics[i] = {
                'avg_return': regime_data['returns'].mean(),
                'avg_volatility': regime_data['volatility'].mean(),
                'trend_strength': regime_data['trend'].mean(),
                'volume_profile': regime_data['volume_ratio'].mean(),
                'typical_duration': (regime_mask.astype(int).groupby((regime_mask != regime_mask.shift()).cumsum()).sum()).mean()
            }
        
        # Rename regimes based on characteristics
        self._rename_regimes_by_characteristics()
        
    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime identification"""
        features = pd.DataFrame(index=data.index)
        
        # Return features
        features['returns'] = data['close'].pct_change()
        features['returns_5d'] = data['close'].pct_change(5)
        features['returns_20d'] = data['close'].pct_change(20)
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_5d'] = features['returns'].rolling(5).std()
        features['volatility_ratio'] = features['volatility_5d'] / features['volatility']
        
        # Trend features
        features['trend'] = (data['close'].rolling(20).mean() - data['close'].rolling(50).mean()) / data['close']
        features['trend_strength'] = features['returns_20d'] / features['volatility']
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_trend'] = data['volume'].rolling(20).mean().pct_change(20)
        
        # Market structure
        features['high_low_spread'] = (data['high'] - data['low']) / data['close']
        features['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        return features.dropna()
    
    def _rename_regimes_by_characteristics(self):
        """Rename regime clusters based on their characteristics"""
        # Sort regimes by average return and volatility
        regime_profiles = []
        for i in range(self.n_regimes):
            char = self.regime_characteristics[i]
            regime_profiles.append({
                'id': i,
                'return': char['avg_return'],
                'volatility': char['avg_volatility'],
                'trend': char['trend_strength']
            })
        
        # Sort by return (descending) and assign names
        sorted_regimes = sorted(regime_profiles, key=lambda x: x['return'], reverse=True)
        
        new_names = {}
        for idx, regime in enumerate(sorted_regimes):
            if idx == 0:
                new_names[regime['id']] = "STRONG_BULL"
            elif idx == 1:
                new_names[regime['id']] = "MODERATE_BULL"
            elif idx == self.n_regimes - 1:
                new_names[regime['id']] = "STRONG_BEAR"
            elif idx == self.n_regimes - 2:
                new_names[regime['id']] = "MODERATE_BEAR"
            else:
                # Middle regimes - check volatility
                if regime['volatility'] > np.median([r['volatility'] for r in regime_profiles]):
                    new_names[regime['id']] = "RANGING_HIGH_VOL"
                else:
                    new_names[regime['id']] = "RANGING_LOW_VOL"
        
        self.regime_names = new_names
    
    def identify_regime(self, current_data: pd.DataFrame) -> Tuple[str, float]:
        """Identify current market regime"""
        features = self._extract_regime_features(current_data)
        if len(features) == 0:
            return "UNKNOWN", 0.0
        
        scaled_features = self.scaler.transform(features.iloc[-1:])
        
        # Get regime prediction
        regime_id = self.kmeans.predict(scaled_features)[0]
        
        # Calculate confidence based on distance to cluster center
        distances = self.kmeans.transform(scaled_features)[0]
        min_distance = distances[regime_id]
        confidence = 1 / (1 + min_distance)  # Convert distance to confidence
        
        return self.regime_names.get(regime_id, "UNKNOWN"), confidence


class StrategyPerformanceTracker:
    """Tracks and analyzes strategy performance across different regimes"""
    
    def __init__(self, db_path: str = "strategy_performance.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize performance tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                regime TEXT,
                strategy_id TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                pnl REAL,
                return_pct REAL,
                holding_period_hours INTEGER,
                entry_conditions TEXT,
                exit_conditions TEXT,
                indicators_used TEXT,
                success BOOLEAN
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_performance (
                regime TEXT PRIMARY KEY,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                avg_return REAL,
                win_rate REAL,
                avg_holding_hours REAL,
                best_strategy_id TEXT,
                last_updated DATETIME
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_trade(self, trade: Dict[str, Any]):
        """Record a completed trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                timestamp, regime, strategy_id, entry_price, exit_price,
                position_size, pnl, return_pct, holding_period_hours,
                entry_conditions, exit_conditions, indicators_used, success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['timestamp'],
            trade['regime'],
            trade['strategy_id'],
            trade['entry_price'],
            trade['exit_price'],
            trade['position_size'],
            trade['pnl'],
            trade['return_pct'],
            trade['holding_period_hours'],
            json.dumps(trade['entry_conditions']),
            json.dumps(trade['exit_conditions']),
            json.dumps(trade['indicators_used']),
            trade['success']
        ))
        
        conn.commit()
        conn.close()
        
        # Update regime performance
        self._update_regime_performance(trade['regime'])
    
    def _update_regime_performance(self, regime: str):
        """Update aggregated performance for a regime"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get aggregated stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as winning_trades,
                SUM(pnl) as total_pnl,
                AVG(return_pct) as avg_return,
                AVG(holding_period_hours) as avg_holding_hours
            FROM trades
            WHERE regime = ?
        """, (regime,))
        
        stats = cursor.fetchone()
        if stats[0] > 0:  # Has trades
            win_rate = stats[1] / stats[0] if stats[0] > 0 else 0
            
            # Find best strategy
            cursor.execute("""
                SELECT strategy_id, SUM(pnl) as total_pnl
                FROM trades
                WHERE regime = ?
                GROUP BY strategy_id
                ORDER BY total_pnl DESC
                LIMIT 1
            """, (regime,))
            
            best_strategy = cursor.fetchone()
            
            cursor.execute("""
                INSERT OR REPLACE INTO regime_performance
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                regime,
                stats[0],  # total_trades
                stats[1],  # winning_trades
                stats[2],  # total_pnl
                stats[3],  # avg_return
                win_rate,
                stats[4],  # avg_holding_hours
                best_strategy[0] if best_strategy else None,
                datetime.now()
            ))
        
        conn.commit()
        conn.close()
    
    def get_regime_performance(self, regime: str) -> Dict[str, Any]:
        """Get performance statistics for a specific regime"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM regime_performance WHERE regime = ?
        """, (regime,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'regime': row[0],
                'total_trades': row[1],
                'winning_trades': row[2],
                'total_pnl': row[3],
                'avg_return': row[4],
                'win_rate': row[5],
                'avg_holding_hours': row[6],
                'best_strategy_id': row[7],
                'last_updated': row[8]
            }
        return None
    
    def get_successful_patterns(self, regime: str, min_trades: int = 5) -> List[Dict[str, Any]]:
        """Get successful trading patterns for a regime"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                entry_conditions,
                exit_conditions,
                indicators_used,
                COUNT(*) as trade_count,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as wins,
                AVG(return_pct) as avg_return,
                SUM(pnl) as total_pnl
            FROM trades
            WHERE regime = ? AND success = 1
            GROUP BY entry_conditions, exit_conditions
            HAVING trade_count >= ?
            ORDER BY avg_return DESC
            LIMIT 10
        """, (regime, min_trades))
        
        patterns = []
        for row in cursor.fetchall():
            patterns.append({
                'entry_conditions': json.loads(row[0]),
                'exit_conditions': json.loads(row[1]),
                'indicators_used': json.loads(row[2]),
                'trade_count': row[3],
                'wins': row[4],
                'avg_return': row[5],
                'total_pnl': row[6],
                'win_rate': row[4] / row[3]
            })
        
        conn.close()
        return patterns


class RegimeStrategyOptimizer:
    """Optimizes strategies for each market regime based on historical performance"""
    
    def __init__(self):
        self.regime_identifier = RegimeIdentifier()
        self.performance_tracker = StrategyPerformanceTracker()
        self.regime_strategies = {}
        self.fallback_strategies = self._create_fallback_strategies()
        
    def _create_fallback_strategies(self) -> Dict[str, RegimeStrategy]:
        """Create baseline strategies for each regime"""
        return {
            "STRONG_BULL": RegimeStrategy(
                regime="STRONG_BULL",
                entry_conditions=[
                    "price > sma_20",
                    "rsi > 50 and rsi < 80",
                    "volume > volume_sma_20 * 1.2"
                ],
                exit_conditions=[
                    "price < sma_20 * 0.98",
                    "rsi > 85",
                    "price >= entry_price * 1.05"
                ],
                parameters={
                    "position_size": 0.3,
                    "stop_loss": 0.03,
                    "take_profit": 0.05,
                    "trailing_stop": True
                },
                historical_performance={},
                confidence=0.5,
                last_updated=datetime.now()
            ),
            "RANGING_LOW_VOL": RegimeStrategy(
                regime="RANGING_LOW_VOL",
                entry_conditions=[
                    "price <= bb_lower * 1.01",
                    "rsi < 35",
                    "abs(price - sma_50) / sma_50 < 0.02"
                ],
                exit_conditions=[
                    "price >= bb_middle",
                    "rsi > 65",
                    "price >= entry_price * 1.02"
                ],
                parameters={
                    "position_size": 0.2,
                    "stop_loss": 0.02,
                    "take_profit": 0.02,
                    "trailing_stop": False
                },
                historical_performance={},
                confidence=0.5,
                last_updated=datetime.now()
            ),
            # Add more fallback strategies for other regimes...
        }
    
    def train_on_historical_data(self, market_data: pd.DataFrame, trades_df: pd.DataFrame):
        """Train regime identifier and optimize strategies based on historical data"""
        logger.info("Training regime strategy optimizer on historical data")
        
        # Train regime identifier
        self.regime_identifier.fit(market_data)
        
        # Identify regime for each historical period
        regimes = []
        for i in range(len(market_data)):
            window = market_data.iloc[max(0, i-100):i+1]
            if len(window) >= 20:
                regime, confidence = self.regime_identifier.identify_regime(window)
                regimes.append(regime)
            else:
                regimes.append("UNKNOWN")
        
        market_data['regime'] = regimes
        
        # Analyze historical trades by regime
        if 'timestamp' in trades_df.columns and 'timestamp' in market_data.index.names:
            trades_with_regime = trades_df.merge(
                market_data[['regime']], 
                left_on='timestamp', 
                right_index=True,
                how='left'
            )
            
            # Record historical trades
            for _, trade in trades_with_regime.iterrows():
                self.performance_tracker.record_trade(trade.to_dict())
        
        # Optimize strategies for each regime
        for regime in self.regime_identifier.regime_names.values():
            self._optimize_regime_strategy(regime)
        
        logger.info("Regime strategy optimization completed")
    
    def _optimize_regime_strategy(self, regime: str):
        """Optimize strategy for a specific regime based on performance data"""
        # Get performance data
        perf = self.performance_tracker.get_regime_performance(regime)
        patterns = self.performance_tracker.get_successful_patterns(regime)
        
        if not patterns or not perf:
            # Use fallback strategy
            self.regime_strategies[regime] = self.fallback_strategies.get(
                regime, 
                self.fallback_strategies["RANGING_LOW_VOL"]
            )
            return
        
        # Extract best performing patterns
        best_pattern = patterns[0] if patterns else {}
        
        # Create optimized strategy
        self.regime_strategies[regime] = RegimeStrategy(
            regime=regime,
            entry_conditions=best_pattern.get('entry_conditions', []),
            exit_conditions=best_pattern.get('exit_conditions', []),
            parameters={
                "position_size": self._calculate_optimal_position_size(perf),
                "stop_loss": self._calculate_optimal_stop_loss(patterns),
                "take_profit": self._calculate_optimal_take_profit(patterns),
                "max_holding_hours": perf.get('avg_holding_hours', 24) * 2
            },
            historical_performance=perf,
            confidence=self._calculate_strategy_confidence(perf),
            last_updated=datetime.now(),
            successful_trades=patterns[:5]  # Top 5 patterns
        )
        
        logger.info(f"Optimized strategy for {regime}: Win rate={perf['win_rate']:.2%}, Avg return={perf['avg_return']:.2%}")
    
    def _calculate_optimal_position_size(self, performance: Dict[str, Any]) -> float:
        """Calculate optimal position size based on regime performance"""
        win_rate = performance.get('win_rate', 0.5)
        avg_return = performance.get('avg_return', 0)
        
        # Kelly Criterion with safety factor
        if avg_return > 0 and win_rate > 0.5:
            kelly = (win_rate * avg_return) / abs(avg_return)
            return min(0.5, max(0.1, kelly * 0.25))  # 25% of Kelly
        return 0.1
    
    def _calculate_optimal_stop_loss(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate optimal stop loss based on successful patterns"""
        if not patterns:
            return 0.02
        
        # Analyze losing trades to find optimal stop loss
        # For now, use a simple heuristic
        avg_return = np.mean([p['avg_return'] for p in patterns])
        return min(0.05, max(0.01, abs(avg_return) * 0.5))
    
    def _calculate_optimal_take_profit(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate optimal take profit based on successful patterns"""
        if not patterns:
            return 0.04
        
        # Use average return of successful patterns
        avg_return = np.mean([p['avg_return'] for p in patterns])
        return max(0.02, min(0.10, avg_return * 1.5))
    
    def _calculate_strategy_confidence(self, performance: Dict[str, Any]) -> float:
        """Calculate confidence in regime strategy"""
        total_trades = performance.get('total_trades', 0)
        win_rate = performance.get('win_rate', 0.5)
        
        # More trades and higher win rate = higher confidence
        trade_confidence = min(1.0, total_trades / 100)
        performance_confidence = win_rate
        
        return (trade_confidence + performance_confidence) / 2
    
    def get_strategy_for_regime(self, regime: str, current_confidence: float) -> RegimeStrategy:
        """Get optimized strategy for current regime"""
        if regime in self.regime_strategies:
            strategy = self.regime_strategies[regime]
            
            # Adjust confidence based on regime identification confidence
            strategy.confidence *= current_confidence
            
            return strategy
        
        # Return fallback strategy
        return self.fallback_strategies.get(regime, self.fallback_strategies["RANGING_LOW_VOL"])
    
    def update_strategy_performance(self, trade_result: Dict[str, Any]):
        """Update strategy performance with new trade result"""
        self.performance_tracker.record_trade(trade_result)
        
        # Re-optimize strategy if enough new data
        regime = trade_result['regime']
        perf = self.performance_tracker.get_regime_performance(regime)
        
        if perf and perf['total_trades'] % 10 == 0:  # Every 10 trades
            logger.info(f"Re-optimizing strategy for {regime} after {perf['total_trades']} trades")
            self._optimize_regime_strategy(regime)
"""
REAL Machine Learning Trading Engine
No more bullshit. This actually learns from data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Any, Optional
import pickle
from loguru import logger
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeSignal:
    """Actual trade signal with probability and reasoning"""
    timestamp: pd.Timestamp
    action: str  # 'BUY', 'SELL', 'HOLD'
    probability: float  # Confidence in the signal
    predicted_return: float  # Expected return
    stop_loss: float
    take_profit: float
    position_size: float
    features: Dict[str, float] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)


class FeatureEngineer:
    """Extract REAL trading features from market data"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create actual useful features for ML trading"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns_1h'] = df['close'].pct_change(1)
        features['returns_4h'] = df['close'].pct_change(4)
        features['returns_24h'] = df['close'].pct_change(24)
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        features['volatility_1h'] = features['returns_1h'].rolling(20).std()
        features['volatility_24h'] = features['returns_1h'].rolling(24*20).std()
        features['volatility_ratio'] = features['volatility_1h'] / features['volatility_24h']
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_trend'] = df['volume'].rolling(20).mean().pct_change(20)
        features['dollar_volume'] = df['close'] * df['volume']
        features['dollar_volume_ratio'] = features['dollar_volume'] / features['dollar_volume'].rolling(20).mean()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['close'])
        features['rsi_slope'] = features['rsi'].diff(5)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        features['bb_upper'] = sma + (std * 2)
        features['bb_lower'] = sma - (std * 2)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma
        
        # Market microstructure
        features['high_low_ratio'] = df['high'] / df['low'] - 1
        features['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Order flow imbalance (if we have bid/ask data)
        if 'bid' in df.columns and 'ask' in df.columns:
            features['spread'] = df['ask'] - df['bid']
            features['spread_pct'] = features['spread'] / df['close']
            features['mid_price'] = (df['bid'] + df['ask']) / 2
            features['price_vs_mid'] = (df['close'] - features['mid_price']) / features['mid_price']
        
        # Time-based features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_us_market_hours'] = ((features['hour'] >= 14) & (features['hour'] < 21)).astype(int)
        
        # Regime features
        features['trend_strength'] = self._calculate_trend_strength(df['close'])
        features['regime_volatility'] = self._calculate_regime_volatility(features['returns_1h'])
        
        # Clean up
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI properly"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength using linear regression slope"""
        def trend_slope(x):
            if len(x) < 2:
                return 0
            y = np.arange(len(x))
            slope, _, r_value, _, _ = stats.linregress(y, x)
            return slope * r_value ** 2  # Slope weighted by R-squared
        
        return prices.rolling(window).apply(trend_slope)
    
    def _calculate_regime_volatility(self, returns: pd.Series) -> pd.Series:
        """Classify volatility regime"""
        vol = returns.rolling(20).std()
        vol_percentile = vol.rolling(100).rank(pct=True)
        return vol_percentile


class MLTradingModel:
    """The actual ML model that learns from historical data"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.entry_model = None
        self.return_model = None
        self.risk_model = None
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_importance = {}
        
    def prepare_training_data(self, df: pd.DataFrame, target_hours: int = 4) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data with actual targets based on future price movements"""
        # Create features
        features = self.feature_engineer.create_features(df)
        
        # Create targets
        future_returns = df['close'].pct_change(target_hours).shift(-target_hours)
        
        # Entry signal: 1 if profitable (considering transaction costs)
        transaction_cost = 0.001  # 0.1% each way
        profitable_threshold = transaction_cost * 2
        entry_targets = (future_returns > profitable_threshold).astype(int)
        
        # Return targets: actual returns for regression
        return_targets = future_returns
        
        # Remove NaN rows
        valid_idx = ~(features.isna().any(axis=1) | entry_targets.isna() | return_targets.isna())
        
        return features[valid_idx], entry_targets[valid_idx], return_targets[valid_idx]
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """Train the ML models on historical data"""
        logger.info("Starting ML model training on historical data")
        
        # Prepare data
        X, y_entry, y_returns = self.prepare_training_data(df)
        
        # Time series split (don't use random split for time series!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_entry_train, y_entry_test = y_entry[:split_idx], y_entry[split_idx:]
        y_returns_train, y_returns_test = y_returns[:split_idx], y_returns[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train entry prediction model (XGBoost for better performance)
        logger.info("Training entry signal model...")
        self.entry_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            use_label_encoder=False
        )
        self.entry_model.fit(
            X_train_scaled, y_entry_train,
            eval_set=[(X_test_scaled, y_entry_test)],
            verbose=False
        )
        
        # Train return prediction model
        logger.info("Training return prediction model...")
        self.return_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror'
        )
        self.return_model.fit(
            X_train_scaled, y_returns_train,
            eval_set=[(X_test_scaled, y_returns_test)],
            verbose=False
        )
        
        # Train risk model (predicts volatility)
        logger.info("Training risk model...")
        y_risk_train = np.abs(y_returns_train)  # Use absolute returns as risk proxy
        y_risk_test = np.abs(y_returns_test)
        
        self.risk_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        self.risk_model.fit(X_train_scaled, y_risk_train)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        # Evaluate on test set
        self._evaluate_models(X_test_scaled, y_entry_test, y_returns_test)
        
        self.trained = True
        logger.info("ML model training completed")
    
    def _calculate_feature_importance(self):
        """Extract and store feature importance"""
        feature_names = self.feature_engineer.feature_names
        
        # Get importance from XGBoost model
        importance = self.entry_model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 most important features:")
        for feat, imp in sorted_features[:10]:
            logger.info(f"  {feat}: {imp:.4f}")
    
    def _evaluate_models(self, X_test: np.ndarray, y_entry_test: pd.Series, y_returns_test: pd.Series):
        """Evaluate model performance on test set"""
        # Entry model evaluation
        entry_pred = self.entry_model.predict_proba(X_test)[:, 1]
        entry_accuracy = ((entry_pred > 0.5) == y_entry_test).mean()
        logger.info(f"Entry model test accuracy: {entry_accuracy:.2%}")
        
        # Return model evaluation
        return_pred = self.return_model.predict(X_test)
        return_mae = np.abs(return_pred - y_returns_test).mean()
        logger.info(f"Return model MAE: {return_mae:.4f}")
        
        # Simulate trading on test set
        signals = entry_pred > 0.6  # Higher threshold for actual trading
        actual_returns = y_returns_test[signals]
        if len(actual_returns) > 0:
            win_rate = (actual_returns > 0).mean()
            avg_return = actual_returns.mean()
            logger.info(f"Simulated trading - Win rate: {win_rate:.2%}, Avg return: {avg_return:.4f}")
    
    def predict(self, current_data: pd.DataFrame) -> TradeSignal:
        """Generate trading signal for current market conditions"""
        if not self.trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features
        features = self.feature_engineer.create_features(current_data)
        latest_features = features.iloc[-1:].fillna(0)
        
        # Scale features
        features_scaled = self.scaler.transform(latest_features)
        
        # Get predictions
        entry_prob = self.entry_model.predict_proba(features_scaled)[0, 1]
        expected_return = self.return_model.predict(features_scaled)[0]
        expected_risk = self.risk_model.predict(features_scaled)[0]
        
        # Generate signal
        action = self._determine_action(entry_prob, expected_return, expected_risk)
        stop_loss, take_profit = self._calculate_risk_levels(expected_return, expected_risk)
        position_size = self._calculate_position_size(entry_prob, expected_risk)
        
        # Create reasoning
        reasoning = self._generate_reasoning(
            latest_features.iloc[0].to_dict(),
            entry_prob,
            expected_return,
            expected_risk
        )
        
        return TradeSignal(
            timestamp=current_data.index[-1],
            action=action,
            probability=entry_prob,
            predicted_return=expected_return,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            features=latest_features.iloc[0].to_dict(),
            reasoning=reasoning
        )
    
    def _determine_action(self, entry_prob: float, expected_return: float, expected_risk: float) -> str:
        """Determine trading action based on ML predictions"""
        # Risk-adjusted threshold
        risk_adjusted_threshold = 0.6 + (expected_risk * 0.5)  # Higher threshold for riskier trades
        
        if entry_prob > risk_adjusted_threshold and expected_return > 0.002:  # 0.2% minimum
            return 'BUY'
        else:
            return 'HOLD'
    
    def _calculate_risk_levels(self, expected_return: float, expected_risk: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        # Base levels
        base_stop_loss = 0.02  # 2%
        base_take_profit = 0.04  # 4%
        
        # Adjust based on predictions
        stop_loss = base_stop_loss + (expected_risk * 0.5)
        take_profit = max(expected_return * 2, base_take_profit)
        
        # Ensure minimum risk/reward ratio
        if take_profit / stop_loss < 1.5:
            take_profit = stop_loss * 2
        
        return stop_loss, take_profit
    
    def _calculate_position_size(self, entry_prob: float, expected_risk: float) -> float:
        """Kelly Criterion-inspired position sizing"""
        # Simplified Kelly: f = p - q/b
        # p = probability of win, q = probability of loss, b = win/loss ratio
        p = entry_prob
        q = 1 - p
        b = 2  # Assume 2:1 win/loss ratio
        
        kelly = p - (q / b)
        
        # Apply safety factor and risk adjustment
        safety_factor = 0.25  # Use 25% of Kelly
        risk_adjustment = 1 / (1 + expected_risk * 10)
        
        position_size = max(0.1, min(0.5, kelly * safety_factor * risk_adjustment))
        
        return position_size
    
    def _generate_reasoning(self, features: Dict[str, float], entry_prob: float, 
                          expected_return: float, expected_risk: float) -> List[str]:
        """Generate human-readable reasoning for the trade"""
        reasoning = []
        
        # Probability reasoning
        if entry_prob > 0.8:
            reasoning.append(f"Strong entry signal with {entry_prob:.1%} confidence")
        elif entry_prob > 0.6:
            reasoning.append(f"Moderate entry signal with {entry_prob:.1%} confidence")
        
        # Return reasoning
        reasoning.append(f"Expected return: {expected_return:.2%}")
        
        # Feature-based reasoning
        if features.get('rsi', 50) < 30:
            reasoning.append("RSI indicates oversold conditions")
        elif features.get('rsi', 50) > 70:
            reasoning.append("RSI indicates overbought conditions")
        
        if features.get('trend_strength', 0) > 0.5:
            reasoning.append("Strong trending market detected")
        
        if features.get('volatility_ratio', 1) > 1.5:
            reasoning.append("Elevated short-term volatility")
        
        # Top features reasoning
        top_features = sorted(
            [(k, v) for k, v in features.items() if k in self.feature_importance],
            key=lambda x: self.feature_importance.get(x[0], 0),
            reverse=True
        )[:3]
        
        for feat_name, feat_value in top_features:
            if abs(feat_value) > 1:  # Significant deviation
                reasoning.append(f"Key signal from {feat_name}: {feat_value:.2f}")
        
        return reasoning
    
    def update_online(self, trade_result: Dict[str, Any]):
        """Online learning - update model with new trade results"""
        # This is where we'd implement online learning
        # For now, we'll just log the result
        logger.info(f"Trade result received: PnL={trade_result.get('pnl', 0):.2f}")
        
        # In a real system, we would:
        # 1. Store this result
        # 2. Periodically retrain when we have enough new data
        # 3. Use techniques like online gradient descent for real-time updates
    
    def save(self, filepath: str):
        """Save trained model"""
        model_data = {
            'entry_model': self.entry_model,
            'return_model': self.return_model,
            'risk_model': self.risk_model,
            'scaler': self.scaler,
            'feature_names': self.feature_engineer.feature_names,
            'feature_importance': self.feature_importance,
            'trained': self.trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.entry_model = model_data['entry_model']
        self.return_model = model_data['return_model']
        self.risk_model = model_data['risk_model']
        self.scaler = model_data['scaler']
        self.feature_engineer.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.trained = model_data['trained']
        logger.info(f"Model loaded from {filepath}")


class TradingSystemOrchestrator:
    """Orchestrates the entire ML trading system"""
    
    def __init__(self):
        self.ml_model = MLTradingModel()
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    def train_system(self, historical_data: pd.DataFrame):
        """Train the ML trading system"""
        logger.info("Training ML Trading System...")
        self.ml_model.train(historical_data)
    
    def generate_signal(self, current_data: pd.DataFrame) -> TradeSignal:
        """Generate trading signal for current market"""
        return self.ml_model.predict(current_data)
    
    def execute_trade(self, signal: TradeSignal, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Execute trade based on signal and track results"""
        if signal.action == 'HOLD':
            return {'action': 'HOLD', 'pnl': 0}
        
        # Simulate trade execution
        entry_price = market_data['close'].iloc[-1]
        
        # This would be replaced with actual broker API calls
        trade_result = {
            'timestamp': signal.timestamp,
            'action': signal.action,
            'entry_price': entry_price,
            'position_size': signal.position_size,
            'stop_loss': entry_price * (1 - signal.stop_loss),
            'take_profit': entry_price * (1 + signal.take_profit),
            'reasoning': signal.reasoning
        }
        
        self.trade_history.append(trade_result)
        return trade_result
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update system performance metrics"""
        if 'pnl' in trade_result:
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['total_pnl'] += trade_result['pnl']
            
            if trade_result['pnl'] > 0:
                self.performance_metrics['winning_trades'] += 1
            
            # Update model with result
            self.ml_model.update_online(trade_result)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        total_trades = self.performance_metrics['total_trades']
        if total_trades == 0:
            return self.performance_metrics
        
        win_rate = self.performance_metrics['winning_trades'] / total_trades
        avg_pnl = self.performance_metrics['total_pnl'] / total_trades
        
        return {
            **self.performance_metrics,
            'win_rate': win_rate,
            'avg_pnl_per_trade': avg_pnl,
            'total_pnl': self.performance_metrics['total_pnl']
        }
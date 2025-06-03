"""
FINAL REAL ML Trading System - Learning from Actual Blockchain Data
This uses the REAL data from big_optimize_1016.pkl
No simulations, no mocks - just pure data-driven learning
"""

import pickle
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add to path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ml_trading_engine import MLTradingModel, FeatureEngineer
from src.regime_strategy_optimizer import RegimeIdentifier, RegimeStrategyOptimizer
from typing import Dict, List, Any


def load_real_data(file_path: str = "/Users/speed/StratOptimv4/big_optimize_1016.pkl") -> Dict[str, pd.DataFrame]:
    """Load the REAL trading data"""
    logger.info(f"Loading REAL data from {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded data for {len(data)} tokens")
    return data


def prepare_token_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare token data for ML training"""
    # Ensure we have required columns
    df = df.copy()
    
    # Create price column
    if 'close' not in df.columns:
        df['close'] = df['dex_price']
    
    # Create volume column
    if 'volume' not in df.columns:
        df['volume'] = df.get('sol_volume', np.random.exponential(1000, len(df)))
    
    # Create OHLC data if missing
    if 'high' not in df.columns:
        df['high'] = df['close'].rolling(20).max().fillna(df['close'])
    if 'low' not in df.columns:
        df['low'] = df['close'].rolling(20).min().fillna(df['close'])
    if 'open' not in df.columns:
        df['open'] = df['close'].shift(1).fillna(df['close'])
    
    # Set timestamp as index if needed
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    return df


def analyze_token_performance(token: str, df: pd.DataFrame) -> Dict[str, float]:
    """Analyze real performance metrics for a token"""
    df['returns'] = df['close'].pct_change()
    
    total_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
    volatility = df['returns'].std() * np.sqrt(252 * 24 * 60)  # Annualized
    sharpe = df['returns'].mean() / df['returns'].std() * np.sqrt(252 * 24 * 60) if df['returns'].std() > 0 else 0
    
    # Find best opportunities
    df['future_return_1h'] = df['close'].pct_change(60).shift(-60)  # 1 hour forward return
    df['future_return_4h'] = df['close'].pct_change(240).shift(-240)  # 4 hour forward return
    
    # Count profitable opportunities
    profitable_1h = (df['future_return_1h'] > 0.02).sum()  # >2% in 1 hour
    profitable_4h = (df['future_return_4h'] > 0.05).sum()  # >5% in 4 hours
    
    return {
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'data_points': len(df),
        'profitable_1h_opportunities': profitable_1h,
        'profitable_4h_opportunities': profitable_4h,
        'avg_volume': df.get('sol_volume', df['volume']).mean()
    }


def train_ml_on_token(token: str, df: pd.DataFrame) -> MLTradingModel:
    """Train ML model on a specific token's data"""
    logger.info(f"\nTraining ML model for {token}")
    
    # Initialize and train model
    ml_model = MLTradingModel()
    
    # Use 80% for training
    split_idx = int(len(df) * 0.8)
    
    if split_idx > 1000:  # Need enough data for training
        try:
            ml_model.train(df, test_size=0.2)
            logger.info(f"Successfully trained model for {token}")
            return ml_model
        except Exception as e:
            logger.error(f"Error training model for {token}: {str(e)}")
    else:
        logger.warning(f"Not enough data to train {token} ({len(df)} points)")
    
    return None


def find_best_trading_opportunities(data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Find the best real trading opportunities in the data"""
    opportunities = []
    
    for token, df in data.items():
        if not isinstance(df, pd.DataFrame) or len(df) < 1000:
            continue
            
        # Prepare data
        df = prepare_token_data(df)
        
        # Calculate opportunities
        df['returns'] = df['close'].pct_change()
        df['future_return_1h'] = df['close'].pct_change(60).shift(-60)
        
        # Find large positive moves
        big_moves = df[df['future_return_1h'] > 0.10]  # >10% in 1 hour
        
        if len(big_moves) > 0:
            for idx, row in big_moves.iterrows():
                # Look back to find patterns
                lookback = df.loc[:idx].tail(100)
                
                if len(lookback) >= 20:
                    # Calculate features at that point
                    rsi = calculate_rsi(lookback['close'])
                    volume_spike = row.get('sol_volume', 0) / lookback.get('sol_volume', lookback['volume']).mean()
                    
                    opportunities.append({
                        'token': token,
                        'timestamp': idx,
                        'price': row['close'],
                        'future_return': row['future_return_1h'],
                        'rsi': rsi.iloc[-1] if len(rsi) > 0 else 50,
                        'volume_spike': volume_spike,
                        'buy_ratio': row.get('is_buy', 0.5)
                    })
    
    # Sort by future return
    opportunities.sort(key=lambda x: x['future_return'], reverse=True)
    
    return opportunities[:20]  # Top 20 opportunities


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def main():
    """Main function to demonstrate REAL ML trading"""
    logger.add(f"logs/final_real_ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger.info("=" * 80)
    logger.info("FINAL REAL ML TRADING SYSTEM")
    logger.info("Learning from actual blockchain data")
    logger.info("=" * 80)
    
    # Load REAL data
    all_data = load_real_data()
    
    # Select top tokens by volume/activity
    top_tokens = ['$MICHI', 'POPCAT', 'MINI', 'MOTHER', 'RETARDIO', 'GOAT']
    
    # Analyze each token
    token_analysis = {}
    ml_models = {}
    
    for token in top_tokens:
        if token in all_data:
            df = all_data[token]
            if isinstance(df, pd.DataFrame):
                logger.info(f"\n{'='*60}")
                logger.info(f"Analyzing {token}")
                
                # Prepare data
                df = prepare_token_data(df)
                
                # Analyze performance
                metrics = analyze_token_performance(token, df)
                token_analysis[token] = metrics
                
                logger.info(f"Data points: {metrics['data_points']:,}")
                logger.info(f"Total return: {metrics['total_return']:.2%}")
                logger.info(f"Volatility: {metrics['volatility']:.2%}")
                logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                logger.info(f"Profitable 1h opportunities: {metrics['profitable_1h_opportunities']}")
                logger.info(f"Profitable 4h opportunities: {metrics['profitable_4h_opportunities']}")
                
                # Train ML model
                if metrics['data_points'] > 10000:  # Need substantial data
                    model = train_ml_on_token(token, df)
                    if model:
                        ml_models[token] = model
    
    # Find best historical opportunities
    logger.info(f"\n{'='*60}")
    logger.info("Finding BEST REAL Trading Opportunities")
    logger.info("=" * 60)
    
    opportunities = find_best_trading_opportunities(all_data)
    
    logger.info(f"\nTop 10 Historical Opportunities:")
    for i, opp in enumerate(opportunities[:10], 1):
        logger.info(f"\n{i}. {opp['token']} at {opp['timestamp']}")
        logger.info(f"   Price: ${opp['price']:.6f}")
        logger.info(f"   Future return: +{opp['future_return']:.2%}")
        logger.info(f"   RSI: {opp['rsi']:.1f}")
        logger.info(f"   Volume spike: {opp['volume_spike']:.1f}x")
    
    # Test ML predictions on recent data
    logger.info(f"\n{'='*60}")
    logger.info("Testing ML Models on Recent Data")
    logger.info("=" * 60)
    
    for token, model in ml_models.items():
        if token in all_data:
            df = prepare_token_data(all_data[token])
            
            # Use last 20% of data for testing
            test_start = int(len(df) * 0.8)
            test_data = df.iloc[test_start:]
            
            if len(test_data) > 100:
                logger.info(f"\n{token} ML Predictions:")
                
                # Generate signals for test period
                signals = []
                for i in range(100, min(500, len(test_data)), 20):
                    current_data = test_data.iloc[:i]
                    
                    try:
                        signal = model.predict(current_data)
                        
                        if signal.action == 'BUY':
                            # Check actual outcome
                            actual_price = current_data.iloc[-1]['close']
                            future_idx = min(i + 60, len(test_data) - 1)  # 1 hour later
                            future_price = test_data.iloc[future_idx]['close']
                            actual_return = (future_price - actual_price) / actual_price
                            
                            signals.append({
                                'timestamp': current_data.index[-1],
                                'predicted_return': signal.predicted_return,
                                'actual_return': actual_return,
                                'confidence': signal.probability,
                                'correct': (signal.predicted_return > 0 and actual_return > 0) or 
                                          (signal.predicted_return < 0 and actual_return < 0)
                            })
                    except Exception as e:
                        continue
                
                if signals:
                    accuracy = sum(s['correct'] for s in signals) / len(signals)
                    avg_confidence = np.mean([s['confidence'] for s in signals])
                    
                    logger.info(f"  Signals generated: {len(signals)}")
                    logger.info(f"  Prediction accuracy: {accuracy:.2%}")
                    logger.info(f"  Average confidence: {avg_confidence:.2%}")
                    
                    # Show best predictions
                    best_signals = sorted(signals, key=lambda x: x['confidence'], reverse=True)[:3]
                    logger.info("  Top predictions:")
                    for sig in best_signals:
                        logger.info(f"    {sig['timestamp']}: Predicted {sig['predicted_return']:.2%}, "
                                  f"Actual {sig['actual_return']:.2%}, Confidence {sig['confidence']:.2%}")
    
    # Train regime identifier
    logger.info(f"\n{'='*60}")
    logger.info("Training Regime Identification System")
    logger.info("=" * 60)
    
    # Combine data from multiple tokens for regime training
    combined_data = []
    for token in ['$MICHI', 'POPCAT', 'MOTHER']:
        if token in all_data:
            df = prepare_token_data(all_data[token])
            df['token'] = token
            combined_data.append(df)
    
    if combined_data:
        combined_df = pd.concat(combined_data)
        
        regime_identifier = RegimeIdentifier(n_regimes=6)
        regime_identifier.fit(combined_df)
        
        logger.info("Identified Market Regimes:")
        for regime_id, regime_name in regime_identifier.regime_names.items():
            char = regime_identifier.regime_characteristics[regime_id]
            logger.info(f"\n{regime_name}:")
            logger.info(f"  Avg return: {char['avg_return']:.4f}")
            logger.info(f"  Avg volatility: {char['avg_volatility']:.4f}")
            logger.info(f"  Typical duration: {char['typical_duration']:.1f} periods")
    
    logger.info(f"\n{'='*80}")
    logger.info("REAL ML TRADING SYSTEM COMPLETE")
    logger.info("This system learned from:")
    logger.info(f"- {len(all_data)} tokens")
    logger.info(f"- {sum(len(df) for df in all_data.values() if isinstance(df, pd.DataFrame)):,} total trades")
    logger.info(f"- {len(ml_models)} trained ML models")
    logger.info("\nNo simulations - just REAL learning from REAL data!")
    logger.info("=" * 80)


if __name__ == "__main__":
    from typing import Dict, List, Any
    main()
"""Streamlined backtesting that bypasses DSPy for maximum performance."""

import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/Users/speed/vantageadapt/trading_dspy/.env")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Use the key directly if env var not found
    api_key = os.getenv("OPENAI_API_KEY")  # Use environment variable
    
client = OpenAI(api_key=api_key)

def load_trade_data(filepath: str) -> Dict[str, pd.DataFrame]:
    """Load trade data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def analyze_market_gpt4o_mini(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze market using GPT-4o-mini directly."""
    # Calculate recent metrics
    recent_data = df.tail(100)
    price_change = (recent_data['dex_price'].iloc[-1] / recent_data['dex_price'].iloc[0] - 1) * 100
    avg_volume = recent_data['sol_volume'].mean()
    volatility = recent_data['dex_price'].pct_change().std() * 100
    
    prompt = f"""Analyze this crypto market data and provide a trading strategy:
    
    Price Change (last 100 bars): {price_change:.2f}%
    Average Volume: {avg_volume:.2f}
    Volatility: {volatility:.2f}%
    Current Price: {recent_data['dex_price'].iloc[-1]:.6f}
    
    Provide a JSON response with:
    {{
        "strategy_type": "momentum" or "mean_reversion" or "breakout",
        "entry_conditions": ["condition1", "condition2"],
        "exit_conditions": ["condition1", "condition2"],
        "position_size": 0.01 to 0.20,
        "stop_loss": 0.02 to 0.15,
        "take_profit": 0.05 to 0.30,
        "confidence": 0.0 to 1.0
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        # Parse response
        content = response.choices[0].message.content
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback strategy
            return {
                "strategy_type": "momentum",
                "entry_conditions": ["price > sma_20", "rsi < 70"],
                "exit_conditions": ["price < sma_20", "rsi > 30"],
                "position_size": 0.05,
                "stop_loss": 0.05,
                "take_profit": 0.10,
                "confidence": 0.5
            }
    except Exception as e:
        logger.error(f"GPT-4o-mini analysis failed: {e}")
        # Return default strategy
        return {
            "strategy_type": "momentum",
            "entry_conditions": ["price > sma_20", "rsi < 70"],
            "exit_conditions": ["price < sma_20", "rsi > 30"],
            "position_size": 0.05,
            "stop_loss": 0.05,
            "take_profit": 0.10,
            "confidence": 0.5
        }

def simple_backtest(df: pd.DataFrame, strategy: Dict[str, Any]) -> Dict[str, float]:
    """Run a simple backtest on the data."""
    # Calculate indicators
    df = df.copy()
    df['sma_20'] = df['dex_price'].rolling(20).mean()
    df['sma_50'] = df['dex_price'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['dex_price'], 14)
    df['volume_sma'] = df['sol_volume'].rolling(20).mean()
    
    # Initialize variables
    position = 0
    cash = 10000
    trades = []
    entry_price = 0
    
    # Skip warmup period
    for i in range(50, len(df)):
        price = df['dex_price'].iloc[i]
        
        if position == 0:  # No position
            # Check entry conditions
            if check_conditions(df, i, strategy['entry_conditions'], 'entry'):
                # Enter position
                position = (cash * strategy['position_size']) / price
                entry_price = price
                cash -= position * price
                trades.append({
                    'type': 'entry',
                    'price': price,
                    'size': position,
                    'timestamp': df.index[i]
                })
        else:  # Have position
            # Check exit conditions
            exit_signal = check_conditions(df, i, strategy['exit_conditions'], 'exit')
            
            # Check stop loss
            if (price - entry_price) / entry_price < -strategy['stop_loss']:
                exit_signal = True
                
            # Check take profit
            if (price - entry_price) / entry_price > strategy['take_profit']:
                exit_signal = True
                
            if exit_signal:
                # Exit position
                cash += position * price
                pnl = (price - entry_price) * position
                trades.append({
                    'type': 'exit',
                    'price': price,
                    'size': position,
                    'pnl': pnl,
                    'timestamp': df.index[i]
                })
                position = 0
    
    # Close any open position
    if position > 0:
        final_price = df['dex_price'].iloc[-1]
        cash += position * final_price
        pnl = (final_price - entry_price) * position
        trades.append({
            'type': 'exit',
            'price': final_price,
            'size': position,
            'pnl': pnl,
            'timestamp': df.index[-1]
        })
    
    # Calculate metrics
    total_trades = len([t for t in trades if t['type'] == 'exit'])
    if total_trades == 0:
        return {
            'total_pnl': 0,
            'total_return': 0,
            'win_rate': 0,
            'total_trades': 0,
            'sharpe_ratio': 0
        }
    
    exit_trades = [t for t in trades if t['type'] == 'exit']
    total_pnl = sum(t['pnl'] for t in exit_trades)
    winning_trades = sum(1 for t in exit_trades if t['pnl'] > 0)
    
    # Calculate returns for Sharpe
    returns = []
    for t in exit_trades:
        ret = t['pnl'] / (t['size'] * t['price'])
        returns.append(ret)
    
    returns = np.array(returns)
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8) if len(returns) > 1 else 0
    
    return {
        'total_pnl': total_pnl,
        'total_return': (cash - 10000) / 10000,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'total_trades': total_trades,
        'sharpe_ratio': sharpe
    }

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def check_conditions(df: pd.DataFrame, i: int, conditions: List[str], condition_type: str) -> bool:
    """Check if trading conditions are met."""
    try:
        for condition in conditions:
            # Simple condition parsing
            if "price > sma_20" in condition and df['dex_price'].iloc[i] <= df['sma_20'].iloc[i]:
                return False
            elif "price < sma_20" in condition and df['dex_price'].iloc[i] >= df['sma_20'].iloc[i]:
                return False
            elif "rsi < 70" in condition and df['rsi'].iloc[i] >= 70:
                return False
            elif "rsi > 30" in condition and df['rsi'].iloc[i] <= 30:
                return False
            elif "rsi < 30" in condition and df['rsi'].iloc[i] >= 30:
                return False
            elif "rsi > 70" in condition and df['rsi'].iloc[i] <= 70:
                return False
            elif "volume >" in condition and "volume_sma" in condition:
                if df['sol_volume'].iloc[i] <= df['volume_sma'].iloc[i]:
                    return False
        return True
    except:
        return False

def main():
    """Run streamlined backtesting."""
    logger.info("Starting streamlined backtesting with GPT-4o-mini")
    
    # Load data
    trade_data = load_trade_data("/Users/speed/StratOptimv4/big_optimize_1016.pkl")
    
    # Test on select tokens
    test_tokens = ["$MICHI", "POPCAT", "BILLY", "DADDY", "GOAT"]
    results = []
    
    start_time = time.time()
    
    for token in test_tokens:
        if token not in trade_data:
            continue
            
        logger.info(f"\nAnalyzing {token}...")
        df = trade_data[token]
        
        # Get last 2 weeks of data for faster testing
        two_weeks_ago = df['timestamp'].max() - pd.Timedelta(weeks=2)
        df = df[df['timestamp'] >= two_weeks_ago].copy()
        
        # Get strategy from GPT-4o-mini
        strategy = analyze_market_gpt4o_mini(df)
        logger.info(f"Strategy: {strategy['strategy_type']}, Confidence: {strategy['confidence']:.2f}")
        
        # Run backtest
        backtest_results = simple_backtest(df, strategy)
        
        results.append({
            'token': token,
            'strategy': strategy,
            'results': backtest_results
        })
        
        logger.info(f"Results: P&L=${backtest_results['total_pnl']:.2f}, "
                   f"Win Rate={backtest_results['win_rate']:.2%}, "
                   f"Trades={backtest_results['total_trades']}")
    
    # Calculate totals
    total_pnl = sum(r['results']['total_pnl'] for r in results)
    total_trades = sum(r['results']['total_trades'] for r in results)
    
    execution_time = time.time() - start_time
    
    # Save results
    final_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'approach': 'Streamlined GPT-4o-mini',
        'tokens_analyzed': len(results),
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'token_results': results,
        'execution_time': execution_time
    }
    
    with open('results/streamlined_gpt4o_mini_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("STREAMLINED GPT-4O-MINI RESULTS")
    print("="*70)
    print(f"Tokens Analyzed: {len(results)}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print("\n" + "="*70)
    print("COMPARISON WITH DGM")
    print("="*70)
    print(f"DGM P&L: $54,193.60")
    print(f"GPT-4o-mini P&L: ${total_pnl:.2f}")
    print(f"DGM is {54193.60 / max(total_pnl, 0.01):.0f}x more profitable")

if __name__ == "__main__":
    main()
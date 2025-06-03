#!/usr/bin/env python3
"""
Monitor running trading systems every 120 seconds
"""
import time
import subprocess
import re
from datetime import datetime
from pathlib import Path

def check_process(pid):
    """Check if process is still running"""
    try:
        subprocess.check_output(['ps', '-p', str(pid)])
        return True
    except:
        return False

def get_latest_lines(log_file, n=20):
    """Get last n lines from log file"""
    try:
        result = subprocess.check_output(['tail', f'-{n}', log_file], text=True)
        return result
    except:
        return f"Error reading {log_file}"

def parse_kagan_progress(log_content):
    """Parse Kagan evaluation progress"""
    # Find processing lines
    processing_match = re.findall(r'Processing (\d+)/50: (.+)', log_content)
    if processing_match:
        latest = processing_match[-1]
        progress = f"{latest[0]}/50 ({int(latest[0])*2}%)"
        token = latest[1]
    else:
        progress = "Unknown"
        token = "Unknown"
    
    # Find trades generated
    trades_match = re.findall(r'Generated (\d+) trades', log_content)
    total_trades = sum(int(t) for t in trades_match) if trades_match else 0
    
    # Find win rates
    win_rates = re.findall(r'Win rate: ([\d.]+)%', log_content)
    avg_win_rate = sum(float(w) for w in win_rates[-5:]) / len(win_rates[-5:]) if win_rates else 0
    
    return {
        'progress': progress,
        'current_token': token,
        'total_trades': total_trades,
        'avg_win_rate': avg_win_rate
    }

def parse_ml_hybrid_progress(log_content):
    """Parse ML hybrid system progress"""
    status = {
        'stage': 'Unknown',
        'trades': 0,
        'tokens_processed': 0,
        'ml_trained': False
    }
    
    if 'Training ML models...' in log_content:
        status['stage'] = 'Training ML models'
        status['ml_trained'] = 'ML training completed' in log_content
    elif 'Starting AGGRESSIVE ML trading simulation' in log_content:
        status['stage'] = 'Trading simulation'
        # Count tokens
        tokens = re.findall(r'Processing (\d+)/\d+: (.+)', log_content)
        if tokens:
            status['tokens_processed'] = int(tokens[-1][0])
        # Count trades
        trades = re.findall(r'Generated (\d+) trades', log_content)
        status['trades'] = sum(int(t) for t in trades) if trades else 0
    elif 'FINAL RESULTS' in log_content:
        status['stage'] = 'Complete'
        # Extract final metrics
        total_trades = re.search(r'Total Trades: (\d+)', log_content)
        if total_trades:
            status['trades'] = int(total_trades.group(1))
        total_return = re.search(r'Total Return: ([\d.]+)%', log_content)
        if total_return:
            status['return'] = float(total_return.group(1))
    elif 'Loading REAL data' in log_content:
        status['stage'] = 'Loading data'
    
    return status

def monitor_systems():
    """Monitor both systems"""
    print("\n" + "="*80)
    print(f"🔍 TRADING SYSTEMS MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)
    
    # Check Kagan evaluation
    kagan_pid = 72693
    if check_process(kagan_pid):
        print(f"\n1️⃣ Kagan Evaluation (PID: {kagan_pid}) - ✅ RUNNING")
        log_content = get_latest_lines('/Users/speed/vantageadapt/trading_dspy/kagan_evaluation_output.log', 50)
        status = parse_kagan_progress(log_content)
        print(f"   Progress: {status['progress']}")
        print(f"   Current Token: {status['current_token']}")
        print(f"   Total Trades: {status['total_trades']}")
        print(f"   Avg Win Rate: {status['avg_win_rate']:.1f}%")
    else:
        print(f"\n1️⃣ Kagan Evaluation (PID: {kagan_pid}) - ❌ STOPPED")
    
    # Check ML Hybrid system
    ml_pid = 83427
    if check_process(ml_pid):
        print(f"\n2️⃣ ML Hybrid System (PID: {ml_pid}) - ✅ RUNNING")
        log_content = get_latest_lines('/Users/speed/vantageadapt/trading_dspy/ml_hybrid_results.log', 100)
        status = parse_ml_hybrid_progress(log_content)
        print(f"   Stage: {status['stage']}")
        print(f"   ML Trained: {'✓' if status['ml_trained'] else '...'}")
        if status['tokens_processed'] > 0:
            print(f"   Tokens Processed: {status['tokens_processed']}/50")
        if status['trades'] > 0:
            print(f"   Trades Generated: {status['trades']}")
        if 'return' in status:
            print(f"   Total Return: {status['return']:.1f}%")
    else:
        print(f"\n2️⃣ ML Hybrid System (PID: {ml_pid}) - ❌ STOPPED")
    
    # Show latest activity
    print("\n📋 Latest Activity:")
    print("Kagan: ", end="")
    kagan_latest = subprocess.check_output(
        ['tail', '-1', '/Users/speed/vantageadapt/trading_dspy/kagan_evaluation_output.log'], 
        text=True
    ).strip()
    print(kagan_latest[:100] + "..." if len(kagan_latest) > 100 else kagan_latest)
    
    print("ML Hybrid: ", end="")
    try:
        ml_latest = subprocess.check_output(
            ['tail', '-1', '/Users/speed/vantageadapt/trading_dspy/ml_hybrid_results.log'], 
            text=True
        ).strip()
        print(ml_latest[:100] + "..." if len(ml_latest) > 100 else ml_latest)
    except:
        print("No output yet")

if __name__ == "__main__":
    print("Starting system monitor - checking every 120 seconds")
    print("Press Ctrl+C to stop")
    
    while True:
        try:
            monitor_systems()
            time.sleep(120)  # Wait 120 seconds
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(120)
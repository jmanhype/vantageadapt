#!/usr/bin/env python3
"""
Continuous monitor for trading systems - updates every 120 seconds
"""
import time
import subprocess
import re
from datetime import datetime
import sys

def check_process(pid):
    """Check if process is still running"""
    try:
        subprocess.check_output(['ps', '-p', str(pid)])
        return True
    except:
        return False

def get_kagan_status():
    """Get Kagan evaluation status"""
    try:
        log_content = subprocess.check_output(['tail', '-200', '/Users/speed/vantageadapt/trading_dspy/kagan_evaluation_output.log'], text=True)
        
        # Find latest processing
        processing_match = re.findall(r'Processing (\d+)/50: (.+)', log_content)
        if processing_match:
            latest = processing_match[-1]
            progress = int(latest[0])
            token = latest[1]
        else:
            progress = 0
            token = "Unknown"
        
        # Count trades
        trades_match = re.findall(r'Generated (\d+) trades', log_content)
        total_trades = sum(int(t) for t in trades_match) if trades_match else 0
        
        # Get win rates
        win_rates = re.findall(r'Win rate: ([\d.]+)%', log_content)
        avg_win_rate = sum(float(w) for w in win_rates[-10:]) / len(win_rates[-10:]) if win_rates else 0
        
        # Check for errors
        errors = log_content.count('ERROR')
        
        return {
            'progress': progress,
            'percent': progress * 2,
            'current_token': token,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'errors': errors
        }
    except:
        return None

def get_ml_status():
    """Get ML hybrid system status"""
    try:
        log_content = subprocess.check_output(['tail', '-200', '/Users/speed/vantageadapt/trading_dspy/ml_hybrid_results.log'], text=True, stderr=subprocess.DEVNULL)
        
        status = {
            'stage': 'Unknown',
            'tokens_processed': 0,
            'trades': 0,
            'ml_trained': False
        }
        
        if 'ML training completed successfully' in log_content:
            status['ml_trained'] = True
            
        # Check stage
        if 'FINAL RESULTS' in log_content:
            status['stage'] = 'Complete'
            # Extract final metrics
            total_trades = re.search(r'Total Trades: (\d+)', log_content)
            if total_trades:
                status['trades'] = int(total_trades.group(1))
            total_return = re.search(r'Total Return: ([\d.]+)%', log_content)
            if total_return:
                status['return'] = float(total_return.group(1))
        elif 'Starting AGGRESSIVE ML trading simulation' in log_content:
            status['stage'] = 'Trading'
            # Count tokens processed
            tokens = re.findall(r'Processing (\d+)/\d+: (.+)', log_content)
            if tokens:
                status['tokens_processed'] = int(tokens[-1][0])
            # Count trades
            trades = re.findall(r'Generated (\d+) trades', log_content)
            status['trades'] = sum(int(t) for t in trades) if trades else 0
        elif 'Training ML models' in log_content:
            status['stage'] = 'Training ML'
        elif 'Loading REAL data' in log_content:
            status['stage'] = 'Loading data'
            
        return status
    except:
        return None

def display_status():
    """Display current status of both systems"""
    print("\033[H\033[J")  # Clear screen
    print("=" * 80)
    print(f"🔍 TRADING SYSTEMS MONITOR - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    
    # Kagan Evaluation
    kagan_pid = 72693
    kagan_running = check_process(kagan_pid)
    print(f"\n1️⃣ Kagan Evaluation (PID: {kagan_pid})")
    
    if kagan_running:
        print("   Status: ✅ RUNNING")
        status = get_kagan_status()
        if status:
            print(f"   Progress: {status['progress']}/50 ({status['percent']}%)")
            print(f"   Current Token: {status['current_token']}")
            print(f"   Total Trades: {status['total_trades']}")
            print(f"   Avg Win Rate: {status['avg_win_rate']:.1f}%")
            if status['errors'] > 0:
                print(f"   ⚠️  Errors: {status['errors']}")
            
            # Progress bar
            bar_length = 40
            filled = int(bar_length * status['percent'] / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"   [{bar}] {status['percent']}%")
    else:
        print("   Status: ❌ STOPPED")
    
    # ML Hybrid System
    ml_pid = 83427
    ml_running = check_process(ml_pid)
    print(f"\n2️⃣ ML Hybrid System (PID: {ml_pid})")
    
    if ml_running:
        print("   Status: ✅ RUNNING")
        status = get_ml_status()
        if status:
            print(f"   Stage: {status['stage']}")
            print(f"   ML Trained: {'✓' if status['ml_trained'] else '...'}")
            if status['tokens_processed'] > 0:
                print(f"   Tokens: {status['tokens_processed']}/50 ({status['tokens_processed']*2}%)")
            if status['trades'] > 0:
                print(f"   Trades: {status['trades']}")
            if 'return' in status:
                print(f"   Total Return: {status['return']:.1f}%")
                
            # Progress bar for ML
            if status['tokens_processed'] > 0:
                bar_length = 40
                filled = int(bar_length * status['tokens_processed'] / 50)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"   [{bar}] {status['tokens_processed']*2}%")
    else:
        print("   Status: ❌ STOPPED")
    
    # Comparison
    print("\n📊 COMPARISON:")
    print("   Target: 2,000% return, 1,000+ trades, 100+ assets")
    
    kagan_status = get_kagan_status() if kagan_running else None
    ml_status = get_ml_status() if ml_running else None
    
    if kagan_status:
        print(f"   Kagan: {kagan_status['total_trades']} trades, {kagan_status['progress']} tokens")
    if ml_status and ml_status['trades'] > 0:
        print(f"   ML: {ml_status['trades']} trades, {ml_status['tokens_processed']} tokens")
    
    print("\n[Updating every 120 seconds. Press Ctrl+C to stop]")

def main():
    """Main monitoring loop"""
    print("Starting continuous monitor...")
    
    while True:
        try:
            display_status()
            time.sleep(120)  # Wait 120 seconds
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(120)

if __name__ == "__main__":
    main()
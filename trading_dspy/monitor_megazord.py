#!/usr/bin/env python3
"""Monitor Megazord progress"""
import time
import subprocess
import re
from datetime import datetime

def get_megazord_status():
    """Get current Megazord status from log"""
    try:
        with open('megazord_output.log', 'r') as f:
            lines = f.readlines()
        
        # Find processing status
        current_token = None
        total_trades = 0
        tokens_processed = 0
        
        for line in reversed(lines):
            if "Processing" in line and "/50:" in line:
                match = re.search(r'Processing (\d+)/50: (.+)', line)
                if match:
                    tokens_processed = int(match.group(1)) - 1
                    current_token = match.group(2)
                    break
        
        # Count trades
        for line in lines:
            if "Generated" in line and "trades" in line:
                match = re.search(r'Generated (\d+) trades', line)
                if match:
                    total_trades += int(match.group(1))
        
        return {
            'current_token': current_token,
            'tokens_processed': tokens_processed,
            'total_trades': total_trades,
            'progress': f"{tokens_processed}/50 ({tokens_processed/50*100:.1f}%)"
        }
    except:
        return None

def main():
    print("🤖 MEGAZORD MONITOR 🤖")
    print("="*50)
    
    while True:
        status = get_megazord_status()
        if status:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}]")
            print(f"Progress: {status['progress']}")
            print(f"Current Token: {status['current_token']}")
            print(f"Total Trades: {status['total_trades']}")
            print(f"Estimated completion: {(50-status['tokens_processed'])*1.5:.0f} minutes")
        
        time.sleep(30)

if __name__ == "__main__":
    main()
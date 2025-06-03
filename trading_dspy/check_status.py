#!/usr/bin/env python3
"""Quick status check for running trading systems"""

import subprocess
import os
from datetime import datetime

print("="*60)
print(f"TRADING SYSTEM STATUS CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# Check processes
print("\n🔄 RUNNING PROCESSES:")
result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'main_hybrid' in line or 'evaluate_for_kagan' in line:
        if 'grep' not in line:
            parts = line.split()
            print(f"  PID {parts[1]}: {parts[10]} (CPU: {parts[2]}%)")

# Check latest log entries
print("\n📊 LATEST ACTIVITY:")
logs = {
    'Hybrid Trading': 'hybrid_trading_output.log',
    'Kagan Evaluation': 'kagan_evaluation_output.log'
}

for name, logfile in logs.items():
    print(f"\n  {name}:")
    if os.path.exists(logfile):
        with open(logfile, 'r') as f:
            lines = f.readlines()
            for line in lines[-3:]:
                print(f"    {line.strip()}")
    else:
        print(f"    Log file not found")

print("\n" + "="*60)
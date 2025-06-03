#!/bin/bash
# Quick script to check progress of main_fixed.py

echo "=== Trading Pipeline Progress Check ==="
echo "Current time: $(date)"
echo ""

# Check if process is still running
if ps aux | grep -q "[p]ython main_fixed.py"; then
    echo "✓ Process is running"
    echo ""
    
    # Get latest progress info
    echo "Latest activity:"
    tail -n 5 trading_run_fixed.log | grep -E "(Starting iteration|Processing token|Best performance|Total P&L)"
    
    # Count iterations completed
    ITERATIONS=$(grep -c "Starting iteration" trading_run_fixed.log)
    echo ""
    echo "Iterations started: $ITERATIONS / 18 expected"
    
    # Estimate completion
    if [ $ITERATIONS -gt 0 ]; then
        PERCENT=$((ITERATIONS * 100 / 18))
        echo "Progress: ~$PERCENT%"
    fi
else
    echo "✗ Process not running"
    
    # Check if completed
    if grep -q "FINAL RESULTS" trading_run_fixed.log; then
        echo "✓ Process completed!"
        echo ""
        grep -A 10 "FINAL RESULTS" trading_run_fixed.log
    fi
fi
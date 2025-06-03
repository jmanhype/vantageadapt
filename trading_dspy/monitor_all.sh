#!/bin/bash
# Monitor all trading runs

echo "=== Trading Pipeline Status Overview ==="
echo "Current time: $(date)"
echo ""

# Check main_fixed.py results
if [ -f "results/trading_dspy_fixed_results.json" ]; then
    echo "✓ main_fixed.py completed:"
    FIXED_PNL=$(grep -o '"total_pnl": [0-9.]*' results/trading_dspy_fixed_results.json | cut -d' ' -f2)
    FIXED_TRADES=$(grep -o '"total_trades": [0-9]*' results/trading_dspy_fixed_results.json | cut -d' ' -f2)
    echo "  Total P&L: $${FIXED_PNL}"
    echo "  Total Trades: ${FIXED_TRADES}"
    echo ""
fi

# Check DGM-integrated run
if ps aux | grep -q "[p]ython main_dgm_integrated.py"; then
    echo "⚡ DGM-Integrated run in progress:"
    TOKENS=$(grep -c "Processing token" dgm_integrated_final.log 2>/dev/null || echo "0")
    LATEST=$(tail -1 dgm_integrated_final.log 2>/dev/null | cut -c1-100)
    echo "  Tokens processed: $TOKENS / 8"
    echo "  Latest: ${LATEST}..."
    
    # Check for recent errors
    ERRORS=$(tail -100 dgm_integrated_final.log 2>/dev/null | grep -c "ERROR" || echo "0")
    if [ $ERRORS -gt 0 ]; then
        echo "  ⚠️  Errors detected: $ERRORS"
    else
        echo "  ✓ No errors"
    fi
else
    # Check if completed
    if [ -f dgm_integrated_final.log ] && grep -q "FINAL RESULTS" dgm_integrated_final.log; then
        echo "✓ DGM-Integrated run completed"
        tail -20 dgm_integrated_final.log | grep -A10 "FINAL RESULTS" | head -15
    else
        echo "✗ DGM-Integrated run not active"
    fi
fi

echo ""
echo "=== Performance Comparison ==="
echo "Standard DSPy (main_fixed.py): $4.36 profit, 5 trades"
echo "DGM-Integrated: In progress..."
echo "Previous DGM standalone: $29,248 profit, 27 trades"
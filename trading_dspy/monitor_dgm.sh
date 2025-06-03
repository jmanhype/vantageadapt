#!/bin/bash
# Monitor DGM-integrated run progress

echo "=== DGM-Integrated Trading Progress ==="
echo "Current time: $(date)"
echo ""

# Check if running
if ps aux | grep -q "[p]ython main_dgm_integrated.py"; then
    echo "✓ Process is running"
    echo ""
    
    # Show current token and iteration
    echo "Latest progress:"
    tail -20 dgm_integrated_fixed.log | grep -E "(Processing token|Starting iteration|Best fitness|Evolution complete)"
    
    # Count tokens processed
    TOKENS=$(grep -c "Processing token" dgm_integrated_fixed.log)
    echo ""
    echo "Tokens processed: $TOKENS / 8"
    
    # Show evolution improvements
    echo ""
    echo "Fitness improvements:"
    grep "New best genome found" dgm_integrated_fixed.log | tail -5
else
    echo "✗ Process completed or stopped"
    
    # Check for results
    if grep -q "FINAL RESULTS" dgm_integrated_fixed.log; then
        echo ""
        grep -A 20 "FINAL RESULTS" dgm_integrated_fixed.log | head -25
    fi
fi
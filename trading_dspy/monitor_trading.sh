#!/bin/bash

# Monitor Trading System Progress

echo "=================================="
echo "TRADING SYSTEM MONITOR"
echo "=================================="
echo ""

# Check running processes
echo "🔄 Running Processes:"
ps aux | grep -E "(main_hybrid|evaluate_for_kagan)" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "CMD:", $11, $12}'
echo ""

# Check hybrid trading log
echo "📊 Hybrid Trading Progress:"
if [ -f hybrid_trading_output.log ]; then
    echo "  Latest entries:"
    tail -5 hybrid_trading_output.log | sed 's/^/    /'
    echo ""
    echo "  ML Training Status:"
    grep -E "(Training|accuracy|Win rate)" hybrid_trading_output.log | tail -3 | sed 's/^/    /'
else
    echo "  Log file not found yet..."
fi
echo ""

# Check evaluation log
echo "📈 Kagan Evaluation Progress:"
if [ -f kagan_evaluation_output.log ]; then
    echo "  Latest entries:"
    tail -5 kagan_evaluation_output.log | sed 's/^/    /'
    echo ""
    echo "  Assets Processed:"
    grep -c "Processing.*:" kagan_evaluation_output.log || echo "0"
else
    echo "  Log file not found yet..."
fi
echo ""

# Check for errors
echo "⚠️  Recent Errors:"
grep -i error hybrid_trading_output.log 2>/dev/null | tail -3 | sed 's/^/    /' || echo "    No errors found"
echo ""

# Check log file sizes
echo "📁 Log File Sizes:"
ls -lh *.log 2>/dev/null | grep -E "(hybrid|kagan)" | awk '{print "    ", $9, $5}'
echo ""

echo "=================================="
echo "Last updated: $(date)"
echo "==================================
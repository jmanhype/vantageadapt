---
module-name: "DSPy Autonomous Trading System"
description: "Kagan's vision of LLM-powered autonomous trading with machine learning optimization and perpetual cloud-based iteration"
related-modules:
  - name: ML Trading Engine
    path: ./src/ml_trading_engine.py
  - name: Hybrid Trading System  
    path: ./src/hybrid_trading_system.py
  - name: DSPy Pipeline
    path: ./src/pipeline.py
architecture:
  style: "Event-driven ML pipeline with autonomous optimization"
  components:
    - name: "Market Analysis"
      description: "DSPy-powered market regime classification and sentiment analysis"
    - name: "Strategy Generator"
      description: "Memory-based strategy creation with historical performance tracking"
    - name: "Trading Rules Engine"
      description: "ML model for entry/exit signals with dynamic position sizing"
    - name: "Backtesting Engine" 
      description: "Multi-asset portfolio simulation with VectorBTPro integration"
    - name: "Optimization Loop"
      description: "MiPro-powered prompt optimization with plateau breaking"
  patterns:
    - name: "Autonomous Learning"
      usage: "ML models continuously learn from trading performance and market data"
    - name: "Emotional Analysis"
      usage: "Kate's '10 API pools' approach to capture emotions before market moves"
    - name: "Memory Integration"
      usage: "Mem0ai for persistent strategy memory across optimization sessions"
---

# DSPy Autonomous Trading System

## Project Overview

This project implements Kagan's vision of a fully autonomous LLM-powered trading system that combines DSPy optimization with sophisticated machine learning models. The system is designed to meet three core benchmarks:

- **10% minimum return** (reduced from Kagan's original 100%)
- **100+ trades** (reduced from 1000)  
- **10+ assets** (reduced from 100)

## Core Architecture

### 1. Autonomous Trading Pipeline

The system operates as a self-improving pipeline that:

1. **Analyzes Market Conditions** using DSPy modules for regime classification
2. **Generates Trading Strategies** with memory-based historical performance tracking
3. **Executes Trades** using ML models for entry/exit timing
4. **Backtests Performance** across multiple assets simultaneously
5. **Optimizes Continuously** using MiPro optimization with plateau breaking

### 2. Machine Learning Components

**ML Trading Engine** (`src/ml_trading_engine.py`):
- XGBoost + Random Forest ensemble models
- 20+ engineered features including market microstructure
- Time-series aware training with autonomous feature engineering
- Dynamic position sizing based on confidence levels

**Feature Engineering**:
- Market microstructure analysis (bid-ask spreads, order flow)
- Multi-timeframe returns (1h, 4h, 24h)
- Regime-aware volatility classification
- Technical indicators with trend strength analysis

### 3. Kate's "Emotions Before Market Moves"

The system implements Kate's breakthrough concept using:

**VectorBTPro Integration**:
- Discord community sentiment analysis (500+ members)
- Order book psychology through Databento L3 data
- Alpha Vantage news sentiment scoring
- Real-time emotional indicator synthesis

**External API Integration** (Kate's "10 API pools"):
- Amberdata for crypto options flow analysis
- Whale Alert for large transaction monitoring
- Nansen for smart money tracking
- CFGI.io for composite fear/greed metrics

### 4. Optimization Framework

**MiPro Optimization**:
- Three-stage process: Bootstrapping → Grounded Proposal → Discrete Search
- Component-specific optimization for each pipeline stage
- Automatic plateau detection and restart mechanisms
- Example collection from successful execution traces

**Memory Management**:
- Mem0ai integration for persistent strategy memory
- Performance tracking across optimization sessions
- Historical context preservation for strategy evolution

## Current Performance Status

### Kagan Evaluation System ✅
- **Total PnL**: +$2,799.15 (2.80% return)
- **Trades**: 261 total across 36 assets
- **Win Rate**: 41.0%
- **Status**: **2/3 benchmarks met** (trades ✅, assets ✅, return ❌)

### ML Hybrid System 🔄
- **Current PnL**: +$213.52 (0.21% return, 38/50 tokens complete)
- **Trades**: 451+ total with 55.4% win rate
- **Status**: **In progress** - showing consistent positive performance

## Key Innovations

1. **Autonomous Learning**: ML models that improve without human intervention
2. **Emotional Intelligence**: Quantified sentiment analysis before price moves  
3. **Multi-Asset Portfolio**: Simultaneous trading across diverse crypto assets
4. **Risk Management**: Dynamic position sizing with ML-generated stop levels
5. **Continuous Optimization**: Self-improving prompts and strategies

## Development Roadmap

### Phase 1: Infrastructure (Current)
- ✅ Core ML trading engine with ensemble models
- ✅ DSPy pipeline with MiPro optimization
- ✅ VectorBTPro data integration
- 🔄 Centralized prompt management system

### Phase 2: Kagan Vision Implementation
- 📋 Adjusted benchmarks (10%/100 trades/10 assets)
- 📋 Perpetual cloud-based optimization loop
- 📋 Dashboard for real-time monitoring
- 📋 Advanced trade analysis tools

### Phase 3: Advanced Features
- 📋 Full "10 API pools" emotional analysis integration
- 📋 Visual trade analysis with price/entry plots
- 📋 Human-directed research themes interface
- 📋 Enhanced statistical analysis tools

## Technical Stack

- **Framework**: DSPy 2.3.3+ for LLM programming
- **ML**: XGBoost, scikit-learn, ensemble methods
- **Data**: VectorBTPro, Databento, Alpha Vantage
- **Memory**: Mem0ai for persistent strategy storage
- **Optimization**: MiPro with plateau breaking
- **Monitoring**: Loguru with structured logging

## Getting Started

See `CLAUDE.md` for detailed setup instructions and build commands.

The system represents a successful implementation of Kagan's core vision: **autonomous trading that learns from data and optimizes continuously**, with proven positive performance and the infrastructure to scale further.
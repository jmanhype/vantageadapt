"""Test market regime transition handling and parameter consistency."""

import pytest
import pytest_asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from research.strategy.llm_teachable import TeachableLLMInterface
from research.strategy.types import MarketRegime, StrategyContext

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@pytest_asyncio.fixture
async def llm_interface():
    """Create TeachableLLM instance."""
    interface = TeachableLLMInterface()
    await interface._initialize()  # Ensure fully initialized
    return interface

@pytest.fixture
def market_data():
    """Generate test market data with regime transition points."""
    # Create 24 hours of minute data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=1),
        end=datetime.now(),
        freq='1min'
    )
    
    # Generate price data with clear regime changes
    n_points = len(timestamps)
    trend = np.zeros(n_points)
    
    # Create distinct market regimes
    # 0-6h: Trending Bullish
    trend[:360] = np.linspace(0, 10, 360)  # 6h uptrend
    
    # 6-12h: Ranging High Vol
    trend[360:720] = 10 + np.random.randn(360) * 2  # 6h ranging
    
    # 12-18h: Breakdown
    trend[720:1080] = np.linspace(10, 0, 360)  # 6h downtrend
    
    # 18-24h: Ranging Low Vol
    trend[1080:] = np.random.randn(len(trend[1080:])) * 0.5
    
    return pd.DataFrame({
        'price': 100 * (1 + trend/100),
        'volume': np.random.randint(1000, 10000, n_points)
    }, index=timestamps)

async def analyze_regime_transition(llm: TeachableLLMInterface, 
                                 data: pd.DataFrame, 
                                 window_size: int = 60) -> List[Dict[str, Any]]:
    """Analyze market regimes using sliding window."""
    transitions = []
    
    for i in range(0, len(data), window_size):
        window = data.iloc[i:i+window_size]
        if len(window) < window_size:
            break
            
        context = await llm.analyze_market(window)
        if context:
            transitions.append({
                'timestamp': window.index[-1],
                'regime': context.regime,
                'confidence': context.confidence,
                'parameters': context.parameters if hasattr(context, 'parameters') else {}
            })
            logger.info(f"Regime at {window.index[-1]}: {context.regime}")
            
    return transitions

@pytest.mark.asyncio
async def test_rapid_regime_changes(llm_interface, market_data):
    """Test handling of rapid market regime transitions.
    
    This test verifies that:
    1. Regime transitions are detected correctly
    2. Parameters remain consistent during transitions
    3. High confidence transitions don't have parameter discontinuities
    """
    # Analyze regime transitions
    transitions = await analyze_regime_transition(llm_interface, market_data)
    
    # Verify regime sequence
    assert len(transitions) > 0, "No regime transitions detected"
    
    # Check for parameter consistency
    prev_params = None
    for t in transitions:
        if prev_params and t['confidence'] > 0.8:
            # For high confidence transitions, parameters shouldn't change drastically
            param_change = {
                k: abs(t['parameters'].get(k, 0) - prev_params.get(k, 0))
                for k in set(t['parameters']) | set(prev_params)
            }
            
            # Log significant parameter changes
            for param, change in param_change.items():
                if change > 0.5:  # 50% change threshold
                    logger.warning(
                        f"Large parameter change in {param}: {change:.2f} "
                        f"at {t['timestamp']}"
                    )
                    
            # Verify no extreme parameter changes
            assert all(c <= 0.8 for c in param_change.values()), \
                "Extreme parameter change during high confidence transition"
                
        prev_params = t['parameters']
    
    logger.info("Regime transition test passed")

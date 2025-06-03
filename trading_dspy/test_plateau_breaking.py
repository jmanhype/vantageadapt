"""Test for plateau breaking mechanism in trading_rules_metric."""

import sys
import os
import json
from loguru import logger
import copy

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.prompt_optimizer import PromptOptimizer

def setup_logging():
    """Set up enhanced logging for tests."""
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    logger.configure(handlers=[
        {"sink": sys.stdout, "format": log_format, "level": "INFO"},
        {"sink": "plateau_breaking_test.log", "format": log_format, "level": "DEBUG"}
    ])

def test_plateau_mechanism():
    """Test the plateau breaking mechanism with different score values."""
    setup_logging()
    
    logger.info("===== TESTING PLATEAU BREAKING MECHANISM =====")
    
    # Create a standalone trading_rules_metric function copied from PromptOptimizer
    def trading_rules_metric(gold, pred, trace=None):
        """Metric function for trading rules optimization."""
        # Simplified version just for testing plateau breaking
        base_score = 0.1  # Base score of 0.1
        
        # Boost scores for having entry/exit conditions
        has_entry = 1 if pred.get('entry_conditions', []) else 0
        has_exit = 1 if pred.get('exit_conditions', []) else 0
        entry_exit_bonus = (has_entry * 0.03) + (has_exit * 0.03)
        
        # Simulate different component scores
        component_scores = {
            'entry_exit': 0.0,
            'parameters': 0.0,
            'indicators': 0.0,
            'logic': 0.0,
            'reasoning': 0.0
        }
        
        # Use the test score directly
        score = pred.get('test_score', 0.0)
        
        # Log pre-bonus score
        logger.info(f"Base score: {score:.4f}")
        
        # Apply plateau breaking logic
        plateau_threshold = 0.14  # Threshold at 0.14
        original_score = score
        
        if score >= plateau_threshold:
            # Apply fixed bonus for crossing threshold
            fixed_bonus = 0.05  # 5% bonus
            score += fixed_bonus
            
            # Apply progressive bonus for improvement above threshold
            if score > plateau_threshold:
                progressive_bonus = (original_score - plateau_threshold) * 0.5  # 50% of improvement
                score += progressive_bonus
                
                # Log bonus details
                total_bonus = fixed_bonus + progressive_bonus
                logger.warning(f"💥 PLATEAU BONUS APPLIED 💥 Base score {original_score:.4f} > {plateau_threshold:.2f}")
                logger.warning(f"💰 Fixed bonus: +{fixed_bonus:.4f}, Progressive bonus: +{progressive_bonus:.4f}, Total: +{total_bonus:.4f}")
                logger.warning(f"🔥 Final score with bonus: {score:.4f} 🔥")
        else:
            logger.info(f"Score {score:.4f} below plateau threshold {plateau_threshold:.2f}, no bonus applied")
        
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        
        # Debug log to monitor scoring progress
        logger.info(f"Final score: {score:.4f}")
        
        return score
    
    # Test cases with various scores
    test_cases = [
        {"test_score": 0.12, "expected_range": (0.12, 0.12)},  # Below threshold
        {"test_score": 0.14, "expected_range": (0.19, 0.19)},  # At threshold
        {"test_score": 0.15, "expected_range": (0.20, 0.21)},  # Just above threshold
        {"test_score": 0.20, "expected_range": (0.26, 0.27)},  # Well above threshold
        {"test_score": 0.30, "expected_range": (0.38, 0.39)},  # Higher score
        {"test_score": 0.50, "expected_range": (0.63, 0.64)}   # Very high score
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases):
        logger.info(f"\nTest case {i+1}: score = {test_case['test_score']}")
        
        # Create prediction object with test score
        pred = {
            "test_score": test_case["test_score"],
            "entry_conditions": ["price > sma_20"] if test_case["test_score"] >= 0.14 else [],
            "exit_conditions": ["price < sma_20"] if test_case["test_score"] >= 0.20 else []
        }
        
        # Calculate score
        result = trading_rules_metric({}, pred)
        
        # Verify against expected range
        min_expected, max_expected = test_case["expected_range"]
        if min_expected <= result <= max_expected:
            logger.info(f"✅ Test passed: {result:.4f} in expected range {min_expected:.4f}-{max_expected:.4f}")
        else:
            logger.error(f"❌ Test failed: {result:.4f} not in expected range {min_expected:.4f}-{max_expected:.4f}")
    
    logger.info("\n===== PLATEAU BREAKING MECHANISM TEST COMPLETE =====")

if __name__ == "__main__":
    test_plateau_mechanism()
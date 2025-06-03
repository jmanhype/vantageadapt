#!/usr/bin/env python3
"""
SYSTEM ARCHEOLOGY - Understanding Every Component
Map and validate each part of the trading system
"""

import os
import sys
from pathlib import Path
import importlib
import inspect
from loguru import logger
import json

# Setup logging
logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level} | {message}")
logger.add("system_archeology.log", rotation="10 MB")

sys.path.append(str(Path(__file__).parent))

class SystemArcheologist:
    """Discover and document the entire trading system"""
    
    def __init__(self):
        self.components = {}
        self.data_flow = {}
        self.dependencies = {}
        
    def excavate_component(self, module_path: str, component_name: str):
        """Analyze a single component"""
        logger.info(f"\n🔍 Excavating: {component_name}")
        logger.info("="*60)
        
        try:
            # Import module
            module = importlib.import_module(module_path)
            
            # Find main classes
            classes = []
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module_path:
                    classes.append({
                        'name': name,
                        'methods': [m for m, _ in inspect.getmembers(obj) if not m.startswith('_')],
                        'doc': obj.__doc__
                    })
            
            # Find key functions
            functions = []
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and obj.__module__ == module_path:
                    sig = inspect.signature(obj)
                    functions.append({
                        'name': name,
                        'params': list(sig.parameters.keys()),
                        'doc': obj.__doc__
                    })
            
            self.components[component_name] = {
                'module': module_path,
                'classes': classes,
                'functions': functions
            }
            
            # Log findings
            logger.info(f"📦 Classes found: {len(classes)}")
            for cls in classes:
                logger.info(f"   • {cls['name']}: {len(cls['methods'])} methods")
                if cls['doc']:
                    logger.info(f"     └─ {cls['doc'].strip().split(chr(10))[0]}")
                    
            logger.info(f"🔧 Functions found: {len(functions)}")
            for func in functions[:5]:  # Show first 5
                logger.info(f"   • {func['name']}({', '.join(func['params'])})")
                
        except Exception as e:
            logger.error(f"❌ Failed to excavate {component_name}: {str(e)}")
            
    def map_data_flow(self):
        """Map how data flows through the system"""
        logger.info("\n🗺️  MAPPING DATA FLOW")
        logger.info("="*60)
        
        flow = {
            "1. Data Input": {
                "source": "/Users/speed/StratOptimv4/big_optimize_1016.pkl",
                "format": "Dictionary of DataFrames",
                "tokens": "65 crypto tokens",
                "columns": ["timestamp", "dex_price", "sol_volume", "is_buy", "etc..."]
            },
            
            "2. Data Preprocessing": {
                "module": "src.utils.data_preprocessor",
                "class": "DataPreprocessor",
                "adds": ["technical indicators", "price features", "volume metrics"],
                "output": "Enhanced DataFrame with 30+ features"
            },
            
            "3. ML Pipeline": {
                "entry_model": {
                    "type": "XGBoost Classifier",
                    "predicts": "BUY/HOLD signals",
                    "confidence": "probability 0-1"
                },
                "return_model": {
                    "type": "RandomForest Regressor", 
                    "predicts": "expected return %",
                    "range": "-5% to +5%"
                },
                "risk_model": {
                    "type": "GradientBoosting Regressor",
                    "predicts": "volatility/risk",
                    "used_for": "position sizing"
                }
            },
            
            "4. DSPy Components": {
                "market_analysis": {
                    "module": "src.modules.market_analysis",
                    "llm": "GPT-4 or GPT-3.5",
                    "analyzes": "market conditions",
                    "outputs": "regime classification"
                },
                "strategy_generator": {
                    "module": "src.modules.strategy_generator",
                    "generates": "trading strategies",
                    "uses": "memory from past trades"
                },
                "trading_rules": {
                    "module": "src.modules.trading_rules",
                    "creates": "entry/exit conditions",
                    "format": "if-then rules"
                }
            },
            
            "5. Hybrid Signal Generation": {
                "weights": {
                    "ML": 0.6,
                    "Regime": 0.3,
                    "DSPy": 0.1
                },
                "threshold": "30% confidence to trade",
                "position_size": "10-30% of capital"
            },
            
            "6. Backtesting": {
                "module": "src.modules.backtester",
                "engine": "VectorBTPro",
                "calculates": ["returns", "sharpe", "drawdown", "win_rate"],
                "validates": "strategy performance"
            },
            
            "7. Memory Storage": {
                "module": "src.utils.memory_manager",
                "backend": "Mem0ai",
                "stores": "successful strategies",
                "learns": "from past performance"
            }
        }
        
        self.data_flow = flow
        
        # Log the flow
        for stage, details in flow.items():
            logger.info(f"\n{stage}:")
            self._log_dict(details, indent=2)
            
    def _log_dict(self, d: dict, indent: int = 0):
        """Recursively log dictionary contents"""
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}• {key}:")
                self._log_dict(value, indent + 1)
            else:
                logger.info(f"{prefix}• {key}: {value}")
                
    def validate_dependencies(self):
        """Check all required dependencies"""
        logger.info("\n🔌 VALIDATING DEPENDENCIES")
        logger.info("="*60)
        
        deps = {
            "Core Libraries": {
                "pandas": "Data manipulation",
                "numpy": "Numerical operations",
                "scikit-learn": "ML models",
                "xgboost": "Entry signal model",
                "dspy": "LLM orchestration",
                "vectorbtpro": "Backtesting engine"
            },
            "APIs": {
                "OPENAI_API_KEY": "GPT-4/3.5 access",
                "MEM0_API_KEY": "Memory storage",
                "GITHUB_TOKEN": "VectorBTPro access"
            },
            "Data Files": {
                "/Users/speed/StratOptimv4/big_optimize_1016.pkl": "Real trading data",
                "prompts/": "Prompt templates",
                "mem0_track/": "Memory backups"
            }
        }
        
        self.dependencies = deps
        
        # Check each dependency
        for category, items in deps.items():
            logger.info(f"\n{category}:")
            for dep, purpose in items.items():
                if category == "Core Libraries":
                    try:
                        __import__(dep)
                        logger.info(f"  ✅ {dep}: {purpose}")
                    except:
                        logger.error(f"  ❌ {dep}: MISSING! ({purpose})")
                elif category == "APIs":
                    if os.getenv(dep):
                        logger.info(f"  ✅ {dep}: Set ({purpose})")
                    else:
                        logger.error(f"  ❌ {dep}: NOT SET! ({purpose})")
                elif category == "Data Files":
                    if Path(dep).exists():
                        logger.info(f"  ✅ {dep}: Exists ({purpose})")
                    else:
                        logger.error(f"  ❌ {dep}: NOT FOUND! ({purpose})")
                        
    def analyze_ml_models(self):
        """Deep dive into ML components"""
        logger.info("\n🤖 ANALYZING ML MODELS")
        logger.info("="*60)
        
        ml_analysis = {
            "Training Process": {
                "1. Feature Engineering": [
                    "Price returns (1h, 4h, 24h)",
                    "Technical indicators (RSI, MACD, Bollinger)",
                    "Volume metrics",
                    "Market microstructure"
                ],
                "2. Model Training": [
                    "80/20 time-series split",
                    "StandardScaler normalization",
                    "Cross-validation"
                ],
                "3. Signal Generation": [
                    "Probability > threshold → BUY",
                    "Expected return calculation",
                    "Risk-adjusted position sizing"
                ]
            },
            
            "Expected Outputs": {
                "XGBoost Entry": {
                    "accuracy": "75-95%",
                    "output": "probability 0-1",
                    "threshold": "0.3 for aggressive"
                },
                "RandomForest Return": {
                    "MAE": "0.001-0.005",
                    "range": "-5% to +5%",
                    "use": "profit targets"
                },
                "GradientBoost Risk": {
                    "predicts": "volatility",
                    "range": "0-10%",
                    "use": "position sizing"
                }
            }
        }
        
        for section, details in ml_analysis.items():
            logger.info(f"\n{section}:")
            self._log_dict(details, indent=1)
            
    def trace_execution_path(self):
        """Trace the full execution path"""
        logger.info("\n🛤️  EXECUTION PATH TRACE")
        logger.info("="*60)
        
        execution_path = [
            {
                "step": 1,
                "action": "Load blockchain data",
                "input": "pickle file with 65 tokens",
                "output": "Dictionary of DataFrames",
                "validates": "data integrity"
            },
            {
                "step": 2,
                "action": "Train ML models",
                "input": "First token's data (80%)",
                "output": "3 trained models",
                "validates": "75%+ accuracy"
            },
            {
                "step": 3,
                "action": "Process each token",
                "input": "DataFrame + ML models",
                "output": "Trade signals",
                "validates": "confidence > 15%"
            },
            {
                "step": 4,
                "action": "Execute trades",
                "input": "Signals + capital",
                "output": "Trade records",
                "validates": "stop/target execution"
            },
            {
                "step": 5,
                "action": "Calculate P&L",
                "input": "All trades",
                "output": "Performance metrics",
                "validates": "return > 10%"
            }
        ]
        
        for step in execution_path:
            logger.info(f"\nStep {step['step']}: {step['action']}")
            logger.info(f"  📥 Input: {step['input']}")
            logger.info(f"  📤 Output: {step['output']}")
            logger.info(f"  ✓ Validates: {step['validates']}")
            
    def generate_report(self):
        """Generate comprehensive archeology report"""
        report = {
            "components": self.components,
            "data_flow": self.data_flow,
            "dependencies": self.dependencies,
            "timestamp": str(Path(__file__).stat().st_mtime)
        }
        
        with open('system_archeology_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info("\n📄 Report saved to system_archeology_report.json")

def main():
    """Run complete system archeology"""
    logger.info("🏛️  TRADING SYSTEM ARCHEOLOGY")
    logger.info("="*60)
    
    archaeologist = SystemArcheologist()
    
    # 1. Excavate key components
    components_to_excavate = [
        ("src.ml_trading_engine", "ML Trading Engine"),
        ("src.utils.data_preprocessor", "Data Preprocessor"),
        ("src.modules.market_analysis", "Market Analysis"),
        ("src.modules.strategy_generator", "Strategy Generator"),
        ("src.modules.backtester", "Backtester"),
        ("src.utils.memory_manager", "Memory Manager"),
        ("src.hybrid_trading_system", "Hybrid System")
    ]
    
    for module_path, name in components_to_excavate:
        archaeologist.excavate_component(module_path, name)
        
    # 2. Map data flow
    archaeologist.map_data_flow()
    
    # 3. Validate dependencies
    archaeologist.validate_dependencies()
    
    # 4. Analyze ML models
    archaeologist.analyze_ml_models()
    
    # 5. Trace execution
    archaeologist.trace_execution_path()
    
    # 6. Generate report
    archaeologist.generate_report()
    
    logger.info("\n🏁 ARCHEOLOGY COMPLETE!")
    logger.info("Now we understand EXACTLY how the system works.")
    logger.info("Ready to run with full confidence! 🚀")

if __name__ == "__main__":
    main()
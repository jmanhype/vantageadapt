# System Design Documentation

## Overview

This document provides a comprehensive overview of the VantageAdapt system architecture and algorithmic flow. The system implements an advanced trading strategy generation and optimization pipeline using LLMs and self-improving code.

## Language & Specialization
- **Primary Language**: Python
- **Focus Area**: High-Level System Design
- **Key Feature**: End-to-end algorithmic flow across multiple system components

## Comprehensive Algorithm

### 1. System Initialization

1. **Load Environment & Database**  
   1.1. `init_db.py` or the Alembic scripts (`migrate`, `create`, etc.) are optionally run to set up or migrate the database schema.  
   1.2. Environment variables (e.g., DB credentials) are read.  

2. **Import Modules & Create Objects**  
   2.1. `main.py` imports:  
       - **Backtester** (for data loading, parameter optimization)  
       - **GodelAgent** (for code-patching improvements)  
       - **StrategicTrader** (for LLM-based strategy logic)  
       - **LLMInterface** (OpenAI function-calling logic)  
       - **DatabaseConnection** (for storing results)  
       - **PromptManager** (for reading and updating `.prompt` files)  
   2.2. `main.py` sets up logging (`logger`) and argument parsing to accept `--theme` and `--data` paths.  

### 2. Data Loading

1. **Argument Parsing**  
   - The user supplies a strategy `theme` (e.g., `"breakout trading"`) and a `data` path (pickled DataFrame dictionaries).  

2. **Load Trade Data**  
   - `backtester.load_trade_data(data_path)` is called.  
   - Returns a dictionary mapping `{asset_symbol -> pd.DataFrame}`, each containing historical price/volume columns like `dex_price`, `timestamp`, etc.

### 3. Core Iterative Loop (from `main.py`)

The main script runs multiple **iterations** to refine or "self-improve" the strategy:

1. **Initialize Key Instances**  
   1.1. Create a **`GodelAgent`** object (e.g., `improvement_threshold=0.1`, `max_iterations=5`).  
   1.2. Create a **`StrategicTrader`** object, which internally initializes a `PromptManager` and `LLMInterface`.  

2. **For each iteration `i` from 1 to `agent.max_iterations`:**  
   2.1. **Analyze Market Context**  
       - Use `StrategicTrader.analyze_market(market_data)` on one representative asset (or a merged set).  
       - This calls `LLMInterface.analyze_market`, sending a function-call prompt to OpenAI with relevant stats (volatility, price change, etc.).  
       - Receives a structured JSON containing `MarketRegime` (e.g., `RANGING_LOW_VOL`), confidence level, risk level, etc.  
       - Stored in a `MarketContext` object.

   2.2. **Generate Strategy Insights**  
       - Call `StrategicTrader.generate_strategy(theme)` with the current `MarketContext`.  
       - Internally, `LLMInterface.generate_strategy` sends prompts (function calls) to get a `StrategyInsight` object describing e.g. `regime_change_probability`, `suggested_position_size`, rules for each regime, etc.

   2.3. **Generate Trading Rules**  
       - Call `StrategicTrader.generate_trading_rules(strategy_insights, market_context)`.  
       - `LLMInterface.generate_trading_rules`:
         - Creates a JSON schema specifying conditions (entry/exit triggers) and parameters (stop_loss, take_profit, etc.).  
         - The LLM is told to return a JSON structure with `conditions` and `parameters`.

   2.4. **Parameter Optimization**  
       - Pass the generated conditions to the backtester's `run_parameter_optimization(trade_data, conditions)`.  
         - Sub-steps in `backtester.py`:  
           1. Define search ranges for take_profit, stop_loss, MACD windows, etc.  
           2. Run random subset optimization (using `vbt.Param`) on a small subset of assets, ranking results by custom "score."  
           3. Pick the best parameters, re-run on all assets.  
           4. Collect aggregated stats (sum of total_return, average sortino_ratio, etc.).  
           5. Return a dictionary with `metrics` (like total_return, sortino_ratio, total_trades) and chosen `parameters`.

   2.5. **Validate Performance**  
       - Compare the new `metrics` with thresholds (e.g., total_return >= 0.5, sortino_ratio >= 2.0, etc.).  
       - If the strategy fails, logs "Strategy did not meet performance requirements...".

   2.6. **Track Performance & Attempt Self-Improvement**  
       - `GodelAgent.track_performance(metrics, parameters, market_regime)` updates the agent's best or last metrics.  
       - If insufficient improvement is detected, the code tries `agent.propose_patch(...)` on modules (`strategy_generator.py`, `backtester.py`, `llm_interface.py`):  
         - The agent forms a "code improvement prompt," sends it to the LLM, and tries to call `_insert_code()`, `_modify_code()`, or `_delete_code()` to rewrite Python source.  
         - If successful patches are found, the script re-imports or reloads the modules via `importlib.reload(...)`.  
         - The iteration is repeated with the new code if improvements are made.

   2.7. **Prompt Modifications** (Optional)  
       - If indicated by the agent, it modifies `.prompt` files using `PromptManager.update_prompt(...)`.  
       - The new or refined instructions might be used in the next iteration.

   2.8. **Check for Next Iteration**  
       - If performance is still below thresholds and `i < max_iterations`, continue.  
       - Otherwise, exit the loop.

### 4. Final Results & Storage

1. **Pick the Best Result**  
   - `main.py` or the GodelAgent tracks the best iteration's metrics.  
   - Once the iteration loop completes, logs "Best metrics achieved: {...}" if any success was found.

2. **Save to Database**  
   - The system may call `db.save_strategy(...)` or a similar method from the `StrategyGenerator` to store:  
     - Final conditions, parameters, metrics, and relevant `MarketContext` & `StrategyInsight` fields.  
     - Insert or update in `strategies` and `performance` tables (via the `DatabaseConnection` class).

3. **Summary**  
   - The script logs a summary of any constraints still unmet (like minimum trades, min win rate).  
   - If they remain unmet but code patches are exhausted, it logs warnings and ends.

### 5. Auxiliary Components

1. **`PromptManager`**  
   - Loads `.prompt` files from category folders (`strategy`, `analysis`, etc.).  
   - Allows dynamic "{placeholder}" formatting with `format_prompt(...)`.  
   - Retains a simple version history, enabling partial or global changes when the agent modifies them.

2. **`models.py`**  
   - Defines the enumerations (`MarketRegime`) and data structures (`MarketContext`, `StrategyInsight`) to unify the LLM's structured JSON with Python objects.

3. **`Database` Scripts**  
   - `init_db.py` sets up the async database with `sqlalchemy.ext.asyncio`.  
   - `alembic` scripts (via Typer) run migrations or rollbacks.  
   - The code uses `asyncpg` or `sqlalchemy` pools for performing reads/writes to the `strategies` and `performance` tables.

### 6. Putting It All Together

**At a high level**:
1. **System Setup**: DB and environment loaded, modules imported.  
2. **Start**: `python main.py --theme <THEME> --data <DATA.pkl>`  
3. **Load** trade data, build `StrategicTrader`, `GodelAgent`.  
4. **Iterate** several times:
   - **Analyze** market via LLM  
   - **Generate** initial insights & rules  
   - **Backtest** & optimize parameters  
   - **Validate** if thresholds are met  
   - If not, attempt code or prompt improvements  
5. **Finish**: Log final best results, store in DB if desired.

When all is done, the system has either:
- Found an acceptable strategy (meeting performance thresholds).  
- Or exhausted its iteration/patch attempts, leaving final metrics below the threshold with a "not met" warning.

## Next Steps

With this architectural overview, you can:
1. Configure performance thresholds
2. Modify the optimization parameters
3. Adjust the LLM prompts
4. Extend the database schema
5. Add new market regimes or strategy types 

## Advanced Performance Metrics and Trade Tracking

1. **Optimized Trade Processing**
   - Numba-accelerated trade tracking using `@njit` decorators
   - Custom `TradeMemory` structure for efficient record keeping
   - Vectorized backtesting via `vectorbtpro` with custom settings
   - Real-time trade metrics logging and analysis

2. **Position Size Management**
   - Dynamic position sizing based on volatility thresholds
   - Risk-adjusted position calculations
   - Default position size modulation using market conditions

3. **Performance Analysis Pipeline**
   - Comprehensive trade statistics calculation
   - Integration with VBT's advanced metrics
   - Custom performance validation thresholds
   - Real-time metric tracking and adjustment

## Default Trading Configuration

1. **Base Trading Rules**
   ```python
   default_rules = {
       "conditions": {
           "entry": ["price > ma_20", "volume > volume_ma"],
           "exit": ["price < ma_20", "drawdown > max_drawdown"]
       },
       "parameters": {
           "take_profit": 0.1,
           "stop_loss": 0.05,
           "order_size": 0.001,
           "max_orders": 3
       }
   }
   ```

2. **Parameter Optimization Ranges**
   - Take profit: 0.05 to 0.3
   - Stop loss: 0.02 to 0.15
   - Order size: 0.0005 to 0.005
   - Maximum concurrent orders: 1 to 5

## Database Integration

1. **Model Structure**
   - Strategy model for storing generated strategies
   - Backtest model for performance results
   - Async SQLAlchemy integration
   - Performance metrics storage and retrieval

2. **Async Operations**
   - Concurrent database operations
   - Efficient batch updates
   - Transaction management
   - Performance metric aggregation

3. **Data Persistence**
   - Strategy configuration storage
   - Performance metrics archival
   - Market context preservation
   - Historical analysis capabilities 
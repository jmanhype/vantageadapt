"""FastAPI endpoints for controlling the trading system."""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import psutil
from research.database.connection import DatabaseConnection
from typing import List, Optional, Dict, Any
import json
from pydantic import BaseModel
import asyncio
import random
from backtester import load_trade_data, from_signals_backtest, calculate_stats
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176", "http://localhost:5177"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ExecuteStrategyRequest(BaseModel):
    strategy_id: int
    data_path: str
    live: bool = False

class StopStrategyRequest(BaseModel):
    strategy_id: int

db = DatabaseConnection()
running_tasks: Dict[int, asyncio.Task] = {}

async def run_backtest(strategy_id: int, data_path: str):
    """Run backtest for a strategy."""
    try:
        # Load trade data
        trade_data_dict = load_trade_data(data_path)
        if not trade_data_dict:
            logger.error("Failed to load trade data")
            await db.update_strategy_status(strategy_id, "error")
            return

        # Use default parameters for now
        params = {
            "take_profit": 0.08,
            "stop_loss": 0.12,
            "sl_window": 400,
            "max_orders": 3,
            "order_size": 0.0025,
            "post_buy_delay": 2,
            "post_sell_delay": 5,
            "macd_signal_fast": 120,
            "macd_signal_slow": 260,
            "macd_signal_signal": 90,
            "min_macd_signal_threshold": 0,
            "max_macd_signal_threshold": 0,
            "enable_sl_mod": False,
            "enable_tp_mod": False,
        }

        # Run backtest on all assets
        test_portfolio = {}
        for asset, trade_data in trade_data_dict.items():
            trade_data = trade_data.copy()
            
            # Trim length to last 2 weeks for faster testing
            two_weeks_ago = trade_data['timestamp'].max() - pd.Timedelta(weeks=2)
            trade_data = trade_data[trade_data['timestamp'] >= two_weeks_ago]

            pf = from_signals_backtest(trade_data, **params)
            if pf is not None:
                test_portfolio[asset] = pf

        # Calculate stats
        all_stats_df = calculate_stats(test_portfolio, trade_data_dict)
        if all_stats_df.empty:
            logger.error("No valid statistics calculated")
            await db.update_strategy_status(strategy_id, "error")
            return
        
        # Update performance in database
        metrics = {
            'total_return': float(all_stats_df['total_return'].sum()),
            'total_pnl': float(all_stats_df['total_pnl'].sum()),
            'avg_pnl_per_trade': float(all_stats_df['avg_pnl_per_trade'].mean()),
            'total_trades': int(all_stats_df['total_trades'].sum()),
            'win_rate': float((all_stats_df['total_return'] > 0).mean()),
            'sortino_ratio': float(all_stats_df['sortino_ratio'].mean()),
        }
        
        async with db.pool.acquire() as conn:
            # Check if performance record exists
            existing = await conn.fetchrow(
                "SELECT id FROM performance WHERE strategy_id = $1",
                strategy_id
            )
            
            if existing:
                # Update existing record
                await conn.execute("""
                    UPDATE performance 
                    SET total_return = $2, sharpe_ratio = $3
                    WHERE strategy_id = $1
                """, strategy_id, metrics['total_return'], metrics['sortino_ratio'])
            else:
                # Insert new record
                await conn.execute("""
                    INSERT INTO performance (strategy_id, total_return, sharpe_ratio)
                    VALUES ($1, $2, $3)
                """, strategy_id, metrics['total_return'], metrics['sortino_ratio'])
            
        # Update strategy status
        await db.update_strategy_status(strategy_id, "inactive")
        
    except Exception as e:
        logger.error(f"Error in backtest for strategy {strategy_id}: {str(e)}")
        logger.exception("Full traceback:")
        await db.update_strategy_status(strategy_id, "error")
        raise

async def run_live_trading(strategy_id: int, data_path: str):
    """Run live trading for a strategy."""
    try:
        while True:
            # Simulate live trading
            await asyncio.sleep(10)  # Update every 10 seconds
            
            # Generate random performance metrics
            total_return = random.uniform(-0.1, 0.2)
            sharpe_ratio = random.uniform(0.8, 1.8)
            
            # Update performance in database
            async with db.pool.acquire() as conn:
                # Check if performance record exists
                existing = await conn.fetchrow(
                    "SELECT id FROM performance WHERE strategy_id = $1",
                    strategy_id
                )
                
                if existing:
                    await conn.execute("""
                        UPDATE performance 
                        SET total_return = $2, sharpe_ratio = $3
                        WHERE strategy_id = $1
                    """, strategy_id, total_return, sharpe_ratio)
                else:
                    await conn.execute("""
                        INSERT INTO performance (strategy_id, total_return, sharpe_ratio)
                        VALUES ($1, $2, $3)
                    """, strategy_id, total_return, sharpe_ratio)
                
    except asyncio.CancelledError:
        await db.update_strategy_status(strategy_id, "inactive")
    except Exception as e:
        logger.error(f"Error in live trading for strategy {strategy_id}: {str(e)}")
        await db.update_strategy_status(strategy_id, "inactive")
        raise

@app.on_event("startup")
async def startup_event():
    await db.init()

@app.on_event("shutdown")
async def shutdown_event():
    # Cancel all running tasks
    for task in running_tasks.values():
        task.cancel()
    await db.close()

@app.get("/api/system/metrics")
async def get_system_metrics():
    try:
        active_strategies = await db.get_active_strategies()
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "active_strategies": len(active_strategies)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategy/list")
async def list_strategies():
    try:
        strategies = await db.get_all_strategies()
        return strategies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategy/generate")
async def generate_strategy(data: Dict[str, str]):
    try:
        theme = data.get("theme")
        if not theme:
            raise HTTPException(status_code=400, detail="Theme is required")
        strategy_id = await db.create_strategy(theme)
        return {"id": strategy_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategy/execute")
async def execute_strategy(request: ExecuteStrategyRequest):
    try:
        # Cancel any existing task for this strategy
        if request.strategy_id in running_tasks:
            running_tasks[request.strategy_id].cancel()
            
        # Update status to active
        await db.update_strategy_status(request.strategy_id, "active")
        
        # Start new task based on mode
        if request.live:
            task = asyncio.create_task(run_live_trading(request.strategy_id, request.data_path))
        else:
            task = asyncio.create_task(run_backtest(request.strategy_id, request.data_path))
            
        running_tasks[request.strategy_id] = task
        return {"status": "success", "message": f"Strategy {request.strategy_id} is now {request.live and 'live trading' or 'backtesting'}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategy/stop")
async def stop_strategy(request: StopStrategyRequest):
    try:
        # Cancel the running task if it exists
        if request.strategy_id in running_tasks:
            running_tasks[request.strategy_id].cancel()
            del running_tasks[request.strategy_id]
            
        await db.update_strategy_status(request.strategy_id, "inactive")
        return {"status": "success", "message": f"Strategy {request.strategy_id} is now inactive"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
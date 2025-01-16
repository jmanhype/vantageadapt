"""Database models for trading strategies and backtests."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import ForeignKey, String, Float, Boolean, JSON, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .base import Base

class Strategy(Base):
    """Trading strategy model."""
    
    # Override tablename
    __tablename__ = 'strategies'
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Required fields
    theme: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    conditions: Mapped[Dict[str, List[str]]] = mapped_column(JSON, nullable=False)
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    market_context: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    strategy_insights: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now(), nullable=False)
    
    # Relationships
    backtests: Mapped[List["Backtest"]] = relationship(
        "Backtest", 
        back_populates="strategy",
        cascade="all, delete-orphan"
    )

class Backtest(Base):
    """Backtest results model."""
    
    # Override tablename
    __tablename__ = 'backtests'
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Required fields
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"), nullable=False)
    metrics: Mapped[Dict[str, float]] = mapped_column(JSON, nullable=False)
    performance_analysis: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    strategy: Mapped["Strategy"] = relationship("Strategy", back_populates="backtests")
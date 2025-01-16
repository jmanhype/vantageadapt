from sqlalchemy import Column, Integer, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from ..models.base import Base

class Backtest(Base):
    """Model representing a backtest run of a strategy."""
    
    __tablename__ = 'backtests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    metrics = Column(JSON, nullable=False)
    performance_analysis = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False) 
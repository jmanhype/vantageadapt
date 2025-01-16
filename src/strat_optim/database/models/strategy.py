from sqlalchemy import Column, Integer, String, JSON
from sqlalchemy.sql import func
from sqlalchemy.types import TIMESTAMP

from ..models.base import Base

class Strategy(Base):
    """Model representing a trading strategy."""
    
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    theme = Column(String, nullable=False)
    description = Column(String, nullable=False)
    conditions = Column(JSON, nullable=False)
    parameters = Column(JSON, nullable=False)
    market_context = Column(JSON, nullable=False)
    strategy_insights = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False) 
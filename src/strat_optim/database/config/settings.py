"""Database configuration settings."""

import os

# Database connection configuration
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "postgres",
    "database": "trading_db"
}

# Pool configuration
POOL_CONFIG = {
    "min_size": 1,
    "max_size": 10,
    "max_queries": 50000,
    "max_inactive_connection_lifetime": 300.0
}

# Query configuration
QUERY_CONFIG = {
    "timeout": 60.0,
    "statement_timeout": 60000,
    "command_timeout": 60.0
}

# Retry configuration
RETRY_CONFIG = {
    "max_attempts": 3,
    "initial_delay": 1.0,
    "max_delay": 5.0
}

# Schema configuration
SCHEMA_CONFIG = {
    "version": "1.0",
    "auto_migrate": True
}

# SQLAlchemy configuration (for legacy support)
SQLALCHEMY_CONFIG = {
    "echo": True,
    "pool_size": 5,
    "max_overflow": 10,
    "pool_timeout": 30,
    "pool_recycle": 1800
}

# Database URL for SQLAlchemy
DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
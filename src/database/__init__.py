"""
Database management module for MLB data.

This module provides DuckDB database connections and table management
for storing transformed MLB API data.
"""

from src.database.duckdb_handler import DuckDBHandler

__all__ = ["DuckDBHandler"]

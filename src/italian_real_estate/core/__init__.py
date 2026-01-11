"""
Core utilities module for the Italian Real Estate pipeline.

This module provides shared utilities used across all pipeline stages,
including database connections, string manipulation, date conversion,
and numeric calculations.

Submodules:
    connections: Database connection management for MongoDB, PostgreSQL, and SQLite.
    string_utils: String manipulation and text preprocessing utilities.
    date_utils: Date parsing and conversion utilities.
    numeric_utils: Mortgage calculations and numeric rounding utilities.
    dict_utils: Dictionary key conversion and JSON encoding utilities.
"""

from .connections import (
    get_mongodb_client,
    get_datalake_collections,
    get_warehouse_collections,
    get_postgres_connection_string,
)
from .string_utils import remove_non_numbers, preprocess_text
from .date_utils import convert_mongo_date
from .numeric_utils import mortgage_monthly_payment, calculate_monthly_payment, random_round
from .dict_utils import convert_dict_keys_to_tuple, convert_dict_keys_to_string, DateEncoder

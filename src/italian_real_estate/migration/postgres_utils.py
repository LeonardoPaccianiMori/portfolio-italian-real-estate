"""
PostgreSQL utility functions for the migration pipeline.

This module provides utility functions for interacting with PostgreSQL
databases, including silent query execution, primary key detection,
and data processing helpers.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import re
from typing import Any, List, Optional, Tuple

from ..config.logging_config import get_logger

# Module-level logger instance for consistent logging throughout.
logger = get_logger(__name__)


def execute_query_silent(
    postgres_hook: Any,
    query: str,
    params: Optional[Tuple] = None
) -> Optional[List[Tuple]]:
    """
    Execute an SQL query without logging the full query text.

    This function is designed to reduce verbose logging in Airflow task logs
    while still executing queries and returning results. It handles both
    SELECT queries (returning results) and INSERT/UPDATE queries (returning None).

    Args:
        postgres_hook: PostgreSQL connection hook (typically from Airflow's
            PostgresHook class). Must have a get_conn() method that returns
            a database connection object.
        query: The SQL query string to execute. Can include %s placeholders
            for parameterized queries.
        params: Optional tuple of parameters to substitute into the query.
            These are safely escaped by the database driver to prevent
            SQL injection attacks.

    Returns:
        A list of tuples containing the query results if the query returns
        data (i.e., has cursor.description set after execution), or None
        if the query doesn't return data (like INSERT/UPDATE statements).

    Raises:
        Exception: Propagates any database exceptions that occur during
            query execution.

    Example:
        >>> hook = PostgresHook(postgres_conn_id='postgres_default')
        >>> result = execute_query_silent(
        ...     hook,
        ...     "SELECT * FROM dim_date WHERE date_value = %s",
        ...     (datetime.date(2024, 1, 1),)
        ... )
        >>> if result:
        ...     print(f"Found {len(result)} rows")
    """
    # Opens a connection to the PostgreSQL database.
    conn = postgres_hook.get_conn()

    # Creates a cursor object for executing queries.
    cursor = conn.cursor()

    try:
        # Executes the query with the provided parameters (or None if not given).
        cursor.execute(query, params)

        # Commits the transaction to persist any changes.
        conn.commit()

        # Checks if the query returned any data by inspecting cursor.description.
        # This attribute is set for SELECT queries and None for INSERT/UPDATE.
        if cursor.description:
            # Fetches and returns all rows from the result set.
            return cursor.fetchall()

        # Returns None for queries that don't produce results.
        return None

    finally:
        # Always closes the cursor to free up database resources.
        cursor.close()


def check_all_nulls(values: List[Any]) -> bool:
    """
    Check if all values in a list are None.

    This helper function is used throughout the dimension processing code
    to detect when a record has all NULL values, which requires special
    handling to avoid creating duplicate NULL records in dimension tables.

    Args:
        values: A list of values to check. Can contain any types including
            None, strings, numbers, booleans, etc.

    Returns:
        True if all values in the list are None, False otherwise.
        Returns True for an empty list as well.

    Example:
        >>> check_all_nulls([None, None, None])
        True
        >>> check_all_nulls([None, "value", None])
        False
        >>> check_all_nulls([])
        True
    """
    # Uses the all() built-in with a generator expression for efficiency.
    # Returns True only if every value v in the list is None.
    return all(v is None for v in values)


def get_existing_null_record_id(
    postgres_hook: Any,
    table_name: str,
    id_column: str,
    condition_columns: Optional[List[str]] = None
) -> Optional[int]:
    """
    Get the ID of an existing record with all NULL values.

    This function searches for a record in a dimension table where all
    specified columns are NULL. This is needed to avoid creating duplicate
    "all NULL" records, which can happen when multiple source records
    have missing data for an entire dimension.

    Args:
        postgres_hook: PostgreSQL connection hook for executing queries.
        table_name: Name of the dimension table to search.
        id_column: Name of the primary key column in the table.
        condition_columns: List of column names to check for NULL values.
            If None, the function will query the information_schema to get
            all columns except the ID column.

    Returns:
        The primary key ID of an existing all-NULL record if one exists,
        or None if no such record is found.

    Example:
        >>> hook = PostgresHook(postgres_conn_id='postgres_default')
        >>> null_id = get_existing_null_record_id(
        ...     hook,
        ...     'dim_seller_type',
        ...     'seller_type_id',
        ...     ['seller_type']
        ... )
        >>> if null_id:
        ...     print(f"Found existing NULL record with ID {null_id}")
    """
    # If no columns specified, query the database schema to get all columns.
    if condition_columns is None:
        # Builds a query to get column names from the information_schema.
        # Excludes the ID column since we want to check other columns for NULL.
        query = f'''
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        AND column_name != '{id_column}'
        '''

        # Executes the schema query to get the list of columns.
        column_results = execute_query_silent(postgres_hook, query)

        # Extracts just the column names from the query results.
        condition_columns = [row[0] for row in column_results]

    # Constructs a WHERE clause that checks if all columns are NULL.
    # Each column gets its own "IS NULL" condition, joined with AND.
    where_clause = " AND ".join([f"{col} IS NULL" for col in condition_columns])

    # Builds the final query to find an existing NULL record.
    query = f'''
    SELECT {id_column}
    FROM {table_name}
    WHERE {where_clause}
    LIMIT 1
    '''

    # Executes the query and extracts the ID if found.
    result = execute_query_silent(postgres_hook, query)

    # Returns the ID from the first row if results exist, otherwise None.
    return result[0][0] if result else None


def get_primary_key_column(
    postgres_hook: Any,
    table_name: str
) -> Optional[str]:
    """
    Get the primary key column name for a table.

    This function queries the PostgreSQL system catalogs to determine
    which column is the primary key for a given table. This is useful
    for dynamic operations where the primary key column name isn't
    known in advance.

    Args:
        postgres_hook: PostgreSQL connection hook for executing queries.
        table_name: Name of the table to inspect.

    Returns:
        The name of the primary key column as a string, or None if the
        table doesn't have a primary key or the table doesn't exist.

    Example:
        >>> hook = PostgresHook(postgres_conn_id='postgres_default')
        >>> pk_col = get_primary_key_column(hook, 'dim_date')
        >>> print(f"Primary key column: {pk_col}")
        Primary key column: date_id
    """
    # Queries the PostgreSQL system catalogs to find the primary key.
    # pg_index contains index metadata, including which indexes are primary keys.
    # pg_attribute contains column information, linked by relation OID and column number.
    query = '''
    SELECT a.attname
    FROM   pg_index i
    JOIN   pg_attribute a ON a.attrelid = i.indrelid
                         AND a.attnum = ANY(i.indkey)
    WHERE  i.indrelid = %s::regclass
    AND    i.indisprimary
    '''

    # Executes the query with the table name as a parameter.
    params = (table_name,)
    result = execute_query_silent(postgres_hook, query, params)

    # Returns the column name from the first result row, or None if not found.
    if result:
        return result[0][0]
    return None


def process_total_room_number(value: Any) -> Optional[int]:
    """
    Process the total_room_number field to handle various formats.

    The total_room_number field in the MongoDB warehouse can contain
    integers, floats, or strings with special formatting like ranges
    ("2 - 5") or open-ended values ("5+"). This function normalizes
    all these formats to a single integer value.

    Args:
        value: The raw total_room_number value from MongoDB. Can be:
            - None: Returns None
            - int or float: Returns as int
            - str: Parses and converts to int, handling ranges and + notation

    Returns:
        An integer representing the room count, or None if the value
        cannot be parsed or is None.

    Processing rules:
        - Simple numbers are returned directly
        - Ranges like "2-5" return the upper bound (5)
        - Open-ended values like "5+" return the value plus one (6)
        - Ranges with + like "2-5+" return upper bound plus one (6)

    Example:
        >>> process_total_room_number(3)
        3
        >>> process_total_room_number("2 - 5")
        5
        >>> process_total_room_number("5+")
        6
        >>> process_total_room_number(None)
        None
    """
    # Handles the None case by returning None.
    if value is None:
        return None

    # If the value is already numeric, converts directly to int.
    if isinstance(value, (int, float)):
        return int(value)

    # Converts to string and removes all whitespace for easier parsing.
    value_str = str(value).replace(" ", "")

    try:
        # First, tries to parse as a simple integer.
        return int(value_str)

    except ValueError:
        # If simple parsing fails, handles more complex formats.

        # Checks for range format (e.g., "2-5" or "2-5+").
        if "-" in value_str:
            # Splits the string by hyphen to get the parts.
            parts = value_str.split("-")

            # Gets the last part of the range (the upper bound).
            last_part = parts[-1]

            # Checks if the upper bound has a + sign (open-ended).
            if "+" in last_part:
                # Removes the + and adds 1 to indicate "more than X".
                num = int(last_part.replace("+", ""))
                return num + 1
            else:
                # Returns the upper bound of the range.
                return int(last_part)

        # Handles standalone + notation (e.g., "5+").
        elif "+" in value_str:
            # Removes the + and adds 1 to the number.
            num = int(value_str.replace("+", ""))
            return num + 1

        # If no pattern matches, logs a warning and returns None.
        logger.warning(f"Could not parse total_room_number value: {value}")
        return None


def extract_numeric_from_string(text: str) -> Optional[float]:
    """
    Extract a numeric value from a string containing text and numbers.

    This utility function is used to parse fields like condominium expenses
    or heating costs where the value might be stored as text with currency
    symbols or units (e.g., "€ 150.00" or "100 €/month").

    Args:
        text: A string potentially containing a numeric value.

    Returns:
        The extracted numeric value as a float, or None if no number
        is found in the string.

    Example:
        >>> extract_numeric_from_string("€ 150.00")
        150.0
        >>> extract_numeric_from_string("No expenses")
        None
        >>> extract_numeric_from_string("1,234.56 euros")
        1234.56
    """
    # Checks if the input is actually a string.
    if not isinstance(text, str):
        return None

    # Uses regex to find a decimal number in the string.
    # Pattern matches: optional negative sign, digits, optional decimal part.
    numeric_match = re.search(r'(\d+(?:\.\d+)?)', text)

    # Returns the matched number as float, or None if no match.
    return float(numeric_match.group(1)) if numeric_match else None

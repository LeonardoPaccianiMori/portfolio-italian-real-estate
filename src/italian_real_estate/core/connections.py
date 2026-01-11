"""
Database connection utilities for the Italian Real Estate pipeline.

This module provides centralized database connection management for MongoDB,
PostgreSQL, and SQLite. It implements connection pooling, context managers,
and helper functions for consistent database access across the pipeline.

The module supports three database systems:
    - MongoDB: Used for the datalake (raw HTML) and warehouse (transformed features)
    - PostgreSQL: Used for the final dimensional data warehouse
    - SQLite: Used for caching (e.g., translation cache)

Usage:
    from italian_real_estate.core.connections import (
        get_mongodb_client,
        get_datalake_collections,
        get_postgres_connection_string
    )

    # MongoDB access
    client = get_mongodb_client()
    sale_coll, rent_coll, auction_coll = get_datalake_collections(client)

    # PostgreSQL connection string
    conn_string = get_postgres_connection_string()

Author: Leonardo Pacciani-Mori
License: MIT
"""

import sqlite3
from contextlib import contextmanager
from typing import Tuple, Optional, Any

import pymongo
from pymongo import MongoClient

from italian_real_estate.config.settings import (
    MONGODB_HOST,
    MONGODB_PORT,
    MONGODB_USER,
    MONGODB_PASSWORD,
    MONGODB_AUTH_SOURCE,
    MONGODB_DATALAKE_NAME,
    MONGODB_WAREHOUSE_NAME,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DATABASE,
)


# =============================================================================
# MONGODB CONNECTION UTILITIES
# =============================================================================

def _get_mongo_auth_kwargs(
    username: Optional[str],
    password: Optional[str],
    auth_source: Optional[str],
) -> dict:
    if username and password:
        return {
            "username": username,
            "password": password,
            "authSource": auth_source or "admin",
        }
    return {}


def get_mongodb_client(
    host: str = MONGODB_HOST,
    port: int = MONGODB_PORT,
    username: Optional[str] = MONGODB_USER,
    password: Optional[str] = MONGODB_PASSWORD,
    auth_source: Optional[str] = MONGODB_AUTH_SOURCE,
    timeout_ms: Optional[int] = None,
) -> MongoClient:
    """
    Create and return a MongoDB client connection.

    This function establishes a connection to the MongoDB server using the
    provided host and port parameters. The returned client can be used to
    access both the datalake and warehouse databases.

    The connection uses default MongoDB settings for connection pooling.
    For long-running processes, consider using the context manager version
    to ensure proper cleanup.

    Args:
        host: The hostname or IP address of the MongoDB server.
            Defaults to the value in settings.MONGODB_HOST.
        port: The port number of the MongoDB server.
            Defaults to the value in settings.MONGODB_PORT.

    Returns:
        MongoClient: A connected MongoDB client instance that can be used
            to access databases and collections.

    Example:
        >>> client = get_mongodb_client()
        >>> db = client["listing_website_datalake"]
        >>> collection = db["sale"]
        >>> # Don't forget to close when done
        >>> client.close()
    """
    # Create the MongoDB client with the specified connection parameters.
    # PyMongo handles connection pooling automatically.
    kwargs = _get_mongo_auth_kwargs(username, password, auth_source)
    if timeout_ms is not None:
        kwargs["serverSelectionTimeoutMS"] = timeout_ms
    client = MongoClient(host, port, **kwargs)
    return client


@contextmanager
def mongodb_connection(
    host: str = MONGODB_HOST,
    port: int = MONGODB_PORT,
    username: Optional[str] = MONGODB_USER,
    password: Optional[str] = MONGODB_PASSWORD,
    auth_source: Optional[str] = MONGODB_AUTH_SOURCE,
    timeout_ms: Optional[int] = None,
):
    """
    Context manager for MongoDB connections with automatic cleanup.

    This context manager ensures that MongoDB connections are properly closed
    after use, even if an exception occurs. It is the recommended way to use
    MongoDB connections for short-lived operations.

    Args:
        host: The hostname or IP address of the MongoDB server.
        port: The port number of the MongoDB server.

    Yields:
        MongoClient: A connected MongoDB client instance.

    Example:
        >>> with mongodb_connection() as client:
        ...     db = client["listing_website_datalake"]
        ...     count = db["sale"].count_documents({})
        ...     print(f"Found {count} sale listings")
        # Connection is automatically closed here
    """
    # Create the client within the context manager
    kwargs = _get_mongo_auth_kwargs(username, password, auth_source)
    if timeout_ms is not None:
        kwargs["serverSelectionTimeoutMS"] = timeout_ms
    client = MongoClient(host, port, **kwargs)

    try:
        # Yield the client for use in the with block
        yield client
    finally:
        # Ensure the connection is closed even if an exception occurs.
        # This prevents connection leaks in long-running processes.
        client.close()


def get_datalake_collections(
    client: MongoClient
) -> Tuple[Any, Any, Any]:
    """
    Get the three main collections from the MongoDB datalake.

    The datalake stores raw HTML data scraped from listing.website. Each
    collection corresponds to a listing type: sale, rent, and auction.
    Documents contain the raw HTML source code along with metadata like
    province name and scraping date.

    Args:
        client: An active MongoDB client connection obtained from
            get_mongodb_client() or mongodb_connection().

    Returns:
        Tuple[Collection, Collection, Collection]: A tuple containing three
            MongoDB collection objects in order: (sale, rent, auction).

    Example:
        >>> client = get_mongodb_client()
        >>> sale_coll, rent_coll, auction_coll = get_datalake_collections(client)
        >>> print(f"Sale listings: {sale_coll.count_documents({})}")
    """
    # Access the datalake database
    datalake = client[MONGODB_DATALAKE_NAME]

    # Return the three collection objects.
    # Each collection stores documents with structure:
    # {
    #     "_id": listing_id,
    #     "province_name": "...",
    #     "province_code": "...",
    #     "parent_listing": null or listing_id,
    #     "child_listings": null or [listing_ids],
    #     "data": [{"scraping_date": "...", "html_source": "..."}]
    # }
    sale_collection = datalake.sale
    rent_collection = datalake.rent
    auction_collection = datalake.auction

    return sale_collection, rent_collection, auction_collection


def get_warehouse_collections(
    client: MongoClient
) -> Tuple[Any, Any, Any]:
    """
    Get the three main collections from the MongoDB warehouse.

    The warehouse stores transformed and structured data extracted from
    the raw HTML in the datalake. Each collection corresponds to a listing
    type: sale, rent, and auction. Documents contain extracted features
    like price, surface area, location, and property characteristics.

    Args:
        client: An active MongoDB client connection obtained from
            get_mongodb_client() or mongodb_connection().

    Returns:
        Tuple[Collection, Collection, Collection]: A tuple containing three
            MongoDB collection objects in order: (sale, rent, auction).

    Example:
        >>> client = get_mongodb_client()
        >>> sale_coll, rent_coll, auction_coll = get_warehouse_collections(client)
        >>> # Query transformed data
        >>> listing = sale_coll.find_one({"_id": 12345})
        >>> print(f"Price: {listing['data'][0]['listing_features']['price']}")
    """
    # Access the warehouse database
    warehouse = client[MONGODB_WAREHOUSE_NAME]

    # Return the three collection objects.
    # Each collection stores documents with structure:
    # {
    #     "_id": listing_id,
    #     "province_name": "...",
    #     "province_code": "...",
    #     "parent_listing": null or listing_id,
    #     "child_listings": null or [listing_ids],
    #     "data": [{"scraping_date": "...", "listing_features": {...}}]
    # }
    sale_collection = warehouse.sale
    rent_collection = warehouse.rent
    auction_collection = warehouse.auction

    return sale_collection, rent_collection, auction_collection


def get_collection_for_listing_type(
    client: MongoClient,
    listing_type: str,
    database: str = "datalake"
) -> Any:
    """
    Get a specific collection based on listing type and database.

    This utility function provides a convenient way to get a single collection
    when you know which listing type you need. It validates the listing type
    and database parameters to catch errors early.

    Args:
        client: An active MongoDB client connection.
        listing_type: The type of listings to access. Must be one of:
            "sale", "rent", or "auction".
        database: Which database to access. Either "datalake" for raw HTML
            data or "warehouse" for transformed features. Defaults to "datalake".

    Returns:
        Collection: The MongoDB collection for the specified listing type.

    Raises:
        ValueError: If listing_type is not one of the valid types, or if
            database is not "datalake" or "warehouse".

    Example:
        >>> client = get_mongodb_client()
        >>> rent_coll = get_collection_for_listing_type(client, "rent", "warehouse")
        >>> count = rent_coll.count_documents({})
    """
    # Validate the listing_type argument
    valid_listing_types = ["sale", "rent", "auction"]
    if listing_type not in valid_listing_types:
        raise ValueError(
            f"listing_type must be one of {valid_listing_types}, got '{listing_type}'"
        )

    # Validate the database argument
    if database not in ["datalake", "warehouse"]:
        raise ValueError(
            f"database must be 'datalake' or 'warehouse', got '{database}'"
        )

    # Get the appropriate collections tuple based on database type
    if database == "datalake":
        sale_coll, rent_coll, auction_coll = get_datalake_collections(client)
    else:
        sale_coll, rent_coll, auction_coll = get_warehouse_collections(client)

    # Return the collection for the specified listing type
    collection_map = {
        "sale": sale_coll,
        "rent": rent_coll,
        "auction": auction_coll
    }
    return collection_map[listing_type]


# =============================================================================
# POSTGRESQL CONNECTION UTILITIES
# =============================================================================

def get_postgres_connection_string(
    host: str = POSTGRES_HOST,
    port: str = POSTGRES_PORT,
    user: str = POSTGRES_USER,
    password: str = POSTGRES_PASSWORD,
    database: str = POSTGRES_DATABASE
) -> str:
    """
    Build a PostgreSQL connection string for SQLAlchemy.

    This function constructs a properly formatted connection string that can
    be used with SQLAlchemy's create_engine() function. The string uses the
    psycopg2 driver for PostgreSQL connectivity.

    Args:
        host: The hostname or IP address of the PostgreSQL server.
        port: The port number of the PostgreSQL server (as string).
        user: The username for authentication.
        password: The password for authentication.
        database: The name of the database to connect to.

    Returns:
        str: A SQLAlchemy-compatible connection string in the format:
            postgresql+psycopg2://user:password@host:port/database

    Example:
        >>> from sqlalchemy import create_engine
        >>> conn_string = get_postgres_connection_string()
        >>> engine = create_engine(conn_string)
        >>> with engine.connect() as conn:
        ...     result = conn.execute("SELECT 1")
    """
    # Build the connection string using SQLAlchemy's URL format.
    # psycopg2 is the standard PostgreSQL adapter for Python.
    connection_string = (
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    )
    return connection_string


def get_postgres_db_params() -> dict:
    """
    Get PostgreSQL connection parameters as a dictionary.

    This function returns the connection parameters in a format suitable
    for libraries that accept individual parameters rather than a connection
    string, such as psycopg2.connect() directly.

    Returns:
        dict: A dictionary containing the connection parameters with keys:
            - host: Server hostname
            - port: Server port (as string)
            - user: Username
            - password: Password
            - dbname: Database name

    Example:
        >>> import psycopg2
        >>> params = get_postgres_db_params()
        >>> conn = psycopg2.connect(**params)
    """
    return {
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "dbname": POSTGRES_DATABASE
    }


# =============================================================================
# SQLITE CONNECTION UTILITIES
# =============================================================================

@contextmanager
def sqlite_connection(db_path: str):
    """
    Context manager for SQLite database connections.

    This context manager provides automatic connection management for SQLite
    databases. It ensures that connections are properly closed and changes
    are committed even if an exception occurs.

    The translation cache uses SQLite to store translated texts, avoiding
    redundant calls to the LibreTranslate API.

    Args:
        db_path: The file path to the SQLite database. The database will
            be created if it doesn't exist.

    Yields:
        Tuple[sqlite3.Connection, sqlite3.Cursor]: A tuple containing the
            SQLite connection and cursor objects.

    Example:
        >>> with sqlite_connection("cache.db") as (conn, cursor):
        ...     cursor.execute("SELECT * FROM cache WHERE key = ?", (key,))
        ...     result = cursor.fetchone()
        # Connection is committed and closed automatically
    """
    # Create or open the SQLite database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Yield both connection and cursor for flexibility
        yield conn, cursor
        # Commit any pending changes on successful completion
        conn.commit()
    except Exception:
        # Rollback on error to maintain database consistency
        conn.rollback()
        raise
    finally:
        # Always close the cursor and connection
        cursor.close()
        conn.close()


def create_sqlite_database(db_path: str, schema_sql: str) -> None:
    """
    Create a SQLite database with the specified schema.

    This function creates a new SQLite database file (or opens an existing
    one) and executes the provided SQL schema to create tables. It is safe
    to call multiple times as it uses "CREATE TABLE IF NOT EXISTS".

    Args:
        db_path: The file path where the SQLite database should be created.
        schema_sql: SQL statements to create the database schema.
            Should use "IF NOT EXISTS" clauses for idempotency.

    Returns:
        None

    Example:
        >>> schema = '''
        ...     CREATE TABLE IF NOT EXISTS cache (
        ...         key TEXT PRIMARY KEY,
        ...         value TEXT,
        ...         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        ...     )
        ... '''
        >>> create_sqlite_database("cache.db", schema)
    """
    with sqlite_connection(db_path) as (conn, cursor):
        # Execute the schema SQL.
        # executescript handles multiple statements and commits automatically.
        cursor.executescript(schema_sql)

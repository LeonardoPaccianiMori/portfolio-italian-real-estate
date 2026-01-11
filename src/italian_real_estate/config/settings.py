"""
Configuration settings for the Italian Real Estate pipeline.

This module centralizes all configuration constants, database parameters,
and default values used throughout the pipeline. Settings are grouped by
their functional area for easy maintenance.

Configuration includes:
    - Database connection parameters (MongoDB, PostgreSQL)
    - Batch processing settings
    - HTTP client settings
    - Translation service settings
    - Airflow DAG default arguments

Note:
    Some values are hardcoded as per the original implementation.
    Modify these values according to your environment.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import os
from datetime import timedelta

try:
    from airflow.utils.dates import days_ago
except ImportError:
    try:
        from airflow.sdk import timezone

        def days_ago(n: int):
            return timezone.utcnow() - timedelta(days=n)
    except ImportError:
        from datetime import datetime, timezone as dt_timezone

        def days_ago(n: int):
            return datetime.now(dt_timezone.utc) - timedelta(days=n)

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
# All database settings support environment variables for Docker deployment.
# Fallback values are provided for local development.

# -----------------------------------------------------------------------------
# MongoDB Configuration
# -----------------------------------------------------------------------------
# Connection parameters for the MongoDB server hosting the datalake and warehouse.
# The datalake stores raw HTML data, while the warehouse stores transformed features.
# Use MONGODB_HOST=mongodb for Docker, or 127.0.0.1 for local development.
MONGODB_HOST = os.getenv("MONGODB_HOST", "127.0.0.1")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
MONGODB_USER = os.getenv("MONGODB_USER", "")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD", "")
MONGODB_AUTH_SOURCE = os.getenv("MONGODB_AUTH_SOURCE", "admin")

# Database names in MongoDB
MONGODB_DATALAKE_NAME = os.getenv("MONGODB_DATALAKE_NAME", "listing_website_datalake")
MONGODB_WAREHOUSE_NAME = os.getenv("MONGODB_WAREHOUSE_NAME", "listing_website_warehouse")

# Collection names (same for both datalake and warehouse)
COLLECTION_NAMES = ["sale", "rent", "auction"]

# -----------------------------------------------------------------------------
# PostgreSQL Configuration
# -----------------------------------------------------------------------------
# Connection parameters for the PostgreSQL data warehouse.
# Uses dimensional modeling (star schema) with fact and dimension tables.
# Use POSTGRES_HOST=postgres for Docker, or localhost for local development.
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "lpm")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "LeonardoPostgreSQL")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "listing_website_warehouse")

# Consolidated connection parameters dictionary for psycopg2.
# Used by modules that need to establish PostgreSQL connections.
POSTGRES_CONNECTION_PARAMS = {
    "host": POSTGRES_HOST,
    "port": POSTGRES_PORT,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "database": POSTGRES_DATABASE,
}

# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================

# Number of records to process in each batch during migration.
# Larger batches are faster but use more memory.
BATCH_SIZE = 10000

# Maximum number of records per collection to process.
# Set to None to process all records, or a specific number for testing.
MAX_RECORDS_PER_COLLECTION = None

# =============================================================================
# HTTP CLIENT CONFIGURATION
# =============================================================================

# Polite scraping mode (slower, sequential, retries on block pages).
SCRAPING_POLITE_MODE = os.getenv("SCRAPING_POLITE_MODE", "false").lower() in (
    "1",
    "true",
    "yes",
)
SCRAPING_POLITE_MIN_WAIT = float(os.getenv("SCRAPING_POLITE_MIN_WAIT", "1.0"))
SCRAPING_POLITE_MAX_WAIT = float(os.getenv("SCRAPING_POLITE_MAX_WAIT", "5.0"))
SCRAPING_POLITE_MAX_RETRIES = int(os.getenv("SCRAPING_POLITE_MAX_RETRIES", "5"))
SCRAPING_POLITE_BACKOFF_BASE = float(
    os.getenv("SCRAPING_POLITE_BACKOFF_BASE", "2.0")
)

# Maximum number of concurrent HTTP requests.
# Controls the load on the target server during scraping.
HTTP_SEMAPHORE_LIMIT = int(
    os.getenv("HTTP_SEMAPHORE_LIMIT", "1" if SCRAPING_POLITE_MODE else "50")
)

# Timeout in seconds for HTTP requests.
# Prevents indefinite hanging on slow or unresponsive servers.
HTTP_TIMEOUT_SECONDS = 60

# Scraping request headers (helps avoid basic bot blocks).
SCRAPING_USER_AGENT = os.getenv(
    "SCRAPING_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
)
SCRAPING_ACCEPT_LANGUAGE = os.getenv(
    "SCRAPING_ACCEPT_LANGUAGE",
    "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
)
SCRAPING_HEADERS = {
    "User-Agent": SCRAPING_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": SCRAPING_ACCEPT_LANGUAGE,
}
SCRAPING_USE_SELENIUM = os.getenv("SCRAPING_USE_SELENIUM", "false").lower() in (
    "1",
    "true",
    "yes",
)
SCRAPING_SELENIUM_HEADLESS = os.getenv("SCRAPING_SELENIUM_HEADLESS", "true").lower() in (
    "1",
    "true",
    "yes",
)
SCRAPING_SELENIUM_PAGE_DELAY = float(os.getenv("SCRAPING_SELENIUM_PAGE_DELAY", "1.0"))
SCRAPING_SELENIUM_PAGE_LOAD_TIMEOUT = int(
    os.getenv("SCRAPING_SELENIUM_PAGE_LOAD_TIMEOUT", "30")
)
SCRAPING_HUMANIZE = os.getenv("SCRAPING_HUMANIZE", "false").lower() in (
    "1",
    "true",
    "yes",
)
SCRAPING_HUMANIZE_MIN_WAIT = float(os.getenv("SCRAPING_HUMANIZE_MIN_WAIT", "0.5"))
SCRAPING_HUMANIZE_MAX_WAIT = float(os.getenv("SCRAPING_HUMANIZE_MAX_WAIT", "3.0"))
SCRAPING_HUMANIZE_SCROLL_MIN = int(os.getenv("SCRAPING_HUMANIZE_SCROLL_MIN", "1"))
SCRAPING_HUMANIZE_SCROLL_MAX = int(os.getenv("SCRAPING_HUMANIZE_SCROLL_MAX", "5"))
SCRAPING_HUMANIZE_SCROLL_WAIT_MIN = float(
    os.getenv("SCRAPING_HUMANIZE_SCROLL_WAIT_MIN", "0.5")
)
SCRAPING_HUMANIZE_SCROLL_WAIT_MAX = float(
    os.getenv("SCRAPING_HUMANIZE_SCROLL_WAIT_MAX", "1.5")
)

# Batch size for processing URLs in the HTTP client.
# Groups URLs into batches for more efficient async processing.
HTTP_BATCH_SIZE = 50

# Maximum number of connections per TCP connector.
# Limits concurrent connections to prevent resource exhaustion.
HTTP_CONNECTION_LIMIT = 25

# =============================================================================
# TRANSLATION SERVICE CONFIGURATION
# =============================================================================

# URL of the LibreTranslate API instance.
# Uses a locally-hosted LibreTranslate server for Italian-to-English translation.
# Use LIBRETRANSLATE_URL=http://libretranslate:5000/translate for Docker.
LIBRETRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "http://localhost:5000/translate")

# Number of texts to translate per API batch request.
# Balances API efficiency with memory usage.
TRANSLATION_BATCH_SIZE = 100

# Maximum retry attempts for failed translation requests.
# Provides resilience against transient API failures.
TRANSLATION_MAX_RETRIES = 10

# Delay in seconds between retry attempts.
TRANSLATION_RETRY_DELAY = 2

# =============================================================================
# PRICE RANGE CONFIGURATION
# =============================================================================

# Price ranges for scraping sale/auction listings (in euros).
# The scraper iterates through these ranges to handle the website's pagination limits.
SALE_PRICE_MIN = 50000
SALE_PRICE_MAX = 1000000
SALE_PRICE_STEP = 50000

# Price ranges for scraping rent listings (in euros per month).
RENT_PRICE_MIN = 200
RENT_PRICE_MAX = 5000
RENT_PRICE_STEP = 400

# =============================================================================
# AIRFLOW DAG CONFIGURATION
# =============================================================================

# Default arguments for all Airflow DAGs in this pipeline.
# These settings control task execution, retries, and scheduling.
DAG_DEFAULT_ARGS = {
    # Owner of the DAG for tracking purposes
    "owner": "Leonardo Pacciani-Mori",

    # Start date for the DAG (days_ago(0) = today)
    "start_date": days_ago(0),

    # Maximum number of tasks that can run concurrently within the DAG
    "max_active_tasks": 16,

    # Number of retry attempts for failed tasks
    "retries": 5,

    # Delay between retry attempts
    "retry_delay": timedelta(minutes=1),
}

# DAG IDs for each pipeline stage
DAG_ID_SCRAPING = "listing.website_datalake_population_DAG_webscraping"
DAG_ID_ETL = "listing.website_datalake_ETL_warehouse_MongoDB_DAG"
DAG_ID_MIGRATION = "listing.website_MongoDB_to_PostgreSQL_migration"

# =============================================================================
# FILE PATHS
# =============================================================================

# Path to the CSV file containing Italian province data.
# Bundled with the package for portability (no user-specific paths).
from importlib import resources as _resources

# Use importlib.resources to avoid hardcoded absolute paths.
PROVINCES_CSV_PATH = _resources.files("italian_real_estate").joinpath("data/provinces.csv")

# =============================================================================
# LISTING TYPE CONSTANTS
# =============================================================================

# Valid listing types used throughout the pipeline.
# Must be one of these values when specifying a listing type.
VALID_LISTING_TYPES = ["sale", "rent", "auction"]

# URL path segments for each listing type on listing.website
LISTING_TYPE_URL_SEGMENTS = {
    "sale": "vendita-case",
    "rent": "affitto-case",
    "auction": "aste-immobiliari",
}

# =============================================================================
# SYNTHETIC DATA GENERATION CONFIGURATION
# =============================================================================

# Default number of synthetic samples to generate for each listing type.
# These values maintain similar proportions to the original dataset.
DEFAULT_NUM_SYNTHETIC_RENT = 80000
DEFAULT_NUM_SYNTHETIC_AUCTION = 120000
DEFAULT_NUM_SYNTHETIC_SALE = 850000

# KNN parameters for synthetic data generation.
# k=5 neighbors provides a good balance of diversity and similarity.
KNN_NEIGHBORS = 5

# Chunk size for processing synthetic data generation.
# Smaller chunks prevent GPU memory exhaustion.
# Reduced from 10000 to 5000 to prevent OOM errors on GPUs with limited memory.
SYNTHETIC_DATA_CHUNK_SIZE = 5000

# Batch size for GPU processing within each chunk.
# Reduced from 50 to 25 to lower peak GPU memory usage per batch.
SYNTHETIC_DATA_BATCH_SIZE = 25

# =============================================================================
# MACHINE LEARNING CONFIGURATION
# =============================================================================

# Test set proportion for train/test split.
ML_TEST_SIZE = 0.3

# Random seed for reproducibility.
ML_RANDOM_STATE = 2025

# RandomForest hyperparameters.
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1

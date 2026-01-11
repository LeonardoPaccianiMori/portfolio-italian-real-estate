"""
SQLite-based translation cache for the migration pipeline.

This module provides functions for caching translations in a SQLite
database. The cache helps avoid re-translating the same text multiple
times, significantly speeding up the migration process.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import os
import sqlite3
from typing import Optional

from ...config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def initialize_translation_cache(base_path: Optional[str] = None) -> str:
    """
    Create and initialize a SQLite database for translation caching.

    This function creates a SQLite database file and the necessary table
    structure for storing translation mappings. If the database already
    exists, it will reuse it (table creation uses IF NOT EXISTS).

    Args:
        base_path: Optional base directory for the cache file. If not
            provided, uses the directory containing this script file.

    Returns:
        The full path to the initialized cache database file.

    Table structure:
        - source_text (TEXT PRIMARY KEY): The original text to translate
        - translated_text (TEXT): The translated text
        - source_lang (TEXT): Source language code (e.g., "it")
        - target_lang (TEXT): Target language code (e.g., "en")
        - timestamp (TIMESTAMP): When the translation was cached

    Example:
        >>> cache_path = initialize_translation_cache()
        >>> print(f"Cache initialized at {cache_path}")
    """
    # Determines the directory for the cache file.
    if base_path is None:
        # Uses the directory containing this Python file.
        cache_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        cache_dir = base_path

    # Constructs the full path to the cache database file.
    db_path = os.path.join(
        cache_dir,
        'mongoDB_PostgreSQL_migration_translation_cache.db'
    )

    # Opens a connection to the SQLite database (creates if doesn't exist).
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Creates the translation cache table if it doesn't already exist.
    # The source_text column is the primary key since each unique text
    # should only have one cached translation.
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS translation_cache (
        source_text TEXT PRIMARY KEY,
        translated_text TEXT,
        source_lang TEXT,
        target_lang TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Commits the table creation and closes the connection.
    conn.commit()
    conn.close()

    logger.info(f"Translation cache initialized at {db_path}")
    return db_path


def get_from_cache(
    text: str,
    source_lang: str = "it",
    target_lang: str = "en",
    cache_path: Optional[str] = None
) -> Optional[str]:
    """
    Retrieve a translation from the cache.

    This function looks up a previously cached translation for the
    given source text. It's used to avoid re-translating text that
    has already been processed.

    Args:
        text: The source text to look up in the cache.
        source_lang: The source language code (default: "it").
        target_lang: The target language code (default: "en").
        cache_path: Path to the SQLite cache database file.

    Returns:
        The cached translated text if found, or None if the text
        hasn't been cached or if an error occurs.

    Example:
        >>> cached = get_from_cache("appartamento", cache_path="/path/to/cache.db")
        >>> if cached:
        ...     print(f"Found cached translation: {cached}")
    """
    # Returns None if no cache path is provided.
    if not cache_path:
        return None

    try:
        # Opens a connection to the SQLite cache database.
        conn = sqlite3.connect(cache_path)
        cursor = conn.cursor()

        # Queries for the cached translation matching the source text,
        # source language, and target language.
        cursor.execute(
            """
            SELECT translated_text
            FROM translation_cache
            WHERE source_text=? AND source_lang=? AND target_lang=?
            """,
            (text, source_lang, target_lang)
        )

        # Fetches the result (will be None if not found).
        result = cursor.fetchone()

        # Closes the database connection.
        conn.close()

        # Returns the translated text if found, otherwise None.
        if result:
            return result[0]
        return None

    except Exception as e:
        # Logs any errors that occur during cache lookup.
        logger.error(f"Error retrieving from translation cache: {str(e)}")
        return None


def save_to_cache(
    text: str,
    translated_text: str,
    source_lang: str = "it",
    target_lang: str = "en",
    cache_path: Optional[str] = None
) -> None:
    """
    Save a translation to the cache.

    This function stores a translation in the SQLite cache database
    for future reuse. If the text already exists in the cache, it
    will be replaced with the new translation (INSERT OR REPLACE).

    Args:
        text: The original source text.
        translated_text: The translated text to cache.
        source_lang: The source language code (default: "it").
        target_lang: The target language code (default: "en").
        cache_path: Path to the SQLite cache database file.

    Example:
        >>> save_to_cache(
        ...     "appartamento",
        ...     "apartment",
        ...     cache_path="/path/to/cache.db"
        ... )
    """
    # Does nothing if no cache path is provided.
    if not cache_path:
        return

    try:
        # Opens a connection to the SQLite cache database.
        conn = sqlite3.connect(cache_path)
        cursor = conn.cursor()

        # Inserts the translation into the cache, replacing any existing entry.
        # INSERT OR REPLACE handles the case where the text is already cached.
        cursor.execute(
            """
            INSERT OR REPLACE INTO translation_cache
            (source_text, translated_text, source_lang, target_lang)
            VALUES (?, ?, ?, ?)
            """,
            (text, translated_text, source_lang, target_lang)
        )

        # Commits the changes to persist the cache entry.
        conn.commit()

        # Closes the database connection.
        conn.close()

    except Exception as e:
        # Logs any errors that occur during cache save.
        logger.error(f"Error saving to translation cache: {str(e)}")


def get_cache_stats(cache_path: str) -> dict:
    """
    Get statistics about the translation cache.

    This function returns information about the cache contents,
    including the total number of cached translations and counts
    by language pair.

    Args:
        cache_path: Path to the SQLite cache database file.

    Returns:
        A dictionary containing cache statistics:
        - total_entries: Total number of cached translations
        - by_language_pair: Dict mapping (source, target) to count

    Example:
        >>> stats = get_cache_stats("/path/to/cache.db")
        >>> print(f"Total cached: {stats['total_entries']}")
    """
    try:
        # Opens a connection to the SQLite cache database.
        conn = sqlite3.connect(cache_path)
        cursor = conn.cursor()

        # Gets total count of cached translations.
        cursor.execute("SELECT COUNT(*) FROM translation_cache")
        total_entries = cursor.fetchone()[0]

        # Gets counts by language pair.
        cursor.execute(
            """
            SELECT source_lang, target_lang, COUNT(*)
            FROM translation_cache
            GROUP BY source_lang, target_lang
            """
        )
        by_language_pair = {
            (row[0], row[1]): row[2] for row in cursor.fetchall()
        }

        # Closes the database connection.
        conn.close()

        return {
            'total_entries': total_entries,
            'by_language_pair': by_language_pair
        }

    except Exception as e:
        logger.error(f"Error getting cache statistics: {str(e)}")
        return {'total_entries': 0, 'by_language_pair': {}}

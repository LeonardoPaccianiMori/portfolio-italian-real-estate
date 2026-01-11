"""
Batch translation utilities for efficient translation processing.

This module provides functions for translating multiple texts efficiently
using batch requests to LibreTranslate and parallel processing with
ThreadPoolExecutor.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

import requests

from ...config.settings import LIBRETRANSLATE_URL
from ...config.logging_config import get_logger
from ...core.string_utils import preprocess_text
from .custom_translations import CUSTOM_TRANSLATIONS
from .cache import get_from_cache, save_to_cache
from .translator import translate_text

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def batch_translate(
    texts: List[str],
    source_lang: str = "it",
    target_lang: str = "en",
    batch_size: int = 100,
    max_retries: int = 10,
    cache_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Translate a batch of texts efficiently using LibreTranslate.

    This function translates multiple texts at once by:
    1. Checking the cache for existing translations
    2. Applying custom translations for domain-specific terms
    3. Sending batch requests to LibreTranslate
    4. Caching successful translations

    Args:
        texts: List of Italian texts to translate.
        source_lang: Source language code (default: "it").
        target_lang: Target language code (default: "en").
        batch_size: Maximum number of texts per API request.
        max_retries: Maximum retry attempts for failed requests.
        cache_path: Path to SQLite cache database.

    Returns:
        Dictionary mapping original texts to their translations.

    Example:
        >>> texts = ["appartamento", "villa", "bilocale"]
        >>> translations = batch_translate(texts, cache_path="/path/to/cache.db")
        >>> for orig, trans in translations.items():
        ...     print(f"{orig} -> {trans}")
    """
    # Returns empty dict for empty input.
    if not texts:
        return {}

    results = {}

    # First, checks cache for any existing translations.
    cached_results = {}
    texts_to_translate = []

    for text in texts:
        # Tries to get each text from the cache.
        cached = get_from_cache(text, source_lang, target_lang, cache_path)
        if cached:
            cached_results[text] = cached
        else:
            texts_to_translate.append(text)

    # Returns early if all translations were found in cache.
    if not texts_to_translate:
        return cached_results

    # For texts that need translation, prepares placeholders for custom terms.
    text_placeholders = {}
    placeholder_texts = []

    for text in texts_to_translate:
        # Preprocesses the text to improve translation quality.
        preprocessed = preprocess_text(text)

        # Checks if the entire text matches a custom translation pattern.
        direct_match = False
        for pattern, translation in CUSTOM_TRANSLATIONS.items():
            if re.match(f'^{pattern}$', preprocessed.lower(), re.IGNORECASE):
                # If it's a direct match, uses the custom translation.
                cached_results[text] = translation
                save_to_cache(text, translation, source_lang, target_lang, cache_path)
                direct_match = True
                break

        if direct_match:
            continue

        # Creates placeholders for custom translation patterns.
        placeholders = {}
        placeholder_text = preprocessed

        # Replaces each custom translation pattern with a unique placeholder.
        for pattern, translation in CUSTOM_TRANSLATIONS.items():
            # Adds word boundaries to patterns that don't already have them.
            if not (pattern.startswith(r'\b') or pattern.endswith(r'\b')):
                search_pattern = r'\b' + pattern + r'\b'
            else:
                search_pattern = pattern

            # Finds all matches of this pattern.
            matches = list(re.finditer(
                search_pattern, placeholder_text, flags=re.IGNORECASE
            ))

            # Processes matches in reverse order to avoid index shifting.
            for i, match in enumerate(reversed(matches)):
                placeholder = f"[PLACEHOLDER_{len(placeholders)}]"
                placeholders[placeholder] = translation
                start, end = match.span()
                placeholder_text = (
                    placeholder_text[:start] + placeholder + placeholder_text[end:]
                )

        # Stores the placeholder mapping for this text.
        text_placeholders[text] = placeholders
        placeholder_texts.append((text, placeholder_text))

    # Processes translations in batches to avoid overwhelming the API.
    for i in range(0, len(placeholder_texts), batch_size):
        batch = placeholder_texts[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                # Prepares the request payload with all texts in this batch.
                payload = {
                    "q": [item[1] for item in batch],
                    "source": source_lang,
                    "target": target_lang
                }

                # Sends the batch request to LibreTranslate.
                response = requests.post(
                    LIBRETRANSLATE_URL, json=payload, timeout=30
                )

                if response.status_code == 200:
                    # Parses the response to get translated texts.
                    translated_batch = response.json().get("translatedText", [])

                    # Processes each translation and replaces placeholders.
                    for j, (original_text, placeholder_text) in enumerate(batch):
                        if j < len(translated_batch):
                            translated = translated_batch[j]

                            # Replaces placeholders with custom translations.
                            placeholders = text_placeholders.get(original_text, {})
                            for placeholder, custom_translation in placeholders.items():
                                translated = translated.replace(
                                    placeholder, custom_translation
                                )

                            # Stores the result and saves to cache.
                            cached_results[original_text] = translated
                            save_to_cache(
                                original_text, translated,
                                source_lang, target_lang, cache_path
                            )

                    # Breaks out of retry loop on success.
                    break

                else:
                    # Logs warning and retries on non-200 status.
                    logger.warning(
                        f"Batch translation attempt {attempt + 1} failed with "
                        f"status code {response.status_code}: {response.text}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2)

            except Exception as e:
                # Logs error and retries on exception.
                logger.error(
                    f"Error during batch translation attempt {attempt + 1}: {str(e)}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)

    # For any texts that failed batch translation, tries individual translation.
    for text in texts_to_translate:
        if text not in cached_results:
            translated = translate_text(
                text, source_lang, target_lang, max_retries, cache_path
            )
            cached_results[text] = translated

    return cached_results


def translate_values_parallel(
    values: List[str],
    source_lang: str = "it",
    target_lang: str = "en",
    max_workers: int = 10,
    cache_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Translate a list of values in parallel using ThreadPoolExecutor.

    This function combines batch translation with parallel processing
    for efficient translation of large value lists. It first attempts
    batch translation, then uses parallel workers for any remaining
    texts.

    Args:
        values: List of strings to translate.
        source_lang: Source language code (default: "it").
        target_lang: Target language code (default: "en").
        max_workers: Maximum number of parallel translation workers.
        cache_path: Path to SQLite cache database.

    Returns:
        Dictionary mapping original values to their translations.

    Example:
        >>> values = ["appartamento", "villa", "bilocale"]
        >>> translations = translate_values_parallel(values, max_workers=5)
    """
    # Returns empty dict for empty input.
    if not values:
        return {}

    # Deduplicates values to avoid translating the same text multiple times.
    unique_values = list(set(values))
    logger.info(
        f"Translating {len(unique_values)} unique values in parallel "
        f"(max workers: {max_workers})"
    )

    results = {}

    # Tries batch translation first (more efficient for most cases).
    batch_results = batch_translate(
        unique_values, source_lang, target_lang, cache_path=cache_path
    )

    # Identifies any values that weren't translated in the batch.
    values_to_translate = [v for v in unique_values if v not in batch_results]

    if values_to_translate:
        logger.info(
            f"Batch translation completed. {len(values_to_translate)} values "
            "need individual translation."
        )

        # Defines a worker function for parallel translation.
        def translate_worker(text):
            """Translates a single text and returns both original and result."""
            return text, translate_text(
                text, source_lang, target_lang, cache_path=cache_path
            )

        # Uses ThreadPoolExecutor to translate remaining texts in parallel.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for original, translated in executor.map(
                translate_worker, values_to_translate
            ):
                results[original] = translated

    # Combines batch results with individual results.
    results.update(batch_results)

    return results


def resolve_conflicts_in_batch(
    postgres_hook: Any,
    table: str,
    primary_key: str,
    conflicts: Dict[str, List[int]],
    fk_relationships: List[Dict[str, str]]
) -> None:
    """
    Resolve translation conflicts in batch mode.

    When multiple Italian terms translate to the same English term,
    this creates duplicate records in dimension tables. This function
    resolves these conflicts by:
    1. Keeping one "primary" record per translated value
    2. Updating all foreign key references to point to the primary record
    3. Deleting the secondary (duplicate) records

    Args:
        postgres_hook: PostgreSQL connection hook for executing queries.
        table: Name of the dimension table with conflicts.
        primary_key: Name of the primary key column.
        conflicts: Dict mapping translated values to lists of conflicting
            record IDs. The first ID in each list is kept as primary.
        fk_relationships: List of dicts with foreign key info:
            - referencing_table: Table that references this dimension
            - referencing_column: Column containing the foreign key

    Example:
        >>> conflicts = {"apartment": [1, 2, 3]}  # IDs 2, 3 are duplicates
        >>> fk_rels = [{"referencing_table": "fact_listing",
        ...             "referencing_column": "type_of_property_id"}]
        >>> resolve_conflicts_in_batch(hook, "dim_type_of_property",
        ...                            "type_of_property_id", conflicts, fk_rels)
    """
    start_time = time.time()
    logger.info(
        f"Starting batch conflict resolution for {len(conflicts)} conflicts in {table}"
    )

    # Gets a database connection.
    conn = postgres_hook.get_conn()

    try:
        # Groups update operations by referencing table to minimize transactions.
        updates_by_table = {}

        for translated_value, conflicting_ids in conflicts.items():
            # The first ID is the one we keep (primary).
            primary_id = conflicting_ids[0]

            # All other IDs are secondary (to be merged into primary).
            secondary_ids = conflicting_ids[1:]

            if not secondary_ids:
                continue

            # Formats secondary IDs for SQL IN clause.
            secondary_ids_str = ','.join(str(id) for id in secondary_ids)

            # For each table that references this dimension table.
            for fk_rel in fk_relationships:
                referencing_table = fk_rel['referencing_table']
                referencing_column = fk_rel['referencing_column']

                if referencing_table not in updates_by_table:
                    updates_by_table[referencing_table] = []

                updates_by_table[referencing_table].append({
                    'primary_id': primary_id,
                    'secondary_ids': secondary_ids_str,
                    'referencing_column': referencing_column
                })

        # Applies updates efficiently, one referencing table at a time.
        with conn.cursor() as cursor:
            for referencing_table, updates in updates_by_table.items():
                if not updates:
                    continue

                logger.info(
                    f"Processing {len(updates)} conflict resolutions "
                    f"for table {referencing_table}"
                )

                # Creates a temporary table for bulk ID mapping.
                cursor.execute('''
                    CREATE TEMP TABLE temp_id_mapping (
                        old_id INTEGER NOT NULL,
                        new_id INTEGER NOT NULL
                    ) ON COMMIT DROP
                ''')

                # Builds batch insert of all ID mappings.
                mapping_values = []
                mapping_params = []
                for update_info in updates:
                    for secondary_id in update_info['secondary_ids'].split(','):
                        if secondary_id:  # Skips empty strings.
                            mapping_values.append("(%s, %s)")
                            mapping_params.extend([
                                int(secondary_id), update_info['primary_id']
                            ])

                if mapping_values:
                    # Inserts all mappings in one statement.
                    cursor.execute(
                        f"INSERT INTO temp_id_mapping (old_id, new_id) "
                        f"VALUES {', '.join(mapping_values)}",
                        mapping_params
                    )

                    # Gets the referencing column name.
                    referencing_column = updates[0]['referencing_column']

                    # Updates references using a JOIN with the temporary table.
                    cursor.execute(f'''
                        UPDATE {referencing_table} AS t
                        SET {referencing_column} = m.new_id
                        FROM temp_id_mapping AS m
                        WHERE t.{referencing_column} = m.old_id
                    ''')

                    updated_rows = cursor.rowcount
                    logger.info(f"Updated {updated_rows} rows in {referencing_table}")

            # Deletes the secondary records that are no longer referenced.
            all_secondary_ids = []
            for translated_value, conflicting_ids in conflicts.items():
                all_secondary_ids.extend(conflicting_ids[1:])

            if all_secondary_ids:
                # Deletes in batches to avoid huge IN clauses.
                BATCH_SIZE = 1000
                for i in range(0, len(all_secondary_ids), BATCH_SIZE):
                    batch_ids = all_secondary_ids[i:i + BATCH_SIZE]
                    ids_str = ','.join(str(id) for id in batch_ids)

                    try:
                        cursor.execute(
                            f"DELETE FROM {table} WHERE {primary_key} IN ({ids_str})"
                        )
                        deleted_rows = cursor.rowcount
                        logger.info(
                            f"Deleted {deleted_rows} secondary records from {table} "
                            f"(batch {i // BATCH_SIZE + 1})"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not delete some secondary records: {str(e)}"
                        )

            # Commits all changes.
            conn.commit()

    except Exception as e:
        logger.error(f"Error during batch conflict resolution: {str(e)}")
        conn.rollback()
        raise

    finally:
        conn.close()

    elapsed_time = time.time() - start_time
    logger.info(f"Batch conflict resolution completed in {elapsed_time:.2f} seconds")


def bulk_update_translations(
    postgres_hook: Any,
    table: str,
    column: str,
    primary_key: str,
    id_to_translated: Dict[int, str],
    conflicts: Dict[str, List[int]]
) -> None:
    """
    Update translations in bulk in the PostgreSQL database.

    This function efficiently updates translated values for multiple
    records in a dimension table, skipping any records that are
    marked as secondary in the conflicts dictionary.

    Args:
        postgres_hook: PostgreSQL connection hook for executing queries.
        table: Name of the dimension table to update.
        column: Name of the column containing translated text.
        primary_key: Name of the primary key column.
        id_to_translated: Dict mapping record IDs to translated values.
        conflicts: Dict of translation conflicts (secondary IDs skipped).

    Example:
        >>> id_to_translated = {1: "apartment", 2: "villa"}
        >>> bulk_update_translations(
        ...     hook, "dim_type_of_property", "type_of_property",
        ...     "type_of_property_id", id_to_translated, {}
        ... )
    """
    start_time = time.time()
    logger.info(f"Starting bulk update of translations for {table}.{column}")

    # Creates a set of secondary IDs to skip.
    skip_ids = []
    for translated_value, conflicting_ids in conflicts.items():
        skip_ids.extend(conflicting_ids[1:])
    skip_ids_set = set(skip_ids)

    # Builds the list of updates, skipping secondary IDs.
    update_params = []
    for record_id, translated_value in id_to_translated.items():
        if record_id not in skip_ids_set:
            update_params.append((translated_value, record_id))

    if not update_params:
        logger.info(f"No updates needed for {table}.{column}")
        return

    # Executes updates in batches.
    BATCH_SIZE = 1000
    conn = postgres_hook.get_conn()
    update_query = f"UPDATE {table} SET {column} = %s WHERE {primary_key} = %s"

    try:
        with conn.cursor() as cursor:
            total_updated = 0

            for i in range(0, len(update_params), BATCH_SIZE):
                batch = update_params[i:i + BATCH_SIZE]

                # Uses executemany for efficient batch updates.
                cursor.executemany(update_query, batch)
                total_updated += cursor.rowcount

                # Logs progress for large updates.
                if (i + BATCH_SIZE) % 10000 == 0 or i + BATCH_SIZE >= len(update_params):
                    logger.info(
                        f"Updated {i + len(batch)}/{len(update_params)} "
                        f"translations in {table}.{column}"
                    )

            conn.commit()
            logger.info(
                f"Total of {total_updated} translations updated in {table}.{column}"
            )

    except Exception as e:
        logger.error(f"Error during bulk translation update: {str(e)}")
        conn.rollback()
        raise

    finally:
        conn.close()

    elapsed_time = time.time() - start_time
    logger.info(f"Bulk update completed in {elapsed_time:.2f} seconds")

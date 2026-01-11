#!/usr/bin/env python3
"""
CLI script for running the PostgreSQL migration pipeline.

This script provides a command-line interface for migrating data from the
MongoDB warehouse to PostgreSQL using a star schema design. It supports
batch processing, dimension table creation, fact table loading, and
optional Italian-to-English translation.

Usage:
    python run_migration.py --all
    python run_migration.py --batch-size 5000 --max-batches 10
    python run_migration.py --translate-only
    python run_migration.py --stats

Author: Leonardo Pacciani-Mori
License: MIT
"""

import sys
from pathlib import Path

# Allow running without package installation
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
_src_dir = _project_root / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import argparse
import time
from typing import Dict, Any, Optional

import psycopg2


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Migrate data from MongoDB warehouse to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full migration with default settings
    python run_migration.py --all

    # Run migration with custom batch size
    python run_migration.py --all --batch-size 5000

    # Run only specific number of batches (for testing)
    python run_migration.py --all --max-batches 5

    # Skip translation step
    python run_migration.py --all --skip-translation

    # Run translation only on existing data
    python run_migration.py --translate-only

    # Show migration statistics
    python run_migration.py --stats
        """
    )

    # Main operation modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--all",
        action="store_true",
        help="Run full migration (dimensions, facts, translation)"
    )
    mode_group.add_argument(
        "--translate-only",
        action="store_true",
        help="Only run translation on existing PostgreSQL data"
    )
    mode_group.add_argument(
        "--stats",
        action="store_true",
        help="Show migration statistics only"
    )

    # Batch processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of records per batch (default: 10000)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum batches to process (default: all)"
    )

    # Translation options
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip the translation step after migration"
    )

    # Parallel execution options
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for batch processing (default: 8)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress display"
    )

    # Other options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def get_migration_stats() -> Dict[str, Any]:
    """
    Get current migration statistics from PostgreSQL.

    Returns:
        dict: Dictionary with migration statistics.
    """
    from italian_real_estate.config.settings import POSTGRES_CONNECTION_PARAMS
    from italian_real_estate.migration.postgres_utils import execute_query_silent

    stats = {}

    try:
        conn = psycopg2.connect(**POSTGRES_CONNECTION_PARAMS)
        cursor = conn.cursor()

        # Fact table count
        cursor.execute("SELECT COUNT(*) FROM fact_listing")
        result = cursor.fetchone()
        stats["fact_table_records"] = result[0] if result else 0

        # Bridge table counts
        cursor.execute("SELECT COUNT(*) FROM listing_features_bridge")
        result = cursor.fetchone()
        stats["feature_relationships"] = result[0] if result else 0

        cursor.execute("SELECT COUNT(*) FROM surface_composition_bridge")
        result = cursor.fetchone()
        stats["surface_relationships"] = result[0] if result else 0

        # Dimension table counts
        dimension_tables = [
            "dim_date", "dim_listing_type", "dim_seller_type", "dim_category",
            "dim_energy_info", "dim_type_of_property", "dim_condition",
            "dim_rooms_info", "dim_building_info", "dim_location",
            "dim_mortgage_rate", "dim_listing_page", "dim_features",
            "dim_surface_composition"
        ]

        stats["dimension_counts"] = {}
        for table in dimension_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                result = cursor.fetchone()
                stats["dimension_counts"][table] = result[0] if result else 0
            except psycopg2.Error:
                stats["dimension_counts"][table] = "N/A"

        cursor.close()
        conn.close()

    except psycopg2.Error as e:
        stats["error"] = str(e)

    return stats


def print_stats(stats: Dict[str, Any], logger) -> None:
    """Print migration statistics."""
    if "error" in stats:
        logger.error(f"Error getting stats: {stats['error']}")
        return

    logger.info("=" * 60)
    logger.info("PostgreSQL Migration Statistics")
    logger.info("=" * 60)
    logger.info(f"Fact table records: {stats.get('fact_table_records', 0):,}")
    logger.info(f"Feature relationships: {stats.get('feature_relationships', 0):,}")
    logger.info(f"Surface relationships: {stats.get('surface_relationships', 0):,}")

    logger.info("-" * 60)
    logger.info("Dimension table counts:")
    for table, count in stats.get("dimension_counts", {}).items():
        if isinstance(count, int):
            logger.info(f"  {table}: {count:,}")
        else:
            logger.info(f"  {table}: {count}")
    logger.info("=" * 60)


def run_translation(logger, max_workers: int = 20) -> bool:
    """
    Run translation on existing PostgreSQL data.

    Args:
        logger: Logger instance.
        max_workers: Number of parallel translation workers.

    Returns:
        bool: True if successful, False otherwise.
    """
    from italian_real_estate.config.settings import POSTGRES_CONNECTION_PARAMS
    from italian_real_estate.migration.translation import (
        initialize_translation_cache,
        start_libretranslate_service,
        stop_libretranslate_service,
        translate_values_parallel,
        resolve_conflicts_in_batch,
        bulk_update_translations,
    )
    from italian_real_estate.migration.postgres_utils import (
        execute_query_silent,
        get_primary_key_column,
    )

    logger.info("Starting translation of database fields from Italian to English...")
    start_time = time.time()

    # Initialize translation cache
    cache_path = initialize_translation_cache()

    # Start LibreTranslate service
    libretranslate_process = None
    try:
        libretranslate_process = start_libretranslate_service()

        conn = psycopg2.connect(**POSTGRES_CONNECTION_PARAMS)

        # Define columns to translate and their corresponding tables
        columns_to_translate = {
            'dim_seller_type': ['seller_type'],
            'dim_category': ['category_name'],
            'dim_energy_info': ['heating_type', 'air_conditioning'],
            'dim_type_of_property': ['type_of_property'],
            'dim_condition': ['condition'],
            'dim_rooms_info': ['kitchen_status', 'garage'],
            'dim_building_info': ['building_usage'],
            'dim_features': ['feature_name']
        }

        total_translations = 0

        # Cache for foreign key relationships
        fk_relationship_cache = {}

        for table, columns in columns_to_translate.items():
            cursor = conn.cursor()

            # Get primary key
            primary_key = get_primary_key_column(conn, table)
            if not primary_key:
                logger.warning(f"Could not determine primary key for table {table}, skipping")
                continue

            # Get foreign key relationships
            fk_query = f"""
                SELECT
                    tc.constraint_name,
                    tc.table_name as referencing_table,
                    kcu.column_name as referencing_column
                FROM
                    information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE
                    tc.constraint_type = 'FOREIGN KEY'
                    AND ccu.table_name = '{table}'
                    AND ccu.column_name = '{primary_key}'
            """
            cursor.execute(fk_query)
            fk_results = cursor.fetchall()

            fk_relationship_cache[table] = []
            for fk_row in fk_results:
                constraint_name, referencing_table, referencing_column = fk_row
                fk_relationship_cache[table].append({
                    'constraint_name': constraint_name,
                    'referencing_table': referencing_table,
                    'referencing_column': referencing_column
                })

            logger.info(f"Found {len(fk_relationship_cache[table])} FK relationships for {table}")

            for column in columns:
                logger.info(f"Processing translations for {table}.{column}")

                # Get all non-null distinct values
                query = f"SELECT DISTINCT {primary_key}, {column} FROM {table} WHERE {column} IS NOT NULL"
                cursor.execute(query)
                results = cursor.fetchall()

                if not results:
                    logger.info(f"No non-null values found for {table}.{column}")
                    continue

                logger.info(f"Found {len(results)} distinct values for {table}.{column}")

                # Extract values for translation
                original_values = [row[1] for row in results]

                # Translate in parallel
                translations = translate_values_parallel(
                    original_values,
                    max_workers=max_workers,
                    cache_path=cache_path
                )

                # Map IDs to translations
                id_to_translated = {row[0]: translations.get(row[1], row[1]) for row in results}
                total_translations += len(set(translations.values()))

                # Check for conflicts
                translated_to_ids = {}
                for record_id, translated_value in id_to_translated.items():
                    if translated_value not in translated_to_ids:
                        translated_to_ids[translated_value] = []
                    translated_to_ids[translated_value].append(record_id)

                conflicts = {t: ids for t, ids in translated_to_ids.items() if len(ids) > 1}

                if conflicts:
                    logger.info(f"Found {len(conflicts)} translation conflicts for {table}.{column}")
                    resolve_conflicts_in_batch(
                        conn,
                        table,
                        primary_key,
                        conflicts,
                        fk_relationship_cache[table]
                    )

                # Bulk update
                bulk_update_translations(
                    conn,
                    table,
                    column,
                    primary_key,
                    id_to_translated,
                    conflicts
                )

                logger.info(f"Completed processing for {table}.{column}")

            cursor.close()

        conn.close()

        elapsed_time = time.time() - start_time
        logger.info(f"Translation completed in {elapsed_time:.2f} seconds.")
        logger.info(f"Translated {total_translations} unique values across the database.")
        return True

    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return False

    finally:
        if libretranslate_process:
            stop_libretranslate_service(libretranslate_process)


def run_migration(
    args: argparse.Namespace,
    logger
) -> int:
    """
    Run the full migration pipeline.

    Args:
        args: Parsed command-line arguments.
        logger: Logger instance.

    Returns:
        int: Exit code (0 for success).
    """
    from italian_real_estate.migration import (
        setup_batch_processing,
        process_batch,
        load_fact_table_for_batch,
        count_batches,
        finalize_migration,
    )
    from italian_real_estate.utils.progress import ParallelExecutor

    logger.info("Starting PostgreSQL migration...")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max batches: {args.max_batches or 'all'}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Translation: {'disabled' if args.skip_translation else 'enabled'}")

    if args.dry_run:
        logger.info("Dry run mode - showing what would be done")

        # Count batches
        num_batches = count_batches(args.batch_size)
        if args.max_batches:
            num_batches = min(num_batches, args.max_batches)

        logger.info(f"Would process {num_batches} batches")
        return 0

    # Step 1: Setup batch processing
    logger.info("Setting up batch processing...")
    batch_info = setup_batch_processing(batch_size=args.batch_size)
    total_batches = batch_info.get("num_batches", 0)

    if args.max_batches:
        total_batches = min(total_batches, args.max_batches)

    logger.info(f"Will process {total_batches} batches")

    # Step 2: Process batches (dimensions and facts)
    # Each batch needs: process_batch -> load_fact_table_for_batch
    batch_numbers = list(range(total_batches))

    def process_single_batch(batch_num: int) -> Dict[str, Any]:
        """Process dimensions and load facts for a single batch."""
        # Process dimensions
        dimension_mappings = process_batch(batch_num)

        # Load fact table
        records_loaded = load_fact_table_for_batch(batch_num, dimension_mappings)

        return {
            "batch_num": batch_num,
            "records_loaded": records_loaded
        }

    logger.info("Processing batches...")

    if args.workers > 1 and not args.no_progress:
        # Parallel execution with progress display
        executor = ParallelExecutor(
            max_workers=args.workers,
            title="Migration - Processing Batches",
            show_progress=not args.no_progress
        )

        results = executor.run(
            items=batch_numbers,
            job_func=process_single_batch,
            job_name_func=lambda b: f"Batch {b}",
            on_complete=lambda b, r: r.get("records_loaded", 0)
        )
    else:
        # Sequential execution
        results = {}
        for batch_num in batch_numbers:
            logger.info(f"Processing batch {batch_num}/{total_batches - 1}...")
            result = process_single_batch(batch_num)
            results[f"Batch {batch_num}"] = result

    # Calculate totals
    total_records = sum(r.get("records_loaded", 0) for r in results.values())
    logger.info(f"Total records loaded: {total_records:,}")

    # Step 3: Translation (unless skipped)
    if not args.skip_translation:
        logger.info("Starting translation step...")
        success = run_translation(logger, max_workers=20)
        if not success:
            logger.warning("Translation step had errors, but migration continued")

    # Step 4: Finalize
    logger.info("Finalizing migration...")
    final_stats = finalize_migration()

    logger.info("Migration completed successfully!")
    print_stats(get_migration_stats(), logger)

    return 0


def main() -> int:
    """
    Main entry point for the migration CLI.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = parse_arguments()

    # Set up logging
    from italian_real_estate.config.logging_config import setup_logging
    import logging

    if args.verbose:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)

    from italian_real_estate.config.logging_config import get_logger
    logger = get_logger(__name__)

    try:
        # Stats-only mode
        if args.stats:
            stats = get_migration_stats()
            print_stats(stats, logger)
            return 0

        # Translate-only mode
        if args.translate_only:
            success = run_translation(logger)
            return 0 if success else 1

        # Full migration mode
        if args.all:
            return run_migration(args, logger)

        # Should not reach here
        logger.error("No operation specified")
        return 1

    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Migration failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

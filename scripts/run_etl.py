#!/usr/bin/env python3
"""
CLI script for running the ETL pipeline from datalake to warehouse.

This script provides a command-line interface for extracting, transforming,
and loading data from the MongoDB datalake to the MongoDB warehouse.

Usage:
    python run_etl.py --listing-type sale
    python run_etl.py --all
    python run_etl.py --stats

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
from typing import List


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract, transform, and load data from datalake to warehouse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run ETL for a specific listing type
    python run_etl.py --listing-type sale

    # Run ETL for all listing types
    python run_etl.py --all

    # Show warehouse statistics only
    python run_etl.py --stats

    # Run cleaning task only
    python run_etl.py --clean-only
        """
    )

    # Listing type selection (mutually exclusive with --all)
    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        "--listing-type",
        type=str,
        choices=["sale", "rent", "auction"],
        help="Type of listings to process"
    )
    type_group.add_argument(
        "--all",
        action="store_true",
        help="Process all listing types"
    )

    # Statistics only mode
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Only show warehouse statistics, don't run ETL"
    )

    # Cleaning only mode
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only run data cleaning task"
    )

    # Skip cleaning
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip the data cleaning step after ETL"
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def get_listing_types(args: argparse.Namespace) -> List[str]:
    """
    Determine which listing types to process.

    Args:
        args: The parsed command-line arguments.

    Returns:
        list: A list of listing types to process.
    """
    if args.all:
        return ["rent", "auction", "sale"]
    elif args.listing_type:
        return [args.listing_type]
    else:
        return []


def main() -> int:
    """
    Main entry point for the ETL CLI.

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

    # Import ETL functions
    from italian_real_estate.etl.warehouse_loader import migrate, get_warehouse_stats
    from italian_real_estate.etl.transformers import fix_empty_child_listings

    # Stats-only mode
    if args.stats:
        get_warehouse_stats()
        return 0

    # Clean-only mode
    if args.clean_only:
        logger.info("Running data cleaning task...")
        fixed = fix_empty_child_listings()
        logger.info(f"Cleaning completed. Fixed {fixed} documents.")
        return 0

    # Determine what to process
    listing_types = get_listing_types(args)

    if not listing_types:
        logger.error("No listing types specified. Use --listing-type or --all")
        return 1

    logger.info(f"Will process listing types: {listing_types}")

    try:
        # Run ETL for each listing type
        for listing_type in listing_types:
            logger.info(f"Running ETL for {listing_type} listings...")
            migrate(listing_type)

        # Run cleaning task unless skipped
        if not args.skip_cleaning:
            logger.info("Running data cleaning task...")
            fix_empty_child_listings()

        # Log final statistics
        get_warehouse_stats()

        logger.info("ETL completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("ETL interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"ETL failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

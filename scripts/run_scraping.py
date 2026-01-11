#!/usr/bin/env python3
"""
CLI script for running the web scraping pipeline.

This script provides a command-line interface for scraping real estate
listings from listing.website and populating the MongoDB datalake. It can
be run standalone or called by Airflow.

Usage:
    python run_scraping.py --province Milano --listing-type sale
    python run_scraping.py --all-provinces --listing-type rent
    python run_scraping.py --stats

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
from typing import List, Optional


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Scrape real estate listings from listing.website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scrape sale listings for a single province
    python run_scraping.py --province Milano --listing-type sale

    # Scrape all listing types for a province
    python run_scraping.py --province Roma

    # Scrape all provinces (all listing types)
    python run_scraping.py --all-provinces

    # Show datalake statistics only
    python run_scraping.py --stats
        """
    )

    # Province selection (mutually exclusive group)
    province_group = parser.add_mutually_exclusive_group()
    province_group.add_argument(
        "--province",
        type=str,
        help="Name of the province to scrape (e.g., 'Milano')"
    )
    province_group.add_argument(
        "--all-provinces",
        action="store_true",
        help="Scrape all provinces"
    )

    # Listing type selection
    parser.add_argument(
        "--listing-type",
        type=str,
        choices=["sale", "rent", "auction", "all"],
        default="all",
        help="Type of listings to scrape (default: all)"
    )

    # Statistics only mode
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Only show datalake statistics, don't scrape"
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def get_provinces_to_scrape(args: argparse.Namespace) -> List[str]:
    """
    Determine which provinces to scrape based on arguments.

    Args:
        args: The parsed command-line arguments.

    Returns:
        list: A list of province names to scrape.
    """
    from italian_real_estate.scraping.datalake_populator import get_all_provinces

    if args.all_provinces:
        return get_all_provinces()
    elif args.province:
        return [args.province]
    else:
        return []


def get_listing_types_to_scrape(args: argparse.Namespace) -> List[str]:
    """
    Determine which listing types to scrape based on arguments.

    Args:
        args: The parsed command-line arguments.

    Returns:
        list: A list of listing types to scrape.
    """
    if args.listing_type == "all":
        return ["rent", "auction", "sale"]
    else:
        return [args.listing_type]


def main() -> int:
    """
    Main entry point for the scraping CLI.

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

    # If stats-only mode, just show statistics and exit
    if args.stats:
        from italian_real_estate.scraping.datalake_populator import log_datalake_statistics
        log_datalake_statistics()
        return 0

    # Determine what to scrape
    provinces = get_provinces_to_scrape(args)
    listing_types = get_listing_types_to_scrape(args)

    if not provinces:
        logger.error("No provinces specified. Use --province or --all-provinces")
        return 1

    logger.info(f"Will scrape {len(provinces)} province(s): {provinces[:5]}...")
    logger.info(f"Listing types: {listing_types}")

    # Import scraping functions
    from italian_real_estate.scraping.datalake_populator import (
        datalake_populate_links_province,
        log_datalake_statistics,
    )

    # Run scraping for each province and listing type
    try:
        for province in provinces:
            for listing_type in listing_types:
                logger.info(f"Scraping {listing_type} listings for {province}...")
                datalake_populate_links_province(province, listing_type)

        # Log final statistics
        log_datalake_statistics()

        logger.info("Scraping completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("Scraping interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Scraping failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

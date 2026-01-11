#!/usr/bin/env python3
"""
Pipeline TUI/CLI for the Italian Real Estate project.

This script provides both an interactive menu-based TUI and a traditional
CLI for running all pipeline stages. When run without arguments, it launches
the interactive TUI. When run with arguments, it behaves as a standard CLI.

Pipeline Stages:
    1. SCRAPE      - Web scraping to MongoDB Datalake
    2. ETL         - MongoDB Datalake to MongoDB Warehouse
    3. MIGRATE     - MongoDB Warehouse to PostgreSQL
    4. DATA-EXPORT - Export real or synthetic data to Parquet
    5. TRAIN       - Train ML model and generate dashboard data file

Usage:
    # Launch interactive TUI
    python pipeline_tui.py

    # CLI mode (with arguments)
    python pipeline_tui.py scrape --all-provinces
    python pipeline_tui.py etl --all
    python pipeline_tui.py migrate --all
    python pipeline_tui.py data-export --real
    python pipeline_tui.py data-export --synthetic
    python pipeline_tui.py train
    python pipeline_tui.py all -y
    python pipeline_tui.py status

Author: Leonardo Pacciani-Mori
License: MIT
"""

# Suppress TensorFlow CUDA/cuDNN initialization warnings
# Must be set before any imports that might load TensorFlow
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
# Suppress absl logging (TensorFlow's C++ logging)
os.environ.setdefault('ABSL_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')

import sys
from pathlib import Path

# Allow running without package installation
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
_src_dir = _project_root / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
# Add scripts/ to path for cross-script imports
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import argparse
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Any


class PipelineStage(Enum):
    """Pipeline stages in order of execution."""
    SCRAPE = "scrape"
    ETL = "etl"
    MIGRATE = "migrate"
    DATA_EXPORT = "data-export"
    TRAIN = "train"


@dataclass
class StageInfo:
    """Information about a pipeline stage."""
    name: str
    description: str
    source: str
    destination: str
    parallel: bool = False  # Whether stage supports parallel execution


STAGE_INFO = {
    PipelineStage.SCRAPE: StageInfo(
        name="Web Scraping",
        description="Scrape real estate listings from listing.website",
        source="listing.website",
        destination="MongoDB Datalake",
        parallel=True
    ),
    PipelineStage.ETL: StageInfo(
        name="ETL Processing",
        description="Extract, transform, and load data from datalake to warehouse",
        source="MongoDB Datalake",
        destination="MongoDB Warehouse",
        parallel=False
    ),
    PipelineStage.MIGRATE: StageInfo(
        name="PostgreSQL Migration",
        description="Migrate data to PostgreSQL star schema with translation",
        source="MongoDB Warehouse",
        destination="PostgreSQL",
        parallel=True
    ),
    PipelineStage.DATA_EXPORT: StageInfo(
        name="Data Export",
        description="Export real or synthetic data to Parquet",
        source="PostgreSQL",
        destination="Parquet file",
        parallel=False
    ),
    PipelineStage.TRAIN: StageInfo(
        name="ML Model Training",
        description="Train rent prediction model and create dashboard data",
        source="Parquet file",
        destination="Model + Dashboard CSV",
        parallel=False
    ),
}


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser with all subcommands.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Italian Real Estate Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
    1. scrape       Web scraping from listing.website to MongoDB Datalake
    2. etl          Extract, transform, load to MongoDB Warehouse
    3. migrate      Migrate to PostgreSQL with star schema
    4. data-export  Export real or synthetic data to Parquet
    5. train        Train ML model and create dashboard data file

Examples:
    # Run specific stage
    python pipeline_tui.py scrape --all-provinces
    python pipeline_tui.py etl --all
    python pipeline_tui.py migrate --all
    python pipeline_tui.py data-export --real
    python pipeline_tui.py data-export --synthetic
    python pipeline_tui.py train

    # Run all stages (interactive - confirm each stage)
    python pipeline_tui.py all

    # Run all stages (automated - no confirmations)
    python pipeline_tui.py all -y

    # Run range of stages
    python pipeline_tui.py --from scrape --to migrate

    # Run specific stages
    python pipeline_tui.py --stages scrape,etl,migrate

    # Show pipeline status
    python pipeline_tui.py status

    # Interactive database explorer
    python pipeline_tui.py db
        """
    )

    # Global arguments
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompts (automated mode)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress display"
    )

    # Range selection (alternative to subcommands)
    parser.add_argument(
        "--from",
        dest="from_stage",
        type=str,
        choices=["scrape", "etl", "migrate", "data-export", "train"],
        help="Start from this stage (inclusive)"
    )
    parser.add_argument(
        "--to",
        dest="to_stage",
        type=str,
        choices=["scrape", "etl", "migrate", "data-export", "train"],
        help="Stop at this stage (inclusive)"
    )
    parser.add_argument(
        "--stages",
        type=str,
        help="Comma-separated list of stages to run"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")

    # All command
    all_parser = subparsers.add_parser("all", help="Run all pipeline stages")

    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Run web scraping stage")
    scrape_group = scrape_parser.add_mutually_exclusive_group()
    scrape_group.add_argument(
        "--province",
        type=str,
        help="Name of province to scrape"
    )
    scrape_group.add_argument(
        "--all-provinces",
        action="store_true",
        help="Scrape all provinces"
    )
    scrape_parser.add_argument(
        "--listing-type",
        type=str,
        choices=["sale", "rent", "auction", "all"],
        default="all",
        help="Type of listings to scrape (default: all)"
    )

    # ETL command
    etl_parser = subparsers.add_parser("etl", help="Run ETL stage")
    etl_group = etl_parser.add_mutually_exclusive_group()
    etl_group.add_argument(
        "--listing-type",
        type=str,
        choices=["sale", "rent", "auction"],
        help="Type of listings to process"
    )
    etl_group.add_argument(
        "--all",
        action="store_true",
        help="Process all listing types"
    )
    etl_parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip the data cleaning step"
    )

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run PostgreSQL migration")
    migrate_parser.add_argument(
        "--all",
        action="store_true",
        help="Run full migration"
    )
    migrate_parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for migration (default: 10000)"
    )
    migrate_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum batches to process"
    )
    migrate_parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip translation step"
    )

    # Data Export command
    data_export_parser = subparsers.add_parser("data-export", help="Export real or synthetic data")
    data_type_group = data_export_parser.add_mutually_exclusive_group()
    data_type_group.add_argument(
        "--real",
        action="store_true",
        help="Export real data from PostgreSQL (preserves NaN in numerical columns)"
    )
    data_type_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data using KNN interpolation (default)"
    )
    data_export_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: data/real_data.parquet or data/synthetic_data.parquet)"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input data file (default: auto-detect parquet/csv in data/)"
    )
    train_parser.add_argument(
        "--output-dashboard",
        type=str,
        default="data/dashboard_data.csv",
        help="Output dashboard data file"
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test set proportion (default: 0.3)"
    )

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Initialize database schemas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Initialize databases for the Italian Real Estate pipeline.

Creates MongoDB databases/collections and PostgreSQL schema with all tables.
This command should be run on a fresh installation before running the pipeline.

Examples:
    # Interactive setup (prompts for PostgreSQL credentials)
    python pipeline_tui.py setup

    # Check database status only
    python pipeline_tui.py setup --check

    # Setup PostgreSQL only
    python pipeline_tui.py setup --postgres

    # Setup MongoDB only
    python pipeline_tui.py setup --mongodb

    # Reconfigure credentials
    python pipeline_tui.py setup --reconfigure
        """
    )
    setup_parser.add_argument(
        "--check",
        action="store_true",
        help="Check database status without making changes"
    )
    setup_parser.add_argument(
        "--postgres",
        action="store_true",
        help="Set up PostgreSQL only"
    )
    setup_parser.add_argument(
        "--mongodb",
        action="store_true",
        help="Set up MongoDB only"
    )
    setup_parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Reconfigure database credentials"
    )

    # Database explorer command
    db_parser = subparsers.add_parser(
        "db",
        help="Interactive database explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive mode (default):
    python pipeline_tui.py db

    Commands in interactive mode:
      use <database>     - Connect to database (mongo-datalake, mongo-warehouse, postgres)
      collections        - List collections/tables
      sample <name> [n]  - Show n sample records
      count <name>       - Count records
      schema <name>      - Show schema/fields
      query <SQL>        - Run SQL query (PostgreSQL only)
      find <coll> <json> - Run MongoDB find query
      stats              - Show database statistics
      help               - Show help
      exit               - Exit explorer

Non-interactive mode:
    python pipeline_tui.py db --database mongo-datalake --collections
    python pipeline_tui.py db --database postgres --sample fact_listing 5
    python pipeline_tui.py db --database postgres --query "SELECT * FROM dim_location LIMIT 5"
        """
    )
    db_parser.add_argument(
        "--database", "-d",
        type=str,
        choices=["mongo-datalake", "mongo-warehouse", "postgres"],
        help="Database to query (required for non-interactive mode)"
    )
    db_parser.add_argument(
        "--collections", "--tables",
        action="store_true",
        help="List collections/tables"
    )
    db_parser.add_argument(
        "--sample",
        nargs="+",
        metavar=("NAME", "N"),
        help="Sample records: --sample <collection> [n]"
    )
    db_parser.add_argument(
        "--count",
        type=str,
        metavar="NAME",
        help="Count records in collection/table"
    )
    db_parser.add_argument(
        "--schema",
        type=str,
        metavar="NAME",
        help="Show schema for collection/table"
    )
    db_parser.add_argument(
        "--query",
        type=str,
        metavar="SQL",
        help="Run SQL query (PostgreSQL only)"
    )
    db_parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics"
    )

    return parser


def get_pipeline_status(logger) -> Dict[str, Any]:
    """
    Get the current status of all data stores in the pipeline.

    Returns:
        dict: Status information for each data store.
    """
    import os

    status = {}

    # MongoDB Datalake status
    try:
        from italian_real_estate.core.connections import get_mongodb_client
        from italian_real_estate.config.settings import (
            MONGODB_HOST, MONGODB_PORT, MONGODB_DATALAKE_NAME, COLLECTION_NAMES
        )
        client = get_mongodb_client(MONGODB_HOST, MONGODB_PORT)
        db = client[MONGODB_DATALAKE_NAME]
        status["mongodb_datalake"] = {
            "connected": True,
            "collections": {name: db[name].count_documents({}) for name in COLLECTION_NAMES}
        }
        client.close()
    except Exception as e:
        status["mongodb_datalake"] = {"connected": False, "error": str(e)}

    # MongoDB Warehouse status
    try:
        from italian_real_estate.core.connections import get_mongodb_client
        from italian_real_estate.config.settings import (
            MONGODB_HOST, MONGODB_PORT, MONGODB_WAREHOUSE_NAME, COLLECTION_NAMES
        )
        client = get_mongodb_client(MONGODB_HOST, MONGODB_PORT)
        db = client[MONGODB_WAREHOUSE_NAME]
        status["mongodb_warehouse"] = {
            "connected": True,
            "collections": {name: db[name].count_documents({}) for name in COLLECTION_NAMES}
        }
        client.close()
    except Exception as e:
        status["mongodb_warehouse"] = {"connected": False, "error": str(e)}

    # PostgreSQL status
    try:
        import psycopg2
        from italian_real_estate.config.settings import POSTGRES_CONNECTION_PARAMS
        conn = psycopg2.connect(**POSTGRES_CONNECTION_PARAMS)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM fact_listing")
        fact_count = cursor.fetchone()[0]
        status["postgresql"] = {
            "connected": True,
            "fact_table_records": fact_count
        }
        cursor.close()
        conn.close()
    except Exception as e:
        status["postgresql"] = {"connected": False, "error": str(e)}

    # Check data files
    data_files = [
        "data/real_data.parquet",
        "data/synthetic_data.parquet",
        "data/synthetic_data.csv",  # Legacy
        "data/dashboard_data.csv",
    ]
    status["data_files"] = {}
    for filepath in data_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            status["data_files"][filepath] = {"exists": True, "size_mb": size / (1024 * 1024)}
        else:
            status["data_files"][filepath] = {"exists": False}

    return status


def print_pipeline_status(status: Dict[str, Any], logger) -> None:
    """Print formatted pipeline status."""
    logger.info("=" * 70)
    logger.info("Italian Real Estate Pipeline Status")
    logger.info("=" * 70)

    # MongoDB Datalake
    logger.info("\n[Stage 1: SCRAPE] MongoDB Datalake")
    logger.info("-" * 40)
    dl = status.get("mongodb_datalake", {})
    if dl.get("connected"):
        for coll, count in dl.get("collections", {}).items():
            logger.info(f"  {coll}: {count:,} documents")
    else:
        logger.info(f"  Not connected: {dl.get('error', 'Unknown error')}")

    # MongoDB Warehouse
    logger.info("\n[Stage 2: ETL] MongoDB Warehouse")
    logger.info("-" * 40)
    wh = status.get("mongodb_warehouse", {})
    if wh.get("connected"):
        for coll, count in wh.get("collections", {}).items():
            logger.info(f"  {coll}: {count:,} documents")
    else:
        logger.info(f"  Not connected: {wh.get('error', 'Unknown error')}")

    # PostgreSQL
    logger.info("\n[Stage 3: MIGRATE] PostgreSQL")
    logger.info("-" * 40)
    pg = status.get("postgresql", {})
    if pg.get("connected"):
        logger.info(f"  fact_listing: {pg.get('fact_table_records', 0):,} records")
    else:
        logger.info(f"  Not connected: {pg.get('error', 'Unknown error')}")

    # Data files
    logger.info("\n[Stages 4-5: DATA EXPORT/TRAIN] Data Files")
    logger.info("-" * 40)
    for filepath, info in status.get("data_files", {}).items():
        if info.get("exists"):
            logger.info(f"  {filepath}: {info.get('size_mb', 0):.2f} MB")
        else:
            logger.info(f"  {filepath}: Not found")

    logger.info("\n" + "=" * 70)


def confirm_stage(stage: PipelineStage, auto_confirm: bool) -> bool:
    """
    Prompt user to confirm running a stage.

    Args:
        stage: Pipeline stage to confirm.
        auto_confirm: If True, skip confirmation.

    Returns:
        bool: True if confirmed, False otherwise.
    """
    if auto_confirm:
        return True

    info = STAGE_INFO[stage]
    print(f"\n{'=' * 60}")
    print(f"Stage: {info.name}")
    print(f"Description: {info.description}")
    print(f"Source: {info.source} -> Destination: {info.destination}")
    print(f"{'=' * 60}")

    while True:
        response = input("Continue? [Y/n/q] ").strip().lower()
        if response in ("", "y", "yes"):
            return True
        elif response in ("n", "no"):
            print("Skipping this stage...")
            return False
        elif response in ("q", "quit"):
            print("Quitting pipeline...")
            sys.exit(0)
        else:
            print("Please enter 'y' (yes), 'n' (no), or 'q' (quit)")


def run_scrape_stage(args: argparse.Namespace, logger) -> int:
    """Run the web scraping stage."""
    from italian_real_estate.scraping.datalake_populator import (
        datalake_populate_links_province,
        get_all_provinces,
        log_datalake_statistics,
    )
    from italian_real_estate.utils.progress import ParallelExecutor

    # Determine provinces
    if hasattr(args, 'all_provinces') and args.all_provinces:
        provinces = get_all_provinces()
    elif hasattr(args, 'province') and args.province:
        provinces = [args.province]
    else:
        provinces = get_all_provinces()

    # Determine listing types
    listing_type = getattr(args, 'listing_type', 'all')
    if listing_type == "all":
        listing_types = ["rent", "auction", "sale"]
    else:
        listing_types = [listing_type]

    logger.info(f"Scraping {len(provinces)} provinces, listing types: {listing_types}")

    # Create all jobs (province, listing_type combinations)
    jobs = [(prov, lt) for prov in provinces for lt in listing_types]

    def scrape_job(prov_type: tuple) -> None:
        """Scrape a single (province, listing_type) pair into the datalake."""
        province, listing_type = prov_type
        datalake_populate_links_province(province, listing_type)

    show_progress = not getattr(args, 'no_progress', False)
    workers = getattr(args, 'workers', 8)

    executor = ParallelExecutor(
        max_workers=workers,
        title=f"Scraping - {len(jobs)} jobs",
        show_progress=show_progress
    )

    results = executor.run(
        items=jobs,
        job_func=scrape_job,
        job_name_func=lambda j: f"{j[0][:15]}-{j[1]}"
    )

    log_datalake_statistics()
    return 0


def run_etl_stage(args: argparse.Namespace, logger) -> int:
    """Run the ETL stage."""
    from italian_real_estate.etl.warehouse_loader import migrate, get_warehouse_stats
    from italian_real_estate.etl.transformers import fix_empty_child_listings

    # Determine listing types
    if hasattr(args, 'all') and args.all:
        listing_types = ["rent", "auction", "sale"]
    elif hasattr(args, 'listing_type') and args.listing_type:
        listing_types = [args.listing_type]
    else:
        listing_types = ["rent", "auction", "sale"]

    logger.info(f"Running ETL for listing types: {listing_types}")

    for listing_type in listing_types:
        logger.info(f"Processing {listing_type} listings...")
        migrate(listing_type)

    # Run cleaning unless skipped
    skip_cleaning = getattr(args, 'skip_cleaning', False)
    if not skip_cleaning:
        logger.info("Running data cleaning task...")
        fix_empty_child_listings()

    get_warehouse_stats()
    return 0


def run_migrate_stage(args: argparse.Namespace, logger) -> int:
    """Run the PostgreSQL migration stage."""
    from italian_real_estate.migration import (
        setup_batch_processing,
        process_batch,
        load_fact_table_for_batch,
        finalize_migration,
    )
    from italian_real_estate.utils.progress import ParallelExecutor

    # Import translation for later use
    from run_migration import run_translation

    batch_size = getattr(args, 'batch_size', 10000)
    max_batches = getattr(args, 'max_batches', None)
    skip_translation = getattr(args, 'skip_translation', False)
    workers = getattr(args, 'workers', 8)
    show_progress = not getattr(args, 'no_progress', False)

    logger.info(f"Setting up migration (batch_size={batch_size})...")
    batch_info = setup_batch_processing(batch_size=batch_size)
    total_batches = batch_info.get("num_batches", 0)

    if max_batches:
        total_batches = min(total_batches, max_batches)

    logger.info(f"Processing {total_batches} batches...")

    def process_single_batch(batch_num: int) -> Dict[str, Any]:
        """Process one migration batch and return bookkeeping stats."""
        dimension_mappings = process_batch(batch_num)
        records_loaded = load_fact_table_for_batch(batch_num, dimension_mappings)
        return {"batch_num": batch_num, "records_loaded": records_loaded}

    batch_numbers = list(range(total_batches))

    executor = ParallelExecutor(
        max_workers=workers,
        title="Migration - Processing Batches",
        show_progress=show_progress
    )

    results = executor.run(
        items=batch_numbers,
        job_func=process_single_batch,
        job_name_func=lambda b: f"Batch {b}",
        on_complete=lambda b, r: r.get("records_loaded", 0)
    )

    total_records = sum(r.get("records_loaded", 0) for r in results.values())
    logger.info(f"Total records loaded: {total_records:,}")

    # Translation
    if not skip_translation:
        logger.info("Running translation step...")
        run_translation(logger)

    finalize_migration()
    return 0


def run_data_export_stage(args: argparse.Namespace, logger) -> int:
    """Run the data export stage (real or synthetic)."""
    # Determine data type
    is_real = getattr(args, 'real', False)

    if is_real:
        return _run_real_data_export(args, logger)
    else:
        return _run_synthetic_data_export(args, logger)


def _run_real_data_export(args: argparse.Namespace, logger) -> int:
    """Export real data from PostgreSQL to Parquet."""
    from italian_real_estate.synthetic_data import export_real_data

    output_path = getattr(args, 'output', None) or 'data/real_data.parquet'

    logger.info(f"Exporting real data to {output_path}...")
    df = export_real_data(output_path)
    logger.info(f"Exported {len(df):,} records to {output_path}")

    return 0


def _run_synthetic_data_export(args: argparse.Namespace, logger) -> int:
    """Generate synthetic data and export to Parquet."""
    from italian_real_estate.synthetic_data import (
        extract_data_from_postgres,
        extract_features_from_postgres,
        merge_features_with_data,
        preprocess_data,
        split_by_listing_type,
        process_dataset,
        generate_synthetic_data_in_chunks,
        postprocess_synthetic_data,
        combine_synthetic_data,
        configure_gpu_memory_growth,
        get_gpu_info,
        get_rounded_listing_counts,
    )

    output_path = getattr(args, 'output', None) or 'data/synthetic_data.parquet'

    # Auto-calculate sample counts from database
    logger.info("Calculating sample counts from database...")
    counts = get_rounded_listing_counts()
    num_rent = counts["rent"]
    num_auction = counts["auction"]
    num_sale = counts["sale"]
    logger.info(f"Sample counts: rent={num_rent:,}, auction={num_auction:,}, sale={num_sale:,}")

    # GPU setup
    gpu_info = get_gpu_info()
    logger.info(f"GPU available: {gpu_info['gpu_available']}")
    if gpu_info['gpu_available']:
        configure_gpu_memory_growth()

    # Extract and preprocess data
    logger.info("Extracting data from PostgreSQL...")
    raw_data = extract_data_from_postgres()
    features_df = extract_features_from_postgres()
    data_with_features = merge_features_with_data(raw_data, features_df)

    logger.info("Preprocessing data...")
    preprocessed_data, numerical_columns, categorical_columns = preprocess_data(data_with_features)

    # Split by listing type
    rent_data, auction_data, sale_data = split_by_listing_type(preprocessed_data)

    # Remove listing_type from categorical_columns since it's dropped after split
    categorical_columns = [c for c in categorical_columns if c != 'listing_type']

    # Process datasets for KNN
    distance_columns = ["price", "latitude", "longitude", "surface"]

    logger.info("Processing datasets for KNN...")
    rent_transformed, rent_cat_idx, _, rent_num_trans, rent_cats = process_dataset(
        rent_data, "rent", numerical_columns, categorical_columns
    )
    auction_transformed, auction_cat_idx, _, auction_num_trans, auction_cats = process_dataset(
        auction_data, "auction", numerical_columns, categorical_columns
    )
    sale_transformed, sale_cat_idx, _, sale_num_trans, sale_cats = process_dataset(
        sale_data, "sale", numerical_columns, categorical_columns
    )

    # Generate synthetic data
    logger.info(f"Generating {num_rent:,} synthetic rent samples...")
    synthetic_rent = generate_synthetic_data_in_chunks(
        rent_data, rent_transformed, rent_transformed[:, :len(distance_columns)],
        num_rent, rent_cat_idx, rent_cats, rent_num_trans, numerical_columns, categorical_columns
    )

    logger.info(f"Generating {num_auction:,} synthetic auction samples...")
    synthetic_auction = generate_synthetic_data_in_chunks(
        auction_data, auction_transformed, auction_transformed[:, :len(distance_columns)],
        num_auction, auction_cat_idx, auction_cats, auction_num_trans, numerical_columns, categorical_columns
    )

    logger.info(f"Generating {num_sale:,} synthetic sale samples...")
    synthetic_sale = generate_synthetic_data_in_chunks(
        sale_data, sale_transformed, sale_transformed[:, :len(distance_columns)],
        num_sale, sale_cat_idx, sale_cats, sale_num_trans, numerical_columns, categorical_columns
    )

    # Postprocess and combine
    synthetic_rent = postprocess_synthetic_data(synthetic_rent)
    synthetic_auction = postprocess_synthetic_data(synthetic_auction)
    synthetic_sale = postprocess_synthetic_data(synthetic_sale)

    combined_data = combine_synthetic_data(synthetic_rent, synthetic_auction, synthetic_sale)

    # Save to Parquet
    logger.info(f"Saving {len(combined_data):,} records to {output_path}...")
    combined_data.to_parquet(output_path, index=False, engine='pyarrow')

    return 0


def run_train_stage(args: argparse.Namespace, logger) -> int:
    """Run the ML training stage."""
    import os
    import pandas as pd
    from italian_real_estate.ml import (
        engineer_all_features,
        prepare_rent_training_data,
        train_rent_model,
        evaluate_model,
        print_metrics,
        create_dashboard_data,
    )

    input_path = getattr(args, 'input', None)
    output_dashboard = getattr(args, 'output_dashboard', 'data/dashboard_data.csv')
    test_size = getattr(args, 'test_size', 0.3)

    # Auto-detect input file if not specified
    if not input_path:
        # Preference order: synthetic parquet, real parquet, synthetic csv
        candidates = [
            "data/synthetic_data.parquet",
            "data/real_data.parquet",
            "data/synthetic_data.csv",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                input_path = candidate
                break
        if not input_path:
            logger.error("No data file found. Run data-export stage first.")
            return 1

    logger.info(f"Loading data from {input_path}...")
    if input_path.endswith('.parquet'):
        data = pd.read_parquet(input_path, engine='pyarrow')
    else:
        data = pd.read_csv(input_path)
    logger.info(f"Loaded {len(data):,} records")

    logger.info("Applying feature engineering...")
    engineered_data = engineer_all_features(data)

    logger.info("Preparing training data...")
    X, Y = prepare_rent_training_data(engineered_data)
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")

    logger.info("Training RandomForest model...")
    model, X_train, X_test, Y_train, Y_test = train_rent_model(
        X, Y, test_size=test_size, random_state=2025
    )

    logger.info("Evaluating model...")
    Y_pred = model.predict(X_test)
    metrics = evaluate_model(Y_test, Y_pred)
    print_metrics(metrics)

    logger.info("Creating dashboard data...")
    dashboard_data = create_dashboard_data(model, engineered_data, output_path=output_dashboard)
    logger.info(f"Dashboard saved to {output_dashboard} ({len(dashboard_data)} records)")

    return 0


STAGE_RUNNERS = {
    PipelineStage.SCRAPE: run_scrape_stage,
    PipelineStage.ETL: run_etl_stage,
    PipelineStage.MIGRATE: run_migrate_stage,
    PipelineStage.DATA_EXPORT: run_data_export_stage,
    PipelineStage.TRAIN: run_train_stage,
}


def get_stages_to_run(args: argparse.Namespace) -> List[PipelineStage]:
    """
    Determine which stages to run based on arguments.

    Args:
        args: Parsed arguments.

    Returns:
        List of stages to run in order.
    """
    all_stages = list(PipelineStage)

    # If specific command given, run that stage only
    if args.command and args.command not in ("all", "status", "db"):
        return [PipelineStage(args.command)]

    # If 'all' command or --stages or --from/--to
    if args.command == "all":
        return all_stages

    # Range selection
    if args.from_stage or args.to_stage:
        start_idx = 0
        end_idx = len(all_stages)

        if args.from_stage:
            start_idx = [s.value for s in all_stages].index(args.from_stage)
        if args.to_stage:
            end_idx = [s.value for s in all_stages].index(args.to_stage) + 1

        return all_stages[start_idx:end_idx]

    # Comma-separated list
    if args.stages:
        stage_names = [s.strip() for s in args.stages.split(",")]
        return [PipelineStage(name) for name in stage_names]

    # Default: show help
    return []


def main() -> int:
    """
    Main entry point for the pipeline orchestrator.

    If no arguments are provided, launches the interactive TUI.
    Otherwise, runs in CLI mode with the provided arguments.

    Returns:
        int: Exit code.
    """
    # If no arguments provided, launch TUI
    if len(sys.argv) == 1:
        from italian_real_estate.utils.tui import PipelineTUI
        tui = PipelineTUI()
        return tui.run()

    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    from italian_real_estate.config.logging_config import setup_logging
    import logging

    if args.verbose:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)

    from italian_real_estate.config.logging_config import get_logger
    logger = get_logger(__name__)

    # Handle status command
    if args.command == "status":
        status = get_pipeline_status(logger)
        print_pipeline_status(status, logger)
        return 0

    # Handle setup command
    if args.command == "setup":
        from setup_databases import (
            setup_mongodb, setup_postgresql,
            check_mongodb_status, check_postgresql_status,
            get_mongodb_config, get_postgres_credentials,
            display_status, display_setup_results
        )
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print(Panel.fit(
            "[bold]Italian Real Estate Database Setup[/bold]",
            border_style="blue"
        ))

        # Determine which databases to work with
        do_mongodb = not args.postgres or args.mongodb
        do_postgres = not args.mongodb or args.postgres
        if not args.postgres and not args.mongodb:
            do_mongodb = do_postgres = True

        if args.check:
            # Check status only
            console.print("\n[bold]Checking database status...[/bold]\n")

            mongo_status = check_mongodb_status() if do_mongodb else {}
            pg_status = check_postgresql_status() if do_postgres else {}

            if do_mongodb and do_postgres:
                display_status(mongo_status, pg_status)
            elif do_mongodb:
                display_status(mongo_status, {"connected": False, "error": "Skipped"})
            else:
                display_status({"connected": False, "error": "Skipped"}, pg_status)

            return 0

        # Setup databases
        if do_mongodb:
            console.print("\n[bold]Setting up MongoDB...[/bold]")
            mongo_config = get_mongodb_config(force_prompt=args.reconfigure)
            mongo_results = setup_mongodb(mongo_config)
            display_setup_results(mongo_results, "MongoDB")

        if do_postgres:
            console.print("\n[bold]Setting up PostgreSQL...[/bold]")
            pg_credentials = get_postgres_credentials(
                force_prompt=args.reconfigure,
                interactive=True
            )
            pg_results = setup_postgresql(pg_credentials)
            display_setup_results(pg_results, "PostgreSQL")

        console.print("\n[bold green]Setup complete![/bold green]")
        console.print("Run with --check to verify database status.\n")

        return 0

    # Handle db command
    if args.command == "db":
        from italian_real_estate.utils.db_explorer import run_explorer, run_single_command

        # Check if any non-interactive option is specified
        has_query = any([
            getattr(args, 'collections', False),
            getattr(args, 'sample', None),
            getattr(args, 'count', None),
            getattr(args, 'schema', None),
            getattr(args, 'query', None),
            getattr(args, 'stats', False),
        ])

        if has_query:
            # Non-interactive mode - requires database
            database = getattr(args, 'database', None)
            if not database:
                logger.error("Database required for non-interactive mode. Use --database/-d")
                return 1

            if args.collections:
                return run_single_command(database, "collections", [])
            elif args.sample:
                return run_single_command(database, "sample", args.sample)
            elif args.count:
                return run_single_command(database, "count", [args.count])
            elif args.schema:
                return run_single_command(database, "schema", [args.schema])
            elif args.query:
                return run_single_command(database, "query", [args.query])
            elif args.stats:
                return run_single_command(database, "stats", [])
        else:
            # Interactive mode
            run_explorer()
            return 0

    # Determine stages to run
    stages = get_stages_to_run(args)

    if not stages:
        parser.print_help()
        return 1

    # Dry run
    if args.dry_run:
        logger.info("Dry run mode - showing stages that would run:")
        for i, stage in enumerate(stages, 1):
            info = STAGE_INFO[stage]
            logger.info(f"  {i}. {info.name}: {info.source} -> {info.destination}")
        return 0

    # Run stages
    logger.info(f"Running {len(stages)} pipeline stage(s)...")
    start_time = time.time()

    try:
        for stage in stages:
            # Confirm before running (unless -y flag)
            if not confirm_stage(stage, args.yes):
                continue

            info = STAGE_INFO[stage]
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting: {info.name}")
            logger.info(f"{'='*60}")

            stage_start = time.time()
            runner = STAGE_RUNNERS[stage]
            exit_code = runner(args, logger)

            stage_elapsed = time.time() - stage_start
            logger.info(f"Stage completed in {stage_elapsed:.1f} seconds")

            if exit_code != 0:
                logger.error(f"Stage {stage.value} failed with exit code {exit_code}")
                return exit_code

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline completed successfully in {elapsed:.1f} seconds")
        logger.info(f"{'='*60}")
        return 0

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

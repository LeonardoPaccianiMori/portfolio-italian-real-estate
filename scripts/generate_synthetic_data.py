#!/usr/bin/env python3
"""
CLI script for generating synthetic real estate data.

This script provides a command-line interface for generating synthetic
real estate listings using K-Nearest Neighbors interpolation. The synthetic
data is used to augment the training data for machine learning models.

Usage:
    python generate_synthetic_data.py --num-rent 80000 --num-sale 850000
    python generate_synthetic_data.py --output data/synthetic_data.csv

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

from italian_real_estate.config.settings import (
    SYNTHETIC_DATA_CHUNK_SIZE,
    SYNTHETIC_DATA_BATCH_SIZE,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic real estate listings using KNN interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate with default sample counts
    python generate_synthetic_data.py

    # Specify sample counts for each listing type
    python generate_synthetic_data.py --num-rent 80000 --num-auction 120000 --num-sale 850000

    # Specify output file
    python generate_synthetic_data.py --output data/my_synthetic_data.csv

    # Use specific chunk and batch sizes for GPU memory management
    python generate_synthetic_data.py --chunk-size 5000 --batch-size 25
        """
    )

    # Sample count arguments
    parser.add_argument(
        "--num-rent",
        type=int,
        default=80000,
        help="Number of synthetic rent samples to generate (default: 80000)"
    )
    parser.add_argument(
        "--num-auction",
        type=int,
        default=120000,
        help="Number of synthetic auction samples to generate (default: 120000)"
    )
    parser.add_argument(
        "--num-sale",
        type=int,
        default=850000,
        help="Number of synthetic sale samples to generate (default: 850000)"
    )

    # Output file
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/synthetic_data.csv",
        help="Output file path (default: data/synthetic_data.csv)"
    )

    # Processing parameters
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=SYNTHETIC_DATA_CHUNK_SIZE,
        help=f"Chunk size for batch processing (default: {SYNTHETIC_DATA_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SYNTHETIC_DATA_BATCH_SIZE,
        help=f"Batch size for GPU processing (default: {SYNTHETIC_DATA_BATCH_SIZE})"
    )

    # KNN parameters
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN interpolation (default: 5)"
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually generating data"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the synthetic data generation CLI.

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

    # Log configuration
    logger.info("Synthetic Data Generation Configuration:")
    logger.info(f"  Rent samples: {args.num_rent}")
    logger.info(f"  Auction samples: {args.num_auction}")
    logger.info(f"  Sale samples: {args.num_sale}")
    logger.info(f"  Output file: {args.output}")
    logger.info(f"  Chunk size: {args.chunk_size}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  K neighbors: {args.k_neighbors}")

    total_samples = args.num_rent + args.num_auction + args.num_sale
    logger.info(f"  Total samples: {total_samples}")

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - no data will be generated")
        return 0

    try:
        # Imports synthetic data generation functions from the module.
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
        )

        # Step 1: Check GPU availability.
        gpu_info = get_gpu_info()
        logger.info(f"TensorFlow version: {gpu_info['tensorflow_version']}")
        logger.info(f"GPU available: {gpu_info['gpu_available']}")
        if gpu_info['gpu_available']:
            logger.info(f"GPU count: {gpu_info['gpu_count']}")
            configure_gpu_memory_growth()

        # Step 2: Extract data from PostgreSQL.
        logger.info("Extracting data from PostgreSQL...")
        raw_data = extract_data_from_postgres()
        logger.info(f"Extracted {len(raw_data)} records")

        # Step 3: Extract and merge features.
        logger.info("Extracting features...")
        features_df = extract_features_from_postgres()
        data_with_features = merge_features_with_data(raw_data, features_df)
        logger.info(f"Merged features. Shape: {data_with_features.shape}")

        # Step 4: Preprocess data.
        logger.info("Preprocessing data...")
        preprocessed_data, numerical_columns, categorical_columns = preprocess_data(
            data_with_features
        )
        logger.info(f"Preprocessed data. Shape: {preprocessed_data.shape}")

        # Step 5: Split by listing type.
        rent_data, auction_data, sale_data = split_by_listing_type(preprocessed_data)

        # Step 6: Process each dataset for KNN.
        logger.info("Processing datasets for KNN...")

        # Defines distance columns for KNN (price, location, surface).
        distance_columns = ["price", "latitude", "longitude", "surface"]

        # Process rent data.
        logger.info("Processing rent data...")
        rent_transformed, rent_cat_idx, _, rent_num_trans, rent_cats = process_dataset(
            rent_data, "rent", numerical_columns, categorical_columns
        )

        # Process auction data.
        logger.info("Processing auction data...")
        auction_transformed, auction_cat_idx, _, auction_num_trans, auction_cats = process_dataset(
            auction_data, "auction", numerical_columns, categorical_columns
        )

        # Process sale data.
        logger.info("Processing sale data...")
        sale_transformed, sale_cat_idx, _, sale_num_trans, sale_cats = process_dataset(
            sale_data, "sale", numerical_columns, categorical_columns
        )

        # Step 7: Generate synthetic data for each listing type.
        logger.info(f"Generating {args.num_rent} synthetic rent samples...")
        synthetic_rent = generate_synthetic_data_in_chunks(
            rent_data,
            rent_transformed,
            rent_transformed[:, :len(distance_columns)],
            args.num_rent,
            rent_cat_idx,
            rent_cats,
            rent_num_trans,
            numerical_columns,
            categorical_columns,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size
        )

        logger.info(f"Generating {args.num_auction} synthetic auction samples...")
        synthetic_auction = generate_synthetic_data_in_chunks(
            auction_data,
            auction_transformed,
            auction_transformed[:, :len(distance_columns)],
            args.num_auction,
            auction_cat_idx,
            auction_cats,
            auction_num_trans,
            numerical_columns,
            categorical_columns,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size
        )

        logger.info(f"Generating {args.num_sale} synthetic sale samples...")
        synthetic_sale = generate_synthetic_data_in_chunks(
            sale_data,
            sale_transformed,
            sale_transformed[:, :len(distance_columns)],
            args.num_sale,
            sale_cat_idx,
            sale_cats,
            sale_num_trans,
            numerical_columns,
            categorical_columns,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size
        )

        # Step 8: Postprocess synthetic data.
        logger.info("Postprocessing synthetic data...")
        synthetic_rent = postprocess_synthetic_data(synthetic_rent)
        synthetic_auction = postprocess_synthetic_data(synthetic_auction)
        synthetic_sale = postprocess_synthetic_data(synthetic_sale)

        # Step 9: Combine all synthetic data.
        logger.info("Combining synthetic data...")
        combined_data = combine_synthetic_data(
            synthetic_rent,
            synthetic_auction,
            synthetic_sale
        )

        # Step 10: Save to CSV.
        logger.info(f"Saving synthetic data to {args.output}...")
        combined_data.to_csv(args.output, index=False)

        logger.info(f"Synthetic data saved to: {args.output}")
        logger.info(f"Total records: {len(combined_data)}")
        logger.info("Synthetic data generation completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Generation failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

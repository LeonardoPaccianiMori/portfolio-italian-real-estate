#!/usr/bin/env python3
"""
CLI script for training the rent prediction machine learning model.

This script provides a command-line interface for training a RandomForest
model to predict rent values based on property characteristics. The trained
model can then be applied to sale/auction listings to estimate potential
rental income.

Usage:
    python train_model.py --input data/synthetic_data.csv
    python train_model.py --test-size 0.3 --random-state 2025

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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a rent prediction model using synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default settings
    python train_model.py

    # Specify input data file
    python train_model.py --input data/my_synthetic_data.csv

    # Specify test size and random state
    python train_model.py --test-size 0.3 --random-state 2025

    # Specify output dashboard file
    python train_model.py --output-dashboard data/dashboard_data.csv

    # Enable evaluation plots
    python train_model.py --show-plots
        """
    )

    # Input/Output files
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/synthetic_data.csv",
        help="Input synthetic data file (default: data/synthetic_data.csv)"
    )
    parser.add_argument(
        "--output-dashboard",
        type=str,
        default="data/dashboard_data.csv",
        help="Output dashboard data file (default: data/dashboard_data.csv)"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        help="Output path to save the trained model (optional)"
    )

    # Model parameters
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Proportion of data to use for testing (default: 0.3)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=2025,
        help="Random state for reproducibility (default: 2025)"
    )

    # RandomForest parameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the forest (default: 100)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of trees (default: None = unlimited)"
    )

    # Visualization
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display evaluation plots"
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        default=None,
        help="Directory to save evaluation plots"
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
        help="Show configuration without training"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the model training CLI.

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
    logger.info("Model Training Configuration:")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Output dashboard: {args.output_dashboard}")
    logger.info(f"  Test size: {args.test_size}")
    logger.info(f"  Random state: {args.random_state}")
    logger.info(f"  N estimators: {args.n_estimators}")
    logger.info(f"  Max depth: {args.max_depth}")

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - no training will be performed")
        return 0

    try:
        # Imports ML functions from the ml module.
        import pandas as pd

        from italian_real_estate.ml import (
            engineer_all_features,
            prepare_rent_training_data,
            train_rent_model,
            evaluate_model,
            print_metrics,
            plot_predictions,
            plot_residuals,
            plot_feature_importance,
            save_model,
            create_dashboard_data,
        )

        # Step 1: Load synthetic data.
        logger.info(f"Loading synthetic data from {args.input}...")
        synthetic_data = pd.read_csv(args.input)
        logger.info(f"Loaded {len(synthetic_data)} records")

        # Step 2: Apply feature engineering.
        logger.info("Applying feature engineering...")
        engineered_data = engineer_all_features(synthetic_data)
        logger.info(f"Feature engineering complete. Shape: {engineered_data.shape}")

        # Step 3: Prepare training data (filters to rent, removes outliers, encodes).
        logger.info("Preparing training data...")
        X, Y = prepare_rent_training_data(engineered_data)
        logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")

        # Step 4: Train the model.
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'random_state': args.random_state,
        }

        logger.info("Training RandomForest model...")
        model, X_train, X_test, Y_train, Y_test = train_rent_model(
            X, Y,
            test_size=args.test_size,
            random_state=args.random_state,
            model_params=model_params
        )
        logger.info("Model training complete")

        # Step 5: Evaluate the model.
        logger.info("Evaluating model performance...")
        Y_pred = model.predict(X_test)
        metrics = evaluate_model(Y_test, Y_pred)
        print_metrics(metrics)

        # Step 6: Generate and save plots if requested.
        if args.show_plots or args.save_plots:
            pred_path = f"{args.save_plots}/predicted_vs_actual.png" if args.save_plots else None
            resid_path = f"{args.save_plots}/residuals.png" if args.save_plots else None
            import_path = f"{args.save_plots}/feature_importance.png" if args.save_plots else None

            plot_predictions(Y_test, Y_pred, save_path=pred_path, show=args.show_plots)
            plot_residuals(Y_test, Y_pred, save_path=resid_path, show=args.show_plots)
            plot_feature_importance(
                model, X_test, Y_test,
                save_path=import_path,
                show=args.show_plots
            )

        # Step 7: Save the model if requested.
        if args.output_model:
            save_model(model, args.output_model, X.columns.tolist())
            logger.info(f"Model saved to {args.output_model}")

        # Step 8: Create dashboard data with predicted rent.
        logger.info("Creating dashboard data with predicted rent...")
        dashboard_data = create_dashboard_data(
            model,
            engineered_data,
            output_path=args.output_dashboard
        )
        logger.info(f"Dashboard data saved to {args.output_dashboard}")
        logger.info(f"Dashboard has {len(dashboard_data)} sale/auction listings")

        logger.info("Model training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Rent prediction for sale and auction listings.

This module provides functions for applying the trained rent prediction
model to sale and auction listings to generate dashboard-ready data.

The key workflow is:
1. Load trained model
2. Engineer features for sale/auction data
3. Predict rent prices
4. Generate dashboard CSV with predicted rents

Author: Leonardo Pacciani-Mori
License: MIT
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config.logging_config import get_logger
from .data_preparation import one_hot_encode_at_position, DEFAULT_CATEGORICAL_COLUMNS

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def predict_rent_for_listings(
    model: RandomForestRegressor,
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    round_to_decade: bool = True
) -> np.ndarray:
    """
    Predict rent for listings using the trained model.

    This function prepares features for sale/auction listings and
    uses the trained model to predict what rent they could command.

    Args:
        model: Trained RandomForestRegressor model.
        df: DataFrame of listings (already feature-engineered but not
            one-hot encoded).
        categorical_columns: List of categorical columns to encode.
            Defaults to DEFAULT_CATEGORICAL_COLUMNS.
        round_to_decade: Whether to round predictions to nearest decade.
            Default is True.

    Returns:
        Array of predicted rent values.

    Example:
        >>> predictions = predict_rent_for_listings(model, sale_data)
        >>> print(f"Mean predicted rent: {predictions.mean():.2f}")
    """
    # Uses default categorical columns if not specified.
    if categorical_columns is None:
        categorical_columns = DEFAULT_CATEGORICAL_COLUMNS

    # Removes listing_type and price to get features.
    features_to_drop = ['listing_type', 'price']
    X_data = df.drop(
        columns=[col for col in features_to_drop if col in df.columns],
        errors='ignore'
    )

    # One-hot encodes categorical columns.
    # Filters to only columns that exist in the DataFrame.
    existing_cat_cols = [col for col in categorical_columns if col in X_data.columns]
    X_encoded = one_hot_encode_at_position(X_data, existing_cat_cols)

    logger.info(f"Predicting rent for {len(X_encoded)} listings...")

    # Makes predictions (model outputs log-transformed values).
    Y_pred_log = model.predict(X_encoded)

    # Inverts log transformation.
    Y_pred = np.expm1(Y_pred_log)

    # Rounds to nearest decade if requested.
    if round_to_decade:
        Y_pred = np.round(Y_pred / 10) * 10

    logger.info(f"Predictions complete. Mean: {Y_pred.mean():.2f}")

    return Y_pred


def create_dashboard_data(
    model: RandomForestRegressor,
    synthetic_data: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create dashboard-ready data with predicted rents.

    This function filters the synthetic data to sale and auction listings,
    predicts rent for each, and creates a DataFrame ready for visualization.

    The predicted rent is inserted as the first column to make it prominent
    in the output.

    Args:
        model: Trained RandomForestRegressor model.
        synthetic_data: Full synthetic data DataFrame (feature-engineered
            but not one-hot encoded).
        categorical_columns: List of categorical columns to encode.
            Defaults to DEFAULT_CATEGORICAL_COLUMNS.
        output_path: Path to save the dashboard CSV. If None, file is
            not saved.

    Returns:
        DataFrame with predicted_rent as the first column, containing
        only sale and auction listings.

    Example:
        >>> dashboard = create_dashboard_data(
        ...     model, synthetic_data,
        ...     output_path='data/dashboard_data.csv'
        ... )
        >>> print(f"Dashboard has {len(dashboard)} rows")
    """
    # Filters to sale and auction listings only.
    dashboard_data = synthetic_data[
        synthetic_data["listing_type"] != "rent"
    ].copy()

    logger.info(
        f"Creating dashboard data for {len(dashboard_data)} "
        "sale/auction listings"
    )

    # Predicts rent for these listings.
    predicted_rent = predict_rent_for_listings(
        model,
        dashboard_data,
        categorical_columns=categorical_columns
    )

    # Inserts predicted_rent as the first column.
    dashboard_data.insert(0, 'predicted_rent', predicted_rent)

    # Saves to file if path provided.
    if output_path:
        dashboard_data.to_csv(output_path, index=False)
        logger.info(f"Dashboard data saved to {output_path}")

    return dashboard_data


def get_rent_summary_by_region(
    dashboard_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Get summary statistics of predicted rent by region.

    This function aggregates the predicted rent by region, providing
    useful summary statistics for analysis.

    Args:
        dashboard_data: DataFrame with predicted_rent column.

    Returns:
        DataFrame with summary statistics per region.

    Example:
        >>> summary = get_rent_summary_by_region(dashboard_data)
        >>> print(summary.head())
    """
    if 'predicted_rent' not in dashboard_data.columns:
        logger.error("predicted_rent column not found")
        raise ValueError("Dashboard data must have predicted_rent column")

    summary = dashboard_data.groupby('region').agg({
        'predicted_rent': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).round(2)

    # Flattens column names.
    summary.columns = [
        'mean_rent', 'median_rent', 'std_rent',
        'min_rent', 'max_rent', 'count'
    ]

    summary = summary.sort_values('mean_rent', ascending=False)

    return summary


def get_rent_summary_by_listing_type(
    dashboard_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Get summary statistics of predicted rent by listing type.

    This function compares predicted rents between sale and auction
    listings.

    Args:
        dashboard_data: DataFrame with predicted_rent and listing_type columns.

    Returns:
        DataFrame with summary statistics per listing type.

    Example:
        >>> summary = get_rent_summary_by_listing_type(dashboard_data)
        >>> print(summary)
    """
    if 'predicted_rent' not in dashboard_data.columns:
        logger.error("predicted_rent column not found")
        raise ValueError("Dashboard data must have predicted_rent column")

    summary = dashboard_data.groupby('listing_type').agg({
        'predicted_rent': ['mean', 'median', 'std', 'min', 'max', 'count'],
        'price': ['mean', 'median']
    }).round(2)

    # Flattens column names.
    summary.columns = [
        'mean_rent', 'median_rent', 'std_rent', 'min_rent', 'max_rent', 'count',
        'mean_price', 'median_price'
    ]

    return summary


def calculate_rent_to_price_ratio(
    dashboard_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the annual rent-to-price ratio (gross yield).

    This function adds a column showing the estimated annual gross yield
    if the property were purchased and rented out.

    Formula: (predicted_rent * 12) / price * 100

    Args:
        dashboard_data: DataFrame with predicted_rent and price columns.

    Returns:
        DataFrame with annual_yield_pct column added.

    Example:
        >>> dashboard = calculate_rent_to_price_ratio(dashboard)
        >>> print(f"Mean yield: {dashboard['annual_yield_pct'].mean():.2f}%")
    """
    result = dashboard_data.copy()

    # Calculates annual rent.
    annual_rent = result['predicted_rent'] * 12

    # Calculates gross yield percentage.
    result['annual_yield_pct'] = (annual_rent / result['price'] * 100).round(2)

    # Handles any infinite values from zero prices.
    result['annual_yield_pct'] = result['annual_yield_pct'].replace(
        [np.inf, -np.inf], np.nan
    )

    logger.info(
        f"Calculated rent-to-price ratio. "
        f"Mean yield: {result['annual_yield_pct'].mean():.2f}%"
    )

    return result

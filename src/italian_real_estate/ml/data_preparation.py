"""
Data preparation utilities for ML model training.

This module provides functions for preparing data for machine learning,
including one-hot encoding, outlier removal, and data type casting.

The key functions are:
1. one_hot_encode_at_position: Encodes categorical columns in place
2. one_hot_encode_in_place: Alternative encoding preserving column position
3. remove_outliers: Removes extreme values based on percentiles
4. prepare_features_and_target: Prepares X and Y for training
5. cast_data_types: Ensures proper data types for all columns

Author: Leonardo Pacciani-Mori
License: MIT
"""

from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

from ..config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


# Default list of columns that should be float type.
DEFAULT_FLOAT_COLUMNS: List[str] = [
    "price", "surface", "condominium_monthly_expenses",
    "heating_yearly_expenses", "latitude", "longitude"
]


# Default categorical columns for one-hot encoding.
DEFAULT_CATEGORICAL_COLUMNS: List[str] = [
    "seller_type", "type_of_property", "condition", "category_name",
    "kitchen_status", "building_usage", "heating_type", "heating_delivery",
    "heating_power", "air_conditioning_type", "region", "province",
    "window_frames_glass", "window_frames_material"
]


def one_hot_encode_at_position(
    df: pd.DataFrame,
    categorical_columns: List[str]
) -> pd.DataFrame:
    """
    One-hot encode categorical columns and place them at original positions.

    This function performs one-hot encoding on specified categorical columns
    while preserving the original column positions. The encoded columns
    replace the original categorical column at its position.

    Args:
        df: The input DataFrame.
        categorical_columns: List of column names to one-hot encode.

    Returns:
        DataFrame with one-hot encoded columns at the positions of the
        original columns.

    Example:
        >>> df = pd.DataFrame({'type': ['A', 'B', 'A'], 'value': [1, 2, 3]})
        >>> encoded = one_hot_encode_at_position(df, ['type'])
        >>> encoded.columns.tolist()
        ['type_A', 'type_B', 'value']
    """
    # Creates a copy of the dataframe to avoid modifying the original.
    result_df = df.copy()

    # Processes each categorical column.
    for col in categorical_columns:
        # Skips if column doesn't exist.
        if col not in result_df.columns:
            logger.warning(f"Column '{col}' not found, skipping encoding")
            continue

        # Gets the index position of the current column.
        col_idx = list(result_df.columns).index(col)

        # One-hot encodes the column using pandas get_dummies.
        one_hot = pd.get_dummies(result_df[col], prefix=col)

        # Drops the original column.
        result_df = result_df.drop(col, axis=1)

        # Inserts the one-hot encoded columns at the original position.
        for i, one_hot_col in enumerate(one_hot.columns):
            result_df.insert(col_idx + i, one_hot_col, one_hot[one_hot_col])

    logger.info(f"One-hot encoded {len(categorical_columns)} columns")
    return result_df


def one_hot_encode_in_place(
    df: pd.DataFrame,
    column_name: str,
    prefix: Optional[str] = None,
    drop_first: bool = False,
    dummy_na: bool = False
) -> pd.DataFrame:
    """
    One-hot encode a single column and place results at original position.

    This function is similar to one_hot_encode_at_position but works on
    a single column and provides more options for encoding.

    Args:
        df: The DataFrame to process.
        column_name: Name of the column to one-hot encode.
        prefix: Prefix for the new column names. Defaults to column_name.
        drop_first: Whether to drop the first category (avoid dummy trap).
        dummy_na: Whether to create a column for NaN values.

    Returns:
        DataFrame with the column one-hot encoded in place.

    Example:
        >>> df = pd.DataFrame({'type': ['A', 'B', 'A', 'C']})
        >>> encoded = one_hot_encode_in_place(df, 'type')
        >>> print(encoded.columns.tolist())
        ['type_A', 'type_B', 'type_C']
    """
    # Gets the position of the column.
    if column_name not in df.columns:
        logger.warning(f"Column '{column_name}' not found")
        return df

    column_position = df.columns.get_loc(column_name)

    # Splits the DataFrame around the target column.
    df_before = df.iloc[:, :column_position]
    df_after = df.iloc[:, column_position + 1:]

    # One-hot encodes the column.
    prefix = prefix or column_name
    dummies = pd.get_dummies(
        df[column_name],
        prefix=prefix,
        drop_first=drop_first,
        dummy_na=dummy_na
    )

    # Concatenates the parts back together.
    result = pd.concat([df_before, dummies, df_after], axis=1)

    return result


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame based on percentile thresholds.

    This function removes rows where the specified column's value falls
    outside the given percentile range. This helps prevent extreme values
    from affecting model training.

    Args:
        df: The DataFrame to process.
        column: The column name to check for outliers.
        lower_percentile: Lower percentile threshold (default 0.01 = 1%).
        upper_percentile: Upper percentile threshold (default 0.99 = 99%).

    Returns:
        DataFrame with outliers removed.

    Example:
        >>> df = pd.DataFrame({'price': [100, 200, 300, 10000, 50]})
        >>> cleaned = remove_outliers(df, 'price', 0.1, 0.9)
        >>> len(cleaned)
        3
    """
    # Calculates the threshold values.
    lower = df[column].quantile(lower_percentile)
    upper = df[column].quantile(upper_percentile)

    # Filters the DataFrame.
    original_len = len(df)
    result = df[(df[column] >= lower) & (df[column] <= upper)]

    # Logs the number of rows removed.
    removed = original_len - len(result)
    logger.info(
        f"Removed {removed} outliers from '{column}' "
        f"(range: {lower:.2f} - {upper:.2f})"
    )

    return result


def cast_data_types(
    df: pd.DataFrame,
    float_columns: Optional[List[str]] = None,
    skip_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Cast DataFrame columns to appropriate data types.

    This function ensures that:
    - Specified float columns are cast to float
    - All other numeric columns (except skipped) are cast to int

    Args:
        df: The DataFrame to process.
        float_columns: List of columns to keep as float. Defaults to
            DEFAULT_FLOAT_COLUMNS.
        skip_columns: List of columns to skip (not cast). Defaults to
            ['listing_type'].

    Returns:
        DataFrame with corrected data types.

    Example:
        >>> df = cast_data_types(df)
    """
    result = df.copy()

    # Uses defaults if not specified.
    if float_columns is None:
        float_columns = DEFAULT_FLOAT_COLUMNS
    if skip_columns is None:
        skip_columns = ["listing_type"]

    # Casts the specified columns to float.
    for col in float_columns:
        if col in result.columns:
            result[col] = result[col].astype(float)

    # Gets list of all columns that should be int.
    columns_to_skip = float_columns + skip_columns
    int_columns = [col for col in result.columns if col not in columns_to_skip]

    # Converts other columns to int where possible.
    for col in int_columns:
        # Ensures the column contains numeric data before converting to int.
        if pd.api.types.is_numeric_dtype(result[col]):
            try:
                result[col] = result[col].astype(int)
            except ValueError as e:
                # This catches errors like NaN values that can't be converted.
                logger.warning(f"Could not convert column '{col}' to int: {e}")

    logger.info("Cast data types complete")
    return result


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str = 'price',
    log_transform_target: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target (Y) for model training.

    This function splits the DataFrame into features and target,
    optionally applying log transformation to the target variable.

    Args:
        df: The DataFrame containing features and target.
        target_column: Name of the target column. Default is 'price'.
        log_transform_target: Whether to apply log1p transformation to
            the target. Default is True (recommended for price prediction).

    Returns:
        Tuple of (X, Y) where X is features DataFrame and Y is target Series.

    Example:
        >>> X, Y = prepare_features_and_target(rent_data)
        >>> X.shape
        (80000, 150)
        >>> Y.shape
        (80000,)
    """
    # Splits features and target.
    X = df.drop([target_column], axis=1)
    Y = df[target_column].copy()

    # Applies log transformation if requested.
    if log_transform_target:
        Y = np.log1p(Y)
        logger.info("Applied log1p transformation to target")

    logger.info(f"Prepared features: {X.shape[1]} columns, {X.shape[0]} rows")
    return X, Y


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify one-hot encoded and numeric columns in a DataFrame.

    This function analyzes the DataFrame to determine which columns are
    one-hot encoded (binary 0/1) and which are numeric.

    Args:
        df: The DataFrame to analyze.

    Returns:
        Tuple of (one_hot_cols, numeric_cols).

    Example:
        >>> one_hot, numeric = get_column_types(X)
        >>> print(f"One-hot: {len(one_hot)}, Numeric: {len(numeric)}")
    """
    # Identifies one-hot columns (only 0 and 1 values).
    one_hot_cols = [
        col for col in df.columns
        if set(df[col].dropna().unique()).issubset({0, 1})
    ]

    # Identifies numeric columns (non-binary).
    numeric_cols = [
        col for col in df.columns
        if not set(df[col].dropna().unique()).issubset({0, 1})
    ]

    return one_hot_cols, numeric_cols


def prepare_rent_training_data(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare rent data for model training.

    This function performs the full data preparation pipeline:
    1. Filters to rent listings only
    2. Removes outliers in price and surface
    3. One-hot encodes categorical columns
    4. Casts data types
    5. Prepares features and target

    Args:
        df: DataFrame with all listings (after feature engineering).
        categorical_columns: List of categorical columns to encode.
            Defaults to DEFAULT_CATEGORICAL_COLUMNS.

    Returns:
        Tuple of (X, Y) ready for model training.

    Example:
        >>> X, Y = prepare_rent_training_data(engineered_df)
    """
    # Uses default categorical columns if not specified.
    if categorical_columns is None:
        categorical_columns = DEFAULT_CATEGORICAL_COLUMNS

    # Filters to rent listings only.
    rent_data = df[df['listing_type'] == 'rent'].drop('listing_type', axis=1)
    logger.info(f"Filtered to {len(rent_data)} rent listings")

    # Removes outliers from price and surface.
    rent_data = remove_outliers(rent_data, 'price')
    rent_data = remove_outliers(rent_data, 'surface')

    # One-hot encodes categorical columns.
    # Filters to only columns that exist in the DataFrame.
    existing_cat_cols = [col for col in categorical_columns if col in rent_data.columns]
    encoded_data = one_hot_encode_at_position(rent_data, existing_cat_cols)

    # Casts data types.
    encoded_data = cast_data_types(encoded_data)

    # Prepares features and target.
    X, Y = prepare_features_and_target(encoded_data)

    return X, Y

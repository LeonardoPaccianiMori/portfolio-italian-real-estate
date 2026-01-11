"""
Data preprocessing functions for synthetic data generation.

This module provides functions for cleaning and preparing real estate
data before synthetic data generation. This includes handling missing
values, type conversions, and categorical value mapping.

Author: Leonardo Pacciani-Mori
License: MIT
"""

from typing import List, Tuple

import pandas as pd
import numpy as np

from ..config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def preprocess_data(
    df: pd.DataFrame,
    fill_numerical_nan: bool = True
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Preprocess raw data for synthetic data generation or real data export.

    This function performs the following preprocessing steps:
    1. Removes spurious columns
    2. Fills NaN values with medians for numerical columns (if fill_numerical_nan=True)
    3. Fills NaN values with "unknown" for categorical columns
    4. Converts boolean columns to 0/1
    5. Maps energy classes to numeric values
    6. Cleans type_of_property values
    7. Removes rows with null prices/regions/provinces
    8. Removes geographic outliers

    Args:
        df: Raw DataFrame from PostgreSQL extraction.
        fill_numerical_nan: If True, fills NaN values in numerical columns
            with medians (required for synthetic data generation). If False,
            preserves NaN values (useful for real data export).

    Returns:
        Tuple containing:
        - Preprocessed DataFrame
        - List of numerical column names
        - List of categorical column names

    Example:
        >>> raw_df = extract_data_from_postgres()
        >>> # For synthetic data generation (fill NaN):
        >>> cleaned_df, num_cols, cat_cols = preprocess_data(raw_df, fill_numerical_nan=True)
        >>> # For real data export (preserve NaN):
        >>> cleaned_df, num_cols, cat_cols = preprocess_data(raw_df, fill_numerical_nan=False)
    """
    logger.info("Starting data preprocessing...")
    raw_data = df.copy()

    # Removes spurious column if present.
    if "db fullname external fixtures. id 7m" in raw_data.columns:
        raw_data = raw_data.drop(columns=["db fullname external fixtures. id 7m"])

    # Fills NaNs with medians in numerical columns (only if requested).
    if fill_numerical_nan:
        numerical_fill_columns = [
            'surface', 'condominium_monthly_expenses', 'heating_yearly_expenses',
            'total_room_number', 'bathrooms_number', 'floor', 'building_year',
            'latitude', 'longitude'
        ]
        for col in numerical_fill_columns:
            if col in raw_data.columns:
                raw_data[col] = raw_data[col].fillna(raw_data[col].median())

        # Handles negative building_year values.
        if 'building_year' in raw_data.columns:
            raw_data.loc[raw_data['building_year'] < 0, 'building_year'] = \
                raw_data['building_year'].median()
    else:
        logger.info("Preserving NaN values in numerical columns (fill_numerical_nan=False)")

    # Fills NaNs with "unknown" in categorical columns.
    categorical_fill_columns = [
        'condition', 'kitchen_status', 'building_usage',
        'heating_type', 'air_conditioning'
    ]
    for col in categorical_fill_columns:
        if col in raw_data.columns:
            raw_data[col] = raw_data[col].fillna("unknown")

    # Converts boolean columns to 0/1.
    if 'has_elevator' in raw_data.columns:
        raw_data['has_elevator'] = raw_data['has_elevator'].map(
            lambda x: 1 if x is True else 0
        )
    if 'is_zero_energy' in raw_data.columns:
        raw_data['is_zero_energy'] = raw_data['is_zero_energy'].map(
            lambda x: 1 if x is True else 0
        )

    # Converts garage to binary (1 if text present, 0 otherwise).
    if 'garage' in raw_data.columns:
        raw_data['garage'] = raw_data['garage'].apply(
            lambda x: 1 if isinstance(x, str) else 0
        )

    # Maps energy_class to numeric values.
    if 'energy_class' in raw_data.columns:
        energy_class_mapping = {
            'a4': 1, 'a3': 2, 'a2': 3, 'a1': 4, 'a+': 4,
            'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10,
            None: None
        }
        raw_data['energy_class'] = raw_data['energy_class'].map(energy_class_mapping)
        raw_data['energy_class'] = raw_data['energy_class'].fillna(
            raw_data['energy_class'].median()
        )

    # Cleans type_of_property values.
    if 'type_of_property' in raw_data.columns:
        raw_data['type_of_property'] = (
            raw_data['type_of_property'].str.split('|', n=1).str[0].str.strip()
        )
        property_replacements = {
            'attic - attic': 'attic',
            'mansarda': 'attic',
            '': 'unknown',
            '♪': 'unknown',
            'holiday in villa': 'vacation villa',
            'vacation in villa': 'vacation villa',
            'holiday in apartment': 'vacation apartment',
            'holiday in the house for holidays': 'vacation house',
            'holiday home for holidays': 'vacation house',
            'holiday home': 'vacation house',
            'vacation in the house for holidays': 'vacation house',
            'holidays in bed & breakfast': 'bed & breakfast',
            'holidays in farmhouse': 'vacation farmhouse',
            'holidays in residence': 'vacation residence',
            'warehouse or warehouse': 'warehouse',
            'land - agricultural land': 'agricultural land',
            'estate or estate': 'estate',
            'rustic - building land residential': 'rustic',
            'historical abode': 'historical residence',
            'warehouse or storage': 'storage'
        }
        raw_data['type_of_property'] = raw_data['type_of_property'].replace(
            property_replacements
        )

    # Removes rows with null values in essential columns.
    raw_data.dropna(subset=['price', 'region', 'province'], inplace=True)

    # Removes geographic outliers (points outside Italy).
    raw_data = raw_data[
        (raw_data['longitude'] >= 6) & (raw_data['longitude'] <= 19) &
        (raw_data['latitude'] >= 35) & (raw_data['latitude'] <= 48)
    ]

    # Defines numerical and categorical columns.
    numerical_columns = [
        "listing_id", "price", "surface",
        "condominium_monthly_expenses", "heating_yearly_expenses",
        "building_year", "latitude", "longitude"
    ]
    categorical_columns = [
        col for col in raw_data.columns if col not in numerical_columns
    ]

    logger.info(f"Preprocessing complete. {len(raw_data)} records remaining.")
    return raw_data, numerical_columns, categorical_columns


def one_hot_encode_in_place(
    df: pd.DataFrame,
    column_name: str,
    prefix: str = None,
    drop_first: bool = False,
    dummy_na: bool = False
) -> pd.DataFrame:
    """
    One-hot encode a column and place the resulting columns at the original position.

    This function performs pandas one-hot encoding (get_dummies) while
    preserving the column order by placing the new columns at the position
    of the original column.

    Args:
        df: The DataFrame to process.
        column_name: Name of the column to one-hot encode.
        prefix: Prefix for the new column names. Defaults to column_name.
        drop_first: Whether to drop the first category (avoid dummy variable trap).
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


def split_by_listing_type(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into separate DataFrames by listing type.

    Args:
        df: DataFrame containing all listing types.

    Returns:
        Tuple of (rent_data, auction_data, sale_data) DataFrames,
        each without the listing_type column.

    Example:
        >>> rent, auction, sale = split_by_listing_type(df)
        >>> print(f"Rent: {len(rent)}, Auction: {len(auction)}, Sale: {len(sale)}")
    """
    rent_data = df[df['listing_type'] == 'rent'].drop('listing_type', axis=1)
    auction_data = df[df['listing_type'] == 'auction'].drop('listing_type', axis=1)
    sale_data = df[df['listing_type'] == 'sale'].drop('listing_type', axis=1)

    logger.info(
        f"Split data: rent={len(rent_data)}, auction={len(auction_data)}, "
        f"sale={len(sale_data)}"
    )

    return rent_data, auction_data, sale_data

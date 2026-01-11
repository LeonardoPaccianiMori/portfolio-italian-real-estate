"""
Post-processing functions for synthetic data.

This module provides functions for cleaning and rounding synthetic data
after generation. This ensures the synthetic data has appropriate values
(e.g., integer room counts, rounded prices).

Author: Leonardo Pacciani-Mori
License: MIT
"""

from typing import List

import numpy as np
import pandas as pd

from ..config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def random_round(x: float) -> int:
    """
    Round a number, randomly deciding direction when exactly at midpoint.

    When a number is exactly at a midpoint (e.g., 2.5), standard rounding
    can introduce bias. This function randomly rounds up or down with
    equal probability at midpoints.

    Args:
        x: The number to round.

    Returns:
        The rounded integer value.

    Example:
        >>> random_round(2.5)  # Returns 2 or 3 with equal probability
        2
        >>> random_round(2.3)  # Always returns 2
        2
    """
    floor = np.floor(x)

    # If the number is at an exact midpoint.
    if x == floor + 0.5:
        # 50% chance of rounding up or down.
        return int(floor + np.random.randint(0, 2))
    else:
        return int(round(x))


def postprocess_synthetic_data(
    df: pd.DataFrame,
    float_columns: List[str] = None,
    int_columns: List[str] = None
) -> pd.DataFrame:
    """
    Post-process synthetic data by rounding and constraining values.

    This function performs the following post-processing:
    1. Converts columns to appropriate data types (float/int)
    2. Rounds price to nearest decade
    3. Rounds surface to nearest integer
    4. Rounds expenses to 2 decimal places
    5. Constrains room counts to valid ranges
    6. Constrains energy class to 1-10 range
    7. Fixes province name inconsistencies

    Args:
        df: Synthetic DataFrame to post-process.
        float_columns: List of columns to convert to float. If None, uses defaults.
        int_columns: List of columns to convert to int. If None, uses defaults.

    Returns:
        Post-processed DataFrame.

    Example:
        >>> processed_df = postprocess_synthetic_data(synthetic_df)
    """
    logger.info("Starting post-processing of synthetic data...")

    # Default float columns if not specified.
    if float_columns is None:
        float_columns = [
            "price", "surface", "condominium_monthly_expenses",
            "heating_yearly_expenses", "latitude", "longitude"
        ]

    # Default int columns if not specified.
    if int_columns is None:
        int_columns = [
            'listing_id', 'total_room_number', 'bathrooms_number', 'garage',
            'floor', 'has_elevator', 'building_year', 'is_zero_energy',
            'energy_class', 'access for disabled', 'alarm system',
            'armored door', 'attic', 'balcony', 'bathroom for disabled',
            'bike parking', 'cellar', 'central tv system', 'common garden',
            'doorman half day', 'doorman whole day', 'double exposure',
            'driveway', 'electric gate', 'external exposure', 'fenced property',
            'fiber optics', 'fireplace', 'floating floor', 'front desk',
            'furnished', 'gym', 'internal exposure', 'jacuzzi', 'kitchen',
            'land owned', 'only furnished kitchen', 'partially furnished',
            'private garden', 'supervision cctv', 'swimming pool', 'tavern',
            'tennis court', 'terrace', 'tv system with satellite dish',
            'video intercom', 'wall cabinet',
            'window frames double glass / metal exterior',
            'window frames exterior double glass / wood',
            'window frames exterior glass / metal',
            'window frames exterior glass / wood',
            'window frames exterior in triple glass / metal',
            'window frames exterior in triple glass / pvc',
            'window frames exterior in triple glass / wood',
            'window frames external double glass / pvc', 'wired'
        ]

    result = df.copy()

    # Converts listing_id to int.
    if 'listing_id' in result.columns:
        result['listing_id'] = result['listing_id'].astype(float).astype(int)

    # Converts float columns.
    logger.info("Casting numeric columns...")
    for col in float_columns:
        if col in result.columns:
            result[col] = result[col].astype(float)

    # Converts int columns.
    logger.info("Casting integer columns...")
    for col in int_columns:
        if col in result.columns:
            result[col] = result[col].astype(float).astype(int)

    # Rounds price to nearest decade.
    if 'price' in result.columns:
        result['price'] = result['price'].apply(lambda x: round(x / 10) * 10)

    # Rounds surface to nearest integer.
    if 'surface' in result.columns:
        result['surface'] = result['surface'].apply(random_round)

    # Rounds expenses to 2 decimal places.
    if 'condominium_monthly_expenses' in result.columns:
        result['condominium_monthly_expenses'] = \
            result['condominium_monthly_expenses'].round(2)

    if 'heating_yearly_expenses' in result.columns:
        result['heating_yearly_expenses'] = \
            result['heating_yearly_expenses'].round(2)

    # Constrains total_room_number between 1 and 6.
    if 'total_room_number' in result.columns:
        result['total_room_number'] = result['total_room_number'].apply(
            lambda x: np.clip(random_round(x), 1, 6)
        )

    # Constrains bathrooms_number between 1 and 4.
    if 'bathrooms_number' in result.columns:
        result['bathrooms_number'] = result['bathrooms_number'].apply(
            lambda x: np.clip(random_round(x), 1, 4)
        )

    # Rounds floor to nearest integer.
    if 'floor' in result.columns:
        result['floor'] = result['floor'].apply(random_round)

    # Rounds building_year to nearest integer.
    if 'building_year' in result.columns:
        result['building_year'] = result['building_year'].apply(random_round)

    # Constrains energy_class between 1 and 10.
    if 'energy_class' in result.columns:
        result['energy_class'] = result['energy_class'].apply(
            lambda x: np.clip(random_round(x), 1, 10)
        )

    # Fixes province name inconsistencies.
    if 'province' in result.columns:
        result['province'] = result['province'].str.replace(
            "l'aquila", "aquila", regex=False
        )
        result['province'] = result['province'].str.replace(
            "monza-e-brianza", "monza-brianza", regex=False
        )

    logger.info("Post-processing complete.")
    return result


def combine_synthetic_data(
    rent_df: pd.DataFrame,
    auction_df: pd.DataFrame,
    sale_df: pd.DataFrame,
    shuffle: bool = True,
    random_state: int = 2025
) -> pd.DataFrame:
    """
    Combine synthetic data from all listing types into one DataFrame.

    This function adds the listing_type column back to each DataFrame,
    concatenates them, and optionally shuffles the result.

    Args:
        rent_df: Synthetic rent listings DataFrame.
        auction_df: Synthetic auction listings DataFrame.
        sale_df: Synthetic sale listings DataFrame.
        shuffle: Whether to shuffle the combined DataFrame.
        random_state: Random seed for shuffling (for reproducibility).

    Returns:
        Combined DataFrame with all listing types.

    Example:
        >>> combined = combine_synthetic_data(rent, auction, sale)
        >>> print(f"Total records: {len(combined)}")
    """
    # Adds listing_type column to each DataFrame.
    rent_df = rent_df.copy()
    auction_df = auction_df.copy()
    sale_df = sale_df.copy()

    rent_df['listing_type'] = 'rent'
    auction_df['listing_type'] = 'auction'
    sale_df['listing_type'] = 'sale'

    # Repositions listing_type column after listing_id.
    for df in [rent_df, auction_df, sale_df]:
        listing_type = df.pop('listing_type')

        if 'listing_id' in df.columns:
            pos = list(df.columns).index('listing_id') + 1
        elif 'price' in df.columns:
            pos = list(df.columns).index('price')
        else:
            pos = 0

        df.insert(pos, 'listing_type', listing_type)

    # Concatenates DataFrames.
    combined = pd.concat(
        [rent_df, auction_df, sale_df],
        ignore_index=True
    )

    # Shuffles if requested.
    if shuffle:
        combined = combined.sample(frac=1, random_state=random_state).reset_index(
            drop=True
        )

    logger.info(
        f"Combined synthetic data: {len(combined)} total records "
        f"(rent={len(rent_df)}, auction={len(auction_df)}, sale={len(sale_df)})"
    )

    return combined

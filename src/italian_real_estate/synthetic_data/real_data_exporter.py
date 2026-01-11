"""
Real data export functions for the Italian Real Estate pipeline.

This module provides functions for exporting real (non-synthetic) data
from PostgreSQL to Parquet format, with preprocessing applied but
preserving NaN values in numerical columns.

Author: Leonardo Pacciani-Mori
License: MIT
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..config.settings import POSTGRES_CONNECTION_PARAMS
from ..config.logging_config import get_logger
from .data_extractor import (
    extract_data_from_postgres,
    extract_features_from_postgres,
    merge_features_with_data,
)
from .preprocessor import preprocess_data

logger = get_logger(__name__)


def round_to_myriad(n: int) -> int:
    """
    Round a number to the nearest 10,000 (myriad).

    Args:
        n: The number to round.

    Returns:
        The number rounded to the nearest 10,000.

    Examples:
        >>> round_to_myriad(75721)
        80000
        >>> round_to_myriad(114595)
        110000
        >>> round_to_myriad(834729)
        830000
    """
    return round(n / 10000) * 10000


def get_listing_counts() -> Dict[str, int]:
    """
    Get the count of listings by type from PostgreSQL.

    Queries the PostgreSQL database to count the number of unique listings
    for each listing type (rent, auction, sale).

    Returns:
        Dictionary with keys 'rent', 'auction', 'sale' and their counts.

    Example:
        >>> counts = get_listing_counts()
        >>> print(counts)
        {'rent': 75721, 'auction': 114595, 'sale': 834729}
    """
    from sqlalchemy import create_engine

    logger.info("Querying PostgreSQL for listing counts by type...")

    # Build connection string
    conn_string = (
        f"postgresql+psycopg2://{POSTGRES_CONNECTION_PARAMS['user']}:"
        f"{POSTGRES_CONNECTION_PARAMS['password']}@"
        f"{POSTGRES_CONNECTION_PARAMS['host']}:"
        f"{POSTGRES_CONNECTION_PARAMS['port']}/"
        f"{POSTGRES_CONNECTION_PARAMS['database']}"
    )
    engine = create_engine(conn_string)

    # Query to count unique listings by type (using earliest date per listing)
    query = """
    WITH earliest_listings AS (
        SELECT
            fl.listing_id,
            dlt.listing_type
        FROM
            fact_listing fl
        JOIN
            dim_date dd ON fl.date_id = dd.date_id
        JOIN
            dim_listing_type dlt ON fl.listing_type_id = dlt.listing_type_id
        WHERE (fl.listing_id, dd.date_value) IN (
            SELECT fl2.listing_id, MIN(dd2.date_value)
            FROM fact_listing fl2
            JOIN dim_date dd2 ON fl2.date_id = dd2.date_id
            GROUP BY fl2.listing_id
        )
    )
    SELECT listing_type, COUNT(*) as count
    FROM earliest_listings
    GROUP BY listing_type
    ORDER BY listing_type;
    """

    with engine.connect() as conn:
        result = pd.read_sql_query(query, conn.connection)

    # Convert to dictionary
    counts = {
        "rent": 0,
        "auction": 0,
        "sale": 0,
    }
    for _, row in result.iterrows():
        listing_type = row["listing_type"]
        if listing_type in counts:
            counts[listing_type] = int(row["count"])

    logger.info(f"Listing counts: rent={counts['rent']:,}, auction={counts['auction']:,}, sale={counts['sale']:,}")
    return counts


def get_rounded_listing_counts() -> Dict[str, int]:
    """
    Get listing counts rounded to the nearest 10,000.

    This is useful for determining synthetic data sample sizes that
    approximate the real data distribution.

    Returns:
        Dictionary with keys 'rent', 'auction', 'sale' and rounded counts.

    Example:
        >>> counts = get_rounded_listing_counts()
        >>> print(counts)
        {'rent': 80000, 'auction': 110000, 'sale': 830000}
    """
    counts = get_listing_counts()
    rounded = {
        key: round_to_myriad(value) for key, value in counts.items()
    }
    logger.info(f"Rounded counts: rent={rounded['rent']:,}, auction={rounded['auction']:,}, sale={rounded['sale']:,}")
    return rounded


def export_real_data(
    output_path: str = "data/real_data.parquet",
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Export real data from PostgreSQL to Parquet format.

    This function extracts data from PostgreSQL, applies preprocessing
    (except filling NaN in numerical columns), and saves to Parquet.
    The output schema matches the synthetic data format.

    Args:
        output_path: Path for the output Parquet file.
        limit: Optional limit on number of records to extract.

    Returns:
        The exported DataFrame.

    Example:
        >>> df = export_real_data("data/real_data.parquet")
        >>> print(f"Exported {len(df)} records")
    """
    logger.info("Starting real data export...")

    # Extract raw data from PostgreSQL
    logger.info("Extracting data from PostgreSQL...")
    raw_data = extract_data_from_postgres(limit=limit)
    features_df = extract_features_from_postgres(limit=limit)
    data_with_features = merge_features_with_data(raw_data, features_df)

    # Preprocess WITHOUT filling numerical NaN values
    logger.info("Preprocessing data (preserving NaN in numerical columns)...")
    preprocessed_data, numerical_columns, categorical_columns = preprocess_data(
        data_with_features,
        fill_numerical_nan=False  # Key difference from synthetic flow
    )

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to Parquet
    logger.info(f"Saving {len(preprocessed_data):,} records to {output_path}...")
    preprocessed_data.to_parquet(output_path, index=False, engine='pyarrow')

    logger.info(f"Real data export complete: {len(preprocessed_data):,} records saved to {output_path}")
    return preprocessed_data

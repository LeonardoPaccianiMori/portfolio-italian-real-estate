"""
Data extraction functions for synthetic data generation.

This module provides functions for extracting real estate data from the
PostgreSQL data warehouse. The extracted data is used as the basis for
generating synthetic listings.

Author: Leonardo Pacciani-Mori
License: MIT
"""

from typing import Optional, Tuple
import warnings

import pandas as pd
from sqlalchemy import create_engine

# Suppress pandas warning about DBAPI2 connections (we're using psycopg2 via SQLAlchemy)
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

from ..config.settings import POSTGRES_CONNECTION_PARAMS
from ..config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def extract_data_from_postgres(
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract listing data from the PostgreSQL data warehouse.

    This function extracts all real estate listings from the PostgreSQL
    warehouse, joining across multiple dimension tables to get a
    denormalized view of each listing. For listings that were scraped
    on multiple dates, only the earliest occurrence is returned.

    Args:
        limit: Optional maximum number of records to extract. If None,
            extracts all records from the database.

    Returns:
        A pandas DataFrame containing listing data with columns for:
        - listing_id: Unique identifier
        - listing_type: rent, sale, or auction
        - price: Listing price
        - surface: Property surface area in square meters
        - seller_type: Type of seller (agency, private, etc.)
        - condominium_monthly_expenses: Monthly condo fees
        - heating_yearly_expenses: Annual heating costs
        - type_of_property: Property classification
        - condition: Property condition
        - category_name: Property category
        - total_room_number: Number of rooms
        - bathrooms_number: Number of bathrooms
        - kitchen_status: Kitchen type/status
        - garage: Garage information
        - floor: Floor number
        - has_elevator: Whether building has elevator
        - building_usage: Building usage type
        - building_year: Year of construction
        - is_zero_energy: Whether zero energy building
        - heating_type: Type of heating system
        - energy_class: Energy efficiency class
        - air_conditioning: AC information
        - latitude, longitude: Geographic coordinates
        - region, province: Geographic location

    Example:
        >>> df = extract_data_from_postgres(limit=1000)
        >>> print(f"Extracted {len(df)} records")
    """
    logger.info(
        f"Extracting data from PostgreSQL"
        f"{' (limited to ' + str(limit) + ' records)' if limit else ''}..."
    )

    # Builds connection string from parameters.
    conn_string = (
        f"postgresql+psycopg2://{POSTGRES_CONNECTION_PARAMS['user']}:"
        f"{POSTGRES_CONNECTION_PARAMS['password']}@"
        f"{POSTGRES_CONNECTION_PARAMS['host']}:"
        f"{POSTGRES_CONNECTION_PARAMS['port']}/"
        f"{POSTGRES_CONNECTION_PARAMS['database']}"
    )
    engine = create_engine(conn_string)

    # SQL query to extract data of interest.
    # For listings scraped on multiple dates, only gets earliest occurrence.
    query = """
    WITH earliest_date AS (
        SELECT
            fl.listing_id,
            MIN(dd.date_value) AS earliest_date
        FROM
            fact_listing fl
        JOIN
            dim_date dd ON fl.date_id = dd.date_id
        GROUP BY
            fl.listing_id
    )
    SELECT
        fl.listing_id,
        dlt.listing_type,
        fl.price,
        fl.surface,
        dst.seller_type,
        dac.condominium_monthly_expenses,
        dac.heating_yearly_expenses,
        dtp.type_of_property,
        dc.condition,
        dcat.category_name,
        dri.total_room_number,
        dri.bathrooms_number,
        dri.kitchen_status,
        dri.garage,
        dri.floor,
        dbi.has_elevator,
        dbi.building_usage,
        dbi.building_year,
        dei.is_zero_energy_building AS is_zero_energy,
        dei.heating_type,
        dei.energy_class,
        dei.air_conditioning,
        dli2.latitude,
        dli2.longitude,
        dli2.region,
        dli2.province
    FROM
        fact_listing fl
    JOIN
        earliest_date ed ON fl.listing_id = ed.listing_id
    JOIN
        dim_date dd ON fl.date_id = dd.date_id AND dd.date_value = ed.earliest_date
    JOIN
        dim_listing_type dlt ON fl.listing_type_id = dlt.listing_type_id
    LEFT JOIN
        dim_listing_info dli ON fl.listing_info_id = dli.listing_info_id
    LEFT JOIN
        dim_seller_type dst ON dli.seller_type_id = dst.seller_type_id
    LEFT JOIN
        dim_additional_costs dac ON fl.additional_costs_id = dac.additional_costs_id
    LEFT JOIN
        dim_type_of_property dtp ON fl.type_of_property_id = dtp.type_of_property_id
    LEFT JOIN
        dim_condition dc ON fl.condition_id = dc.condition_id
    LEFT JOIN
        dim_category dcat ON fl.category_id = dcat.category_id
    LEFT JOIN
        dim_rooms_info dri ON fl.rooms_info_id = dri.rooms_info_id
    LEFT JOIN
        dim_building_info dbi ON fl.building_info_id = dbi.building_info_id
    LEFT JOIN
        dim_energy_info dei ON fl.energy_info_id = dei.energy_info_id
    LEFT JOIN
        dim_location_info dli2 ON fl.location_info_id = dli2.location_info_id
    """

    # Adds limit clause if specified.
    if limit is not None:
        query += f" LIMIT {limit}"

    # Executes query and loads results into pandas DataFrame.
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn.connection)

    logger.info(f"Extracted {len(df)} records")
    return df


def extract_features_from_postgres(
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract feature data from the PostgreSQL listing_features_bridge table.

    This function extracts the many-to-many relationship between listings
    and their additional features (balcony, garage, elevator, etc.) from
    the bridge table.

    Args:
        limit: Optional maximum number of listings to consider. If None,
            extracts features for all listings.

    Returns:
        A pandas DataFrame with columns:
        - listing_id: The listing identifier
        - feature_name: The name of the feature

    Example:
        >>> features_df = extract_features_from_postgres(limit=1000)
        >>> print(features_df.head())
    """
    logger.info("Extracting features from PostgreSQL...")

    # Builds connection string.
    conn_string = (
        f"postgresql+psycopg2://{POSTGRES_CONNECTION_PARAMS['user']}:"
        f"{POSTGRES_CONNECTION_PARAMS['password']}@"
        f"{POSTGRES_CONNECTION_PARAMS['host']}:"
        f"{POSTGRES_CONNECTION_PARAMS['port']}/"
        f"{POSTGRES_CONNECTION_PARAMS['database']}"
    )
    engine = create_engine(conn_string)

    # Base features query.
    features_query = """
    WITH earliest_date AS (
        SELECT
            fl.listing_id,
            MIN(dd.date_value) AS earliest_date
        FROM
            fact_listing fl
        JOIN
            dim_date dd ON fl.date_id = dd.date_id
        GROUP BY
            fl.listing_id
    )
    SELECT
        lfb.listing_id,
        df.feature_name
    FROM
        listing_features_bridge lfb
    JOIN
        earliest_date ed ON lfb.listing_id = ed.listing_id
    JOIN
        fact_listing fl ON lfb.listing_id = fl.listing_id
    JOIN
        dim_date dd ON fl.date_id = dd.date_id AND dd.date_value = ed.earliest_date
    JOIN
        dim_features df ON lfb.feature_id = df.feature_id
    """

    # If limit is specified, adjusts the query.
    if limit is not None:
        # Gets the main data query with limit.
        main_query = extract_data_from_postgres.__doc__  # Just for reference.
        features_query = f"""
        WITH limited_listings AS (
            SELECT listing_id FROM (
                SELECT fl.listing_id
                FROM fact_listing fl
                LIMIT {limit}
            ) as limited
        ),
        earliest_listings AS (
            SELECT
                fl.listing_id,
                fl.date_id,
                MIN(dd.date_value) as earliest_date
            FROM
                fact_listing fl
            JOIN
                dim_date dd ON fl.date_id = dd.date_id
            WHERE
                fl.listing_id IN (SELECT listing_id FROM limited_listings)
            GROUP BY
                fl.listing_id, fl.date_id
        )
        SELECT
            lfb.listing_id,
            df.feature_name
        FROM
            listing_features_bridge lfb
        JOIN
            earliest_listings el ON lfb.listing_id = el.listing_id
                                AND lfb.date_id = el.date_id
        JOIN
            dim_features df ON lfb.feature_id = df.feature_id
        """

    # Executes query.
    with engine.connect() as conn:
        features_df = pd.read_sql_query(features_query, conn.connection)

    logger.info(f"Extracted {len(features_df)} feature records")
    return features_df


def merge_features_with_data(
    df: pd.DataFrame,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge feature data with main listing data using pivot.

    This function converts the long-format features DataFrame into
    one-hot encoded columns and merges them with the main listing data.

    Args:
        df: Main listing DataFrame from extract_data_from_postgres.
        features_df: Features DataFrame from extract_features_from_postgres.

    Returns:
        DataFrame with main listing data plus one-hot encoded feature columns.

    Example:
        >>> df = extract_data_from_postgres()
        >>> features = extract_features_from_postgres()
        >>> merged = merge_features_with_data(df, features)
    """
    if features_df.empty:
        logger.warning("Features DataFrame is empty, returning original data")
        return df

    # Pivots features into one-hot encoded columns.
    features_pivot = pd.crosstab(
        index=features_df['listing_id'],
        columns=features_df['feature_name'],
        values=1,
        aggfunc='max'
    ).fillna(0).astype(int)

    # Resets index to merge with main dataframe.
    features_pivot = features_pivot.reset_index()

    # Merges with main dataframe.
    df = pd.merge(df, features_pivot, on='listing_id', how='left')

    # Fills NaN values in feature columns with 0.
    feature_cols = features_pivot.columns.tolist()
    feature_cols.remove('listing_id')
    df[feature_cols] = df[feature_cols].fillna(0)

    logger.info(f"Merged {len(feature_cols)} feature columns with main data")
    return df

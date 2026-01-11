"""
Dimension table processors for the PostgreSQL migration.

This module provides functions for processing dimension tables during
the migration from MongoDB warehouse to PostgreSQL data warehouse. It
handles all 14 dimension tables in the star schema, plus bridge tables.

The dimension tables are:
- dim_date: Scraping date information
- dim_listing_type: Listing type (rent, sale, auction)
- dim_seller_type: Seller type (agency, private, etc.)
- dim_listing_info: Listing metadata
- dim_availability: Property availability status
- dim_type_of_property: Property type classification
- dim_condition: Property condition
- dim_features: Additional property features (bridge table)
- dim_additional_costs: Cost information
- dim_category: Property category
- dim_rooms_info: Room information
- dim_building_info: Building information
- dim_energy_info: Energy efficiency information
- dim_location_info: Geographic location
- dim_surface_composition: Surface breakdown (bridge table)
- dim_auction_info: Auction-specific information
- dim_cadastral_info: Cadastral registry information

Author: Leonardo Pacciani-Mori
License: MIT
"""

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable

from ..config.logging_config import get_logger
from ..core.date_utils import convert_mongo_date
from ..core.numeric_utils import calculate_monthly_payment
from .postgres_utils import (
    execute_query_silent,
    check_all_nulls,
    get_existing_null_record_id,
    process_total_room_number,
)

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def process_dimensions_for_batch(batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process all dimension tables for a batch of listings.

    This is the main entry point for dimension processing. It takes a batch
    of listing data from MongoDB and creates/retrieves records in all
    PostgreSQL dimension tables, returning a mapping of dimension keys
    to their database IDs.

    Args:
        batch_data: List of listing dictionaries from MongoDB, each containing
            'data' array with scraping entries and 'collection_name'.

    Returns:
        Dictionary with dimension mappings:
        - 'date': {date_string: date_id}
        - 'listing_type': {type_string: listing_type_id}
        - 'seller_type': {seller_type_string: seller_type_id}
        - ... and so on for all dimensions

    Example:
        >>> batch = [{'data': [...], 'collection_name': 'sale'}]
        >>> mappings = process_dimensions_for_batch(batch)
        >>> date_id = mappings['date']['2024-01-15']
    """
    logger.info("Processing dimensions for batch data...")
    start_time = time.time()

    # Gets PostgreSQL connection hook.
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    # Sets transaction isolation level for consistency.
    execute_query_silent(
        postgres_hook,
        "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"
    )

    # Loads adjusted mortgage rates for cost calculations.
    adjusted_rates = json.loads(Variable.get("adjusted_mortgage_rates", "{}"))
    logger.info(f"Loaded adjusted mortgage rates for {len(adjusted_rates)} dates")

    # Dictionary to store all dimension mappings.
    dimension_mappings: Dict[str, Any] = {}
    dimension_counts: Dict[str, int] = {}
    total_listings = len(batch_data)

    # Process each dimension table.
    # The order matters because some dimensions reference others (e.g., listing_info -> seller_type).

    # 1. Process date dimension
    dimension_mappings['date'] = _process_date_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 2. Process listing_type dimension
    dimension_mappings['listing_type'] = _process_listing_type_dimension(
        postgres_hook, dimension_counts
    )

    # 3. Process seller_type dimension
    dimension_mappings['seller_type'] = _process_seller_type_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 4. Process listing_info dimension (depends on seller_type)
    dimension_mappings['listing_info'] = _process_listing_info_dimension(
        batch_data, postgres_hook, dimension_mappings['seller_type'], dimension_counts
    )

    # 5. Process availability dimension
    dimension_mappings['availability'] = _process_availability_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 6. Process type_of_property dimension
    dimension_mappings['type_of_property'] = _process_type_of_property_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 7. Process condition dimension
    dimension_mappings['condition'] = _process_condition_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 8. Process features dimension
    dimension_mappings['features'] = _process_features_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 9. Process additional_costs dimension
    dimension_mappings['additional_costs'] = _process_additional_costs_dimension(
        batch_data, postgres_hook, adjusted_rates, dimension_counts
    )

    # 10. Process category dimension
    dimension_mappings['category'] = _process_category_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 11. Process rooms_info dimension
    dimension_mappings['rooms_info'] = _process_rooms_info_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 12. Process building_info dimension
    dimension_mappings['building_info'] = _process_building_info_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 13. Process energy_info dimension
    dimension_mappings['energy_info'] = _process_energy_info_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 14. Process location_info dimension
    dimension_mappings['location_info'] = _process_location_info_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 15. Process surface_composition dimension
    dimension_mappings['surface_composition'] = _process_surface_composition_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 16. Process auction_info dimension
    dimension_mappings['auction_info'] = _process_auction_info_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    # 17. Process cadastral_info dimension
    dimension_mappings['cadastral_info'] = _process_cadastral_info_dimension(
        batch_data, postgres_hook, dimension_counts
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Dimension processing complete in {elapsed_time:.2f} seconds")
    logger.info(f"Dimension counts: {dimension_counts}")

    return dimension_mappings


def _process_date_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[str, int]:
    """
    Process the dim_date dimension table.

    Extracts all unique scraping dates from the batch and ensures they
    exist in the dim_date table.

    Args:
        batch_data: List of listing dictionaries.
        postgres_hook: PostgreSQL connection hook.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping date strings to date_id values.
    """
    logger.info("Processing date dimension...")
    date_mappings: Dict[str, int] = {}
    date_rows: Set[str] = set()

    # Collects all unique dates from the batch.
    for listing in batch_data:
        for data_entry in listing.get('data', []):
            scraping_date = data_entry.get('scraping_date')
            if scraping_date:
                date_rows.add(scraping_date)

    initial_count = len(date_rows)
    processed_count = 0

    for date_str in date_rows:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == initial_count:
            logger.info(f"Processing date dimension: {processed_count}/{initial_count}")

        # Converts string to date object.
        date_obj = convert_mongo_date(date_str)
        if date_obj:
            # Gets month name in lowercase.
            month_name = date_obj.strftime('%B').lower()

            # Uses INSERT ... ON CONFLICT DO NOTHING for upsert behavior.
            query = '''
            WITH inserted AS (
                INSERT INTO dim_date (date_value, year, month_number, month_name, day_of_month)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (date_value) DO NOTHING
                RETURNING date_id
            )
            SELECT date_id FROM inserted
            UNION ALL
            SELECT date_id FROM dim_date WHERE date_value = %s
            LIMIT 1
            '''
            params = (
                date_obj, date_obj.year, date_obj.month,
                month_name, date_obj.day, date_obj
            )
            result = execute_query_silent(postgres_hook, query, params)

            date_id = result[0][0] if result else None
            if date_id:
                date_mappings[date_str] = date_id
            else:
                logger.warning(f"Failed to get date_id for {date_obj}")

    dimension_counts['date'] = len(date_mappings)
    return date_mappings


def _process_listing_type_dimension(
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[str, int]:
    """
    Process the dim_listing_type dimension table.

    Ensures the three listing types (rent, auction, sale) exist in the
    dimension table.

    Args:
        postgres_hook: PostgreSQL connection hook.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping listing type strings to listing_type_id values.
    """
    logger.info("Processing listing_type dimension...")
    listing_type_mappings: Dict[str, int] = {}
    listing_types = {'rent', 'auction', 'sale'}

    processed_count = 0
    total_count = len(listing_types)

    for lt in listing_types:
        processed_count += 1
        logger.info(f"Processing listing_type: {processed_count}/{total_count}")

        lt_lower = lt.lower()

        # Uses INSERT ... ON CONFLICT DO NOTHING for upsert behavior.
        query = '''
        WITH inserted AS (
            INSERT INTO dim_listing_type (listing_type)
            VALUES (%s)
            ON CONFLICT (listing_type) DO NOTHING
            RETURNING listing_type_id
        )
        SELECT listing_type_id FROM inserted
        UNION ALL
        SELECT listing_type_id FROM dim_listing_type WHERE listing_type = %s
        LIMIT 1
        '''
        params = (lt_lower, lt_lower)
        result = execute_query_silent(postgres_hook, query, params)

        listing_type_id = result[0][0] if result else None
        if listing_type_id:
            listing_type_mappings[lt_lower] = listing_type_id
        else:
            logger.warning(f"Failed to get listing_type_id for {lt_lower}")

    dimension_counts['listing_type'] = len(listing_type_mappings)
    return listing_type_mappings


def _process_seller_type_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Optional[str], int]:
    """
    Process the dim_seller_type dimension table.

    Extracts unique seller types from the batch and ensures they exist
    in the dimension table.

    Args:
        batch_data: List of listing dictionaries.
        postgres_hook: PostgreSQL connection hook.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping seller type strings (or None) to seller_type_id values.
    """
    logger.info("Processing seller_type dimension...")
    seller_type_mappings: Dict[Optional[str], int] = {}
    seller_types: Set[str] = set()

    # Checks for existing NULL seller_type record.
    null_seller_type_id = get_existing_null_record_id(
        postgres_hook, 'dim_seller_type', 'seller_type_id', ['seller_type']
    )
    if null_seller_type_id:
        seller_type_mappings[None] = null_seller_type_id

    # Collects unique seller types from batch.
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing seller_type: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if listing_features:
                listing_info = listing_features.get('listing_info')
                if listing_info:
                    seller_type = listing_info.get('seller_type')
                    if seller_type:
                        seller_types.add(seller_type.lower())

    processed_count = 0
    total_count = len(seller_types)

    for st in seller_types:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(f"Processing seller_type values: {processed_count}/{total_count}")

        query = '''
        WITH inserted AS (
            INSERT INTO dim_seller_type (seller_type)
            VALUES (%s)
            ON CONFLICT (seller_type) DO NOTHING
            RETURNING seller_type_id
        )
        SELECT seller_type_id FROM inserted
        UNION ALL
        SELECT seller_type_id FROM dim_seller_type WHERE seller_type = %s
        LIMIT 1
        '''
        params = (st, st)
        result = execute_query_silent(postgres_hook, query, params)

        seller_type_id = result[0][0] if result else None
        if seller_type_id:
            seller_type_mappings[st] = seller_type_id
        else:
            logger.warning(f"Failed to get seller_type_id for {st}")

    dimension_counts['seller_type'] = len(seller_type_mappings)
    return seller_type_mappings


def _process_listing_info_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    seller_type_mappings: Dict[Optional[str], int],
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """
    Process the dim_listing_info dimension table.

    Extracts listing metadata (age, last update, photos, seller type)
    and ensures they exist in the dimension table.

    Args:
        batch_data: List of listing dictionaries.
        postgres_hook: PostgreSQL connection hook.
        seller_type_mappings: Mapping from seller types to IDs.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping (listing_age, last_update, photos, seller_type_id)
        tuples to listing_info_id values.
    """
    logger.info("Processing listing_info dimension...")
    listing_info_mappings: Dict[Tuple, int] = {}
    processed_count = 0

    # Defines the all-NULL key for special handling.
    all_null_key = (None, None, None, None)

    # Checks for existing all-NULL record.
    listing_info_columns = [
        'listing_age', 'listing_last_update', 'number_of_pictures', 'seller_type_id'
    ]
    null_listing_info_id = get_existing_null_record_id(
        postgres_hook, 'dim_listing_info', 'listing_info_id', listing_info_columns
    )

    if null_listing_info_id:
        listing_info_mappings[all_null_key] = null_listing_info_id
        logger.info(
            f"Found existing all-NULL listing_info record with ID {null_listing_info_id}"
        )

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing listing_info: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            listing_info = listing_features.get('listing_info')
            if not listing_info:
                continue

            # Gets seller_type_id from mapping.
            seller_type = listing_info.get('seller_type')
            if seller_type and isinstance(seller_type, str):
                seller_type = seller_type.lower()
            seller_type_id = seller_type_mappings.get(seller_type)

            # Falls back to NULL mapping if no seller_type_id.
            if not seller_type_id and None in seller_type_mappings:
                seller_type_id = seller_type_mappings[None]

            if seller_type_id:
                listing_age = listing_info.get('listing_age')
                listing_last_update = listing_info.get('listing_last_update')
                number_of_photos = listing_info.get('number_of_photos')

                # Creates unique key for this combination.
                listing_info_key = (
                    listing_age,
                    listing_last_update,
                    number_of_photos,
                    seller_type_id
                )

                # Handles all-NULL case specially.
                if (listing_age is None and listing_last_update is None
                        and number_of_photos is None):
                    if all_null_key in listing_info_mappings:
                        listing_info_mappings[listing_info_key] = listing_info_mappings[all_null_key]
                        continue

                # Skips if already processed.
                if listing_info_key in listing_info_mappings:
                    continue

                processed_count += 1

                # Uses INSERT with complex WHERE for null handling.
                query = '''
                WITH inserted AS (
                    INSERT INTO dim_listing_info (
                        listing_age, listing_last_update, number_of_pictures, seller_type_id
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (listing_age, listing_last_update, number_of_pictures, seller_type_id)
                    DO NOTHING
                    RETURNING listing_info_id
                )
                SELECT listing_info_id FROM inserted
                UNION ALL
                SELECT listing_info_id FROM dim_listing_info
                WHERE (listing_age IS NULL AND %s IS NULL OR listing_age = %s)
                AND (listing_last_update IS NULL AND %s IS NULL OR listing_last_update = %s)
                AND (number_of_pictures IS NULL AND %s IS NULL OR number_of_pictures = %s)
                AND seller_type_id = %s
                LIMIT 1
                '''
                params = (
                    listing_age, listing_last_update, number_of_photos, seller_type_id,
                    listing_age, listing_age,
                    listing_last_update, listing_last_update,
                    number_of_photos, number_of_photos,
                    seller_type_id
                )
                result = execute_query_silent(postgres_hook, query, params)

                listing_info_id = result[0][0] if result else None
                if listing_info_id:
                    listing_info_mappings[listing_info_key] = listing_info_id

    dimension_counts['listing_info'] = len(listing_info_mappings)
    return listing_info_mappings


def _process_availability_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Optional[str], int]:
    """
    Process the dim_availability dimension table.

    Args:
        batch_data: List of listing dictionaries.
        postgres_hook: PostgreSQL connection hook.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping availability strings to availability_id values.
    """
    logger.info("Processing availability dimension...")
    availability_mappings: Dict[Optional[str], int] = {}
    availability_values: Set[str] = set()

    # Checks for existing NULL availability.
    null_availability_id = get_existing_null_record_id(
        postgres_hook, 'dim_availability', 'availability_id', ['availability']
    )
    if null_availability_id:
        availability_mappings[None] = null_availability_id

    # Collects unique availability values.
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing availability: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if listing_features:
                availability = listing_features.get('availability')
                if availability:
                    availability_values.add(availability.lower())

    processed_count = 0
    total_count = len(availability_values)

    for availability in availability_values:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(
                f"Processing availability values: {processed_count}/{total_count}"
            )

        query = '''
        WITH inserted AS (
            INSERT INTO dim_availability (availability)
            VALUES (%s)
            ON CONFLICT (availability) DO NOTHING
            RETURNING availability_id
        )
        SELECT availability_id FROM inserted
        UNION ALL
        SELECT availability_id FROM dim_availability WHERE availability = %s
        LIMIT 1
        '''
        params = (availability, availability)
        result = execute_query_silent(postgres_hook, query, params)

        availability_id = result[0][0] if result else None
        if availability_id:
            availability_mappings[availability] = availability_id

    dimension_counts['availability'] = len(availability_mappings)
    return availability_mappings


def _process_type_of_property_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Optional[str], int]:
    """
    Process the dim_type_of_property dimension table.

    Args:
        batch_data: List of listing dictionaries.
        postgres_hook: PostgreSQL connection hook.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping property type strings to type_of_property_id values.
    """
    logger.info("Processing type_of_property dimension...")
    type_of_property_mappings: Dict[Optional[str], int] = {}
    type_of_property_values: Set[str] = set()

    # Checks for existing NULL type_of_property.
    null_type_id = get_existing_null_record_id(
        postgres_hook, 'dim_type_of_property', 'type_of_property_id', ['type_of_property']
    )
    if null_type_id:
        type_of_property_mappings[None] = null_type_id

    # Collects unique property types.
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(
                f"Processing type_of_property: listing #{idx+1}/{len(batch_data)}"
            )

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            type_of_property = listing_features.get('type_of_property')
            if (type_of_property and type_of_property.get('class')
                    and isinstance(type_of_property.get('class'), str)):
                type_of_property_values.add(type_of_property.get('class').lower())

    processed_count = 0
    total_count = len(type_of_property_values)

    for property_type in type_of_property_values:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(
                f"Processing type_of_property values: {processed_count}/{total_count}"
            )

        query = '''
        WITH inserted AS (
            INSERT INTO dim_type_of_property (type_of_property)
            VALUES (%s)
            ON CONFLICT (type_of_property) DO NOTHING
            RETURNING type_of_property_id
        )
        SELECT type_of_property_id FROM inserted
        UNION ALL
        SELECT type_of_property_id FROM dim_type_of_property WHERE type_of_property = %s
        LIMIT 1
        '''
        params = (property_type, property_type)
        result = execute_query_silent(postgres_hook, query, params)

        type_id = result[0][0] if result else None
        if type_id:
            type_of_property_mappings[property_type] = type_id

    dimension_counts['type_of_property'] = len(type_of_property_mappings)
    return type_of_property_mappings


def _process_condition_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Optional[str], int]:
    """
    Process the dim_condition dimension table.

    Args:
        batch_data: List of listing dictionaries.
        postgres_hook: PostgreSQL connection hook.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping condition strings to condition_id values.
    """
    logger.info("Processing condition dimension...")
    condition_mappings: Dict[Optional[str], int] = {}
    condition_values: Set[str] = set()

    # Checks for existing NULL condition.
    null_condition_id = get_existing_null_record_id(
        postgres_hook, 'dim_condition', 'condition_id', ['condition']
    )
    if null_condition_id:
        condition_mappings[None] = null_condition_id

    # Collects unique condition values.
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing condition: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if listing_features:
                condition = listing_features.get('condition')
                if condition:
                    condition_values.add(condition.lower())

    processed_count = 0
    total_count = len(condition_values)

    for condition in condition_values:
        processed_count += 1
        if processed_count % 10 == 0 or processed_count == total_count:
            logger.info(f"Processing condition values: {processed_count}/{total_count}")

        query = '''
        WITH inserted AS (
            INSERT INTO dim_condition (condition)
            VALUES (%s)
            ON CONFLICT (condition) DO NOTHING
            RETURNING condition_id
        )
        SELECT condition_id FROM inserted
        UNION ALL
        SELECT condition_id FROM dim_condition WHERE condition = %s
        LIMIT 1
        '''
        params = (condition, condition)
        result = execute_query_silent(postgres_hook, query, params)

        condition_id = result[0][0] if result else None
        if condition_id:
            condition_mappings[condition] = condition_id

    dimension_counts['condition'] = len(condition_mappings)
    return condition_mappings


def _process_features_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[str, int]:
    """
    Process the dim_features dimension table.

    Extracts unique feature names from additional_features arrays
    and ensures they exist in the dimension table.

    Args:
        batch_data: List of listing dictionaries.
        postgres_hook: PostgreSQL connection hook.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping feature name strings to feature_id values.
    """
    logger.info("Processing features dimension...")
    features_mappings: Dict[str, int] = {}
    feature_values: Set[str] = set()

    # Collects unique features.
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(f"Processing features: listing #{idx+1}/{len(batch_data)}")

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if listing_features:
                additional_features = listing_features.get('additional_features', [])
                if additional_features:
                    for feature in additional_features:
                        if feature:
                            feature_values.add(feature.lower())

    processed_count = 0
    total_count = len(feature_values)

    for feature in feature_values:
        processed_count += 1
        if processed_count % 50 == 0 or processed_count == total_count:
            logger.info(f"Processing features values: {processed_count}/{total_count}")

        query = '''
        WITH inserted AS (
            INSERT INTO dim_features (feature_name)
            VALUES (%s)
            ON CONFLICT (feature_name) DO NOTHING
            RETURNING feature_id
        )
        SELECT feature_id FROM inserted
        UNION ALL
        SELECT feature_id FROM dim_features WHERE feature_name = %s
        LIMIT 1
        '''
        params = (feature, feature)
        result = execute_query_silent(postgres_hook, query, params)

        feature_id = result[0][0] if result else None
        if feature_id:
            features_mappings[feature] = feature_id

    dimension_counts['features'] = len(features_mappings)
    return features_mappings


def _process_additional_costs_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    adjusted_rates: Dict[str, float],
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """
    Process the dim_additional_costs dimension table.

    Extracts cost information including mortgage rates and monthly payments.

    Args:
        batch_data: List of listing dictionaries.
        postgres_hook: PostgreSQL connection hook.
        adjusted_rates: Dictionary of adjusted mortgage rates by date.
        dimension_counts: Dictionary to update with counts.

    Returns:
        Dictionary mapping cost tuples to additional_costs_id values.
    """
    logger.info("Processing additional_costs dimension...")
    additional_costs_mappings: Dict[Tuple, int] = {}

    # Checks for existing all-NULL record.
    all_null_key = (None, None, None, None)
    additional_costs_columns = [
        'condominium_monthly_expenses', 'heating_yearly_expenses',
        'mortgage_rate', 'estimated_monthly_payment'
    ]
    null_additional_costs_id = get_existing_null_record_id(
        postgres_hook, 'dim_additional_costs', 'additional_costs_id',
        additional_costs_columns
    )
    if null_additional_costs_id:
        additional_costs_mappings[all_null_key] = null_additional_costs_id

    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(
                f"Processing additional_costs: listing #{idx+1}/{len(batch_data)}"
            )

        collection_name = listing.get('collection_name', '').lower()

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            additional_costs = listing_features.get('additional_costs', {})

            condominium_monthly_expenses = additional_costs.get('condominium_monthly_expenses')
            heating_yearly_expenses = additional_costs.get('heating_yearly_expenses')

            # Gets mortgage rate for sale/auction listings.
            mortgage_rate = None
            estimated_monthly_payment = None

            if collection_name in ['sale', 'auction']:
                scraping_date = data_entry.get('scraping_date')
                mortgage_rate = adjusted_rates.get(scraping_date)

                if mortgage_rate is not None:
                    price = listing_features.get('price')
                    if price is not None:
                        estimated_monthly_payment = calculate_monthly_payment(
                            price, mortgage_rate
                        )

            costs_key = (
                condominium_monthly_expenses,
                heating_yearly_expenses,
                mortgage_rate,
                estimated_monthly_payment
            )

            # Handles all-NULL case.
            if check_all_nulls(list(costs_key)):
                if all_null_key in additional_costs_mappings:
                    additional_costs_mappings[costs_key] = additional_costs_mappings[all_null_key]
                    continue

            if costs_key in additional_costs_mappings:
                continue

            query = '''
            WITH inserted AS (
                INSERT INTO dim_additional_costs (
                    condominium_monthly_expenses, heating_yearly_expenses,
                    mortgage_rate, estimated_monthly_payment
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING additional_costs_id
            )
            SELECT additional_costs_id FROM inserted
            UNION ALL
            SELECT additional_costs_id FROM dim_additional_costs
            WHERE (condominium_monthly_expenses IS NULL AND %s IS NULL
                   OR condominium_monthly_expenses = %s)
            AND (heating_yearly_expenses IS NULL AND %s IS NULL
                 OR heating_yearly_expenses = %s)
            AND (mortgage_rate IS NULL AND %s IS NULL OR mortgage_rate = %s)
            AND (estimated_monthly_payment IS NULL AND %s IS NULL
                 OR estimated_monthly_payment = %s)
            LIMIT 1
            '''
            params = (
                condominium_monthly_expenses, heating_yearly_expenses,
                mortgage_rate, estimated_monthly_payment,
                condominium_monthly_expenses, condominium_monthly_expenses,
                heating_yearly_expenses, heating_yearly_expenses,
                mortgage_rate, mortgage_rate,
                estimated_monthly_payment, estimated_monthly_payment
            )
            result = execute_query_silent(postgres_hook, query, params)

            costs_id = result[0][0] if result else None
            if costs_id:
                additional_costs_mappings[costs_key] = costs_id

    dimension_counts['additional_costs'] = len(additional_costs_mappings)
    return additional_costs_mappings


def _process_category_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Optional[str], int]:
    """Process the dim_category dimension table."""
    logger.info("Processing category dimension...")
    category_mappings: Dict[Optional[str], int] = {}
    category_values: Set[str] = set()

    null_category_id = get_existing_null_record_id(
        postgres_hook, 'dim_category', 'category_id', ['category_name']
    )
    if null_category_id:
        category_mappings[None] = null_category_id

    for listing in batch_data:
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if listing_features:
                category = listing_features.get('category')
                if category and category.get('name'):
                    category_values.add(category.get('name').lower())

    for category in category_values:
        query = '''
        WITH inserted AS (
            INSERT INTO dim_category (category_name)
            VALUES (%s)
            ON CONFLICT (category_name) DO NOTHING
            RETURNING category_id
        )
        SELECT category_id FROM inserted
        UNION ALL
        SELECT category_id FROM dim_category WHERE category_name = %s
        LIMIT 1
        '''
        params = (category, category)
        result = execute_query_silent(postgres_hook, query, params)

        category_id = result[0][0] if result else None
        if category_id:
            category_mappings[category] = category_id

    dimension_counts['category'] = len(category_mappings)
    return category_mappings


def _process_rooms_info_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """Process the dim_rooms_info dimension table."""
    logger.info("Processing rooms_info dimension...")
    rooms_info_mappings: Dict[Tuple, int] = {}

    for listing in batch_data:
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            rooms_info = listing_features.get('rooms_info', {})

            total_room_number = process_total_room_number(
                rooms_info.get('total_room_number')
            )
            bathrooms_number = rooms_info.get('bathrooms_number')
            kitchen_status = rooms_info.get('kitchen_status')
            if kitchen_status and isinstance(kitchen_status, str):
                kitchen_status = kitchen_status.lower()
            garage = rooms_info.get('garage')
            if garage and isinstance(garage, str):
                garage = garage.lower()
            floor = rooms_info.get('floor')

            rooms_key = (
                total_room_number, bathrooms_number, kitchen_status, garage, floor
            )

            if rooms_key in rooms_info_mappings:
                continue

            query = '''
            WITH inserted AS (
                INSERT INTO dim_rooms_info (
                    total_room_number, bathrooms_number, kitchen_status, garage, floor
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING rooms_info_id
            )
            SELECT rooms_info_id FROM inserted
            UNION ALL
            SELECT rooms_info_id FROM dim_rooms_info
            WHERE (total_room_number IS NULL AND %s IS NULL OR total_room_number = %s)
            AND (bathrooms_number IS NULL AND %s IS NULL OR bathrooms_number = %s)
            AND (kitchen_status IS NULL AND %s IS NULL OR kitchen_status = %s)
            AND (garage IS NULL AND %s IS NULL OR garage = %s)
            AND (floor IS NULL AND %s IS NULL OR floor = %s)
            LIMIT 1
            '''
            params = (
                total_room_number, bathrooms_number, kitchen_status, garage, floor,
                total_room_number, total_room_number,
                bathrooms_number, bathrooms_number,
                kitchen_status, kitchen_status,
                garage, garage,
                floor, floor
            )
            result = execute_query_silent(postgres_hook, query, params)

            rooms_id = result[0][0] if result else None
            if rooms_id:
                rooms_info_mappings[rooms_key] = rooms_id

    dimension_counts['rooms_info'] = len(rooms_info_mappings)
    return rooms_info_mappings


def _process_building_info_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """Process the dim_building_info dimension table."""
    logger.info("Processing building_info dimension...")
    building_info_mappings: Dict[Tuple, int] = {}

    for listing in batch_data:
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            building_info = listing_features.get('building_info', {})

            total_building_floors = building_info.get('total_building_floors')
            has_elevator = building_info.get('has_elevator')
            building_usage = building_info.get('building_usage')
            if building_usage and isinstance(building_usage, str):
                building_usage = building_usage.lower()
            building_year = building_info.get('building_year')

            building_key = (
                total_building_floors, has_elevator, building_usage, building_year
            )

            if building_key in building_info_mappings:
                continue

            query = '''
            WITH inserted AS (
                INSERT INTO dim_building_info (
                    total_building_floors, has_elevator, building_usage, building_year
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING building_info_id
            )
            SELECT building_info_id FROM inserted
            UNION ALL
            SELECT building_info_id FROM dim_building_info
            WHERE (total_building_floors IS NULL AND %s IS NULL
                   OR total_building_floors = %s)
            AND (has_elevator IS NULL AND %s IS NULL OR has_elevator = %s)
            AND (building_usage IS NULL AND %s IS NULL OR building_usage = %s)
            AND (building_year IS NULL AND %s IS NULL OR building_year = %s)
            LIMIT 1
            '''
            params = (
                total_building_floors, has_elevator, building_usage, building_year,
                total_building_floors, total_building_floors,
                has_elevator, has_elevator,
                building_usage, building_usage,
                building_year, building_year
            )
            result = execute_query_silent(postgres_hook, query, params)

            building_id = result[0][0] if result else None
            if building_id:
                building_info_mappings[building_key] = building_id

    dimension_counts['building_info'] = len(building_info_mappings)
    return building_info_mappings


def _process_energy_info_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """Process the dim_energy_info dimension table."""
    logger.info("Processing energy_info dimension...")
    energy_info_mappings: Dict[Tuple, int] = {}

    for listing in batch_data:
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            energy_info = listing_features.get('energy_info', {})

            is_zero_energy_building = energy_info.get('is_zero_energy_building')
            heating_type = energy_info.get('heating_type')
            if heating_type and isinstance(heating_type, str):
                heating_type = heating_type.lower()
            energy_class = energy_info.get('energy_class')
            if energy_class and isinstance(energy_class, str):
                energy_class = energy_class.lower()
            air_conditioning = energy_info.get('air_conditioning')
            if air_conditioning and isinstance(air_conditioning, str):
                air_conditioning = air_conditioning.lower()

            energy_key = (
                is_zero_energy_building, heating_type, energy_class, air_conditioning
            )

            if energy_key in energy_info_mappings:
                continue

            query = '''
            WITH inserted AS (
                INSERT INTO dim_energy_info (
                    is_zero_energy_building, heating_type, energy_class, air_conditioning
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING energy_info_id
            )
            SELECT energy_info_id FROM inserted
            UNION ALL
            SELECT energy_info_id FROM dim_energy_info
            WHERE (is_zero_energy_building IS NULL AND %s IS NULL
                   OR is_zero_energy_building = %s)
            AND (heating_type IS NULL AND %s IS NULL OR heating_type = %s)
            AND (energy_class IS NULL AND %s IS NULL OR energy_class = %s)
            AND (air_conditioning IS NULL AND %s IS NULL OR air_conditioning = %s)
            LIMIT 1
            '''
            params = (
                is_zero_energy_building, heating_type, energy_class, air_conditioning,
                is_zero_energy_building, is_zero_energy_building,
                heating_type, heating_type,
                energy_class, energy_class,
                air_conditioning, air_conditioning
            )
            result = execute_query_silent(postgres_hook, query, params)

            energy_id = result[0][0] if result else None
            if energy_id:
                energy_info_mappings[energy_key] = energy_id

    dimension_counts['energy_info'] = len(energy_info_mappings)
    return energy_info_mappings


def _process_location_info_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """Process the dim_location_info dimension table."""
    logger.info("Processing location_info dimension...")
    location_info_mappings: Dict[Tuple, int] = {}

    for listing in batch_data:
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            location_info = listing_features.get('location_info', {})

            latitude = location_info.get('latitude')
            longitude = location_info.get('longitude')
            region = location_info.get('region')
            if region and isinstance(region, str):
                region = region.lower()
            province = location_info.get('province')
            if province and isinstance(province, str):
                province = province.lower()

            location_key = (latitude, longitude, region, province)

            if location_key in location_info_mappings:
                continue

            query = '''
            WITH inserted AS (
                INSERT INTO dim_location_info (latitude, longitude, region, province)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING location_info_id
            )
            SELECT location_info_id FROM inserted
            UNION ALL
            SELECT location_info_id FROM dim_location_info
            WHERE (latitude IS NULL AND %s IS NULL OR latitude = %s)
            AND (longitude IS NULL AND %s IS NULL OR longitude = %s)
            AND (region IS NULL AND %s IS NULL OR region = %s)
            AND (province IS NULL AND %s IS NULL OR province = %s)
            LIMIT 1
            '''
            params = (
                latitude, longitude, region, province,
                latitude, latitude,
                longitude, longitude,
                region, region,
                province, province
            )
            result = execute_query_silent(postgres_hook, query, params)

            location_id = result[0][0] if result else None
            if location_id:
                location_info_mappings[location_key] = location_id

    dimension_counts['location_info'] = len(location_info_mappings)
    return location_info_mappings


def _process_surface_composition_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """Process the dim_surface_composition dimension table."""
    logger.info("Processing surface_composition dimension...")
    surface_composition_mappings: Dict[Tuple, int] = {}

    for listing in batch_data:
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            surface_composition = listing_features.get('surface_composition')
            if not surface_composition:
                continue

            element_name = surface_composition.get('element')
            if element_name and isinstance(element_name, str):
                element_name = element_name.lower()

            floor_str = surface_composition.get('floor')
            floor = None
            if floor_str:
                if isinstance(floor_str, str):
                    floor_str_lower = floor_str.lower()
                    if "rialzato" in floor_str_lower or "terra" in floor_str_lower:
                        floor = 0
                    elif "seminterrato" in floor_str_lower or "interrato" in floor_str_lower:
                        floor = -1
                    else:
                        floor_match = re.search(r'(\d+)', floor_str)
                        if floor_match:
                            floor = int(floor_match.group(1))
                elif isinstance(floor_str, (int, float)):
                    floor = int(floor_str)

            surface = surface_composition.get('surface')
            percentage = surface_composition.get('percentage')
            commercial_surface = surface_composition.get('commercial_surface')

            surface_key = (element_name, floor, surface, percentage, commercial_surface)

            if surface_key in surface_composition_mappings:
                continue

            query = '''
            WITH inserted AS (
                INSERT INTO dim_surface_composition (
                    element_name, floor, surface, percentage, commercial_surface
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING surface_composition_id
            )
            SELECT surface_composition_id FROM inserted
            UNION ALL
            SELECT surface_composition_id FROM dim_surface_composition
            WHERE (element_name IS NULL AND %s IS NULL OR element_name = %s)
            AND (floor IS NULL AND %s IS NULL OR floor = %s)
            AND (surface IS NULL AND %s IS NULL OR surface = %s)
            AND (percentage IS NULL AND %s IS NULL OR percentage = %s)
            AND (commercial_surface IS NULL AND %s IS NULL OR commercial_surface = %s)
            LIMIT 1
            '''
            params = (
                element_name, floor, surface, percentage, commercial_surface,
                element_name, element_name,
                floor, floor,
                surface, surface,
                percentage, percentage,
                commercial_surface, commercial_surface
            )
            result = execute_query_silent(postgres_hook, query, params)

            surface_id = result[0][0] if result else None
            if surface_id:
                surface_composition_mappings[surface_key] = surface_id

    dimension_counts['surface_composition'] = len(surface_composition_mappings)
    return surface_composition_mappings


def _process_auction_info_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """Process the dim_auction_info dimension table."""
    logger.info("Processing auction_info dimension...")
    auction_info_mappings: Dict[Tuple, int] = {}

    for listing in batch_data:
        collection_name = listing.get('collection_name', '').lower()
        if collection_name != 'auction':
            continue

        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            auction_info = listing_features.get('auction_info', {})
            if not auction_info:
                continue

            auction_end_date = auction_info.get('auction_end_date')
            deposit_modality = auction_info.get('deposit_modality')
            if deposit_modality and isinstance(deposit_modality, str):
                deposit_modality = deposit_modality.lower()

            # Creates hash for long text fields.
            deposit_modality_hash = None
            if deposit_modality:
                deposit_modality_hash = hashlib.md5(
                    deposit_modality.encode()
                ).hexdigest()[:16]

            auction_type = auction_info.get('auction_type')
            if auction_type and isinstance(auction_type, str):
                auction_type = auction_type.lower()

            is_open = auction_info.get('is_open')
            minimum_offer = auction_info.get('minimum_offer')

            procedure_number = auction_info.get('procedure_number')
            if procedure_number and isinstance(procedure_number, str):
                procedure_number = procedure_number.lower()

            auction_court = auction_info.get('auction_court')
            if auction_court and isinstance(auction_court, str):
                auction_court = auction_court.lower()

            lot_category = auction_info.get('lot_category')
            lot_category_id = None
            lot_category_name = None
            if lot_category:
                lot_category_id = lot_category.get('id')
                lot_category_name = lot_category.get('name')
                if lot_category_name and isinstance(lot_category_name, str):
                    lot_category_name = lot_category_name.lower()

            auction_key = (
                auction_end_date, deposit_modality, deposit_modality_hash,
                auction_type, is_open, minimum_offer, procedure_number,
                auction_court, lot_category_id, lot_category_name
            )

            if auction_key in auction_info_mappings:
                continue

            # Simplified query for auction_info.
            query = '''
            INSERT INTO dim_auction_info (
                auction_end_date, deposit_modality, deposit_modality_hash,
                auction_type, is_open, minimum_offer, procedure_number,
                auction_court, lot_category_id, lot_category_name
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING auction_info_id
            '''
            params = (
                auction_end_date, deposit_modality, deposit_modality_hash,
                auction_type, is_open, minimum_offer, procedure_number,
                auction_court, lot_category_id, lot_category_name
            )
            result = execute_query_silent(postgres_hook, query, params)

            if result:
                auction_info_mappings[auction_key] = result[0][0]
            else:
                # Try to find existing record.
                find_query = '''
                SELECT auction_info_id FROM dim_auction_info
                WHERE (deposit_modality_hash = %s OR (deposit_modality_hash IS NULL AND %s IS NULL))
                LIMIT 1
                '''
                find_result = execute_query_silent(
                    postgres_hook, find_query, (deposit_modality_hash, deposit_modality_hash)
                )
                if find_result:
                    auction_info_mappings[auction_key] = find_result[0][0]

    dimension_counts['auction_info'] = len(auction_info_mappings)
    return auction_info_mappings


def _process_cadastral_info_dimension(
    batch_data: List[Dict],
    postgres_hook: Any,
    dimension_counts: Dict[str, int]
) -> Dict[Tuple, int]:
    """Process the dim_cadastral_info dimension table."""
    logger.info("Processing cadastral_info dimension...")
    cadastral_info_mappings: Dict[Tuple, int] = {}

    for listing in batch_data:
        for data_entry in listing.get('data', []):
            listing_features = data_entry.get('listing_features')
            if not listing_features:
                continue

            cadastral_info = listing_features.get('cadastral_info', {})
            if not cadastral_info:
                continue

            cadastral = cadastral_info.get('cadastral')
            if cadastral and isinstance(cadastral, str):
                cadastral = cadastral.lower()

            cadastral_additional_info = cadastral_info.get('cadastral_additional_info')
            if cadastral_additional_info and isinstance(cadastral_additional_info, str):
                cadastral_additional_info = cadastral_additional_info.lower()

            sub_cadastral_info = cadastral_info.get('sub_cadastral_info')
            if sub_cadastral_info and isinstance(sub_cadastral_info, str):
                sub_cadastral_info = sub_cadastral_info.lower()

            cadastral_key = (cadastral, cadastral_additional_info, sub_cadastral_info)

            if cadastral_key in cadastral_info_mappings:
                continue

            query = '''
            WITH inserted AS (
                INSERT INTO dim_cadastral_info (
                    cadastral, cadastral_additional_info, sub_cadastral_info
                ) VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING cadastral_info_id
            )
            SELECT cadastral_info_id FROM inserted
            UNION ALL
            SELECT cadastral_info_id FROM dim_cadastral_info
            WHERE (cadastral IS NULL AND %s IS NULL OR cadastral = %s)
            AND (cadastral_additional_info IS NULL AND %s IS NULL
                 OR cadastral_additional_info = %s)
            AND (sub_cadastral_info IS NULL AND %s IS NULL OR sub_cadastral_info = %s)
            LIMIT 1
            '''
            params = (
                cadastral, cadastral_additional_info, sub_cadastral_info,
                cadastral, cadastral,
                cadastral_additional_info, cadastral_additional_info,
                sub_cadastral_info, sub_cadastral_info
            )
            result = execute_query_silent(postgres_hook, query, params)

            cadastral_id = result[0][0] if result else None
            if cadastral_id:
                cadastral_info_mappings[cadastral_key] = cadastral_id

    dimension_counts['cadastral_info'] = len(cadastral_info_mappings)
    return cadastral_info_mappings

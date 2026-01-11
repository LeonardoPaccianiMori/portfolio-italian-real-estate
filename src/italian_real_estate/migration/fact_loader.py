"""
Fact table loading for the PostgreSQL migration.

This module provides functions for loading data into the fact_listing table
and related bridge tables (listing_features_bridge, surface_composition_bridge)
during the migration from MongoDB warehouse to PostgreSQL data warehouse.

The fact_listing table is the central fact table in the star schema,
containing foreign key references to all dimension tables plus the
price and surface metrics.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable

from ..config.logging_config import get_logger
from ..core.dict_utils import convert_nested_dict_str_keys_to_tuple
from .postgres_utils import execute_query_silent, process_total_room_number

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def load_fact_table_for_batch(batch_num: int = 0, **kwargs) -> str:
    """
    Load data into the fact table for a specific batch.

    This function takes the processed dimension mappings and batch data
    from XCom, creates the appropriate relationships between dimensions,
    and inserts records into the fact_listing table and bridge tables.

    Args:
        batch_num: The batch number to process (0-indexed).
        **kwargs: Airflow task context including 'ti' (TaskInstance)
            for XCom operations.

    Returns:
        Status message indicating completion and count of records loaded.

    Side Effects:
        - Inserts records into fact_listing table
        - Inserts records into listing_features_bridge table
        - Inserts records into surface_composition_bridge table

    Example:
        >>> result = load_fact_table_for_batch(batch_num=0, ti=task_instance)
        >>> print(result)
        "Batch 0: Loaded 500 fact records"
    """
    ti = kwargs['ti']
    logger.info(f"Loading fact table for batch {batch_num}...")
    start_time = time.time()

    # Retrieves dimension mappings from XCom.
    dimension_mappings_serialized = ti.xcom_pull(
        key=f'batch_{batch_num}_dimension_mappings'
    )
    batch_data = ti.xcom_pull(key=f'batch_{batch_num}_data')

    if not dimension_mappings_serialized or not batch_data:
        logger.error(f"Missing data for batch {batch_num}")
        return f"Batch {batch_num}: No data to process"

    # Converts string keys back to tuples.
    dimension_mappings = convert_nested_dict_str_keys_to_tuple(
        dimension_mappings_serialized
    )

    # Gets PostgreSQL connection.
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')

    # Counters for logging.
    records_loaded = 0
    feature_relationships = 0
    surface_relationships = 0

    # Processes each listing in the batch.
    for idx, listing in enumerate(batch_data):
        if (idx + 1) % 100 == 0 or idx + 1 == len(batch_data):
            logger.info(
                f"Loading fact records: listing #{idx+1}/{len(batch_data)}"
            )

        collection_name = listing.get('collection_name', '').lower()
        listing_id = listing.get('listing_id')

        # Processes each data entry (scraping) for this listing.
        for data_entry in listing.get('data', []):
            # Extracts scraping date for date dimension lookup.
            scraping_date = data_entry.get('scraping_date')
            date_id = dimension_mappings.get('date', {}).get(scraping_date)

            if not date_id:
                logger.warning(f"No date_id for scraping_date: {scraping_date}")
                continue

            # Gets listing type ID.
            listing_type_id = dimension_mappings.get(
                'listing_type', {}
            ).get(collection_name)

            if not listing_type_id:
                logger.warning(f"No listing_type_id for: {collection_name}")
                continue

            listing_features = data_entry.get('listing_features', {})

            # Looks up dimension IDs for this listing.
            dimension_ids = _lookup_dimension_ids(
                listing_features,
                dimension_mappings,
                collection_name,
                scraping_date
            )

            # Gets price and surface values.
            price = listing_features.get('price')
            surface = listing_features.get('surface')

            # Inserts fact record.
            fact_id = _insert_fact_record(
                postgres_hook,
                listing_id,
                date_id,
                listing_type_id,
                dimension_ids,
                price,
                surface
            )

            if fact_id:
                records_loaded += 1

                # Inserts feature relationships into bridge table.
                additional_features = listing_features.get('additional_features', [])
                features_mappings = dimension_mappings.get('features', {})

                for feature in additional_features:
                    if feature:
                        feature_lower = feature.lower()
                        feature_id = features_mappings.get(feature_lower)

                        if feature_id:
                            _insert_feature_bridge(
                                postgres_hook, fact_id, feature_id
                            )
                            feature_relationships += 1

                # Inserts surface composition relationships.
                surface_composition = listing_features.get('surface_composition')
                if surface_composition:
                    surface_key = _build_surface_composition_key(surface_composition)
                    surface_comp_id = dimension_mappings.get(
                        'surface_composition', {}
                    ).get(surface_key)

                    if surface_comp_id:
                        _insert_surface_bridge(
                            postgres_hook, fact_id, surface_comp_id
                        )
                        surface_relationships += 1

    elapsed_time = time.time() - start_time
    logger.info(
        f"Batch {batch_num} fact loading complete in {elapsed_time:.2f} seconds. "
        f"Records: {records_loaded}, Features: {feature_relationships}, "
        f"Surfaces: {surface_relationships}"
    )

    return f"Batch {batch_num}: Loaded {records_loaded} fact records"


def _lookup_dimension_ids(
    listing_features: Dict[str, Any],
    dimension_mappings: Dict[str, Any],
    collection_name: str,
    scraping_date: str
) -> Dict[str, Optional[int]]:
    """
    Look up all dimension IDs for a listing.

    Args:
        listing_features: Dictionary of listing features from MongoDB.
        dimension_mappings: Processed dimension mappings from batch.
        collection_name: The collection name (rent, sale, auction).
        scraping_date: The scraping date string.

    Returns:
        Dictionary mapping dimension names to their IDs.
    """
    dimension_ids: Dict[str, Optional[int]] = {}

    # Seller type lookup.
    listing_info = listing_features.get('listing_info', {})
    seller_type = listing_info.get('seller_type')
    if seller_type and isinstance(seller_type, str):
        seller_type = seller_type.lower()
    seller_type_mappings = dimension_mappings.get('seller_type', {})
    seller_type_id = seller_type_mappings.get(seller_type) or seller_type_mappings.get(None)
    dimension_ids['seller_type_id'] = seller_type_id

    # Listing info lookup.
    listing_age = listing_info.get('listing_age')
    listing_last_update = listing_info.get('listing_last_update')
    number_of_photos = listing_info.get('number_of_photos')
    listing_info_key = (listing_age, listing_last_update, number_of_photos, seller_type_id)
    listing_info_mappings = dimension_mappings.get('listing_info', {})
    dimension_ids['listing_info_id'] = listing_info_mappings.get(listing_info_key)

    # Availability lookup.
    availability = listing_features.get('availability')
    if availability and isinstance(availability, str):
        availability = availability.lower()
    availability_mappings = dimension_mappings.get('availability', {})
    dimension_ids['availability_id'] = (
        availability_mappings.get(availability) or availability_mappings.get(None)
    )

    # Type of property lookup.
    type_of_property = listing_features.get('type_of_property', {})
    property_class = type_of_property.get('class')
    if property_class and isinstance(property_class, str):
        property_class = property_class.lower()
    type_of_property_mappings = dimension_mappings.get('type_of_property', {})
    dimension_ids['type_of_property_id'] = (
        type_of_property_mappings.get(property_class)
        or type_of_property_mappings.get(None)
    )

    # Condition lookup.
    condition = listing_features.get('condition')
    if condition and isinstance(condition, str):
        condition = condition.lower()
    condition_mappings = dimension_mappings.get('condition', {})
    dimension_ids['condition_id'] = (
        condition_mappings.get(condition) or condition_mappings.get(None)
    )

    # Additional costs lookup.
    additional_costs = listing_features.get('additional_costs', {})
    condominium_monthly_expenses = additional_costs.get('condominium_monthly_expenses')
    heating_yearly_expenses = additional_costs.get('heating_yearly_expenses')

    # Gets mortgage rate for sale/auction.
    adjusted_rates = json.loads(Variable.get("adjusted_mortgage_rates", "{}"))
    mortgage_rate = None
    estimated_monthly_payment = None

    if collection_name in ['sale', 'auction']:
        mortgage_rate = adjusted_rates.get(scraping_date)
        if mortgage_rate is not None:
            price = listing_features.get('price')
            if price is not None:
                from ..core.numeric_utils import calculate_monthly_payment
                estimated_monthly_payment = calculate_monthly_payment(
                    price, mortgage_rate
                )

    costs_key = (
        condominium_monthly_expenses, heating_yearly_expenses,
        mortgage_rate, estimated_monthly_payment
    )
    additional_costs_mappings = dimension_mappings.get('additional_costs', {})
    dimension_ids['additional_costs_id'] = additional_costs_mappings.get(costs_key)

    # Category lookup.
    category = listing_features.get('category', {})
    category_name = category.get('name')
    if category_name and isinstance(category_name, str):
        category_name = category_name.lower()
    category_mappings = dimension_mappings.get('category', {})
    dimension_ids['category_id'] = (
        category_mappings.get(category_name) or category_mappings.get(None)
    )

    # Rooms info lookup.
    rooms_info = listing_features.get('rooms_info', {})
    total_room_number = process_total_room_number(rooms_info.get('total_room_number'))
    bathrooms_number = rooms_info.get('bathrooms_number')
    kitchen_status = rooms_info.get('kitchen_status')
    if kitchen_status and isinstance(kitchen_status, str):
        kitchen_status = kitchen_status.lower()
    garage = rooms_info.get('garage')
    if garage and isinstance(garage, str):
        garage = garage.lower()
    floor = rooms_info.get('floor')
    rooms_key = (total_room_number, bathrooms_number, kitchen_status, garage, floor)
    rooms_info_mappings = dimension_mappings.get('rooms_info', {})
    dimension_ids['rooms_info_id'] = rooms_info_mappings.get(rooms_key)

    # Building info lookup.
    building_info = listing_features.get('building_info', {})
    total_building_floors = building_info.get('total_building_floors')
    has_elevator = building_info.get('has_elevator')
    building_usage = building_info.get('building_usage')
    if building_usage and isinstance(building_usage, str):
        building_usage = building_usage.lower()
    building_year = building_info.get('building_year')
    building_key = (total_building_floors, has_elevator, building_usage, building_year)
    building_info_mappings = dimension_mappings.get('building_info', {})
    dimension_ids['building_info_id'] = building_info_mappings.get(building_key)

    # Energy info lookup.
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
    energy_key = (is_zero_energy_building, heating_type, energy_class, air_conditioning)
    energy_info_mappings = dimension_mappings.get('energy_info', {})
    dimension_ids['energy_info_id'] = energy_info_mappings.get(energy_key)

    # Location info lookup.
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
    location_info_mappings = dimension_mappings.get('location_info', {})
    dimension_ids['location_info_id'] = location_info_mappings.get(location_key)

    # Cadastral info lookup.
    cadastral_info = listing_features.get('cadastral_info', {})
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
    cadastral_info_mappings = dimension_mappings.get('cadastral_info', {})
    dimension_ids['cadastral_info_id'] = cadastral_info_mappings.get(cadastral_key)

    # Auction info lookup (only for auction listings).
    dimension_ids['auction_info_id'] = None
    if collection_name == 'auction':
        auction_info = listing_features.get('auction_info', {})
        if auction_info:
            # Builds the auction key.
            import hashlib
            deposit_modality = auction_info.get('deposit_modality')
            if deposit_modality and isinstance(deposit_modality, str):
                deposit_modality = deposit_modality.lower()
            deposit_modality_hash = None
            if deposit_modality:
                deposit_modality_hash = hashlib.md5(
                    deposit_modality.encode()
                ).hexdigest()[:16]

            auction_type = auction_info.get('auction_type')
            if auction_type and isinstance(auction_type, str):
                auction_type = auction_type.lower()

            lot_category = auction_info.get('lot_category', {})
            lot_category_id = lot_category.get('id') if lot_category else None
            lot_category_name = lot_category.get('name') if lot_category else None
            if lot_category_name and isinstance(lot_category_name, str):
                lot_category_name = lot_category_name.lower()

            auction_court = auction_info.get('auction_court')
            if auction_court and isinstance(auction_court, str):
                auction_court = auction_court.lower()

            procedure_number = auction_info.get('procedure_number')
            if procedure_number and isinstance(procedure_number, str):
                procedure_number = procedure_number.lower()

            auction_key = (
                auction_info.get('auction_end_date'),
                deposit_modality,
                deposit_modality_hash,
                auction_type,
                auction_info.get('is_open'),
                auction_info.get('minimum_offer'),
                procedure_number,
                auction_court,
                lot_category_id,
                lot_category_name
            )
            auction_info_mappings = dimension_mappings.get('auction_info', {})
            dimension_ids['auction_info_id'] = auction_info_mappings.get(auction_key)

    return dimension_ids


def _insert_fact_record(
    postgres_hook: Any,
    listing_id: str,
    date_id: int,
    listing_type_id: int,
    dimension_ids: Dict[str, Optional[int]],
    price: Optional[float],
    surface: Optional[float]
) -> Optional[int]:
    """
    Insert a record into the fact_listing table.

    Args:
        postgres_hook: PostgreSQL connection hook.
        listing_id: The listing ID from MongoDB.
        date_id: The date dimension ID.
        listing_type_id: The listing type dimension ID.
        dimension_ids: Dictionary of other dimension IDs.
        price: The listing price.
        surface: The listing surface area.

    Returns:
        The fact_listing_id of the inserted record, or None on failure.
    """
    query = '''
    INSERT INTO fact_listing (
        listing_id, date_id, listing_type_id,
        listing_info_id, availability_id, type_of_property_id,
        condition_id, additional_costs_id, category_id,
        rooms_info_id, building_info_id, energy_info_id,
        location_info_id, cadastral_info_id, auction_info_id,
        price, surface
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT DO NOTHING
    RETURNING fact_listing_id
    '''

    params = (
        listing_id,
        date_id,
        listing_type_id,
        dimension_ids.get('listing_info_id'),
        dimension_ids.get('availability_id'),
        dimension_ids.get('type_of_property_id'),
        dimension_ids.get('condition_id'),
        dimension_ids.get('additional_costs_id'),
        dimension_ids.get('category_id'),
        dimension_ids.get('rooms_info_id'),
        dimension_ids.get('building_info_id'),
        dimension_ids.get('energy_info_id'),
        dimension_ids.get('location_info_id'),
        dimension_ids.get('cadastral_info_id'),
        dimension_ids.get('auction_info_id'),
        price,
        surface
    )

    result = execute_query_silent(postgres_hook, query, params)
    return result[0][0] if result else None


def _insert_feature_bridge(
    postgres_hook: Any,
    fact_listing_id: int,
    feature_id: int
) -> None:
    """
    Insert a record into the listing_features_bridge table.

    Args:
        postgres_hook: PostgreSQL connection hook.
        fact_listing_id: The fact table record ID.
        feature_id: The feature dimension ID.
    """
    query = '''
    INSERT INTO listing_features_bridge (fact_listing_id, feature_id)
    VALUES (%s, %s)
    ON CONFLICT DO NOTHING
    '''
    execute_query_silent(postgres_hook, query, (fact_listing_id, feature_id))


def _insert_surface_bridge(
    postgres_hook: Any,
    fact_listing_id: int,
    surface_composition_id: int
) -> None:
    """
    Insert a record into the surface_composition_bridge table.

    Args:
        postgres_hook: PostgreSQL connection hook.
        fact_listing_id: The fact table record ID.
        surface_composition_id: The surface composition dimension ID.
    """
    query = '''
    INSERT INTO surface_composition_bridge (fact_listing_id, surface_composition_id)
    VALUES (%s, %s)
    ON CONFLICT DO NOTHING
    '''
    execute_query_silent(
        postgres_hook, query, (fact_listing_id, surface_composition_id)
    )


def _build_surface_composition_key(
    surface_composition: Dict[str, Any]
) -> Tuple:
    """
    Build a lookup key for surface composition.

    Args:
        surface_composition: Dictionary with surface composition data.

    Returns:
        Tuple key for surface composition mapping lookup.
    """
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

    return (element_name, floor, surface, percentage, commercial_surface)

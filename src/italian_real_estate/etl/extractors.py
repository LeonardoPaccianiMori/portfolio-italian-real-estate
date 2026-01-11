"""
Data extraction utilities for the Italian Real Estate ETL pipeline.

This module provides functions for extracting structured data from raw HTML
stored in the MongoDB datalake. It parses the embedded JSON metadata and
transforms it into a normalized feature dictionary suitable for storage
in the MongoDB warehouse.

The extraction process handles various data quality issues including:
    - Missing fields (defaults to None)
    - Inconsistent data types
    - Nested JSON structures
    - Multiple date formats

Author: Leonardo Pacciani-Mori
License: MIT
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

from bs4 import BeautifulSoup

from italian_real_estate.config.logging_config import get_logger
from italian_real_estate.core.string_utils import remove_non_numbers, safe_lower
from italian_real_estate.core.numeric_utils import (
    mortgage_monthly_payment,
    calculate_price_per_sqm,
    safe_float_conversion,
    safe_int_conversion,
)
from italian_real_estate.core.date_utils import (
    calculate_date_difference_days,
    parse_timestamp_to_date,
    parse_work_dates,
    calculate_work_completion,
)

# Initialize module logger
logger = get_logger(__name__)


async def extract_and_transform_data(
    listing: Dict[str, Any],
    listing_type: str
) -> Dict[str, Any]:
    """
    Extract and transform all features from a listing's raw HTML data.

    This is the main extraction function that processes a listing document
    from the MongoDB datalake and returns a structured dictionary of features.
    It parses the embedded JSON metadata and extracts ~40 different features
    organized into logical groups.

    The function handles all three listing types (sale, rent, auction) and
    extracts features that may or may not be present depending on the type.
    Missing values are represented as None.

    Args:
        listing: A MongoDB document from the datalake containing:
            - _id: The listing ID
            - province_name: Province where the listing is located
            - data: List of scraping entries with html_source and scraping_date
            - parent_listing: ID of parent listing if this is a sub-listing
            - child_listings: List of sub-listing IDs if this has children
        listing_type: The type of listing ("sale", "rent", or "auction").

    Returns:
        dict: A dictionary containing all extracted features organized into
            groups: price info, listing info, additional costs, property
            details, room info, building info, energy info, location info,
            auction info (if applicable), and text descriptions.

    Example:
        >>> listing = datalake_collection.find_one({"_id": 12345})
        >>> features = await extract_and_transform_data(listing, "sale")
        >>> print(features['price'])  # 250000
    """
    # Get the most recent scraping data
    source_code = listing["data"][-1]["html_source"]
    last_date_scraped = listing["data"][-1]["scraping_date"]

    # Parse the HTML and extract the metadata JSON
    soup = BeautifulSoup(source_code, 'html.parser')

    # Find the script tag containing the page metadata
    script = soup.find("script", id="__NEXT_DATA__")

    if not script or not script.contents:
        logger.warning(f"No metadata found for listing {listing['_id']}")
        return _get_empty_features()

    try:
        script_content = script.contents[0].replace("\n", "").replace("   ", "").replace("  ", "")
        data_dict = json.loads(script_content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for listing {listing['_id']}: {str(e)}")
        return _get_empty_features()

    # Navigate to the property data sections
    page_props = data_dict.get('props', {}).get('pageProps', {})
    detail_data = page_props.get('detailData', {})
    real_estate = detail_data.get('realEstate', {})
    properties = real_estate.get('properties', [{}])[0] if real_estate.get('properties') else {}

    # For readability, create shortcuts to commonly accessed dictionaries
    properties_general = real_estate
    prop_gen_keys = properties_general.keys()
    properties_deeper = properties
    prop_deeper_keys = properties_deeper.keys()

    # =========================================================================
    # LISTING INFORMATION
    # =========================================================================

    # Extract listing creation date and calculate listing age
    listing_creation_date = None
    listing_age = None

    if 'createdAt' in prop_gen_keys:
        timestamp = properties_general.get('createdAt')
        if timestamp:
            listing_creation_date = parse_timestamp_to_date(timestamp)

    if listing_creation_date:
        listing_age = calculate_date_difference_days(
            listing_creation_date, last_date_scraped
        )

    # Extract last update date and calculate days since update
    last_update = None
    listing_last_update = None

    if 'lastUpdate' in prop_deeper_keys:
        last_update_str = properties_deeper.get('lastUpdate', '')
        if last_update_str:
            # Parse the date from format like "Aggiornato il 15/01/2024"
            date_parts = last_update_str.split(" ")[-1].split("/")
            if len(date_parts) == 3:
                last_update = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                # Fix abbreviated year format
                if len(last_update) == 8:
                    last_update = "20" + last_update

    if last_update:
        listing_last_update = calculate_date_difference_days(
            last_update, last_date_scraped
        )

    # Check if listing is new
    is_listing_new = properties_general.get('isNew')

    # Determine parent/child relationships
    has_parent = listing.get("parent_listing") is not None
    has_child = listing.get("child_listings") is not None

    # Extract seller type
    seller_type = None
    advertiser = properties_general.get('advertiser')
    if advertiser:
        if 'agency' in advertiser:
            seller_type = advertiser['agency'].get('label')
        else:
            seller_type = 'privato'

    # Count photos
    number_of_photos = None
    multimedia = properties_deeper.get('multimedia', {})
    if multimedia and 'photos' in multimedia:
        number_of_photos = len(multimedia.get('photos', []))

    # =========================================================================
    # PRICE AND COSTS
    # =========================================================================

    # Extract price
    price = None
    price_data = properties_deeper.get('price', {})
    if price_data:
        price = price_data.get('value')

    # Extract additional costs
    additional_costs = None
    costs_data = properties_deeper.get('costs')
    if costs_data:
        additional_costs = {
            'condominium_expenses': costs_data.get('condominiumExpenses'),
            'heating_expenses': costs_data.get('heatingExpenses')
        }

    # Extract mortgage rate
    mortgage_rate = None
    mortgage_data = properties_general.get('mortgage', {})
    if mortgage_data and mortgage_data.get('mortgageWidget'):
        rates_data = mortgage_data['mortgageWidget'].get('rates', [])
        # Find the 80% LTV rate for 30-year mortgage
        for rate_group in rates_data:
            if rate_group.get('percent') == 80:
                for rate_entry in rate_group.get('rates', []):
                    if rate_entry.get('year') == 30:
                        mortgage_rate = rate_entry.get('rate')
                        break

    # Mortgage payment will be calculated during PostgreSQL migration
    mortgage_payment = None

    # =========================================================================
    # PROPERTY DETAILS
    # =========================================================================

    # Extract total surface
    total_surface = _extract_surface(properties_deeper)

    # Calculate price per square meter
    price_per_sq_mt = calculate_price_per_sqm(price, total_surface)

    # Extract surface composition
    surface_composition = _extract_surface_composition(properties_deeper)

    # Extract availability
    availability = properties_deeper.get('availability')

    # Extract category
    category = None
    category_data = properties_deeper.get('category', {})
    if category_data:
        category = safe_lower(category_data.get('name'))

    # Extract type of property
    type_of_property = safe_lower(properties_deeper.get('typologyValue'))

    # Check if luxury
    is_luxury = properties_general.get('luxury')

    # Extract condition
    condition = safe_lower(properties_deeper.get('condition'))

    # =========================================================================
    # ROOM INFORMATION
    # =========================================================================

    # Extract bathroom count
    bathrooms_number = _parse_room_count(properties_deeper.get('bathrooms'))

    # Extract bedroom count
    bedrooms_number = safe_int_conversion(properties_deeper.get('bedRoomsNumber'))

    # Extract total room number
    total_room_number = _parse_room_count(properties_deeper.get('rooms'))

    # Extract rooms description
    rooms_description = safe_lower(properties_deeper.get('roomsValue'))

    # Extract kitchen status
    kitchen_status = safe_lower(properties_deeper.get('kitchenStatus'))

    # Extract garage info
    garage = properties_deeper.get('garage')

    # Extract floor
    property_floor = _extract_floor(properties_deeper)

    # =========================================================================
    # ADDITIONAL FEATURES
    # =========================================================================

    additional_features = properties_deeper.get('ga4features')

    # =========================================================================
    # ENERGY INFORMATION
    # =========================================================================

    energy_info = _extract_energy_info(properties_deeper)

    # =========================================================================
    # LOCATION INFORMATION
    # =========================================================================

    location_info = _extract_location_info(properties_deeper)

    # =========================================================================
    # BUILDING INFORMATION
    # =========================================================================

    # Extract elevator status
    has_elevator = properties_deeper.get('elevator')

    # Extract building usage
    building_usage = None
    building_usage_data = properties_deeper.get('buildingUsage')
    if building_usage_data:
        building_usage = building_usage_data.get('value')

    # Extract building year
    building_year = safe_int_conversion(properties_deeper.get('buildingYear'))

    # Extract total floors
    total_building_floors = None
    floors_str = properties_deeper.get('floors')
    if floors_str:
        floors_num = remove_non_numbers(str(floors_str))
        if floors_num:
            total_building_floors = int(float(floors_num))

    # Extract residential units
    total_number_of_residential_units = properties_deeper.get('residentialUnits')

    # Extract work dates and completion
    work_start_date, work_end_date = parse_work_dates(
        properties_deeper.get('workDates', '')
    )

    work_completion = calculate_work_completion(
        last_date_scraped, work_start_date, work_end_date
    )

    work_progress = properties_deeper.get('workProgress')

    # =========================================================================
    # AUCTION INFORMATION (auction listings only)
    # =========================================================================

    auction_info = _extract_auction_info(properties_deeper)

    # =========================================================================
    # CADASTRAL INFORMATION
    # =========================================================================

    cadastral_info = _extract_cadastral_info(properties_deeper)

    # =========================================================================
    # TEXT DESCRIPTIONS
    # =========================================================================

    title = properties_general.get('title')
    caption = properties_deeper.get('caption')
    description = properties_deeper.get('description')

    # =========================================================================
    # ASSEMBLE FEATURE DICTIONARY
    # =========================================================================

    features = {
        "price": price,
        "total_surface": total_surface,
        "price_per_sq_mt": price_per_sq_mt,
        "listing_info": {
            "listing_age": listing_age,
            "listing_last_update": listing_last_update,
            "is_listing_new": is_listing_new,
            "has_parent": has_parent,
            "has_child": has_child,
            "seller_type": seller_type,
            "number_of_photos": number_of_photos
        },
        "additional_costs": {
            "other_costs": additional_costs,
            "mortgage_rate": mortgage_rate
        },
        "auction_info": auction_info,
        "cadastral_info": cadastral_info,
        "surface_composition": surface_composition,
        "availability": availability,
        "type_of_property": {
            "class": type_of_property,
            "category": category,
            "is_luxury": is_luxury
        },
        "condition": condition,
        "rooms_info": {
            "bathrooms_number": bathrooms_number,
            "bedrooms_number": bedrooms_number,
            "total_room_number": total_room_number,
            "rooms_description": rooms_description,
            "kitchen_status": kitchen_status,
            "garage": garage,
            "property_floor": property_floor
        },
        "additional_features": additional_features,
        "building_info": {
            "has_elevator": has_elevator,
            "building_usage": building_usage,
            "building_year": building_year,
            "total_building_floors": total_building_floors,
            "total_number_of_residential_units": total_number_of_residential_units,
            "work_start_date": work_start_date,
            "work_end_date": work_end_date,
            "work_completion": work_completion,
            "work_progress": work_progress
        },
        "energy_info": energy_info,
        "location_info": location_info,
        "text_info": {
            "title": title,
            "caption": caption,
            "description": description
        }
    }

    return features


def _get_empty_features() -> Dict[str, Any]:
    """
    Return an empty features dictionary with all keys set to None.

    This is used when extraction fails to ensure consistent structure.

    Returns:
        dict: A features dictionary with all values set to None.
    """
    return {
        "price": None,
        "total_surface": None,
        "price_per_sq_mt": None,
        "listing_info": None,
        "additional_costs": None,
        "auction_info": None,
        "cadastral_info": None,
        "surface_composition": None,
        "availability": None,
        "type_of_property": None,
        "condition": None,
        "rooms_info": None,
        "additional_features": None,
        "building_info": None,
        "energy_info": None,
        "location_info": None,
        "text_info": None
    }


def _extract_surface(properties: Dict[str, Any]) -> Optional[float]:
    """Extract and parse the total surface area."""
    surface = properties.get('surface') or properties.get('surfaceValue')

    if surface is None:
        return None

    # Parse numeric value from string like "120 m²"
    surface_str = remove_non_numbers(str(surface))
    if surface_str:
        # Remove the ² symbol and any periods
        surface_str = surface_str.replace("²", "").replace(".", "")
        try:
            return float(surface_str)
        except ValueError:
            return None

    return None


def _extract_surface_composition(properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the breakdown of surface areas (apartment, garage, etc.)."""
    composition_data = properties.get('surfaceConstitution')

    if not composition_data:
        return None

    elements = composition_data.get('surfaceConstitutionElements', [])
    if not elements:
        return None

    # Process each element in the composition
    result = {}
    for element in elements:
        element_type = element.get('constitution')
        if element_type:
            result[element_type] = {
                'floor': element.get('floor', {}).get('value'),
                'surface': _parse_surface_value(element.get('surface')),
                'percentage': element.get('percentage'),
                'commercial_surface': _parse_surface_value(element.get('commercialSurface')),
                'surface_type': element.get('surfaceType')
            }

    return result if result else None


def _parse_surface_value(value: Optional[str]) -> Optional[float]:
    """Parse a surface value from string format."""
    if not value:
        return None

    # Remove thousands separator and get numeric part
    clean_value = remove_non_numbers(str(value).split(",")[0].replace(".", ""))

    if clean_value:
        try:
            return float(clean_value)
        except ValueError:
            return None

    return None


def _parse_room_count(value: Any) -> Optional[int]:
    """Parse room count, handling special values like '3+'."""
    if value is None:
        return None

    if isinstance(value, int):
        return value

    value_str = str(value)

    # Handle "3+" notation
    if value_str == "3+":
        return 4
    if value_str == "5+":
        return 6

    try:
        return int(value_str)
    except ValueError:
        return None


def _extract_floor(properties: Dict[str, Any]) -> Optional[int]:
    """Extract the floor number from property data."""
    floor_data = properties.get('floor')

    if not floor_data:
        return None

    # Ground floor is encoded as 'T' or contains "terra"
    abbrev = floor_data.get('abbreviation', '')
    floor_value = floor_data.get('floorOnlyValue', '')

    if abbrev == 'T' or (floor_value and 'terra' in str(floor_value).lower()):
        return 0

    # Try to get numeric floor
    if isinstance(floor_value, int):
        return floor_value

    if isinstance(floor_value, str):
        try:
            return int(floor_value)
        except ValueError:
            return None

    return None


def _extract_energy_info(properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract energy efficiency information."""
    energy_data = properties.get('energy')

    if not energy_data:
        return None

    return {
        'is_zero_energy_building': energy_data.get('zeroEnergyBuilding'),
        'heating_type': energy_data.get('heatingType'),
        'energy_class': energy_data.get('class'),
        'energy_consumption': _parse_energy_consumption(energy_data.get('epi')),
        'air_conditioning': energy_data.get('airConditioning')
    }


def _parse_energy_consumption(value: Any) -> Optional[float]:
    """Parse energy consumption value (kWh/m²)."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        numeric = remove_non_numbers(value)
        if numeric:
            try:
                return float(numeric)
            except ValueError:
                return None

    return None


def _extract_location_info(properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract location information."""
    location_data = properties.get('location')

    if not location_data:
        return None

    return {
        'latitude': location_data.get('latitude'),
        'longitude': location_data.get('longitude'),
        'region': safe_lower(location_data.get('region', '')).replace(" ", "-"),
        'province': safe_lower(location_data.get('province', '')).replace(" ", "-"),
        'province_code': location_data.get('provinceId'),
        'city': location_data.get('city'),
        'macrozone': location_data.get('macrozone'),
        'microzone': location_data.get('microzone'),
        'locality': location_data.get('locality'),
        'address': location_data.get('address'),
        'street_number': location_data.get('streetNumber')
    }


def _extract_auction_info(properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract auction-specific information."""
    auction_data = properties.get('auction')

    if not auction_data:
        return None

    # Parse minimum offer amount
    minimum_offer = None
    min_offer_str = auction_data.get('minimumOffer', '')
    if min_offer_str:
        numeric = remove_non_numbers(min_offer_str.split(",")[0].replace(".", ""))
        if numeric:
            try:
                minimum_offer = float(numeric)
            except ValueError:
                pass

    return {
        'auction_end_date': auction_data.get('saleDateValue'),
        'deposit_modality': auction_data.get('modalityDeposit'),
        'auction_type': auction_data.get('saleType'),
        'is_open': auction_data.get('saleState', {}).get('isAvailable'),
        'minimum_offer': minimum_offer,
        'procedure_number': auction_data.get('procedureNumber'),
        'auction_court': auction_data.get('auctionCourt'),
        'lot_category': auction_data.get('lotCategory')
    }


def _extract_cadastral_info(properties: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract cadastral information."""
    cadastrals = properties.get('cadastrals', [])

    if not cadastrals:
        return None

    # Get the first cadastral entry
    cadastral_data = cadastrals[0] if cadastrals else {}

    if not cadastral_data:
        return None

    return {
        'cadastral': cadastral_data.get('cadastral'),
        'cadastral_additional_info': cadastral_data.get('cadastralInfo'),
        'sub_cadastral_info': cadastral_data.get('subCadastral')
    }

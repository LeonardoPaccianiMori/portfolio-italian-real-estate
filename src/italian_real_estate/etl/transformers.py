"""
Data transformation utilities for the Italian Real Estate ETL pipeline.

This module provides functions for cleaning and transforming data during
the ETL process. It handles data quality issues like empty arrays that
should be null, and provides utilities for data validation.

Author: Leonardo Pacciani-Mori
License: MIT
"""

from typing import Dict, Any, List

from italian_real_estate.config.logging_config import get_logger
from italian_real_estate.core.connections import (
    get_mongodb_client,
    get_warehouse_collections,
)

# Initialize module logger
logger = get_logger(__name__)


def fix_empty_child_listings() -> int:
    """
    Replace empty arrays in child_listings with null values.

    Some listings that should have null values in the child_listings field
    end up having empty arrays due to the upsert logic. This function
    iterates over all documents in all warehouse collections and fixes
    this data quality issue.

    The distinction matters for downstream processing:
        - None: The listing was never checked for children
        - Empty array []: The listing was checked and has no children

    For consistency, we want empty arrays to be null.

    Returns:
        int: The total number of documents fixed across all collections.

    Example:
        >>> fixed_count = fix_empty_child_listings()
        >>> print(f"Fixed {fixed_count} documents")
    """
    logger.info("Starting to fix empty child_listings arrays")

    # Connect to MongoDB warehouse
    client = get_mongodb_client()
    sale_coll, rent_coll, auction_coll = get_warehouse_collections(client)

    collections = {
        "sale": sale_coll,
        "rent": rent_coll,
        "auction": auction_coll
    }

    total_fixed = 0

    # Process each collection
    for coll_name, collection in collections.items():
        fixed_count = 0

        # Find documents with empty arrays in child_listings
        cursor = collection.find({"child_listings": []})

        for doc in cursor:
            # Replace the empty array with null
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"child_listings": None}}
            )
            fixed_count += 1

        logger.info(
            f"Fixed {fixed_count} documents with empty child_listings "
            f"arrays in {coll_name} collection"
        )
        total_fixed += fixed_count

    # Close MongoDB connection
    client.close()

    logger.info(
        f"Completed fixing empty child_listings arrays. Total fixed: {total_fixed}"
    )

    return total_fixed


def validate_listing_data(features: Dict[str, Any]) -> List[str]:
    """
    Validate the extracted features for a listing.

    This function checks the extracted features for common issues and
    returns a list of validation warnings. It doesn't modify the data
    but helps identify data quality issues.

    Args:
        features: The extracted features dictionary from extract_and_transform_data.

    Returns:
        list: A list of validation warning messages. Empty if no issues found.

    Example:
        >>> features = await extract_and_transform_data(listing, "sale")
        >>> warnings = validate_listing_data(features)
        >>> if warnings:
        ...     for warning in warnings:
        ...         logger.warning(warning)
    """
    warnings = []

    # Check for missing critical fields
    if features.get('price') is None:
        warnings.append("Missing price")

    if features.get('total_surface') is None:
        warnings.append("Missing total surface")

    # Check for unreasonable values
    price = features.get('price')
    if price is not None and price <= 0:
        warnings.append(f"Invalid price: {price}")

    surface = features.get('total_surface')
    if surface is not None and surface <= 0:
        warnings.append(f"Invalid surface: {surface}")

    # Check location info
    location = features.get('location_info')
    if location:
        if location.get('latitude') is None or location.get('longitude') is None:
            warnings.append("Missing coordinates")

    # Check for extremely high price per sqm (potential data error)
    price_per_sqm = features.get('price_per_sq_mt')
    if price_per_sqm is not None and price_per_sqm > 50000:
        warnings.append(f"Unusually high price per sqm: {price_per_sqm}")

    return warnings


def clean_text_fields(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and normalize text fields in the features dictionary.

    This function performs basic text cleaning on text fields including:
        - Stripping leading/trailing whitespace
        - Normalizing multiple spaces
        - Converting to lowercase where appropriate

    Args:
        features: The extracted features dictionary.

    Returns:
        dict: The features dictionary with cleaned text fields.

    Example:
        >>> features = await extract_and_transform_data(listing, "sale")
        >>> features = clean_text_fields(features)
    """
    # Make a copy to avoid modifying the original
    cleaned = features.copy()

    # Clean text_info fields
    text_info = cleaned.get('text_info', {})
    if text_info:
        for key in ['title', 'caption', 'description']:
            if text_info.get(key):
                text_info[key] = _clean_text(text_info[key])
        cleaned['text_info'] = text_info

    return cleaned


def _clean_text(text: str) -> str:
    """
    Clean a single text string.

    Args:
        text: The text to clean.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return text

    # Strip whitespace
    text = text.strip()

    # Normalize multiple spaces to single space
    import re
    text = re.sub(r'\s+', ' ', text)

    return text


def filter_invalid_listings(
    listings: List[Dict[str, Any]],
    require_price: bool = True,
    require_surface: bool = True,
    require_location: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter out listings that don't meet minimum data requirements.

    This function removes listings that are missing critical data fields.
    It's useful for ensuring data quality before downstream processing.

    Args:
        listings: A list of listing feature dictionaries.
        require_price: If True, exclude listings without a price.
        require_surface: If True, exclude listings without a surface area.
        require_location: If True, exclude listings without coordinates.

    Returns:
        list: The filtered list of listings that meet all requirements.

    Example:
        >>> all_listings = [extract_features(l) for l in raw_listings]
        >>> valid_listings = filter_invalid_listings(all_listings)
    """
    filtered = []

    for listing in listings:
        # Check price requirement
        if require_price and listing.get('price') is None:
            continue

        # Check surface requirement
        if require_surface and listing.get('total_surface') is None:
            continue

        # Check location requirement
        if require_location:
            location = listing.get('location_info', {})
            if not location or location.get('latitude') is None:
                continue

        filtered.append(listing)

    removed_count = len(listings) - len(filtered)
    if removed_count > 0:
        logger.info(f"Filtered out {removed_count} invalid listings")

    return filtered


def merge_listing_versions(listings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple versions of a listing into a single record.

    When a listing is scraped multiple times, each version is stored with
    its scraping date. This function merges them into a single record,
    taking the most recent values for most fields.

    Args:
        listings: A list of feature dictionaries for the same listing,
            each from a different scraping date.

    Returns:
        dict: A merged feature dictionary with the most recent values.

    Example:
        >>> versions = [features_jan, features_feb, features_mar]
        >>> merged = merge_listing_versions(versions)
    """
    if not listings:
        return {}

    if len(listings) == 1:
        return listings[0]

    # Sort by any timestamp indicator if available
    # For now, just return the last one (most recent)
    return listings[-1]

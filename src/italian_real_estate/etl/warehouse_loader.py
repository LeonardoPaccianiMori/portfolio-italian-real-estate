"""
MongoDB warehouse loading utilities for the Italian Real Estate ETL pipeline.

This module provides functions for loading transformed data into the MongoDB
warehouse. It handles the migration from the datalake (raw HTML) to the
warehouse (structured features).

The warehouse uses the same collection structure as the datalake (sale, rent,
auction) but stores extracted features instead of raw HTML.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import asyncio
from typing import Dict, Any, Optional

from italian_real_estate.config.logging_config import get_logger
from italian_real_estate.config.settings import VALID_LISTING_TYPES
from italian_real_estate.core.connections import (
    get_mongodb_client,
    get_datalake_collections,
    get_warehouse_collections,
    get_collection_for_listing_type,
)
from italian_real_estate.etl.extractors import extract_and_transform_data

# Initialize module logger
logger = get_logger(__name__)


async def migrate_async(listing_type: str) -> None:
    """
    Migrate all listings of a specific type from datalake to warehouse.

    This function iterates over all documents in the specified datalake
    collection, extracts structured features from each, and stores them
    in the corresponding warehouse collection.

    The migration is idempotent - it checks for existing entries with the
    same scraping date and skips them to avoid duplicates.

    Args:
        listing_type: The type of listings to migrate ("sale", "rent", or "auction").

    Returns:
        None

    Raises:
        ValueError: If listing_type is not one of the valid types.

    Example:
        >>> await migrate_async("sale")
        # Migrates all sale listings from datalake to warehouse
    """
    # Validate listing_type
    if listing_type not in VALID_LISTING_TYPES:
        raise ValueError(f"listing_type must be one of {VALID_LISTING_TYPES}")

    logger.info(
        f"Initializing migration from data lake to data warehouse "
        f"for {listing_type} collection"
    )

    # Connect to MongoDB
    client = get_mongodb_client()

    # Get source (datalake) and destination (warehouse) collections
    datalake_collection = get_collection_for_listing_type(
        client, listing_type, "datalake"
    )
    warehouse_collection = get_collection_for_listing_type(
        client, listing_type, "warehouse"
    )

    # Count documents to process
    all_ids = datalake_collection.distinct("_id")
    documents_number = len(all_ids)

    logger.info(
        f"There are {documents_number} documents in the {listing_type} collection"
    )
    logger.info(
        f"Migration to data warehouse started for {listing_type} collection "
        f"(might take a while)"
    )

    # Process each document in the datalake
    counter = 1
    for listing_data in datalake_collection.find():
        logger.info(f"Processing listing #{counter}/{documents_number}")

        # Extract metadata
        listing_id = listing_data["_id"]
        province_name = listing_data.get("province_name")
        province_code = listing_data.get("province_code")
        parent_listing_id = listing_data.get("parent_listing")
        child_listings_id = listing_data.get("child_listings")
        last_scraping_date = listing_data["data"][-1]["scraping_date"]

        # Check if this exact scraping date already exists in warehouse
        existing_doc = warehouse_collection.find_one({
            "_id": listing_id,
            "data.scraping_date": last_scraping_date
        })

        if not existing_doc:
            # Extract features from the listing
            listing_features = await extract_and_transform_data(
                listing_data, listing_type
            )

            # Check if the document exists (without this scraping date)
            doc_exists = warehouse_collection.find_one({"_id": listing_id})

            if not doc_exists:
                # Insert new document
                warehouse_collection.insert_one({
                    "_id": listing_id,
                    "province_name": province_name,
                    "province_code": province_code,
                    "parent_listing": parent_listing_id,
                    "child_listings": child_listings_id,
                    "data": [{
                        "scraping_date": last_scraping_date,
                        "listing_features": listing_features
                    }]
                })
            else:
                # Add new data entry to existing document
                warehouse_collection.update_one(
                    {"_id": listing_id},
                    {
                        "$push": {
                            "data": {
                                "scraping_date": last_scraping_date,
                                "listing_features": listing_features
                            }
                        }
                    }
                )

        counter += 1

    # Close connection
    client.close()

    logger.info(f"Migration to data warehouse finished for {listing_type} collection")


def migrate(listing_type: str) -> None:
    """
    Synchronous wrapper for migrate_async.

    This wrapper function allows the async migration to be called from
    synchronous code, such as Airflow's PythonOperator.

    Args:
        listing_type: The type of listings to migrate.

    Returns:
        None
    """
    asyncio.run(migrate_async(listing_type))


def get_warehouse_stats() -> str:
    """
    Generate and log statistics for the MongoDB warehouse.

    This function connects to the warehouse and reports the count of
    documents in each collection. It is typically called at the end of
    an ETL run to provide a summary.

    Returns:
        str: A formatted statistics message.

    Example:
        >>> stats = get_warehouse_stats()
        >>> print(stats)
    """
    logger.info("Starting to generate data warehouse statistics")

    # Connect to MongoDB
    client = get_mongodb_client()
    sale_coll, rent_coll, auction_coll = get_warehouse_collections(client)

    collections = {
        "sale": sale_coll,
        "rent": rent_coll,
        "auction": auction_coll
    }

    # Count documents in each collection
    collection_counts = {}
    for coll_name, collection in collections.items():
        total_docs = collection.count_documents({})
        collection_counts[coll_name] = total_docs

    # Calculate total
    total_documents = sum(collection_counts.values())

    # Generate statistics message
    stats_msg = f'''
MongoDB data warehouse statistics:
-------------------------
Total records: {total_documents}
Collection breakdown:
- Sale: {collection_counts['sale']} records
- Rent: {collection_counts['rent']} records
- Auction: {collection_counts['auction']} records
'''

    logger.info(stats_msg)

    # Close connection
    client.close()

    logger.info("Success! All tasks were executed successfully")

    return stats_msg


def count_listings_by_province(
    listing_type: str,
    database: str = "warehouse"
) -> Dict[str, int]:
    """
    Count listings by province for a specific listing type.

    This utility function provides a breakdown of listing counts by
    province, which is useful for monitoring scraping coverage.

    Args:
        listing_type: The type of listings to count.
        database: Which database to query ("datalake" or "warehouse").

    Returns:
        dict: A dictionary mapping province names to listing counts.

    Example:
        >>> counts = count_listings_by_province("sale")
        >>> print(f"Milano: {counts.get('Milano', 0)} listings")
    """
    client = get_mongodb_client()
    collection = get_collection_for_listing_type(client, listing_type, database)

    # Use aggregation to count by province
    pipeline = [
        {"$group": {"_id": "$province_name", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]

    results = list(collection.aggregate(pipeline))

    # Convert to dictionary
    counts = {doc["_id"]: doc["count"] for doc in results if doc["_id"]}

    client.close()

    return counts


def get_latest_scraping_dates(listing_type: str) -> Dict[str, str]:
    """
    Get the most recent scraping date for each province.

    This function helps identify which provinces have been recently
    scraped and which may need updating.

    Args:
        listing_type: The type of listings to check.

    Returns:
        dict: A dictionary mapping province names to their most recent
            scraping date.

    Example:
        >>> dates = get_latest_scraping_dates("rent")
        >>> for province, date in sorted(dates.items()):
        ...     print(f"{province}: {date}")
    """
    client = get_mongodb_client()
    collection = get_collection_for_listing_type(
        client, listing_type, "datalake"
    )

    # Use aggregation to find max scraping date per province
    pipeline = [
        {"$unwind": "$data"},
        {
            "$group": {
                "_id": "$province_name",
                "latest_date": {"$max": "$data.scraping_date"}
            }
        },
        {"$sort": {"latest_date": -1}}
    ]

    results = list(collection.aggregate(pipeline))

    # Convert to dictionary
    dates = {doc["_id"]: doc["latest_date"] for doc in results if doc["_id"]}

    client.close()

    return dates

"""
Batch processing utilities for the MongoDB to PostgreSQL migration.

This module provides functions for setting up and executing batch processing
of real estate listings during the migration from MongoDB warehouse to
PostgreSQL data warehouse. It includes mortgage rate computation, batch
counting, and batch parameter setup.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from airflow.models import Variable

from ..config.settings import BATCH_SIZE, MAX_RECORDS_PER_COLLECTION, MONGODB_WAREHOUSE_NAME
from ..config.logging_config import get_logger
from ..core.numeric_utils import calculate_monthly_payment
from ..core.dict_utils import (
    convert_nested_dict_tuple_keys_to_str,
    convert_nested_dict_str_keys_to_tuple,
)
from ..core.connections import get_mongodb_client
from .postgres_utils import execute_query_silent
from .dimension_processors import process_dimensions_for_batch

# Module-level logger for consistent logging.
logger = get_logger(__name__)


def compute_adjusted_mortgage_rates() -> Dict[str, float]:
    """
    Compute mortgage rates to apply to each scraping date.

    This function analyzes mortgage rate data from 'sale' and 'auction'
    listings to determine the appropriate rate for each scraping date.
    When multiple rates exist for a date, it uses their average.

    The mortgage rate is used to calculate monthly payment estimates
    for properties in the PostgreSQL warehouse.

    Returns:
        Dictionary mapping scraping date strings to their adjusted
        mortgage rates as floats.

    Example:
        >>> rates = compute_adjusted_mortgage_rates()
        >>> print(rates.get('2024-01-15'))
        0.0425
    """
    logger.info("Computing adjusted mortgage rates...")

    # Connects to MongoDB warehouse.
    client = get_mongodb_client()
    db = client[MONGODB_WAREHOUSE_NAME]

    # Processes only 'sale' and 'auction' collections (not 'rent').
    collections = ['sale', 'auction']

    # Dictionary to store date -> [mortgage_rates] mapping.
    date_mortgage_rates: Dict[str, List[float]] = {}

    # Iterates through each collection.
    for collection_name in collections:
        logger.info(
            f"Processing data to compute adjusted mortgage rates for "
            f"{collection_name} collection"
        )
        logger.info("Gathering data from mongoDB warehouse...")

        # Gets all documents in the collection.
        cursor = db[collection_name].find({})

        logger.info("Processing data...")

        # Processes each document to extract mortgage rates.
        for document in cursor:
            # Iterates through each data entry in the document.
            for data_entry in document.get('data', []):
                # Gets the scraping date for this entry.
                scraping_date = data_entry.get('scraping_date')
                if not scraping_date:
                    continue

                # Initializes list for this date if needed.
                if scraping_date not in date_mortgage_rates:
                    date_mortgage_rates[scraping_date] = []

                # Extracts mortgage rate from nested structure.
                listing_features = data_entry.get('listing_features', {})
                additional_costs = listing_features.get('additional_costs', {})
                mortgage_rate = additional_costs.get('mortgage_rate')

                # Adds non-null rates only.
                if mortgage_rate is not None:
                    date_mortgage_rates[scraping_date].append(mortgage_rate)

    # Calculates adjusted rates (single value or average).
    adjusted_rates: Dict[str, float] = {}

    for date, rates in date_mortgage_rates.items():
        logger.info(f"Computing adjusted mortgage rates for data scraped on {date}")

        if len(rates) == 1:
            # Single rate: use it directly.
            adjusted_rates[date] = rates[0]
        elif len(rates) > 1:
            # Multiple rates: compute average.
            adjusted_rates[date] = sum(rates) / len(rates)
        else:
            # No rates found for this date.
            logger.warning(f"No non-null mortgage rates found for date {date}")

    logger.info(f"Computed adjusted mortgage rates for {len(adjusted_rates)} dates")
    return adjusted_rates


def count_batches() -> int:
    """
    Count the total number of batches needed for processing.

    This function connects to MongoDB to count documents in each collection
    (rent, auction, sale) and calculates how many batches of BATCH_SIZE
    are needed to process all records.

    Only listings where child_listings is null are counted, as listings
    with child_listings are parent records that shouldn't be processed
    independently.

    Returns:
        Total number of batches needed to process all collections.

    Example:
        >>> total = count_batches()
        >>> print(f"Need to process {total} batches")
    """
    logger.info("Counting total batches needed...")

    # Connects directly to MongoDB.
    client = get_mongodb_client()
    db = client[MONGODB_WAREHOUSE_NAME]

    # Defines the collections to process.
    collections = ['rent', 'auction', 'sale']
    total_batches = 0

    for collection_name in collections:
        # Only counts listings without child_listings.
        query = {"child_listings": None}
        total_count = db[collection_name].count_documents(query)

        # Applies the per-collection limit if set.
        if MAX_RECORDS_PER_COLLECTION:
            process_count = min(total_count, MAX_RECORDS_PER_COLLECTION)
        else:
            process_count = total_count

        # Calculates batches using ceiling division.
        collection_batches = (process_count + BATCH_SIZE - 1) // BATCH_SIZE
        total_batches += collection_batches

        logger.info(
            f"Collection '{collection_name}': Total={total_count}, "
            f"To Process={process_count}, Batches={collection_batches}"
        )

    logger.info(f"Total batches to process: {total_batches}")
    return total_batches


def setup_batch_processing() -> int:
    """
    Set up batch processing parameters and store them for subsequent tasks.

    This function performs initialization for the batch migration process:
    1. Computes and stores adjusted mortgage rates
    2. Counts documents in each collection
    3. Creates batch specifications with skip/limit parameters
    4. Stores all configuration in Airflow Variables for downstream tasks

    Returns:
        Total number of batches to process.

    Side Effects:
        Stores the following Airflow Variables:
        - adjusted_mortgage_rates: JSON dict of date -> rate mappings
        - batch_info: JSON dict with per-collection counts and batch counts
        - all_batches: JSON list of batch specifications

    Example:
        >>> total = setup_batch_processing()
        >>> print(f"Set up {total} batches for processing")
    """
    logger.info("Setting up batch processing parameters...")

    # Computes mortgage rates and stores in Airflow Variables.
    adjusted_rates = compute_adjusted_mortgage_rates()
    Variable.set("adjusted_mortgage_rates", json.dumps(adjusted_rates))
    logger.info(f"Stored adjusted mortgage rates for {len(adjusted_rates)} dates")

    # Connects to MongoDB to get collection counts.
    client = get_mongodb_client()
    db = client[MONGODB_WAREHOUSE_NAME]

    collections = ['rent', 'auction', 'sale']
    batch_info: Dict[str, Dict[str, int]] = {}

    for collection_name in collections:
        # Counts documents without child_listings.
        query = {"child_listings": None}
        total_count = db[collection_name].count_documents(query)

        # Applies per-collection limit if configured.
        if MAX_RECORDS_PER_COLLECTION:
            process_count = min(total_count, MAX_RECORDS_PER_COLLECTION)
        else:
            process_count = total_count

        # Calculates number of batches.
        num_batches = (process_count + BATCH_SIZE - 1) // BATCH_SIZE

        # Stores batch info for this collection.
        batch_info[collection_name] = {
            'total_count': total_count,
            'process_count': process_count,
            'num_batches': num_batches
        }

        logger.info(
            f"Collection '{collection_name}': Total={total_count}, "
            f"To Process={process_count}, Batches={num_batches}"
        )

    # Stores batch info in Airflow Variables.
    Variable.set("batch_info", json.dumps(batch_info))

    # Prepares list of all batches with skip/limit parameters.
    all_batches: List[Dict[str, Any]] = []
    batch_num = 0

    for collection_name in collections:
        for i in range(batch_info[collection_name]['num_batches']):
            all_batches.append({
                'collection': collection_name,
                'batch_num': batch_num,
                'skip': i * BATCH_SIZE,
                'limit': BATCH_SIZE
            })
            batch_num += 1

    # Stores batch list in Airflow Variables.
    Variable.set("all_batches", json.dumps(all_batches))
    logger.info(f"Setup complete. Total batches to process: {len(all_batches)}")

    return len(all_batches)


def process_batch(batch_num: int = 0, **kwargs) -> str:
    """
    Process a specific batch of data from MongoDB.

    This function retrieves a batch of listings from MongoDB, processes
    all dimension tables for the batch, and stores the results in Airflow
    XCom for the fact table loading task.

    Args:
        batch_num: The batch number to process (0-indexed).
        **kwargs: Airflow task context including 'ti' (TaskInstance)
            for XCom operations.

    Returns:
        Status message indicating completion.

    Side Effects:
        Pushes to XCom:
        - batch_{batch_num}_dimension_mappings: Dimension ID mappings
        - batch_{batch_num}_data: Raw batch data for fact loading

    Raises:
        ValueError: If batch_num is invalid (>= total batches).

    Example:
        >>> result = process_batch(batch_num=0, ti=task_instance)
        >>> print(result)
        "Batch 0 complete"
    """
    # Gets TaskInstance from kwargs for XCom operations.
    ti = kwargs['ti']

    # Retrieves all batch specifications.
    all_batches = json.loads(Variable.get("all_batches"))

    # Validates batch number.
    if batch_num >= len(all_batches):
        logger.error(
            f"Invalid batch number: {batch_num}. Total batches: {len(all_batches)}"
        )
        raise ValueError(f"Invalid batch number: {batch_num}")

    # Gets batch specification.
    batch = all_batches[batch_num]
    collection_name = batch['collection']
    skip = batch['skip']
    limit = batch['limit']

    logger.info(
        f"Processing batch {batch_num}: Collection={collection_name}, "
        f"Skip={skip}, Limit={limit}"
    )

    # Imports MongoHook here to avoid circular imports.
    from airflow.providers.mongo.hooks.mongo import MongoHook
    from bson import json_util

    # Connects to MongoDB.
    mongo_hook = MongoHook(conn_id='mongo_default')

    # Queries for listings without child_listings.
    query = {"child_listings": None}
    cursor = mongo_hook.find(
        mongo_collection=collection_name,
        query=query,
        mongo_db='listing_website_warehouse',
        skip=skip,
        limit=limit
    )

    # Converts cursor to list.
    batch_listings = list(cursor)
    logger.info(f"Retrieved {len(batch_listings)} listings for this batch")

    # Adds collection name to each listing for processing.
    for listing in batch_listings:
        listing['collection_name'] = collection_name.lower()

    # Converts to JSON to handle MongoDB-specific types (ObjectId, etc.).
    json_str = json_util.dumps(batch_listings)
    batch_data = json.loads(json_str)

    # Processes dimension tables for this batch.
    dimension_mappings = process_dimensions_for_batch(batch_data)

    # Converts tuple keys to strings for XCom serialization.
    serializable_dimension_mappings = convert_nested_dict_tuple_keys_to_str(
        dimension_mappings
    )

    # Stores results in XCom for fact table loading task.
    ti.xcom_push(
        key=f'batch_{batch_num}_dimension_mappings',
        value=serializable_dimension_mappings
    )
    ti.xcom_push(key=f'batch_{batch_num}_data', value=batch_data)

    logger.info(
        f"Batch {batch_num} processing complete. Processed {len(batch_data)} records."
    )
    return f"Batch {batch_num} complete"


def finalize_migration(**kwargs) -> Dict[str, Any]:
    """
    Finalize the migration by reporting statistics and cleaning up.

    This function runs after all batch processing and fact table loading
    is complete. It reports summary statistics and cleans up temporary
    Airflow Variables.

    Args:
        **kwargs: Airflow task context including 'ti' (TaskInstance).

    Returns:
        Dictionary containing migration statistics:
        - batches_processed: Number of batches that were processed
        - fact_table_records: Total records in fact_listing table
        - feature_relationships: Total records in features bridge table
        - surface_relationships: Total records in surface bridge table

    Example:
        >>> stats = finalize_migration(ti=task_instance)
        >>> print(f"Processed {stats['batches_processed']} batches")
    """
    # Imports PostgresHook for database queries.
    from airflow.providers.postgres.hooks.postgres import PostgresHook

    # Gets batch info to know how many batches were processed.
    all_batches = json.loads(Variable.get("all_batches"))
    num_batches = len(all_batches)

    logger.info(f"Migration completed. Total batches processed: {num_batches}")

    # Counts total records in fact table.
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    result = execute_query_silent(
        postgres_hook, "SELECT COUNT(*) FROM fact_listing"
    )
    total_records = result[0][0] if result else 0

    logger.info(f"Total records in fact table: {total_records}")

    # Counts records in bridge tables.
    features_result = execute_query_silent(
        postgres_hook, "SELECT COUNT(*) FROM listing_features_bridge"
    )
    features_count = features_result[0][0] if features_result else 0

    surface_result = execute_query_silent(
        postgres_hook, "SELECT COUNT(*) FROM surface_composition_bridge"
    )
    surface_count = surface_result[0][0] if surface_result else 0

    logger.info(f"Total feature relationships: {features_count}")
    logger.info(f"Total surface composition relationships: {surface_count}")

    # Cleans up Airflow variables.
    try:
        Variable.delete("batch_info")
        Variable.delete("all_batches")
        logger.info("Cleaned up temporary variables")
    except Exception:
        logger.warning("Could not clean up temporary variables")

    return {
        "batches_processed": num_batches,
        "fact_table_records": total_records,
        "feature_relationships": features_count,
        "surface_relationships": surface_count
    }

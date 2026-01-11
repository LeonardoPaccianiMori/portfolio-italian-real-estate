"""
ETL module for the Italian Real Estate pipeline.

This module provides functionality for extracting data from raw HTML in the
MongoDB datalake, transforming it into structured features, and loading it
into the MongoDB warehouse.

Submodules:
    extractors: Data extraction functions for parsing HTML and extracting features.
    transformers: Data transformation and cleaning utilities.
    warehouse_loader: Loading functionality for the MongoDB warehouse.
"""

from .extractors import extract_and_transform_data
from .transformers import fix_empty_child_listings
from .warehouse_loader import migrate, migrate_async, get_warehouse_stats

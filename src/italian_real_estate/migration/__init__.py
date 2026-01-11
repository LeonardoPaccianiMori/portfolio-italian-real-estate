"""
Migration module for the Italian Real Estate pipeline.

This module provides functionality for migrating data from the MongoDB warehouse
to a PostgreSQL data warehouse using a star schema design. It includes batch
processing, dimension table creation, fact table loading, and text translation.

The migration uses a star schema with:
- 14 dimension tables (dim_date, dim_listing_type, dim_seller_type, etc.)
- 1 fact table (fact_listing)
- 2 bridge tables (listing_features_bridge, surface_composition_bridge)

Submodules:
    batch_processor: Batch processing setup and execution.
    dimension_processors: Dimension table processing for all dimension tables.
    fact_loader: Fact table loading functionality.
    translation: Translation services including LibreTranslate integration.
    postgres_utils: PostgreSQL helper utilities.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import importlib

__all__ = [
    # Batch processing
    "setup_batch_processing",
    "process_batch",
    "compute_adjusted_mortgage_rates",
    "count_batches",
    "finalize_migration",
    # Dimension processing
    "process_dimensions_for_batch",
    # Fact loading
    "load_fact_table_for_batch",
    # Utilities
    "execute_query_silent",
    "get_primary_key_column",
    "process_total_room_number",
    "check_all_nulls",
    "get_existing_null_record_id",
    # Schema
    "create_postgresql_schema",
    "drop_all_tables",
    "get_table_list",
    "get_dimension_tables",
    "get_bridge_tables",
    "POSTGRESQL_SCHEMA_STATEMENTS",
]

_ATTR_TO_MODULE = {
    # Batch processing
    "setup_batch_processing": ".batch_processor",
    "process_batch": ".batch_processor",
    "compute_adjusted_mortgage_rates": ".batch_processor",
    "count_batches": ".batch_processor",
    "finalize_migration": ".batch_processor",
    # Dimension processing
    "process_dimensions_for_batch": ".dimension_processors",
    # Fact loading
    "load_fact_table_for_batch": ".fact_loader",
    # Utilities
    "execute_query_silent": ".postgres_utils",
    "get_primary_key_column": ".postgres_utils",
    "process_total_room_number": ".postgres_utils",
    "check_all_nulls": ".postgres_utils",
    "get_existing_null_record_id": ".postgres_utils",
    # Schema
    "create_postgresql_schema": ".schema",
    "drop_all_tables": ".schema",
    "get_table_list": ".schema",
    "get_dimension_tables": ".schema",
    "get_bridge_tables": ".schema",
    "POSTGRESQL_SCHEMA_STATEMENTS": ".schema",
}


def __getattr__(name: str):
    module_name = _ATTR_TO_MODULE.get(name)
    if not module_name:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module(module_name, __name__)
    return getattr(module, name)


def __dir__():
    return sorted(set(list(globals().keys()) + __all__))

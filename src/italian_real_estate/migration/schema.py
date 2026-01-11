"""
PostgreSQL schema definitions for the Italian Real Estate data warehouse.

This module contains all CREATE TABLE statements for the star schema,
including dimension tables, fact table, and bridge tables. The schema
can be created on a fresh PostgreSQL database using create_postgresql_schema().

Schema based on ERD with:
- 17 dimension tables
- 1 fact table (fact_listing)
- 2 bridge tables (listing_features_bridge, surface_composition_bridge)

Author: Leonardo Pacciani-Mori
License: MIT
"""

from typing import Optional
import psycopg2


# =============================================================================
# POSTGRESQL SCHEMA - DIMENSION TABLES
# =============================================================================

SCHEMA_DIM_SELLER_TYPE = """
CREATE TABLE IF NOT EXISTS dim_seller_type (
    seller_type_id SERIAL PRIMARY KEY,
    seller_type CHARACTER VARYING(20)
);
"""

SCHEMA_DIM_LISTING_INFO = """
CREATE TABLE IF NOT EXISTS dim_listing_info (
    listing_info_id SERIAL PRIMARY KEY,
    listing_age INTEGER,
    listing_last_update INTEGER,
    number_of_pictures INTEGER,
    seller_type_id INTEGER NOT NULL,
    FOREIGN KEY (seller_type_id) REFERENCES dim_seller_type(seller_type_id)
);
"""

SCHEMA_DIM_DATE = """
CREATE TABLE IF NOT EXISTS dim_date (
    date_id SERIAL PRIMARY KEY,
    date_value DATE NOT NULL,
    year SMALLINT NOT NULL,
    month_number SMALLINT NOT NULL,
    month_name CHARACTER VARYING(9) NOT NULL,
    day_of_month SMALLINT NOT NULL
);
"""

SCHEMA_DIM_AVAILABILITY = """
CREATE TABLE IF NOT EXISTS dim_availability (
    availability_id SERIAL PRIMARY KEY,
    availability TEXT
);
"""

SCHEMA_DIM_CATEGORY = """
CREATE TABLE IF NOT EXISTS dim_category (
    category_id SERIAL PRIMARY KEY,
    category_name CHARACTER VARYING(30)
);
"""

SCHEMA_DIM_ROOMS_INFO = """
CREATE TABLE IF NOT EXISTS dim_rooms_info (
    rooms_info_id SERIAL PRIMARY KEY,
    bathrooms_number INTEGER,
    bedrooms_number INTEGER,
    total_room_number INTEGER,
    kitchen_status CHARACTER VARYING(50),
    garage TEXT,
    floor INTEGER
);
"""

SCHEMA_DIM_ENERGY_INFO = """
CREATE TABLE IF NOT EXISTS dim_energy_info (
    energy_info_id SERIAL PRIMARY KEY,
    is_zero_energy_building BOOLEAN,
    heating_type TEXT,
    energy_class CHARACTER VARYING(2),
    air_conditioning TEXT
);
"""

SCHEMA_DIM_AUCTION_INFO = """
CREATE TABLE IF NOT EXISTS dim_auction_info (
    auction_info_id SERIAL PRIMARY KEY,
    auction_end_date DATE,
    deposit_modality TEXT,
    deposit_modality_hash TEXT,
    auction_type CHARACTER VARYING(50),
    is_open BOOLEAN,
    minimum_offer NUMERIC(15,2),
    procedure_number CHARACTER VARYING(50),
    auction_court TEXT,
    lot_category_id INTEGER,
    lot_category_name TEXT
);
"""

SCHEMA_DIM_LISTING_TYPE = """
CREATE TABLE IF NOT EXISTS dim_listing_type (
    listing_type_id SERIAL PRIMARY KEY,
    listing_type CHARACTER VARYING(7) NOT NULL
);
"""

SCHEMA_DIM_ADDITIONAL_COSTS = """
CREATE TABLE IF NOT EXISTS dim_additional_costs (
    additional_costs_id SERIAL PRIMARY KEY,
    condominium_monthly_expenses NUMERIC(10,2),
    heating_yearly_expenses NUMERIC(10,2),
    mortgage_rate NUMERIC(5,4),
    monthly_payment NUMERIC(10,2)
);
"""

SCHEMA_DIM_TYPE_OF_PROPERTY = """
CREATE TABLE IF NOT EXISTS dim_type_of_property (
    type_of_property_id SERIAL PRIMARY KEY,
    type_of_property TEXT
);
"""

SCHEMA_DIM_CONDITION = """
CREATE TABLE IF NOT EXISTS dim_condition (
    condition_id SERIAL PRIMARY KEY,
    condition TEXT
);
"""

SCHEMA_DIM_BUILDING_INFO = """
CREATE TABLE IF NOT EXISTS dim_building_info (
    building_info_id SERIAL PRIMARY KEY,
    has_elevator BOOLEAN,
    building_usage CHARACTER VARYING(50),
    building_year INTEGER,
    total_building_floors INTEGER,
    total_number_of_residential_units INTEGER,
    work_start_date DATE,
    work_end_date DATE,
    work_completion REAL
);
"""

SCHEMA_DIM_LOCATION_INFO = """
CREATE TABLE IF NOT EXISTS dim_location_info (
    location_info_id SERIAL PRIMARY KEY,
    latitude NUMERIC(9,6),
    longitude NUMERIC(9,6),
    region CHARACTER VARYING(30),
    province CHARACTER VARYING(30),
    province_code CHARACTER(2),
    city TEXT,
    macrozone TEXT,
    microzone TEXT
);
"""

SCHEMA_DIM_CADASTRAL_INFO = """
CREATE TABLE IF NOT EXISTS dim_cadastral_info (
    cadastral_info_id SERIAL PRIMARY KEY,
    cadastral TEXT,
    cadastral_additional_info TEXT,
    sub_cadastral_info TEXT
);
"""

SCHEMA_DIM_SURFACE_COMPOSITION = """
CREATE TABLE IF NOT EXISTS dim_surface_composition (
    surface_composition_id SERIAL PRIMARY KEY,
    element_name TEXT,
    floor INTEGER,
    surface REAL,
    percentage REAL,
    commercial_surface REAL
);
"""

SCHEMA_DIM_FEATURES = """
CREATE TABLE IF NOT EXISTS dim_features (
    feature_id SERIAL PRIMARY KEY,
    feature_name TEXT
);
"""


# =============================================================================
# POSTGRESQL SCHEMA - FACT TABLE
# =============================================================================

SCHEMA_FACT_LISTING = """
CREATE TABLE IF NOT EXISTS fact_listing (
    listing_id BIGINT NOT NULL,
    date_id INTEGER NOT NULL,
    listing_type_id INTEGER NOT NULL,
    price NUMERIC(15,2),
    surface NUMERIC(15,2),
    price_per_sq_mt NUMERIC(15,2),
    listing_info_id INTEGER,
    additional_costs_id INTEGER,
    availability_id INTEGER,
    type_of_property_id INTEGER,
    category_id INTEGER,
    condition_id INTEGER,
    rooms_info_id INTEGER,
    building_info_id INTEGER,
    energy_info_id INTEGER,
    location_info_id INTEGER,
    auction_info_id INTEGER,
    cadastral_info_id INTEGER,
    PRIMARY KEY (listing_id, date_id),
    FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
    FOREIGN KEY (listing_type_id) REFERENCES dim_listing_type(listing_type_id),
    FOREIGN KEY (listing_info_id) REFERENCES dim_listing_info(listing_info_id),
    FOREIGN KEY (additional_costs_id) REFERENCES dim_additional_costs(additional_costs_id),
    FOREIGN KEY (availability_id) REFERENCES dim_availability(availability_id),
    FOREIGN KEY (type_of_property_id) REFERENCES dim_type_of_property(type_of_property_id),
    FOREIGN KEY (category_id) REFERENCES dim_category(category_id),
    FOREIGN KEY (condition_id) REFERENCES dim_condition(condition_id),
    FOREIGN KEY (rooms_info_id) REFERENCES dim_rooms_info(rooms_info_id),
    FOREIGN KEY (building_info_id) REFERENCES dim_building_info(building_info_id),
    FOREIGN KEY (energy_info_id) REFERENCES dim_energy_info(energy_info_id),
    FOREIGN KEY (location_info_id) REFERENCES dim_location_info(location_info_id),
    FOREIGN KEY (auction_info_id) REFERENCES dim_auction_info(auction_info_id),
    FOREIGN KEY (cadastral_info_id) REFERENCES dim_cadastral_info(cadastral_info_id)
);
"""


# =============================================================================
# POSTGRESQL SCHEMA - BRIDGE TABLES
# =============================================================================

SCHEMA_SURFACE_COMPOSITION_BRIDGE = """
CREATE TABLE IF NOT EXISTS surface_composition_bridge (
    listing_id BIGINT NOT NULL,
    date_id INTEGER NOT NULL,
    surface_composition_id INTEGER NOT NULL,
    PRIMARY KEY (listing_id, date_id, surface_composition_id),
    FOREIGN KEY (listing_id, date_id) REFERENCES fact_listing(listing_id, date_id),
    FOREIGN KEY (surface_composition_id) REFERENCES dim_surface_composition(surface_composition_id)
);
"""

SCHEMA_LISTING_FEATURES_BRIDGE = """
CREATE TABLE IF NOT EXISTS listing_features_bridge (
    listing_id BIGINT NOT NULL,
    date_id INTEGER NOT NULL,
    feature_id INTEGER NOT NULL,
    PRIMARY KEY (listing_id, date_id, feature_id),
    FOREIGN KEY (listing_id, date_id) REFERENCES fact_listing(listing_id, date_id),
    FOREIGN KEY (feature_id) REFERENCES dim_features(feature_id)
);
"""


# =============================================================================
# POSTGRESQL SCHEMA - INDEXES (for performance)
# =============================================================================

SCHEMA_INDEXES = """
-- =============================================================================
-- Indexes for fact_listing foreign keys (all FK columns)
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_fact_listing_listing_id ON fact_listing(listing_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_date_id ON fact_listing(date_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_listing_info_id ON fact_listing(listing_info_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_additional_costs_id ON fact_listing(additional_costs_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_availability_id ON fact_listing(availability_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_type_of_property_id ON fact_listing(type_of_property_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_category_id ON fact_listing(category_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_condition_id ON fact_listing(condition_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_rooms_info_id ON fact_listing(rooms_info_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_building_info_id ON fact_listing(building_info_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_energy_info_id ON fact_listing(energy_info_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_location ON fact_listing(location_info_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_auction_info_id ON fact_listing(auction_info_id);
CREATE INDEX IF NOT EXISTS idx_fact_listing_cadastral_info_id ON fact_listing(cadastral_info_id);

-- Indexes for fact_listing measures (common query filters)
CREATE INDEX IF NOT EXISTS idx_fact_listing_price ON fact_listing(price);
CREATE INDEX IF NOT EXISTS idx_fact_listing_surface ON fact_listing(surface);

-- =============================================================================
-- Indexes for bridge tables
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_surface_composition_bridge_surface_id ON surface_composition_bridge(surface_composition_id);
CREATE INDEX IF NOT EXISTS idx_listing_features_bridge_feature_id ON listing_features_bridge(feature_id);

-- =============================================================================
-- Indexes for dimension lookups
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_location_info_region ON dim_location_info(region);
CREATE INDEX IF NOT EXISTS idx_location_info_province ON dim_location_info(province);
CREATE INDEX IF NOT EXISTS idx_location_info_city ON dim_location_info(city);
CREATE INDEX IF NOT EXISTS idx_dim_listing_info_seller_type_id ON dim_listing_info(seller_type_id);
CREATE INDEX IF NOT EXISTS idx_dim_auction_deposit_modality ON dim_auction_info USING hash(deposit_modality);

-- =============================================================================
-- Unique indexes for dimension deduplication (used with ON CONFLICT)
-- =============================================================================
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_seller_type_seller_type ON dim_seller_type(seller_type);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_date_date_value ON dim_date(date_value);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_availability_availability ON dim_availability(availability);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_category_category_name ON dim_category(category_name);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_listing_type_listing_type ON dim_listing_type(listing_type);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_type_of_property_type_of_property ON dim_type_of_property(type_of_property);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_condition_condition ON dim_condition(condition);
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_features_feature_name ON dim_features(feature_name);

-- Composite unique indexes for multi-column dimension tables
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_listing_info
    ON dim_listing_info(listing_age, listing_last_update, number_of_pictures, seller_type_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_rooms_info
    ON dim_rooms_info(bathrooms_number, bedrooms_number, total_room_number, kitchen_status, garage, floor);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_energy_info
    ON dim_energy_info(is_zero_energy_building, heating_type, energy_class, air_conditioning);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_additional_costs
    ON dim_additional_costs(condominium_monthly_expenses, heating_yearly_expenses, mortgage_rate, monthly_payment);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_building_info
    ON dim_building_info(has_elevator, building_usage, building_year, total_building_floors,
                         total_number_of_residential_units, work_start_date, work_end_date, work_completion);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_location_info
    ON dim_location_info(latitude, longitude, region, province, province_code, city, macrozone, microzone);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_cadastral_info
    ON dim_cadastral_info(cadastral, cadastral_additional_info, sub_cadastral_info);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_surface_composition
    ON dim_surface_composition(element_name, floor, surface, percentage, commercial_surface);

-- Special unique indexes for dim_auction_info (handles NULL deposit_modality_hash)
CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_auction_info
    ON dim_auction_info(auction_end_date, deposit_modality_hash, auction_type, is_open,
                        COALESCE(minimum_offer, 0), procedure_number, auction_court,
                        COALESCE(lot_category_id, 0), lot_category_name)
    WHERE deposit_modality_hash IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_auction_info_null_deposit
    ON dim_auction_info(auction_end_date, auction_type, is_open,
                        COALESCE(minimum_offer, 0), procedure_number, auction_court,
                        COALESCE(lot_category_id, 0), lot_category_name)
    WHERE deposit_modality_hash IS NULL;
"""


# =============================================================================
# COMBINED SCHEMA (in correct order for foreign key dependencies)
# =============================================================================

# Order matters: dimension tables first, then fact table, then bridge tables
POSTGRESQL_SCHEMA_STATEMENTS = [
    # Independent dimension tables (no foreign keys)
    ("dim_seller_type", SCHEMA_DIM_SELLER_TYPE),
    ("dim_date", SCHEMA_DIM_DATE),
    ("dim_availability", SCHEMA_DIM_AVAILABILITY),
    ("dim_category", SCHEMA_DIM_CATEGORY),
    ("dim_rooms_info", SCHEMA_DIM_ROOMS_INFO),
    ("dim_energy_info", SCHEMA_DIM_ENERGY_INFO),
    ("dim_auction_info", SCHEMA_DIM_AUCTION_INFO),
    ("dim_listing_type", SCHEMA_DIM_LISTING_TYPE),
    ("dim_additional_costs", SCHEMA_DIM_ADDITIONAL_COSTS),
    ("dim_type_of_property", SCHEMA_DIM_TYPE_OF_PROPERTY),
    ("dim_condition", SCHEMA_DIM_CONDITION),
    ("dim_building_info", SCHEMA_DIM_BUILDING_INFO),
    ("dim_location_info", SCHEMA_DIM_LOCATION_INFO),
    ("dim_cadastral_info", SCHEMA_DIM_CADASTRAL_INFO),
    ("dim_surface_composition", SCHEMA_DIM_SURFACE_COMPOSITION),
    ("dim_features", SCHEMA_DIM_FEATURES),
    # Dimension tables with foreign keys
    ("dim_listing_info", SCHEMA_DIM_LISTING_INFO),
    # Fact table
    ("fact_listing", SCHEMA_FACT_LISTING),
    # Bridge tables
    ("surface_composition_bridge", SCHEMA_SURFACE_COMPOSITION_BRIDGE),
    ("listing_features_bridge", SCHEMA_LISTING_FEATURES_BRIDGE),
    # Indexes
    ("indexes", SCHEMA_INDEXES),
]


# =============================================================================
# SCHEMA CREATION FUNCTIONS
# =============================================================================

def create_postgresql_schema(conn) -> dict:
    """
    Create all PostgreSQL tables for the star schema.

    Args:
        conn: psycopg2 connection object

    Returns:
        dict: Results with 'created', 'skipped', and 'errors' lists
    """
    results = {
        "created": [],
        "skipped": [],
        "errors": [],
    }

    cursor = conn.cursor()

    for table_name, sql in POSTGRESQL_SCHEMA_STATEMENTS:
        try:
            cursor.execute(sql)
            conn.commit()
            results["created"].append(table_name)
        except psycopg2.Error as e:
            conn.rollback()
            results["errors"].append((table_name, str(e)))

    cursor.close()
    return results


def drop_all_tables(conn, confirm: bool = False) -> dict:
    """
    Drop all tables in the schema (use with caution!).

    Args:
        conn: psycopg2 connection object
        confirm: Must be True to actually drop tables

    Returns:
        dict: Results with 'dropped' and 'errors' lists
    """
    if not confirm:
        return {"error": "Must set confirm=True to drop tables"}

    results = {
        "dropped": [],
        "errors": [],
    }

    # Drop in reverse order (bridge tables first, then fact, then dimensions)
    drop_order = [
        "listing_features_bridge",
        "surface_composition_bridge",
        "fact_listing",
        "dim_listing_info",  # Has FK to dim_seller_type
        "dim_seller_type",
        "dim_date",
        "dim_availability",
        "dim_category",
        "dim_rooms_info",
        "dim_energy_info",
        "dim_auction_info",
        "dim_listing_type",
        "dim_additional_costs",
        "dim_type_of_property",
        "dim_condition",
        "dim_building_info",
        "dim_location_info",
        "dim_cadastral_info",
        "dim_surface_composition",
        "dim_features",
    ]

    cursor = conn.cursor()

    for table_name in drop_order:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            conn.commit()
            results["dropped"].append(table_name)
        except psycopg2.Error as e:
            conn.rollback()
            results["errors"].append((table_name, str(e)))

    cursor.close()
    return results


def get_table_list() -> list:
    """
    Get list of all table names in the schema.

    Returns:
        list: Table names in creation order
    """
    return [name for name, _ in POSTGRESQL_SCHEMA_STATEMENTS if name != "indexes"]


def get_dimension_tables() -> list:
    """Get list of dimension table names."""
    return [name for name, _ in POSTGRESQL_SCHEMA_STATEMENTS
            if name.startswith("dim_")]


def get_bridge_tables() -> list:
    """Get list of bridge table names."""
    return ["surface_composition_bridge", "listing_features_bridge"]

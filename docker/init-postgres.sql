-- Italian Real Estate Pipeline - PostgreSQL Initialization Script
--
-- This script runs automatically when the PostgreSQL container first starts.
-- It creates the necessary databases and users for the pipeline.
--
-- Databases:
--   - listing_website_warehouse: Main data warehouse (created by POSTGRES_DB env var)
--   - airflow: Airflow metadata database
--
-- Note: The star schema tables are created by running
--       scripts/setup_databases.py --postgres (or by calling
--       italian_real_estate.migration.schema.create_postgresql_schema).
--       This script only sets up the database infrastructure.
--
-- Author: Leonardo Pacciani-Mori
-- License: MIT

-- =============================================================================
-- Airflow Metadata Database
-- =============================================================================
-- Airflow's metadata DB and role are created by the `airflow-db-init`
-- service in docker-compose.yml so credentials can be centralized in .env.

-- =============================================================================
-- Configure Main Database
-- =============================================================================
-- The main database (listing_website_warehouse) is created automatically by
-- the POSTGRES_DB environment variable in docker-compose.yml.
-- The star schema tables will be created by the migration stage.

-- Enable useful extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text similarity searches
CREATE EXTENSION IF NOT EXISTS btree_gist;  -- For advanced indexing

-- =============================================================================
-- Summary
-- =============================================================================
-- PostgreSQL initialization complete.
--
-- Created:
--   - Extensions: pg_trgm, btree_gist
--
-- The main warehouse database and star schema will be created by the
-- migration stage when it first runs.

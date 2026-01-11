"""
Italian Real Estate Data Pipeline Package.

This package provides a complete data pipeline for Italian real estate analysis,
including web scraping, ETL processing, database migration, synthetic data
generation, and machine learning model development.

Modules:
    config: Configuration settings and logging setup.
    core: Shared utilities for database connections, string/date/numeric operations.
    scraping: Web scraping functionality for listing.website.
    etl: ETL processing from MongoDB datalake to MongoDB warehouse.
    migration: Data migration from MongoDB to PostgreSQL with translation.
    synthetic_data: KNN-based synthetic data generation.
    ml: Machine learning model training and rent prediction.

Author: Leonardo Pacciani-Mori
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Leonardo Pacciani-Mori"

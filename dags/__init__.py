"""
Airflow DAGs for the Italian Real Estate pipeline.

This module contains all Apache Airflow DAG definitions for orchestrating
the data pipeline stages.

DAGs:
    datalake_population_dag: Web scraping and datalake population.
    mongodb_etl_dag: ETL from datalake to MongoDB warehouse.
    postgres_migration_dag: Migration from MongoDB to PostgreSQL.
"""

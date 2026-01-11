"""
Shared utilities for Airflow DAG definitions.

This module provides factory functions and configuration helpers for
creating consistent Airflow DAGs across the Italian Real Estate pipeline.
It centralizes DAG configuration to ensure consistency and reduce
code duplication.

Author: Leonardo Pacciani-Mori
License: MIT
"""

from datetime import timedelta
from airflow import DAG

try:
    from airflow.utils.dates import days_ago
except ImportError:
    try:
        from airflow.sdk import timezone

        def days_ago(n: int):
            return timezone.utcnow() - timedelta(days=n)
    except ImportError:
        from datetime import datetime, timezone as dt_timezone

        def days_ago(n: int):
            return datetime.now(dt_timezone.utc) - timedelta(days=n)


# =============================================================================
# DEFAULT DAG ARGUMENTS
# =============================================================================

# Default arguments applied to all DAGs in this pipeline.
# These can be overridden when creating individual DAGs.
DAG_DEFAULT_ARGS = {
    # Owner name for tracking and filtering in the Airflow UI
    "owner": "Leonardo Pacciani-Mori",

    # Start date for the DAG schedule (days_ago(0) means today)
    "start_date": days_ago(0),

    # Maximum number of tasks that can run concurrently within this DAG
    "max_active_tasks": 16,

    # Number of times to retry a failed task before marking it as failed
    "retries": 5,

    # How long to wait between retry attempts
    "retry_delay": timedelta(minutes=1),

    # Email settings for failure notifications (configure as needed)
    "email_on_failure": False,
    "email_on_retry": False,
}


def create_dag_with_defaults(
    dag_id: str,
    description: str,
    schedule: str = None,
    max_active_tasks: int = None,
    **kwargs
) -> DAG:
    """
    Create a DAG with standardized default configuration.

    This factory function creates Airflow DAGs with consistent settings,
    reducing boilerplate and ensuring uniformity across the pipeline.

    The function merges the provided arguments with the default configuration,
    allowing selective overrides while maintaining baseline settings.

    Args:
        dag_id: A unique identifier for the DAG. This appears in the Airflow UI
            and is used for triggering and referencing the DAG.
        description: A human-readable description of what the DAG does.
            Displayed in the Airflow UI.
        schedule: The schedule interval for automatic runs. Use None for
            manual-only DAGs, or cron expressions like "0 0 * * *" for daily.
            Defaults to None (manual trigger only).
        max_active_tasks: Override the default maximum concurrent tasks.
            Useful for resource-intensive DAGs that need limiting.
        **kwargs: Additional arguments passed to the DAG constructor.
            These override any defaults if there are conflicts.

    Returns:
        DAG: A configured Airflow DAG instance ready for task definitions.

    Example:
        >>> dag = create_dag_with_defaults(
        ...     dag_id="my_etl_dag",
        ...     description="Daily ETL pipeline for rent listings",
        ...     schedule="0 2 * * *"  # Run at 2 AM daily
        ... )
        >>> with dag:
        ...     task1 = PythonOperator(...)
    """
    # Start with default arguments
    dag_args = DAG_DEFAULT_ARGS.copy()

    # Apply overrides if provided
    if max_active_tasks is not None:
        dag_args["max_active_tasks"] = max_active_tasks

    # Merge any additional kwargs
    dag_args.update(kwargs)

    dag_max_active_tasks = dag_args.pop("max_active_tasks", None)

    # Create and return the DAG
    dag_kwargs = dict(
        dag_id=dag_id,
        default_args=dag_args,
        description=description,
        schedule=schedule,
        catchup=False,  # Don't backfill historical runs
    )
    if dag_max_active_tasks is not None:
        dag_kwargs["max_active_tasks"] = dag_max_active_tasks

    dag = DAG(**dag_kwargs)

    return dag


def get_provinces_list():
    """
    Get the list of Italian provinces for DAG task generation.

    This function loads the province list from the configuration and
    returns it for use in dynamically generating tasks.

    Returns:
        list: A list of province names.

    Example:
        >>> provinces = get_provinces_list()
        >>> for province in provinces:
        ...     task = create_province_task(province)
    """
    import pandas as pd
    from italian_real_estate.config.settings import PROVINCES_CSV_PATH

    try:
        prov_df = pd.read_csv(PROVINCES_CSV_PATH, sep=",", na_filter=False)
        return prov_df["Province"].tolist()
    except FileNotFoundError:
        # Return empty list if file not found (for development/testing)
        return []


def get_listing_types():
    """
    Get the list of valid listing types for DAG task generation.

    Returns:
        list: A list of listing type strings ["sale", "rent", "auction"].
    """
    from italian_real_estate.config.settings import VALID_LISTING_TYPES
    return VALID_LISTING_TYPES

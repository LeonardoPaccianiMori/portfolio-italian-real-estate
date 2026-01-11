"""
Airflow DAG for ETL from MongoDB datalake to MongoDB warehouse.

This DAG orchestrates the extraction, transformation, and loading of data
from the raw HTML datalake to the structured features warehouse. Both
databases are MongoDB collections.

DAG Structure:
    [rent_ETL, auction_ETL, sale_ETL] (parallel) -> cleaning_task -> finalize_task

The three listing types are processed in parallel for efficiency, then
a cleaning task fixes any data quality issues, followed by statistics
reporting.

Author: Leonardo Pacciani-Mori
License: MIT
"""

from airflow.operators.python import PythonOperator

from dag_utils import create_dag_with_defaults, get_listing_types
from italian_real_estate.etl.warehouse_loader import migrate, get_warehouse_stats
from italian_real_estate.etl.transformers import fix_empty_child_listings


# =============================================================================
# DAG DEFINITION
# =============================================================================

# Create the DAG using the factory function.
# max_active_tasks=16 allows parallel processing of listing types
# plus room for any subtasks.
dag = create_dag_with_defaults(
    dag_id="listing.website_datalake_ETL_warehouse_MongoDB_DAG",
    description="DAG to extract, transform and load data from mongoDB datalake to mongoDB warehouse",
    max_active_tasks=16,
)


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

# Create ETL tasks for each listing type.
# These can run in parallel as they operate on separate collections.
etl_tasks = []
for listing_type in get_listing_types():
    task = PythonOperator(
        task_id=f"{listing_type}_ETL",
        python_callable=migrate,
        op_kwargs={"listing_type": listing_type},
        dag=dag
    )
    etl_tasks.append(task)

# Create the cleaning task.
# This fixes data quality issues like empty arrays that should be null.
# It runs after all ETL tasks complete.
cleaning_task = PythonOperator(
    task_id="cleaning_task",
    python_callable=fix_empty_child_listings,
    dag=dag
)

# Create the finalize task.
# This generates and logs warehouse statistics and success message.
finalize_task = PythonOperator(
    task_id="finalize_task",
    python_callable=get_warehouse_stats,
    dag=dag
)


# =============================================================================
# TASK DEPENDENCIES
# =============================================================================

# Set up the DAG structure:
# All ETL tasks run in parallel, then cleaning, then finalize.
# [rent_ETL, auction_ETL, sale_ETL] -> cleaning_task -> finalize_task

for etl_task in etl_tasks:
    etl_task >> cleaning_task

cleaning_task >> finalize_task

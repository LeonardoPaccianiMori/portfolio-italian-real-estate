"""
Airflow DAG for populating the MongoDB datalake with scraped data.

This DAG orchestrates the web scraping process to collect real estate
listings from listing.website and store them in the MongoDB datalake.
It creates tasks for each Italian province and listing type combination.

DAG Structure:
    For each province:
        province_rent -> province_auction -> province_sale -> finalize_task

This sequential execution within each province ensures orderly scraping
while allowing different provinces to run in parallel.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import os

from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator, get_current_context

from dag_utils import create_dag_with_defaults, get_provinces_list
from italian_real_estate.scraping import datalake_populator
from italian_real_estate.scraping.datalake_populator import log_datalake_statistics


# =============================================================================
# DAG DEFINITION
# =============================================================================

# Create the DAG using the factory function with standard defaults.
# max_active_tasks=16 allows 16 concurrent province scrapes.
def _get_max_active_tasks() -> int:
    if os.getenv("SCRAPING_POLITE_MODE", "").strip().lower() in ("1", "true", "yes"):
        return int(os.getenv("SCRAPING_POLITE_MAX_ACTIVE_TASKS", "1"))
    return int(os.getenv("SCRAPING_MAX_ACTIVE_TASKS", "16"))


dag = create_dag_with_defaults(
    dag_id="listing.website_datalake_population_DAG_webscraping",
    description="DAG to populate mongoDB datalake with raw data from rent, auction, and sale listings",
    max_active_tasks=_get_max_active_tasks(),
)

SCRAPE_POOL_NAME = os.getenv("SCRAPING_POOL_NAME", "").strip()


# =============================================================================
# TASK DEFINITIONS
# =============================================================================

def run_scrape_with_filters(province: str, listing_type: str) -> None:
    """
    Run a scrape task if it matches the dag_run.conf filters.

    Supported conf keys:
    - provinces: list of province names to include
    - listing_types: list of listing types to include (rent/auction/sale)
    - use_selenium: bool to enable Selenium scraping
    """
    context = get_current_context()
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}

    selected_provinces = conf.get("provinces")
    selected_listing_types = conf.get("listing_types")
    use_selenium = conf.get("use_selenium")
    polite_mode = os.getenv("SCRAPING_POLITE_MODE", "").strip().lower() in ("1", "true", "yes")

    if polite_mode:
        use_selenium = True

    if use_selenium is not None:
        os.environ["SCRAPING_USE_SELENIUM"] = "1" if use_selenium else "0"
        try:
            from italian_real_estate.config import settings as app_settings
            app_settings.SCRAPING_USE_SELENIUM = bool(use_selenium)
        except Exception:
            pass

    if selected_provinces is not None and province not in selected_provinces:
        raise AirflowSkipException(f"Province '{province}' not selected in dag_run.conf")

    if selected_listing_types is not None and listing_type not in selected_listing_types:
        raise AirflowSkipException(
            f"Listing type '{listing_type}' not selected in dag_run.conf"
        )

    if use_selenium is not None:
        datalake_populator.SCRAPING_USE_SELENIUM = bool(use_selenium)

    run_date = None
    if dag_run is not None:
        run_date = dag_run.start_date or getattr(dag_run, "logical_date", None)
    scraping_date = run_date.strftime("%Y-%m-%d") if run_date else None

    datalake_populator.datalake_populate_links_province(
        province,
        listing_type,
        scraping_date=scraping_date,
    )


# Create the finalize task that runs after all provinces are scraped.
# This task logs statistics about the completed scraping run.
finalize_task = PythonOperator(
    task_id="finalize_task",
    python_callable=log_datalake_statistics,
    dag=dag
)

# Get the list of all Italian provinces
provinces = get_provinces_list()

# Create tasks for each province.
# Each province has three sequential tasks: rent -> auction -> sale
# This ordering is intentional: rent listings are typically fewer and
# serve as a warmup before the larger auction and sale scrapes.
for province in provinces:

    # Task to scrape rent listings for this province
    rent_task_kwargs = {
        "task_id": f"{province}_rents",
        "python_callable": run_scrape_with_filters,
        "op_kwargs": {
            "province": province,
            "listing_type": "rent"
        },
        "dag": dag,
    }
    if SCRAPE_POOL_NAME:
        rent_task_kwargs["pool"] = SCRAPE_POOL_NAME
    rent_task = PythonOperator(**rent_task_kwargs)

    # Task to scrape auction listings for this province
    auction_task_kwargs = {
        "task_id": f"{province}_auctions",
        "python_callable": run_scrape_with_filters,
        "op_kwargs": {
            "province": province,
            "listing_type": "auction"
        },
        "dag": dag,
    }
    if SCRAPE_POOL_NAME:
        auction_task_kwargs["pool"] = SCRAPE_POOL_NAME
    auction_task = PythonOperator(**auction_task_kwargs)

    # Task to scrape sale listings for this province
    sale_task_kwargs = {
        "task_id": f"{province}_sales",
        "python_callable": run_scrape_with_filters,
        "op_kwargs": {
            "province": province,
            "listing_type": "sale"
        },
        "dag": dag,
    }
    if SCRAPE_POOL_NAME:
        sale_task_kwargs["pool"] = SCRAPE_POOL_NAME
    sale_task = PythonOperator(**sale_task_kwargs)

    # Set up task dependencies for this province:
    # rent -> auction -> sale -> finalize
    # This ensures sequential execution within a province while
    # allowing different provinces to run in parallel.
    rent_task >> auction_task >> sale_task >> finalize_task

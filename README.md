# Italian Real Estate Project

This repository contains the code used in [this project](https://leonardo.pm/projects/italian-real-estate) from my personal portfolio.

The code was written by me between January and April 2025, then it was later re-organized and cleaned up in December 2025 using Codex 5.2.

A full data pipeline for Italian real estate analytics: scraping (redacted in the public version), ETL, migration to a star-schema warehouse, synthetic data generation, and ML training. The project includes a Dockerized Airflow stack, an interactive TUI/CLI, and a structured data warehouse design.

## Quick Overview

- **End-to-end pipeline**: scrape -> datalake -> ETL -> PostgreSQL warehouse -> ML training
- **Interactive TUI**: menu-driven orchestration with progress monitoring
- **Docker-first**: reproducible environment with MongoDB, Postgres, Airflow, and the app
- **Public release guardrails**: scraping extraction is intentionally redacted

## Portfolio Caveats (Public Release)

- **Scraping extraction is redacted**: `extract_html_source_code()` raises a runtime error to prevent real scraping out of the box.
- **Only synthetic data is included**: the repo ships with `data/synthetic_data.parquet` only.
- **You must create `.env`**: use `.env.example` and fill in values locally.

If you need a fully functional scraping version, re-implement the redacted sections and ensure compliance with the target website's terms.

---

# Quickstart

## Docker (Recommended)

1) Create your local env file:
```bash
cp .env.example .env
```

2) Start the stack:
```bash
docker compose up -d
```

3) Launch the TUI:
```bash
docker compose exec app python scripts/pipeline_tui.py
```

4) Airflow UI:
- http://localhost:8080
- Credentials are in `.env` (defaults in `.env.example`).

### Note on Airflow Logs
If you see 403 errors for log retrieval, ensure the same `AIRFLOW__WEBSERVER__SECRET_KEY` is set for all Airflow components and restart.

---

# What Still Runs in the Public Version

- **Synthetic data generation**
- **ML training on synthetic data**
- **ETL and migration steps** (if you provide your own data)
- **Database explorer via the TUI**

Scraping is intentionally disabled; the pipeline will fail fast if you try to execute the scrape stage without re-implementing the redacted logic.

---

# Deep Dive

## Architecture

The pipeline is structured into stages:

1) **Scrape**: fetch listing pages and build a MongoDB datalake (redacted here).
2) **ETL**: extract and transform listing metadata into a clean warehouse schema.
3) **Migration**: load ETL results into PostgreSQL star schema.
4) **Data Export**: consolidate datasets for analysis or ML.
5) **Training**: train predictive models on synthetic or real data.

The TUI orchestrates these stages and can trigger Airflow DAGs via the REST API.

## Repository Structure

```
.
├── dags/                       # Airflow DAGs
├── data/                       # Synthetic data only
├── docker/                     # Docker init scripts
├── scripts/                    # CLI entrypoints
├── src/italian_real_estate/    # Core package
└── docs/                       # Additional docs
```

## Pipeline TUI

The main entry point is `scripts/pipeline_tui.py` (also exposed as the `italian-real-estate` console script). It can run interactively or via CLI:

```bash
# Interactive
python scripts/pipeline_tui.py

# Run a specific stage
python scripts/pipeline_tui.py etl --all
```

The TUI provides:
- Stage selection and configuration
- Airflow orchestration and live status updates
- Database explorer for MongoDB and PostgreSQL

## Individual Stage CLIs

```bash
# Scraping (redacted in public version; will fail fast)
python scripts/run_scraping.py --province Milano --listing-type rent

# ETL
python scripts/run_etl.py --all

# Migration
python scripts/run_migration.py --all

# Synthetic Data
python scripts/generate_synthetic_data.py

# ML Training
python scripts/pipeline_tui.py train --input data/synthetic_data.parquet
python scripts/train_model.py --input data/synthetic_data.csv --show-plots
```

Note: `scripts/train_model.py` expects CSV input. Use `pipeline_tui.py train` for Parquet.

## Airflow DAGs

Airflow DAGs are located in `dags/` and include:
- `listing.website_datalake_population_DAG_webscraping`
- `listing.website_datalake_ETL_warehouse_MongoDB_DAG`
- `listing.website_MongoDB_to_PostgreSQL_migration`

If a DAG is queued but not running, unpause it in the UI or via:
```bash
airflow dags unpause listing.website_datalake_population_DAG_webscraping
```

## Configuration

Core configuration lives in `src/italian_real_estate/config/settings.py`. You can override defaults via `.env`.

Key variables:
- `SCRAPING_USE_SELENIUM`: enable Selenium fetching (scraping is still redacted)
- `SCRAPING_POLITE_MODE`: slow down with retries/backoff
- `HTTP_SEMAPHORE_LIMIT`: limit concurrent HTTP requests
- `AIRFLOW__WEBSERVER__SECRET_KEY`: must match across Airflow services

## Data Schema (PostgreSQL Star Schema)

**Fact Table:**
- `fact_listing`

**Dimension Tables:**
- `dim_date`
- `dim_listing_type`
- `dim_seller_type`
- `dim_category`
- `dim_energy_info`
- `dim_type_of_property`
- `dim_condition`
- `dim_rooms_info`
- `dim_building_info`
- `dim_location`
- `dim_mortgage_rate`
- `dim_listing_page`
- `dim_features`
- `dim_surface_composition`

**Bridge Tables:**
- `listing_features_bridge`
- `surface_composition_bridge`

## Scraping Redaction (Public Release)

The public version intentionally redacts the scraping extraction logic. The function
`extract_html_source_code()` now raises a `RuntimeError` and includes a structured
outline of the intended flow. This prevents live scraping while preserving the
pipeline architecture for review.

---

# License

MIT License. See `LICENSE`.

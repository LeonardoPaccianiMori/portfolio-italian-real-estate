# Italian Real Estate Pipeline

Code for the [Italian real-estate portfolio project](https://leonardo.pm/projects/italian-real-estate/): an end-to-end study covering ingestion architecture, MongoDB-based transformation, a PostgreSQL star schema, synthetic-data generation, and regression experiments.

The original implementation was written between January and April 2025. It was reorganized and documented with Codex 5.2 in December 2025 and hardened for public release in August 2026.

## Public-release boundary

This repository contains code, schemas, configuration examples, and documentation. It does **not** contain source listings, scraped HTML, exported warehouse records, or a synthetic listing dataset.

The historical source website was [Immobiliare.it](https://www.immobiliare.it/). Naming the source documents provenance; it does not grant any right to collect or redistribute its content.

- The original extraction implementation is redacted. `extract_html_source_code()` fails explicitly instead of contacting the source website.
- Synthetic-data code remains available for inspection and for use with data you are authorised to process.
- Published portfolio results describe the historical experiment; cloning this repository alone does not reproduce its data or reported score.
- The synthetic-data process was designed to avoid redistributing source listings. It was not evaluated as a formal privacy guarantee.

Anyone adapting the ingestion layer is responsible for the source website's terms, robots policy, database rights, privacy obligations, and applicable law.

## Architecture

```text
authorised input
  -> MongoDB datalake
  -> ETL / MongoDB warehouse
  -> PostgreSQL star schema
  -> local export
  -> synthetic-data generation
  -> regression experiments
```

The repository also includes an interactive TUI, Airflow DAGs, Docker services, and the dimensional warehouse schema used by the project.

## Setup

1. Copy and review the environment template:

   ```bash
   cp .env.example .env
   ```

2. Replace every example password and secret before starting the services.

3. Start the local stack:

   ```bash
   docker compose up -d
   docker compose exec app python scripts/pipeline_tui.py
   ```

The Airflow UI is available at `http://localhost:8080`. Its local credentials come from `.env`.

## Commands

```bash
# ETL and migration require user-supplied local data
python scripts/run_etl.py --all
python scripts/run_migration.py --all

# Generate a local synthetic dataset from an authorised PostgreSQL warehouse
python scripts/generate_synthetic_data.py --output data/synthetic_data.parquet

# Train from a local CSV
python scripts/train_model.py --input /path/to/authorised_data.csv --show-plots
```

The scraping command is retained only to show the orchestration boundary and exits at the redacted extraction step.

## Repository map

```text
dags/                       Airflow orchestration
docker/                     Database initialisation
scripts/                    Command-line entry points
src/italian_real_estate/    Pipeline package
tests/                      Public-boundary and connection tests
data/README.md              Local-data policy (data itself is ignored)
```

## Validation

```bash
python -m compileall -q src scripts dags
pytest -q
```

Full integration runs additionally require Docker and user-supplied data.

## Limitations

- The public checkout is intentionally not a one-command reproduction of the historical study because its inputs are not redistributed.
- The original synthetic-data evaluation did not include a formal disclosure-risk or privacy audit.
- Pipeline services and dependency pins reflect a portfolio project, not a supported production deployment.

## Licence

Original code and documentation are released under the MIT Licence; see [`LICENSE`](LICENSE). No rights to third-party source data are granted, and no project dataset is included.

## Authorship

The original project and analysis are by Leonardo Pacciani-Mori. Codex 5.2 assisted with the later repository reorganisation and documentation cleanup.

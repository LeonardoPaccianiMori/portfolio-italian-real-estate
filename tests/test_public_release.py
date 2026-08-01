from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_checkout_contains_no_dataset():
    data_files = [path for path in (ROOT / "data").rglob("*") if path.is_file()]
    assert data_files == [ROOT / "data" / "README.md"]


def test_scraping_extraction_is_redacted():
    scraping_source = (
        ROOT
        / "src"
        / "italian_real_estate"
        / "scraping"
        / "datalake_populator.py"
    ).read_text()
    assert "RuntimeError" in scraping_source


def test_compose_has_no_fallback_passwords():
    compose = (ROOT / "docker-compose.yml").read_text()
    for weak_default in (
        "AIRFLOW_API_PASSWORD=${AIRFLOW_API_PASSWORD:-admin}",
        "AIRFLOW_ADMIN_PASSWORD:-admin",
        "AIRFLOW_DB_PASSWORD:-airflow",
        "POSTGRES_PASSWORD:-changeme",
        "your-secret-key",
    ):
        assert weak_default not in compose
    assert "127.0.0.1:8080:8080" in compose

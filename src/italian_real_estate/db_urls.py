"""Safe SQLAlchemy URL construction shared by pipeline modules."""

from sqlalchemy.engine import URL

from italian_real_estate.config.settings import (
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


def get_postgres_connection_string(
    host: str = POSTGRES_HOST,
    port: str = POSTGRES_PORT,
    user: str = POSTGRES_USER,
    password: str = POSTGRES_PASSWORD,
    database: str = POSTGRES_DATABASE,
) -> URL:
    """Return a SQLAlchemy URL without manually interpolating credentials."""
    return URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=password,
        host=host,
        port=int(port),
        database=database,
    )

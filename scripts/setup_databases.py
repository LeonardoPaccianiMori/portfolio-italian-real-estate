#!/usr/bin/env python3
"""
Database setup script for the Italian Real Estate pipeline.

This script initializes both MongoDB and PostgreSQL databases with the
required schema. It can be run on a fresh machine to set up all databases.

Features:
- Creates MongoDB databases and collections with indexes
- Creates PostgreSQL database and schema with all tables
- Configures PostgreSQL credentials (prompts if not set)
- Safe to run multiple times (idempotent)

Usage:
    python setup_databases.py              # Interactive setup
    python setup_databases.py --check      # Check database status only
    python setup_databases.py --postgres   # Setup PostgreSQL only
    python setup_databases.py --mongodb    # Setup MongoDB only

Author: Leonardo Pacciani-Mori
License: MIT
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / "src"))

import argparse
import getpass
import json
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm


console = Console()


# =============================================================================
# CONFIGURATION FILE HANDLING
# =============================================================================

def get_config_path() -> Path:
    """Get the path to the configuration file."""
    config_dir = Path.home() / ".config" / "italian-real-estate"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "db_config.json"


def load_config() -> dict:
    """Load configuration from file."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def save_config(config: dict) -> None:
    """Save configuration to file."""
    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    # Set restrictive permissions (owner read/write only)
    config_path.chmod(0o600)


def get_postgres_credentials(force_prompt: bool = False, interactive: bool = True) -> dict:
    """
    Get PostgreSQL credentials from config or prompt user.

    Args:
        force_prompt: If True, always prompt for new credentials
        interactive: If False, never prompt (use defaults/saved config)

    Returns:
        dict with host, port, user, password, database keys
    """
    config = load_config()
    postgres_config = config.get("postgresql", {})

    # Default values - try to get from settings.py first
    try:
        from italian_real_estate.config.settings import POSTGRES_CONNECTION_PARAMS
        defaults = {
            "host": POSTGRES_CONNECTION_PARAMS.get("host", "localhost"),
            "port": POSTGRES_CONNECTION_PARAMS.get("port", "5432"),
            "user": POSTGRES_CONNECTION_PARAMS.get("user", os.getenv("USER", "postgres")),
            "password": POSTGRES_CONNECTION_PARAMS.get("password", ""),
            "database": POSTGRES_CONNECTION_PARAMS.get("database", "listing_website_warehouse"),
        }
    except ImportError:
        defaults = {
            "host": "localhost",
            "port": "5432",
            "user": os.getenv("USER", "postgres"),
            "password": "",
            "database": "listing_website_warehouse",
        }

    # If we have saved config and not forcing prompt, use it
    if postgres_config and not force_prompt:
        return {**defaults, **postgres_config}

    # If not interactive, use defaults
    if not interactive:
        return defaults

    console.print("\n[bold cyan]PostgreSQL Configuration[/bold cyan]")
    console.print("Enter PostgreSQL connection details (press Enter for defaults):\n")

    credentials = {}
    credentials["host"] = Prompt.ask(
        "  Host",
        default=postgres_config.get("host", defaults["host"])
    )
    credentials["port"] = Prompt.ask(
        "  Port",
        default=postgres_config.get("port", defaults["port"])
    )
    credentials["user"] = Prompt.ask(
        "  Username",
        default=postgres_config.get("user", defaults["user"])
    )
    credentials["password"] = Prompt.ask(
        "  Password",
        password=True,
        default=postgres_config.get("password", "")
    )
    credentials["database"] = Prompt.ask(
        "  Database name",
        default=postgres_config.get("database", defaults["database"])
    )

    # Optionally save credentials
    if Confirm.ask("\n  Save these credentials for future use?", default=True):
        config["postgresql"] = credentials
        save_config(config)
        console.print("  [green]Credentials saved to ~/.config/italian-real-estate/db_config.json[/green]")

    return credentials


def get_mongodb_config(force_prompt: bool = False) -> dict:
    """
    Get MongoDB configuration from config or use defaults.

    Args:
        force_prompt: If True, prompt for new configuration

    Returns:
        dict with host, port, datalake_name, warehouse_name keys
    """
    config = load_config()
    mongo_config = config.get("mongodb", {})

    try:
        from italian_real_estate.config.settings import (
            MONGODB_HOST,
            MONGODB_PORT,
            MONGODB_USER,
            MONGODB_PASSWORD,
            MONGODB_AUTH_SOURCE,
            MONGODB_DATALAKE_NAME,
            MONGODB_WAREHOUSE_NAME,
        )
        defaults = {
            "host": MONGODB_HOST,
            "port": MONGODB_PORT,
            "username": MONGODB_USER,
            "password": MONGODB_PASSWORD,
            "auth_source": MONGODB_AUTH_SOURCE,
            "datalake_name": MONGODB_DATALAKE_NAME,
            "warehouse_name": MONGODB_WAREHOUSE_NAME,
        }
    except ImportError:
        defaults = {
            "host": os.getenv("MONGODB_HOST", "127.0.0.1"),
            "port": int(os.getenv("MONGODB_PORT", "27017")),
            "username": os.getenv("MONGODB_USER", ""),
            "password": os.getenv("MONGODB_PASSWORD", ""),
            "auth_source": os.getenv("MONGODB_AUTH_SOURCE", "admin"),
            "datalake_name": os.getenv("MONGODB_DATALAKE_NAME", "listing_website_datalake"),
            "warehouse_name": os.getenv("MONGODB_WAREHOUSE_NAME", "listing_website_warehouse"),
        }

    if mongo_config and not force_prompt:
        return {**defaults, **mongo_config}

    if force_prompt:
        console.print("\n[bold cyan]MongoDB Configuration[/bold cyan]")
        console.print("Enter MongoDB connection details (press Enter for defaults):\n")

        mongo_config = {}
        mongo_config["host"] = Prompt.ask(
            "  Host",
            default=defaults["host"]
        )
        mongo_config["port"] = int(Prompt.ask(
            "  Port",
            default=str(defaults["port"])
        ))
        mongo_config["datalake_name"] = Prompt.ask(
            "  Datalake database name",
            default=defaults["datalake_name"]
        )
        mongo_config["warehouse_name"] = Prompt.ask(
            "  Warehouse database name",
            default=defaults["warehouse_name"]
        )
        mongo_config["username"] = Prompt.ask(
            "  Username (leave blank for none)",
            default=defaults.get("username", "")
        )
        mongo_config["password"] = Prompt.ask(
            "  Password (leave blank for none)",
            default=defaults.get("password", ""),
            password=True,
        )
        mongo_config["auth_source"] = Prompt.ask(
            "  Auth source",
            default=defaults.get("auth_source", "admin"),
        )

        if Confirm.ask("\n  Save this configuration?", default=True):
            config["mongodb"] = mongo_config
            save_config(config)

        return mongo_config

    return defaults


# =============================================================================
# MONGODB SETUP
# =============================================================================

def setup_mongodb(config: Optional[dict] = None) -> dict:
    """
    Set up MongoDB databases and collections.

    MongoDB creates databases and collections automatically on first insert,
    but we create them explicitly here to ensure indexes are set up.

    Args:
        config: MongoDB configuration dict

    Returns:
        dict with setup results
    """
    try:
        from pymongo.errors import ConnectionFailure
        from italian_real_estate.core.connections import get_mongodb_client
    except ImportError:
        return {"success": False, "error": "pymongo not installed"}

    if config is None:
        config = get_mongodb_config()

    results = {
        "success": True,
        "datalake": {"collections": [], "indexes": []},
        "warehouse": {"collections": [], "indexes": []},
    }

    try:
        client = get_mongodb_client(
            config["host"],
            config["port"],
            username=config.get("username") or None,
            password=config.get("password") or None,
            auth_source=config.get("auth_source") or None,
            timeout_ms=5000,
        )
        # Test connection
        client.admin.command("ping")
    except ConnectionFailure as e:
        return {"success": False, "error": f"Cannot connect to MongoDB: {e}"}

    collection_names = ["sale", "rent", "auction"]

    # Setup datalake
    datalake_db = client[config["datalake_name"]]
    for coll_name in collection_names:
        coll = datalake_db[coll_name]
        # Create index on listing_id for fast lookups
        coll.create_index("listing_id", unique=True)
        results["datalake"]["collections"].append(coll_name)
        results["datalake"]["indexes"].append(f"{coll_name}.listing_id")

    # Setup warehouse
    warehouse_db = client[config["warehouse_name"]]
    for coll_name in collection_names:
        coll = warehouse_db[coll_name]
        # Create indexes for warehouse collections
        coll.create_index("listing_id")
        coll.create_index("listing_type")
        coll.create_index("date_scraped")
        results["warehouse"]["collections"].append(coll_name)
        results["warehouse"]["indexes"].extend([
            f"{coll_name}.listing_id",
            f"{coll_name}.listing_type",
            f"{coll_name}.date_scraped",
        ])

    client.close()
    return results


def check_mongodb_status(config: Optional[dict] = None) -> dict:
    """Check MongoDB database status."""
    try:
        from pymongo.errors import ConnectionFailure
        from italian_real_estate.core.connections import get_mongodb_client
    except ImportError:
        return {"connected": False, "error": "pymongo not installed"}

    if config is None:
        config = get_mongodb_config()

    try:
        client = get_mongodb_client(
            config["host"],
            config["port"],
            username=config.get("username") or None,
            password=config.get("password") or None,
            auth_source=config.get("auth_source") or None,
            timeout_ms=5000,
        )
        client.admin.command("ping")
    except ConnectionFailure as e:
        return {"connected": False, "error": str(e)}

    status = {
        "connected": True,
        "host": f"{config['host']}:{config['port']}",
        "databases": {},
    }

    for db_name in [config["datalake_name"], config["warehouse_name"]]:
        db = client[db_name]
        collections = db.list_collection_names()
        db_status = {"collections": {}}
        for coll_name in collections:
            db_status["collections"][coll_name] = db[coll_name].count_documents({})
        status["databases"][db_name] = db_status

    client.close()
    return status


# =============================================================================
# POSTGRESQL SETUP
# =============================================================================

def setup_postgresql(credentials: Optional[dict] = None) -> dict:
    """
    Set up PostgreSQL database and schema.

    Creates the database if it doesn't exist, then creates all tables
    according to the star schema.

    Args:
        credentials: PostgreSQL credentials dict

    Returns:
        dict with setup results
    """
    try:
        import psycopg2
        from psycopg2 import sql
    except ImportError:
        return {"success": False, "error": "psycopg2 not installed"}

    if credentials is None:
        credentials = get_postgres_credentials()

    results = {
        "success": True,
        "database_created": False,
        "tables": {"created": [], "existing": [], "errors": []},
    }

    # First, try to create the database (connect to 'postgres' database)
    try:
        conn = psycopg2.connect(
            host=credentials["host"],
            port=credentials["port"],
            user=credentials["user"],
            password=credentials["password"],
            database="postgres",
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (credentials["database"],)
        )

        if not cursor.fetchone():
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(credentials["database"])
                )
            )
            results["database_created"] = True

        cursor.close()
        conn.close()
    except psycopg2.Error as e:
        return {"success": False, "error": f"Cannot connect to PostgreSQL: {e}"}

    # Now connect to the target database and create schema
    try:
        conn = psycopg2.connect(
            host=credentials["host"],
            port=credentials["port"],
            user=credentials["user"],
            password=credentials["password"],
            database=credentials["database"],
        )

        from italian_real_estate.migration.schema import create_postgresql_schema

        schema_results = create_postgresql_schema(conn)
        results["tables"] = schema_results

        conn.close()
    except psycopg2.Error as e:
        return {"success": False, "error": f"Error creating schema: {e}"}

    return results


def check_postgresql_status(credentials: Optional[dict] = None, interactive: bool = False) -> dict:
    """Check PostgreSQL database status."""
    try:
        import psycopg2
    except ImportError:
        return {"connected": False, "error": "psycopg2 not installed"}

    if credentials is None:
        credentials = get_postgres_credentials(interactive=interactive)

    try:
        conn = psycopg2.connect(
            host=credentials["host"],
            port=credentials["port"],
            user=credentials["user"],
            password=credentials["password"],
            database=credentials["database"],
        )
    except psycopg2.Error as e:
        return {"connected": False, "error": str(e)}

    status = {
        "connected": True,
        "host": f"{credentials['host']}:{credentials['port']}",
        "database": credentials["database"],
        "tables": {},
    }

    cursor = conn.cursor()

    # Get table names and row counts
    cursor.execute("""
        SELECT schemaname, tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename
    """)

    for _, table_name in cursor.fetchall():
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        count = cursor.fetchone()[0]
        status["tables"][table_name] = count

    cursor.close()
    conn.close()
    return status


# =============================================================================
# CLI INTERFACE
# =============================================================================

def display_status(mongodb_status: dict, postgres_status: dict) -> None:
    """Display database status in a formatted table."""
    # MongoDB status
    mongo_table = Table(title="MongoDB Status")
    mongo_table.add_column("Database", style="cyan")
    mongo_table.add_column("Collection", style="green")
    mongo_table.add_column("Documents", justify="right")

    if mongodb_status.get("connected"):
        for db_name, db_info in mongodb_status.get("databases", {}).items():
            first = True
            for coll_name, count in db_info.get("collections", {}).items():
                mongo_table.add_row(
                    db_name if first else "",
                    coll_name,
                    f"{count:,}"
                )
                first = False
            if not db_info.get("collections"):
                mongo_table.add_row(db_name, "(no collections)", "-")
    else:
        mongo_table.add_row(
            "[red]Not connected[/red]",
            mongodb_status.get("error", "Unknown error"),
            "-"
        )

    console.print(mongo_table)
    console.print()

    # PostgreSQL status
    pg_table = Table(title="PostgreSQL Status")
    pg_table.add_column("Table", style="cyan")
    pg_table.add_column("Rows", justify="right")

    if postgres_status.get("connected"):
        for table_name, count in postgres_status.get("tables", {}).items():
            pg_table.add_row(table_name, f"{count:,}")
        if not postgres_status.get("tables"):
            pg_table.add_row("(no tables)", "-")
    else:
        pg_table.add_row(
            "[red]Not connected[/red]",
            postgres_status.get("error", "Unknown error")
        )

    console.print(pg_table)


def display_setup_results(results: dict, db_type: str) -> None:
    """Display setup results."""
    if not results.get("success"):
        console.print(f"[red]Error setting up {db_type}: {results.get('error')}[/red]")
        return

    if db_type == "MongoDB":
        console.print(f"[green]{db_type} setup complete![/green]")
        console.print(f"  Datalake collections: {', '.join(results['datalake']['collections'])}")
        console.print(f"  Warehouse collections: {', '.join(results['warehouse']['collections'])}")
    else:
        console.print(f"[green]{db_type} setup complete![/green]")
        if results.get("database_created"):
            console.print("  [cyan]Database created[/cyan]")
        tables = results.get("tables", {})
        if tables.get("created"):
            console.print(f"  Tables created/verified: {len(tables['created'])}")
        if tables.get("errors"):
            for table, error in tables["errors"]:
                console.print(f"  [red]Error with {table}: {error}[/red]")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up databases for the Italian Real Estate pipeline"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check database status without making changes"
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Set up PostgreSQL only"
    )
    parser.add_argument(
        "--mongodb",
        action="store_true",
        help="Set up MongoDB only"
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Reconfigure database credentials"
    )

    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]Italian Real Estate Database Setup[/bold]",
        border_style="blue"
    ))

    # Determine which databases to work with
    do_mongodb = not args.postgres or args.mongodb
    do_postgres = not args.mongodb or args.postgres
    if not args.postgres and not args.mongodb:
        do_mongodb = do_postgres = True

    if args.check:
        # Check status only
        console.print("\n[bold]Checking database status...[/bold]\n")

        mongo_status = check_mongodb_status() if do_mongodb else {}
        pg_status = check_postgresql_status() if do_postgres else {}

        if do_mongodb and do_postgres:
            display_status(mongo_status, pg_status)
        elif do_mongodb:
            display_status(mongo_status, {"connected": False, "error": "Skipped"})
        else:
            display_status({"connected": False, "error": "Skipped"}, pg_status)

        return 0

    # Setup databases
    if do_mongodb:
        console.print("\n[bold]Setting up MongoDB...[/bold]")
        mongo_config = get_mongodb_config(force_prompt=args.reconfigure)
        mongo_results = setup_mongodb(mongo_config)
        display_setup_results(mongo_results, "MongoDB")

    if do_postgres:
        console.print("\n[bold]Setting up PostgreSQL...[/bold]")
        pg_credentials = get_postgres_credentials(force_prompt=args.reconfigure)
        pg_results = setup_postgresql(pg_credentials)
        display_setup_results(pg_results, "PostgreSQL")

    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("Run with --check to verify database status.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

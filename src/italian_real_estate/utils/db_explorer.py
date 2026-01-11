"""
Interactive database explorer for MongoDB and PostgreSQL.

This module provides a REPL (Read-Eval-Print Loop) interface for exploring
the Italian Real Estate databases, including the MongoDB datalake/warehouse
and PostgreSQL data warehouse.

Usage:
    from italian_real_estate.utils.db_explorer import DatabaseExplorer
    explorer = DatabaseExplorer()
    explorer.run()

Author: Leonardo Pacciani-Mori
License: MIT
"""

import cmd
import json
import os
import readline
import shlex
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from italian_real_estate.config.settings import (
    MONGODB_HOST,
    MONGODB_PORT,
    MONGODB_DATALAKE_NAME,
    MONGODB_WAREHOUSE_NAME,
    POSTGRES_CONNECTION_PARAMS,
    COLLECTION_NAMES,
)
from italian_real_estate.core.connections import get_mongodb_client


# History file for command history persistence
HISTORY_FILE = os.path.expanduser("~/.ire_db_history")


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the database. Returns True if successful."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the database."""
        pass

    @abstractmethod
    def list_collections(self) -> List[Tuple[str, int]]:
        """List all collections/tables with their record counts."""
        pass

    @abstractmethod
    def get_sample(self, name: str, n: int = 5) -> List[Dict]:
        """Get n sample records from a collection/table."""
        pass

    @abstractmethod
    def get_count(self, name: str) -> int:
        """Get the record count for a collection/table."""
        pass

    @abstractmethod
    def get_schema(self, name: str) -> Dict[str, Any]:
        """Get the schema/fields for a collection/table."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this backend."""
        pass

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Short name for prompt display."""
        pass


class MongoDBBackend(DatabaseBackend):
    """MongoDB backend for database exploration."""

    def __init__(self, db_type: str = "datalake"):
        """
        Initialize MongoDB backend.

        Args:
            db_type: Either "datalake" or "warehouse"
        """
        self.db_type = db_type
        self.client = None
        self.db = None
        self._db_name = MONGODB_DATALAKE_NAME if db_type == "datalake" else MONGODB_WAREHOUSE_NAME

    @property
    def name(self) -> str:
        """Human-friendly backend label including the database name."""
        return f"MongoDB {'Datalake' if self.db_type == 'datalake' else 'Warehouse'} ({self._db_name})"

    @property
    def short_name(self) -> str:
        """Short identifier used in prompts and status output."""
        return f"mongo-{self.db_type}"

    def connect(self) -> bool:
        """Open a MongoDB connection and prime the target database handle."""
        try:
            self.client = get_mongodb_client(
                MONGODB_HOST,
                MONGODB_PORT,
                timeout_ms=5000,
            )
            # Test connection
            self.client.server_info()
            self.db = self.client[self._db_name]
            return True
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            return False

    def disconnect(self) -> None:
        """Close any active Mongo client and clear cached handles."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    def list_collections(self) -> List[Tuple[str, int]]:
        """Return collection names with approximate document counts."""
        if not self.db:
            return []
        result = []
        for name in self.db.list_collection_names():
            count = self.db[name].estimated_document_count()
            result.append((name, count))
        return sorted(result, key=lambda x: x[0])

    def get_sample(self, name: str, n: int = 5) -> List[Dict]:
        """Return a random sample of documents from a collection."""
        if not self.db:
            return []
        collection = self.db[name]
        # Use aggregation with $sample for random sampling
        pipeline = [{"$sample": {"size": n}}]
        return list(collection.aggregate(pipeline))

    def get_count(self, name: str) -> int:
        """Return the estimated document count for a collection."""
        if not self.db:
            return 0
        return self.db[name].estimated_document_count()

    def get_schema(self, name: str) -> Dict[str, Any]:
        """Infer schema from sample documents."""
        if not self.db:
            return {}

        # Get a few documents to infer schema
        samples = list(self.db[name].find().limit(10))
        if not samples:
            return {}

        # Collect all keys and their types
        schema = {}
        for doc in samples:
            for key, value in doc.items():
                if key not in schema:
                    schema[key] = {
                        "type": type(value).__name__,
                        "example": str(value)[:50] if value is not None else "null"
                    }

        return schema

    def get_stats(self) -> Dict[str, Any]:
        """Return high-level stats (counts per collection) for the database."""
        if not self.db:
            return {}

        stats = {
            "database": self._db_name,
            "type": self.db_type,
            "collections": {}
        }

        for name in self.db.list_collection_names():
            stats["collections"][name] = {
                "count": self.db[name].estimated_document_count()
            }

        return stats

    def find(self, collection: str, filter_dict: Dict) -> List[Dict]:
        """Execute a find query with a filter."""
        if not self.db:
            return []
        return list(self.db[collection].find(filter_dict).limit(100))

    def wipe_collection(self, collection: str) -> int:
        """Delete all documents from a collection. Returns number deleted."""
        if not self.db:
            return 0
        result = self.db[collection].delete_many({})
        return result.deleted_count

    def wipe_all(self) -> Dict[str, int]:
        """Delete all documents from all collections. Returns counts per collection."""
        if not self.db:
            return {}
        results = {}
        for name in self.db.list_collection_names():
            results[name] = self.wipe_collection(name)
        return results


class PostgreSQLBackend(DatabaseBackend):
    """PostgreSQL backend for database exploration."""

    def __init__(self):
        """Initialize the PostgreSQL backend with connection placeholders."""
        self.conn = None
        self._db_name = POSTGRES_CONNECTION_PARAMS.get("database", "unknown")

    @property
    def name(self) -> str:
        """Human-friendly backend label including the database name."""
        return f"PostgreSQL ({self._db_name})"

    @property
    def short_name(self) -> str:
        """Short identifier used in prompts and status output."""
        return "postgres"

    def connect(self) -> bool:
        """Open a PostgreSQL connection using configured credentials."""
        try:
            import psycopg2
            self.conn = psycopg2.connect(**POSTGRES_CONNECTION_PARAMS)
            return True
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            return False

    def disconnect(self) -> None:
        """Close any active PostgreSQL connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def list_collections(self) -> List[Tuple[str, int]]:
        """List all tables with their row counts."""
        if not self.conn:
            return []

        cursor = self.conn.cursor()

        # Get all tables in public schema
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        result = []
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            result.append((table, count))

        cursor.close()
        return result

    def get_sample(self, name: str, n: int = 5) -> List[Dict]:
        """Return up to n rows from a table as dictionaries."""
        if not self.conn:
            return []

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {name} LIMIT {n}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()

        return [dict(zip(columns, row)) for row in rows]

    def get_count(self, name: str) -> int:
        """Return the row count for a given table."""
        if not self.conn:
            return 0

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {name}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def get_schema(self, name: str) -> Dict[str, Any]:
        """Return column metadata (type/nullability/default) for a table."""
        if not self.conn:
            return {}

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (name,))

        schema = {}
        for row in cursor.fetchall():
            col_name, data_type, nullable, default = row
            schema[col_name] = {
                "type": data_type,
                "nullable": nullable == "YES",
                "default": default
            }

        cursor.close()
        return schema

    def get_stats(self) -> Dict[str, Any]:
        """Return categorized table counts grouped by fact/dimension/bridge tables."""
        if not self.conn:
            return {}

        collections = self.list_collections()

        # Categorize tables
        fact_tables = [(n, c) for n, c in collections if n.startswith("fact_")]
        dim_tables = [(n, c) for n, c in collections if n.startswith("dim_")]
        bridge_tables = [(n, c) for n, c in collections if n.endswith("_bridge")]
        other_tables = [(n, c) for n, c in collections
                       if not n.startswith("fact_")
                       and not n.startswith("dim_")
                       and not n.endswith("_bridge")]

        return {
            "database": self._db_name,
            "fact_tables": dict(fact_tables),
            "dimension_tables": dict(dim_tables),
            "bridge_tables": dict(bridge_tables),
            "other_tables": dict(other_tables),
            "total_tables": len(collections)
        }

    def query(self, sql: str) -> Tuple[List[str], List[Tuple]]:
        """Execute a SQL query and return columns and rows."""
        if not self.conn:
            return [], []

        cursor = self.conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        cursor.close()
        return columns, rows

    def wipe_table(self, table: str) -> int:
        """Delete all rows from a table using TRUNCATE. Returns previous row count."""
        if not self.conn:
            return 0

        cursor = self.conn.cursor()
        # Get count before truncating
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]

        # TRUNCATE CASCADE to handle foreign key constraints
        cursor.execute(f"TRUNCATE TABLE {table} CASCADE")
        self.conn.commit()
        cursor.close()
        return count

    def wipe_all(self) -> Dict[str, int]:
        """Delete all rows from all tables. Returns counts per table."""
        if not self.conn:
            return {}

        # Get table list with proper order (bridge tables first, then fact, then dimensions)
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()

        # Order tables for deletion: bridge -> fact -> dimension
        bridge_tables = [t for t in tables if t.endswith("_bridge")]
        fact_tables = [t for t in tables if t.startswith("fact_")]
        dim_tables = [t for t in tables if t.startswith("dim_")]
        other_tables = [t for t in tables if t not in bridge_tables + fact_tables + dim_tables]

        ordered_tables = bridge_tables + fact_tables + dim_tables + other_tables

        results = {}
        for table in ordered_tables:
            results[table] = self.wipe_table(table)

        return results


class DatabaseExplorer(cmd.Cmd):
    """
    Interactive database explorer REPL.

    Provides commands for exploring MongoDB and PostgreSQL databases.
    """

    intro = """
╔══════════════════════════════════════════════════════════════════╗
║         Italian Real Estate Database Explorer                    ║
║                                                                  ║
║  Type 'help' for available commands, 'exit' to quit.             ║
║  Use 'use <database>' to connect to a database.                  ║
║                                                                  ║
║  Available databases:                                            ║
║    - mongo-datalake   MongoDB raw data                           ║
║    - mongo-warehouse  MongoDB transformed data                   ║
║    - postgres         PostgreSQL star schema                     ║
║                                                                  ║
║  Destructive commands (require confirmation):                    ║
║    - wipe <name>      Delete all data from a table/collection    ║
║    - wipe all         Delete ALL data from current database      ║
╚══════════════════════════════════════════════════════════════════╝
    """

    def __init__(self):
        """Initialize the explorer shell with prompt state and history."""
        super().__init__()
        self.console = Console() if RICH_AVAILABLE else None
        self.current_backend: Optional[DatabaseBackend] = None
        self._update_prompt()

        # Load command history
        self._load_history()

    def _update_prompt(self):
        """Update the prompt based on current database."""
        if self.current_backend:
            self.prompt = f"db[{self.current_backend.short_name}]> "
        else:
            self.prompt = "db> "

    def _load_history(self):
        """Load command history from file."""
        try:
            if os.path.exists(HISTORY_FILE):
                readline.read_history_file(HISTORY_FILE)
        except Exception:
            pass

    def _save_history(self):
        """Save command history to file."""
        try:
            readline.set_history_length(1000)
            readline.write_history_file(HISTORY_FILE)
        except Exception:
            pass

    def _print(self, text: str, style: str = None):
        """Print text, using rich if available."""
        if self.console and RICH_AVAILABLE:
            self.console.print(text, style=style)
        else:
            print(text)

    def _print_table(self, title: str, columns: List[str], rows: List[Tuple]):
        """Print a formatted table."""
        if self.console and RICH_AVAILABLE:
            table = Table(title=title, show_header=True, header_style="bold cyan")
            for col in columns:
                table.add_column(col)
            for row in rows:
                table.add_row(*[str(cell) for cell in row])
            self.console.print(table)
        else:
            print(f"\n{title}")
            print("-" * 60)
            print(" | ".join(columns))
            print("-" * 60)
            for row in rows:
                print(" | ".join(str(cell) for cell in row))
            print()

    def _print_json(self, data: Any, title: str = None):
        """Print JSON data with syntax highlighting."""
        json_str = json.dumps(data, indent=2, default=str)
        if self.console and RICH_AVAILABLE:
            if title:
                self.console.print(Panel(
                    Syntax(json_str, "json", theme="monokai"),
                    title=title
                ))
            else:
                self.console.print(Syntax(json_str, "json", theme="monokai"))
        else:
            if title:
                print(f"\n{title}")
                print("-" * 40)
            print(json_str)

    def _require_connection(self) -> bool:
        """Check if connected to a database."""
        if not self.current_backend:
            self._print("Not connected to any database. Use 'use <database>' first.", "red")
            return False
        return True

    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------

    def do_use(self, arg: str):
        """Switch to a database: use <mongo-datalake|mongo-warehouse|postgres>"""
        arg = arg.strip().lower()

        # Disconnect current backend
        if self.current_backend:
            self.current_backend.disconnect()
            self.current_backend = None

        # Create new backend
        if arg == "mongo-datalake":
            backend = MongoDBBackend("datalake")
        elif arg == "mongo-warehouse":
            backend = MongoDBBackend("warehouse")
        elif arg == "postgres":
            backend = PostgreSQLBackend()
        else:
            self._print(f"Unknown database: {arg}", "red")
            self._print("Available: mongo-datalake, mongo-warehouse, postgres")
            self._update_prompt()
            return

        # Connect
        if backend.connect():
            self.current_backend = backend
            self._print(f"Connected to: {backend.name}", "green")
        else:
            self._print(f"Failed to connect to {arg}", "red")

        self._update_prompt()

    def complete_use(self, text, line, begidx, endidx):
        """Tab completion for use command."""
        options = ["mongo-datalake", "mongo-warehouse", "postgres"]
        return [o for o in options if o.startswith(text)]

    def do_collections(self, arg: str):
        """List all collections/tables in the current database."""
        if not self._require_connection():
            return

        collections = self.current_backend.list_collections()
        if not collections:
            self._print("No collections/tables found.")
            return

        # Format based on backend type
        if isinstance(self.current_backend, PostgreSQLBackend):
            stats = self.current_backend.get_stats()

            self._print("\n[bold]Fact Tables:[/bold]" if RICH_AVAILABLE else "\nFact Tables:")
            for name, count in stats.get("fact_tables", {}).items():
                self._print(f"  - {name} ({count:,} rows)")

            self._print("\n[bold]Dimension Tables:[/bold]" if RICH_AVAILABLE else "\nDimension Tables:")
            for name, count in stats.get("dimension_tables", {}).items():
                self._print(f"  - {name} ({count:,} rows)")

            self._print("\n[bold]Bridge Tables:[/bold]" if RICH_AVAILABLE else "\nBridge Tables:")
            for name, count in stats.get("bridge_tables", {}).items():
                self._print(f"  - {name} ({count:,} rows)")

            if stats.get("other_tables"):
                self._print("\n[bold]Other Tables:[/bold]" if RICH_AVAILABLE else "\nOther Tables:")
                for name, count in stats.get("other_tables", {}).items():
                    self._print(f"  - {name} ({count:,} rows)")
        else:
            self._print("\n[bold]Collections:[/bold]" if RICH_AVAILABLE else "\nCollections:")
            for name, count in collections:
                self._print(f"  - {name} ({count:,} documents)")

        self._print("")

    def do_tables(self, arg: str):
        """Alias for collections (for PostgreSQL users)."""
        self.do_collections(arg)

    def do_sample(self, arg: str):
        """Show sample records: sample <collection/table> [n]"""
        if not self._require_connection():
            return

        parts = arg.split()
        if not parts:
            self._print("Usage: sample <collection/table> [n]", "yellow")
            return

        name = parts[0]
        n = int(parts[1]) if len(parts) > 1 else 5

        try:
            samples = self.current_backend.get_sample(name, n)
            total = self.current_backend.get_count(name)

            if not samples:
                self._print(f"No records found in {name}")
                return

            self._print(f"\n[bold]Sample from {name} ({n} of {total:,}):[/bold]" if RICH_AVAILABLE
                       else f"\nSample from {name} ({n} of {total:,}):")

            for i, record in enumerate(samples, 1):
                self._print_json(record, f"Record {i}")

        except Exception as e:
            self._print(f"Error: {e}", "red")

    def complete_sample(self, text, line, begidx, endidx):
        """Tab completion for sample command."""
        if not self.current_backend:
            return []
        collections = [name for name, _ in self.current_backend.list_collections()]
        return [c for c in collections if c.startswith(text)]

    def do_count(self, arg: str):
        """Get record count: count <collection/table>"""
        if not self._require_connection():
            return

        name = arg.strip()
        if not name:
            self._print("Usage: count <collection/table>", "yellow")
            return

        try:
            count = self.current_backend.get_count(name)
            self._print(f"{name}: {count:,} records")
        except Exception as e:
            self._print(f"Error: {e}", "red")

    def complete_count(self, text, line, begidx, endidx):
        """Tab completion for count command."""
        return self.complete_sample(text, line, begidx, endidx)

    def do_schema(self, arg: str):
        """Show schema for a collection/table: schema <name>"""
        if not self._require_connection():
            return

        name = arg.strip()
        if not name:
            self._print("Usage: schema <collection/table>", "yellow")
            return

        try:
            schema = self.current_backend.get_schema(name)
            if not schema:
                self._print(f"No schema information for {name}")
                return

            self._print(f"\n[bold]Schema for {name}:[/bold]" if RICH_AVAILABLE
                       else f"\nSchema for {name}:")

            if isinstance(self.current_backend, PostgreSQLBackend):
                rows = [(col, info["type"], "Yes" if info["nullable"] else "No")
                       for col, info in schema.items()]
                self._print_table(f"Columns in {name}", ["Column", "Type", "Nullable"], rows)
            else:
                rows = [(col, info["type"], info["example"][:30])
                       for col, info in schema.items()]
                self._print_table(f"Fields in {name}", ["Field", "Type", "Example"], rows)

        except Exception as e:
            self._print(f"Error: {e}", "red")

    def complete_schema(self, text, line, begidx, endidx):
        """Tab completion for schema command."""
        return self.complete_sample(text, line, begidx, endidx)

    def do_query(self, arg: str):
        """Run SQL query (PostgreSQL only): query <SQL>"""
        if not self._require_connection():
            return

        if not isinstance(self.current_backend, PostgreSQLBackend):
            self._print("SQL queries only available for PostgreSQL. Use 'find' for MongoDB.", "yellow")
            return

        sql = arg.strip()
        if not sql:
            self._print("Usage: query <SQL statement>", "yellow")
            return

        try:
            columns, rows = self.current_backend.query(sql)
            if columns:
                self._print_table("Query Result", columns, rows)
                self._print(f"({len(rows)} rows)")
            else:
                self._print("Query executed successfully (no results to display)")
        except Exception as e:
            self._print(f"Query error: {e}", "red")

    def do_find(self, arg: str):
        """Run MongoDB find query: find <collection> <json_filter>"""
        if not self._require_connection():
            return

        if not isinstance(self.current_backend, MongoDBBackend):
            self._print("find command only available for MongoDB. Use 'query' for PostgreSQL.", "yellow")
            return

        parts = arg.split(None, 1)
        if len(parts) < 2:
            self._print("Usage: find <collection> <json_filter>", "yellow")
            self._print("Example: find sale {\"price\": {\"$gt\": 100000}}")
            return

        collection, filter_str = parts
        try:
            filter_dict = json.loads(filter_str)
            results = self.current_backend.find(collection, filter_dict)

            self._print(f"\n[bold]Found {len(results)} documents (max 100 shown):[/bold]" if RICH_AVAILABLE
                       else f"\nFound {len(results)} documents (max 100 shown):")

            for i, doc in enumerate(results[:10], 1):
                self._print_json(doc, f"Document {i}")

            if len(results) > 10:
                self._print(f"... and {len(results) - 10} more")

        except json.JSONDecodeError as e:
            self._print(f"Invalid JSON filter: {e}", "red")
        except Exception as e:
            self._print(f"Error: {e}", "red")

    def complete_find(self, text, line, begidx, endidx):
        """Tab completion for find command."""
        return self.complete_sample(text, line, begidx, endidx)

    def do_stats(self, arg: str):
        """Show database statistics."""
        if not self._require_connection():
            return

        stats = self.current_backend.get_stats()
        self._print_json(stats, f"Statistics for {self.current_backend.name}")

    def do_wipe(self, arg: str):
        """
        Delete all data from a collection/table: wipe <name>
        Or delete ALL data: wipe all

        This permanently deletes all data from the specified collection or table.
        Requires confirmation before executing.
        """
        if not self._require_connection():
            return

        name = arg.strip()
        if not name:
            self._print("Usage: wipe <collection/table> or wipe all", "yellow")
            return

        # Handle "wipe all" as alias for wipeall
        if name.lower() == "all":
            return self.do_wipeall("")

        # Get current count
        try:
            count = self.current_backend.get_count(name)
        except Exception as e:
            self._print(f"Error: {e}", "red")
            return

        if count == 0:
            self._print(f"{name} is already empty.")
            return

        # Confirm action
        self._print(f"\n[bold red]WARNING:[/bold red] This will permanently delete {count:,} records from '{name}'!"
                   if RICH_AVAILABLE else f"\nWARNING: This will permanently delete {count:,} records from '{name}'!")
        self._print("Type 'yes' to confirm: ", style="yellow")

        try:
            confirm = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            self._print("\nCancelled.")
            return

        if confirm != "yes":
            self._print("Cancelled.")
            return

        # Execute wipe
        try:
            if isinstance(self.current_backend, MongoDBBackend):
                deleted = self.current_backend.wipe_collection(name)
            else:
                deleted = self.current_backend.wipe_table(name)

            self._print(f"[green]Deleted {deleted:,} records from '{name}'.[/green]"
                       if RICH_AVAILABLE else f"Deleted {deleted:,} records from '{name}'.")
        except Exception as e:
            self._print(f"Error: {e}", "red")

    def complete_wipe(self, text, line, begidx, endidx):
        """Tab completion for wipe command."""
        return self.complete_sample(text, line, begidx, endidx)

    def do_wipeall(self, arg: str):
        """
        Delete all data from all collections/tables in the current database.

        This permanently deletes ALL data from the current database.
        Requires typing the database name to confirm.
        Use: wipeall  OR  wipe all
        """
        if not self._require_connection():
            return

        # Get all collections/tables with counts
        collections = self.current_backend.list_collections()
        total_records = sum(count for _, count in collections)

        if total_records == 0:
            self._print("Database is already empty.")
            return

        # Display what will be deleted
        self._print(f"\n[bold red]WARNING:[/bold red] This will permanently delete ALL data from '{self.current_backend.short_name}'!"
                   if RICH_AVAILABLE else f"\nWARNING: This will permanently delete ALL data from '{self.current_backend.short_name}'!")
        self._print(f"\nCollections/tables to be wiped:")
        for name, count in collections:
            self._print(f"  - {name}: {count:,} records")
        self._print(f"\n[bold]Total: {total_records:,} records[/bold]" if RICH_AVAILABLE else f"\nTotal: {total_records:,} records")

        # Confirm by typing database name
        self._print(f"\nTo confirm, type the database name '{self.current_backend.short_name}': ", style="yellow")

        try:
            confirm = input().strip()
        except (EOFError, KeyboardInterrupt):
            self._print("\nCancelled.")
            return

        if confirm != self.current_backend.short_name:
            self._print("Cancelled (name did not match).")
            return

        # Execute wipe
        try:
            results = self.current_backend.wipe_all()
            total_deleted = sum(results.values())

            self._print(f"\n[green]Wiped {total_deleted:,} records from {len(results)} tables/collections.[/green]"
                       if RICH_AVAILABLE else f"\nWiped {total_deleted:,} records from {len(results)} tables/collections.")

            for name, count in results.items():
                if count > 0:
                    self._print(f"  - {name}: {count:,} deleted")

        except Exception as e:
            self._print(f"Error: {e}", "red")

    def do_exit(self, arg: str):
        """Exit the database explorer."""
        if self.current_backend:
            self.current_backend.disconnect()
        self._save_history()
        self._print("\nGoodbye!")
        return True

    def do_quit(self, arg: str):
        """Exit the database explorer."""
        return self.do_exit(arg)

    def do_EOF(self, arg: str):
        """Handle Ctrl+D."""
        print()  # Newline for clean exit
        return self.do_exit(arg)

    def default(self, line: str):
        """Handle unknown commands."""
        self._print(f"Unknown command: {line}", "yellow")
        self._print("Type 'help' for available commands.")

    def emptyline(self):
        """Do nothing on empty line."""
        pass

    def postcmd(self, stop, line):
        """Called after each command."""
        self._save_history()
        return stop


def run_explorer():
    """Run the interactive database explorer."""
    explorer = DatabaseExplorer()
    try:
        explorer.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if explorer.current_backend:
            explorer.current_backend.disconnect()


def run_single_command(
    database: str,
    command: str,
    args: List[str]
) -> int:
    """
    Run a single database command and exit.

    Args:
        database: Database identifier (mongo-datalake, mongo-warehouse, postgres)
        command: Command to execute
        args: Command arguments

    Returns:
        Exit code (0 for success)
    """
    console = Console() if RICH_AVAILABLE else None

    # Create and connect backend
    if database == "mongo-datalake":
        backend = MongoDBBackend("datalake")
    elif database == "mongo-warehouse":
        backend = MongoDBBackend("warehouse")
    elif database == "postgres":
        backend = PostgreSQLBackend()
    else:
        print(f"Unknown database: {database}")
        return 1

    if not backend.connect():
        return 1

    try:
        if command == "collections" or command == "tables":
            collections = backend.list_collections()
            for name, count in collections:
                print(f"{name}: {count:,}")

        elif command == "sample":
            if not args:
                print("Usage: --sample <collection> [n]")
                return 1
            name = args[0]
            n = int(args[1]) if len(args) > 1 else 5
            samples = backend.get_sample(name, n)
            for sample in samples:
                print(json.dumps(sample, indent=2, default=str))

        elif command == "count":
            if not args:
                print("Usage: --count <collection>")
                return 1
            count = backend.get_count(args[0])
            print(count)

        elif command == "schema":
            if not args:
                print("Usage: --schema <collection>")
                return 1
            schema = backend.get_schema(args[0])
            print(json.dumps(schema, indent=2))

        elif command == "query":
            if not isinstance(backend, PostgreSQLBackend):
                print("SQL queries only supported for PostgreSQL")
                return 1
            if not args:
                print("Usage: --query <SQL>")
                return 1
            columns, rows = backend.query(" ".join(args))
            print(",".join(columns))
            for row in rows:
                print(",".join(str(c) for c in row))

        elif command == "stats":
            stats = backend.get_stats()
            print(json.dumps(stats, indent=2))

        else:
            print(f"Unknown command: {command}")
            return 1

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1

    finally:
        backend.disconnect()

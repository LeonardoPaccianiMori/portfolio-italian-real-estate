"""
Text User Interface (TUI) for the Italian Real Estate Pipeline.

This module provides an interactive menu-based interface for running
pipeline stages, viewing status, and exploring databases. Uses the
rich library for terminal rendering.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import sys
import os
import csv
import json
import queue
import threading
import time
from pathlib import Path
from contextlib import nullcontext
from urllib.parse import quote
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from rich.console import Console
from rich.columns import Columns
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.style import Style

from ..config.settings import SYNTHETIC_DATA_CHUNK_SIZE, SYNTHETIC_DATA_BATCH_SIZE

AIRFLOW_SCRAPE_DAG_ID = "listing.website_datalake_population_DAG_webscraping"
AIRFLOW_STATE_FILENAME = "airflow_runs.json"


# =============================================================================
# KEYBOARD INPUT HANDLING
# =============================================================================

def get_key() -> str:
    """
    Read a single keypress from the terminal.

    Returns:
        str: The key pressed. Special keys return:
            - 'up', 'down', 'left', 'right' for arrow keys
            - 'enter' for Enter/Return
            - 'q' for quit
            - Single character for letter keys
    """
    if sys.platform == 'win32':
        import msvcrt
        key = msvcrt.getch()
        if key == b'\xe0':  # Special key prefix on Windows
            key = msvcrt.getch()
            if key == b'H':
                return 'up'
            elif key == b'P':
                return 'down'
            elif key == b'K':
                return 'left'
            elif key == b'M':
                return 'right'
        elif key == b'\r':
            return 'enter'
        return key.decode('utf-8', errors='ignore').lower()
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # Escape sequence
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'up'
                    elif ch3 == 'B':
                        return 'down'
                    elif ch3 == 'C':
                        return 'right'
                    elif ch3 == 'D':
                        return 'left'
                return 'escape'
            elif ch == '\r' or ch == '\n':
                return 'enter'
            elif ch == '\x03':  # Ctrl+C
                raise KeyboardInterrupt
            return ch.lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# =============================================================================
# MENU ITEM DEFINITIONS
# =============================================================================

class MenuItemType(Enum):
    """Types of menu items."""
    STAGE = "stage"
    ACTION = "action"
    SEPARATOR = "separator"


@dataclass
class MenuItem:
    """A single menu item."""
    key: str  # Hotkey (e.g., "1", "a", "q")
    label: str  # Display label
    description: str  # Description shown next to label
    item_type: MenuItemType = MenuItemType.ACTION
    stage_name: Optional[str] = None  # For STAGE items, the stage identifier


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    stage_name: str
    display_name: str
    options: Dict[str, List[str]] = field(default_factory=dict)
    selected_options: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# MAIN TUI CLASS
# =============================================================================

class PipelineTUI:
    """
    Interactive menu-based TUI for the Italian Real Estate Pipeline.

    Provides arrow-key navigation through pipeline stages and actions,
    with configuration submenus for each stage.
    """

    MAIN_MENU_ITEMS = [
        MenuItem("1", "SCRAPE", "Web scraping to MongoDB Datalake", MenuItemType.STAGE, "scrape"),
        MenuItem("2", "ETL", "MongoDB Datalake → MongoDB Warehouse", MenuItemType.STAGE, "etl"),
        MenuItem("3", "MIGRATE", "MongoDB Warehouse → PostgreSQL", MenuItemType.STAGE, "migrate"),
        MenuItem("4", "DATA EXPORT", "Export real or synthetic data", MenuItemType.STAGE, "data_export"),
        MenuItem("5", "TRAIN", "Train ML model + create dashboard data file", MenuItemType.STAGE, "train"),
        MenuItem("", "", "", MenuItemType.SEPARATOR),
        MenuItem("a", "Run ALL", "Execute all pipeline stages", MenuItemType.ACTION, "all"),
        MenuItem("s", "STATUS", "Show pipeline status", MenuItemType.ACTION, "status"),
        MenuItem("d", "DATABASE", "Interactive database explorer", MenuItemType.ACTION, "db"),
        MenuItem("i", "INIT DBs", "Initialize database schemas", MenuItemType.ACTION, "init_db"),
        MenuItem("q", "QUIT", "Exit the application", MenuItemType.ACTION, "quit"),
    ]

    STAGE_CONFIGS = {
        "scrape": StageConfig(
            stage_name="scrape",
            display_name="SCRAPE - Web Scraping",
            options={
                "Listing type": ["All", "Sale", "Rent", "Auction"],
                "Provinces": ["All provinces"],
            },
            selected_options={"Listing type": 0, "Provinces": 0},
        ),
        "etl": StageConfig(
            stage_name="etl",
            display_name="ETL - Extract, Transform, Load",
            options={
                "Listing type": ["All", "Sale", "Rent", "Auction"],
                "Cleaning": ["Run cleaning", "Skip cleaning"],
            },
            selected_options={"Listing type": 0, "Cleaning": 0},
        ),
        "migrate": StageConfig(
            stage_name="migrate",
            display_name="MIGRATE - PostgreSQL Migration",
            options={
                "Batch size": ["10000 (default)", "5000", "1000"],
                "Translation": ["Run translation", "Skip translation"],
            },
            selected_options={"Batch size": 0, "Translation": 0},
        ),
        "data_export": StageConfig(
            stage_name="data_export",
            display_name="DATA EXPORT - Data Generation",
            options={
                "Data type": ["Real data", "Synthetic data"],
            },
            selected_options={"Data type": 0},
        ),
        "train": StageConfig(
            stage_name="train",
            display_name="TRAIN - ML Model Training",
            options={
                "Test size": ["30% (default)", "20%", "40%"],
            },
            selected_options={"Test size": 0},
        ),
    }

    def _build_console(self) -> Tuple[Console, bool]:
        stdout_isatty = sys.stdout.isatty()
        force_terminal = None
        force_interactive = None
        if stdout_isatty and os.path.exists("/.dockerenv"):
            force_terminal = True
            force_interactive = True

        force_env = os.getenv("TUI_FORCE_TERMINAL")
        if force_env is not None:
            env_value = force_env.strip().lower()
            if env_value in ("1", "true", "yes"):
                force_terminal = True
                force_interactive = True
            elif env_value in ("0", "false", "no"):
                force_terminal = False
                force_interactive = False

        def _safe_width(value: int) -> Optional[int]:
            if value <= 0:
                return None
            return max(value - 1, 20)

        width = height = None
        size_source = "fallback"
        env_cols = os.getenv("COLUMNS")
        env_lines = os.getenv("LINES")
        if env_cols or env_lines:
            try:
                if env_cols:
                    parsed_cols = int(env_cols)
                    if parsed_cols >= 41:
                        width = _safe_width(parsed_cols)
                if env_lines:
                    parsed_lines = int(env_lines)
                    if parsed_lines >= 10:
                        height = parsed_lines
                if width is not None or height is not None:
                    size_source = "env"
            except ValueError:
                pass
        if stdout_isatty:
            try:
                size = os.get_terminal_size(sys.stdout.fileno())
                if size.columns >= 41:
                    width = _safe_width(size.columns)
                if size.lines >= 10:
                    height = size.lines
                if width is not None and height is not None:
                    size_source = "stdout"
            except OSError:
                pass
        if size_source == "fallback":
            try:
                with open("/dev/tty") as tty_handle:
                    size = os.get_terminal_size(tty_handle.fileno())
                    if size.columns >= 41:
                        width = _safe_width(size.columns)
                    if size.lines >= 10:
                        height = size.lines
                    if width is not None and height is not None:
                        size_source = "tty"
            except OSError:
                pass

        console = Console(
            width=width,
            height=height,
            force_terminal=force_terminal,
            force_interactive=force_interactive,
        )
        panel_expand = width is not None or size_source in ("stdout", "tty")
        return console, panel_expand

    def __init__(self):
        """Initialize the TUI."""
        self.console, self._panel_expand = self._build_console()
        self.selected_index = 0
        self.current_menu = "main"
        self.current_stage_config: Optional[StageConfig] = None
        self.config_selected_index = 0
        self._airflow_state = self._load_airflow_state()
        self._provinces_cache = self._get_provinces_list()
        saved_provinces = self._airflow_state.get("scrape", {}).get("selected_provinces")
        if saved_provinces:
            self.scrape_selected_provinces = [
                province for province in saved_provinces if province in self._provinces_cache
            ]
        else:
            self.scrape_selected_provinces = list(self._provinces_cache)
        self._airflow_last_run_id = self._airflow_state.get("scrape", {}).get("last_run_id")

    def run(self) -> int:
        """
        Main TUI loop.

        Returns:
            int: Exit code (0 for success).
        """
        try:
            while True:
                self._clear_screen()

                if self.current_menu == "main":
                    self._draw_main_menu()
                    action = self._handle_main_menu_input()

                    if action == "quit":
                        return 0
                    elif action == "status":
                        self._run_status()
                    elif action == "db":
                        self._run_db_explorer()
                    elif action == "init_db":
                        self._run_init_databases()
                    elif action == "all":
                        self._run_all_stages()
                    elif action in self.STAGE_CONFIGS:
                        self.current_menu = "config"
                        self.current_stage_config = StageConfig(
                            stage_name=self.STAGE_CONFIGS[action].stage_name,
                            display_name=self.STAGE_CONFIGS[action].display_name,
                            options=dict(self.STAGE_CONFIGS[action].options),
                            selected_options=dict(self.STAGE_CONFIGS[action].selected_options),
                        )
                        self.config_selected_index = 0

                elif self.current_menu == "config":
                    self._draw_config_menu()
                    action = self._handle_config_menu_input()

                    if action == "back":
                        self.current_menu = "main"
                        self.current_stage_config = None
                    elif action == "run":
                        self._run_stage(self.current_stage_config)
                        self.current_menu = "main"
                        self.current_stage_config = None

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Interrupted by user[/yellow]")
            return 130

    def _clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if sys.platform == 'win32' else 'clear')

    def _get_airflow_state_path(self) -> Path:
        return Path.home() / ".config" / "italian-real-estate" / AIRFLOW_STATE_FILENAME

    def _load_airflow_state(self) -> Dict[str, Any]:
        path = self._get_airflow_state_path()
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_airflow_state(self, state: Dict[str, Any]) -> None:
        path = self._get_airflow_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        try:
            path.chmod(0o600)
        except OSError:
            pass

    def _persist_scrape_selection(self) -> None:
        scrape_state = self._airflow_state.setdefault("scrape", {})
        scrape_state["selected_provinces"] = list(self.scrape_selected_provinces)
        self._save_airflow_state(self._airflow_state)

    def _persist_scrape_run(
        self,
        run_id: str,
        provinces: List[str],
        listing_types: List[str],
        use_selenium: Optional[bool] = None,
    ) -> None:
        scrape_state = self._airflow_state.setdefault("scrape", {})
        scrape_state["last_run_id"] = run_id
        scrape_state["selected_provinces"] = list(provinces)
        scrape_state["listing_types"] = list(listing_types)
        if use_selenium is not None:
            scrape_state["use_selenium"] = use_selenium
        self._airflow_last_run_id = run_id
        self._save_airflow_state(self._airflow_state)

    def _get_provinces_list(self) -> List[str]:
        from italian_real_estate.config.settings import PROVINCES_CSV_PATH

        try:
            with PROVINCES_CSV_PATH.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return [row["Province"] for row in reader if row.get("Province")]
        except FileNotFoundError:
            return []

    def _format_province_selection(self) -> str:
        total = len(self._provinces_cache)
        if total == 0:
            return "No provinces found"
        selected = len(self.scrape_selected_provinces)
        if selected >= total:
            return f"All provinces ({total})"
        return f"Selected {selected}/{total}"

    def _configure_provinces(self) -> None:
        provinces = list(self._provinces_cache)
        if not provinces:
            self.console.print("[red]No provinces available to select.[/red]")
            self.console.print("\n[dim]Press any key to continue...[/dim]")
            get_key()
            return

        selected = set(self.scrape_selected_provinces or provinces)
        index = 0
        offset = 0
        page_size = 20

        while True:
            self._clear_screen()
            title = Text("Select Provinces", style="bold cyan")
            self.console.print(Panel(Align.center(title), border_style="cyan"))
            self.console.print()

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Province", width=30)
            table.add_column("Status", width=10)

            if index < offset:
                offset = index
            if index >= offset + page_size:
                offset = index - page_size + 1

            visible = provinces[offset:offset + page_size]
            for i, province in enumerate(visible, start=offset):
                is_selected = i == index
                is_checked = province in selected
                checkbox = "[x]" if is_checked else "[ ]"
                row_style = "bold reverse" if is_selected else ""
                table.add_row(
                    Text(f"{checkbox} {province}", style=row_style),
                    Text("selected" if is_checked else "", style=row_style),
                )

            panel = Panel(
                table,
                title="[bold]Provinces[/bold]",
                subtitle=(
                    "[dim]↑↓ Navigate • Space Toggle • A Select All • N Deselect All "
                    "• S Save • B Back[/dim]"
                ),
                border_style="blue",
            )
            self.console.print(panel)

            key = get_key()
            if key == "up":
                index = max(0, index - 1)
            elif key == "down":
                index = min(len(provinces) - 1, index + 1)
            elif key in (" ", "enter"):
                province = provinces[index]
                if province in selected:
                    selected.remove(province)
                else:
                    selected.add(province)
            elif key == "a":
                selected = set(provinces)
            elif key == "n":
                selected = set()
            elif key == "s":
                if not selected:
                    self.console.print("[red]Select at least one province.[/red]")
                    time.sleep(1)
                    continue
                self.scrape_selected_provinces = [p for p in provinces if p in selected]
                self._persist_scrape_selection()
                return
            elif key in ("b", "q", "escape"):
                return

    def _get_airflow_api_base_url(self) -> str:
        base = os.getenv("AIRFLOW_API_BASE_URL", "http://airflow-webserver:8080/api/v1")
        return base.rstrip("/")

    def _get_airflow_api_auth(self) -> Tuple[str, str]:
        return (
            os.getenv("AIRFLOW_API_USER", "admin"),
            os.getenv("AIRFLOW_API_PASSWORD", "admin"),
        )

    def _airflow_request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self._get_airflow_api_base_url()}{path}"
        auth = self._get_airflow_api_auth()
        try:
            response = requests.request(method, url, auth=auth, timeout=10, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise RuntimeError(f"Airflow API request failed: {exc}") from exc

    def _airflow_trigger_dag_run(self, dag_id: str, conf: Dict[str, Any]) -> str:
        payload = {"conf": conf} if conf else {}
        response = self._airflow_request("POST", f"/dags/{dag_id}/dagRuns", json=payload)
        data = response.json()
        return data.get("dag_run_id") or data.get("run_id")

    def _airflow_get_running_dag_run(self, dag_id: str) -> Optional[Dict[str, Any]]:
        response = self._airflow_request(
            "GET",
            f"/dags/{dag_id}/dagRuns",
            params={"state": "running", "limit": 100},
        )
        runs = response.json().get("dag_runs", [])
        if not runs:
            return None
        runs.sort(
            key=lambda run: run.get("start_date") or run.get("logical_date") or "",
            reverse=True,
        )
        return runs[0]

    def _airflow_get_dag_run(self, dag_id: str, run_id: str) -> Dict[str, Any]:
        response = self._airflow_request("GET", f"/dags/{dag_id}/dagRuns/{quote(run_id)}")
        return response.json()

    def _airflow_get_dag(self, dag_id: str) -> Dict[str, Any]:
        response = self._airflow_request("GET", f"/dags/{dag_id}")
        return response.json()

    def _airflow_set_dag_paused(self, dag_id: str, paused: bool) -> None:
        self._airflow_request("PATCH", f"/dags/{dag_id}", json={"is_paused": paused})

    def _airflow_set_dag_run_state(self, dag_id: str, run_id: str, state: str) -> None:
        self._airflow_request(
            "PATCH",
            f"/dags/{dag_id}/dagRuns/{quote(run_id)}",
            json={"state": state},
        )

    def _airflow_get_task_instances(self, dag_id: str, run_id: str) -> List[Dict[str, Any]]:
        task_instances: List[Dict[str, Any]] = []
        offset = 0
        limit = 1000

        while True:
            response = self._airflow_request(
                "GET",
                f"/dags/{dag_id}/dagRuns/{quote(run_id)}/taskInstances",
                params={"limit": limit, "offset": offset},
            )
            payload = response.json()
            task_instances.extend(payload.get("task_instances", []))

            total = payload.get("total_entries")
            if total is None:
                if len(payload.get("task_instances", [])) < limit:
                    break
            else:
                if len(task_instances) >= total:
                    break

            offset += limit

        return task_instances

    def _format_task_state(self, state: Optional[str]) -> Any:
        normalized = (state or "none").lower()
        if normalized == "running":
            return Spinner("dots", text=Text("RUNNING", style="cyan"), style="cyan")
        if normalized in ("success",):
            return Text("SUCCESS", style="green")
        if normalized in ("failed", "upstream_failed"):
            return Text("FAILED", style="red")
        if normalized in ("skipped",):
            return Text("SKIPPED", style="yellow")
        if normalized in ("queued", "scheduled"):
            return Text("QUEUED", style="yellow")
        if normalized in ("none", "null"):
            return Text("PENDING", style="dim")
        return Text(normalized.upper(), style="dim")

    def _resolve_scrape_provinces(self, selected_provinces: List[str]) -> List[str]:
        selected_set = set(selected_provinces)
        if len(selected_set) == len(self._provinces_cache):
            return list(self._provinces_cache)
        return [province for province in self._provinces_cache if province in selected_set]

    def _get_scrape_max_rows(self) -> Optional[int]:
        height = self.console.size.height
        if not height or height <= 0:
            try:
                height = os.get_terminal_size(sys.stdout.fileno()).lines
            except OSError:
                height = 24
        reserved_lines = 9
        return max(1, height - reserved_lines)

    def _build_scrape_status_panel(
        self,
        run_id: str,
        task_states: Dict[str, Dict[str, Optional[str]]],
        selected_provinces: List[str],
        selected_listing_types: List[str],
        run_state: Optional[str],
        dag_paused: Optional[bool] = None,
        notice: Optional[str] = None,
        refresh_seconds: int = 10,
        scroll_offset: int = 0,
        max_rows: Optional[int] = None,
    ) -> Panel:
        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
            expand=self._panel_expand,
        )
        table.add_column("Province", style="bold", width=20)
        state_width = 12
        table.add_column("Rent", justify="center", width=state_width)
        table.add_column("Auction", justify="center", width=state_width)
        table.add_column("Sale", justify="center", width=state_width)

        selected_set = set(selected_provinces)
        listing_set = set(selected_listing_types)

        display_provinces = self._resolve_scrape_provinces(selected_provinces)
        total_rows = len(display_provinces)
        if max_rows is not None and total_rows > max_rows:
            max_offset = max(0, total_rows - max_rows)
            scroll_offset = max(0, min(scroll_offset, max_offset))
            visible_provinces = display_provinces[scroll_offset:scroll_offset + max_rows]
        else:
            visible_provinces = display_provinces

        for province in visible_provinces:
            if province not in selected_set:
                rent_state = self._format_task_state("skipped")
                auction_state = self._format_task_state("skipped")
                sale_state = self._format_task_state("skipped")
            else:
                province_states = task_states.get(province, {})
                rent_state = (
                    self._format_task_state("skipped")
                    if "rent" not in listing_set
                    else self._format_task_state(province_states.get("rent"))
                )
                auction_state = (
                    self._format_task_state("skipped")
                    if "auction" not in listing_set
                    else self._format_task_state(province_states.get("auction"))
                )
                sale_state = (
                    self._format_task_state("skipped")
                    if "sale" not in listing_set
                    else self._format_task_state(province_states.get("sale"))
                )

            table.add_row(province, rent_state, auction_state, sale_state)

        if dag_paused is None:
            paused_label = "unknown"
        else:
            paused_label = "paused" if dag_paused else "active"
        scroll_hint = ""
        if max_rows is not None and total_rows > max_rows:
            start = min(scroll_offset + 1, total_rows)
            end = min(scroll_offset + max_rows, total_rows)
            scroll_hint = f" • Rows {start}-{end}/{total_rows} • ↑/↓ scroll • U/D page"
        subtitle = (
            f"[dim]Run: {run_id} • State: {run_state or 'unknown'} "
            f"• DAG: {paused_label} • Refresh: {refresh_seconds}s{scroll_hint} "
            "• P pause/unpause • R retrigger • X stop • B/Q back[/dim]"
        )
        if notice:
            subtitle = f"{subtitle}\n[dim]{notice}[/dim]"
        subtitle_text = Text.from_markup(subtitle, overflow="ellipsis")
        subtitle_text.no_wrap = True
        return Panel(
            table,
            title="[bold]Scrape Status (Airflow)[/bold]",
            subtitle=subtitle_text,
            border_style="blue",
            expand=self._panel_expand,
        )

    def _monitor_airflow_scrape(
        self,
        run_id: str,
        selected_provinces: List[str],
        selected_listing_types: List[str],
        use_selenium: Optional[bool] = None,
        refresh_seconds: int = 10,
    ) -> None:
        stop_event = threading.Event()
        key_queue: "queue.Queue[str]" = queue.Queue()
        use_live = self.console.is_terminal and self.console.is_interactive
        live_env = os.getenv("TUI_LIVE_REFRESH")
        if live_env is not None and live_env.strip().lower() in ("0", "false", "no"):
            use_live = False
        all_provinces = self._resolve_scrape_provinces(selected_provinces)
        scroll_offset = 0

        def watch_keys() -> None:
            while not stop_event.is_set():
                try:
                    key = get_key()
                except KeyboardInterrupt:
                    key = "q"
                key_queue.put(key)
                if key in ("b", "q"):
                    stop_event.set()
                    return

        watcher = threading.Thread(target=watch_keys, daemon=True)
        watcher.start()

        def fetch_task_states() -> Tuple[Dict[str, Dict[str, Optional[str]]], Optional[str], Optional[bool]]:
            task_instances = self._airflow_get_task_instances(AIRFLOW_SCRAPE_DAG_ID, run_id)
            run_state = None
            dag_paused = None
            try:
                run_state = self._airflow_get_dag_run(AIRFLOW_SCRAPE_DAG_ID, run_id).get("state")
            except RuntimeError:
                run_state = None
            try:
                dag_paused = bool(self._airflow_get_dag(AIRFLOW_SCRAPE_DAG_ID).get("is_paused"))
            except RuntimeError:
                dag_paused = None

            states: Dict[str, Dict[str, Optional[str]]] = {}
            for instance in task_instances:
                task_id = instance.get("task_id", "")
                state = instance.get("state")
                if task_id.endswith("_rents"):
                    province = task_id[:-6]
                    states.setdefault(province, {})["rent"] = state
                elif task_id.endswith("_auctions"):
                    province = task_id[:-9]
                    states.setdefault(province, {})["auction"] = state
                elif task_id.endswith("_sales"):
                    province = task_id[:-6]
                    states.setdefault(province, {})["sale"] = state
            return states, run_state, dag_paused

        def build_conf() -> Dict[str, Any]:
            if use_selenium is None:
                use_selenium_env = os.getenv("SCRAPING_USE_SELENIUM")
                use_selenium_value = True if use_selenium_env is None else use_selenium_env.lower() in ("1", "true", "yes")
            else:
                use_selenium_value = use_selenium

            return {
                "provinces": list(selected_provinces),
                "listing_types": list(selected_listing_types),
                "use_selenium": use_selenium_value,
            }

        if not use_live:
            self.console.print("[dim]Live refresh disabled (non-interactive output).[/dim]\n")
            self._clear_screen()
            self.console.print(
                self._build_scrape_status_panel(
                    run_id,
                    {},
                    selected_provinces,
                    selected_listing_types,
                    None,
                    refresh_seconds=refresh_seconds,
                    scroll_offset=scroll_offset,
                    max_rows=self._get_scrape_max_rows(),
                )
            )

        live_context = (
            Live(
                self._build_scrape_status_panel(
                    run_id,
                    {},
                    selected_provinces,
                    selected_listing_types,
                    None,
                    refresh_seconds=refresh_seconds,
                    scroll_offset=scroll_offset,
                    max_rows=self._get_scrape_max_rows(),
                ),
                console=self.console,
                refresh_per_second=4,
            )
            if use_live
            else nullcontext()
        )

        with live_context as live:
            notice: Optional[str] = None
            notice_until: Optional[float] = None
            pending_action: Optional[str] = None
            pending_until: Optional[float] = None
            last_states: Dict[str, Dict[str, Optional[str]]] = {}
            last_run_state: Optional[str] = None
            last_dag_paused: Optional[bool] = None

            next_refresh = 0.0
            while not stop_event.is_set():
                while True:
                    try:
                        key = key_queue.get_nowait()
                    except queue.Empty:
                        break

                    now = time.time()
                    if pending_until and now > pending_until:
                        pending_action = None
                        pending_until = None

                    if key in ("b", "q"):
                        stop_event.set()
                        break
                    max_rows = self._get_scrape_max_rows()
                    total_rows = len(all_provinces)
                    max_offset = 0
                    if max_rows is not None and total_rows > max_rows:
                        max_offset = total_rows - max_rows

                    if key in ("up", "k"):
                        scroll_offset = max(0, scroll_offset - 1)
                    if key in ("down", "j"):
                        scroll_offset = min(max_offset, scroll_offset + 1)
                    if key == "u" and max_rows:
                        scroll_offset = max(0, scroll_offset - max_rows)
                    if key == "d" and max_rows:
                        scroll_offset = min(max_offset, scroll_offset + max_rows)

                    if key == "p":
                        try:
                            dag_info = self._airflow_get_dag(AIRFLOW_SCRAPE_DAG_ID)
                            is_paused = bool(dag_info.get("is_paused"))
                            target = not is_paused
                            self._airflow_set_dag_paused(AIRFLOW_SCRAPE_DAG_ID, target)
                            notice = "DAG paused." if target else "DAG unpaused."
                            notice_until = now + 5
                        except RuntimeError as exc:
                            notice = f"Pause/unpause failed: {exc}"
                            notice_until = now + 5

                    if key == "x":
                        if pending_action == "stop" and pending_until and now <= pending_until:
                            try:
                                self._airflow_set_dag_run_state(
                                    AIRFLOW_SCRAPE_DAG_ID,
                                    run_id,
                                    "failed",
                                )
                                notice = "Run marked FAILED."
                            except RuntimeError as exc:
                                notice = f"Stop failed: {exc}"
                            notice_until = now + 5
                            pending_action = None
                            pending_until = None
                        else:
                            pending_action = "stop"
                            pending_until = now + 5
                            notice = "Press X again to stop the run."
                            notice_until = now + 5

                    if key == "r":
                        if pending_action == "retrigger" and pending_until and now <= pending_until:
                            try:
                                conf = build_conf()
                                new_run_id = self._airflow_trigger_dag_run(
                                    AIRFLOW_SCRAPE_DAG_ID,
                                    conf,
                                )
                                if new_run_id:
                                    run_id = new_run_id
                                    self._persist_scrape_run(
                                        run_id,
                                        selected_provinces,
                                        selected_listing_types,
                                        use_selenium=conf.get("use_selenium"),
                                    )
                                    notice = f"Triggered new run {run_id}."
                                else:
                                    notice = "Airflow did not return a run id."
                            except RuntimeError as exc:
                                notice = f"Retrigger failed: {exc}"
                            notice_until = now + 5
                            pending_action = None
                            pending_until = None
                        else:
                            pending_action = "retrigger"
                            pending_until = now + 5
                            notice = "Press R again to retrigger the run."
                            notice_until = now + 5

                now = time.time()
                if notice_until and now > notice_until:
                    notice = None
                    notice_until = None

                if now >= next_refresh:
                    try:
                        task_states, run_state, dag_paused = fetch_task_states()
                        last_states = task_states
                        last_run_state = run_state
                        last_dag_paused = dag_paused
                    except RuntimeError as exc:
                        task_states = last_states
                        run_state = last_run_state
                        dag_paused = last_dag_paused
                        notice = f"Airflow refresh failed: {exc}"
                        notice_until = time.time() + 5
                    panel = self._build_scrape_status_panel(
                        run_id,
                        task_states,
                        selected_provinces,
                        selected_listing_types,
                        run_state,
                        dag_paused,
                        notice,
                        refresh_seconds=refresh_seconds,
                        scroll_offset=scroll_offset,
                        max_rows=self._get_scrape_max_rows(),
                    )
                    if use_live and live is not None:
                        live.update(panel)
                    else:
                        self._clear_screen()
                        self.console.print(panel)
                    next_refresh = now + refresh_seconds

                stop_event.wait(0.1)

    def _draw_main_menu(self) -> None:
        """Render the main menu."""
        # Title
        title = Text("Italian Real Estate Project TUI", style="bold cyan")
        self.console.print(Panel(Align.center(title), border_style="cyan"))
        self.console.print()

        # Menu items
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold yellow", width=5)
        table.add_column("Label", width=15)
        table.add_column("Description", style="dim")

        selectable_items = [item for item in self.MAIN_MENU_ITEMS
                          if item.item_type != MenuItemType.SEPARATOR]

        selectable_index = 0
        for item in self.MAIN_MENU_ITEMS:
            if item.item_type == MenuItemType.SEPARATOR:
                table.add_row("", "", "")
                continue

            # Check if this selectable item is the currently selected one
            is_selected = selectable_index == self.selected_index
            selectable_index += 1

            key_text = f"\\[{item.key}]" if item.key else ""
            label_style = "bold reverse" if is_selected else "bold"
            desc_style = "reverse" if is_selected else "dim"

            prefix = "► " if is_selected else "  "

            table.add_row(
                key_text,
                Text(f"{prefix}{item.label}", style=label_style),
                Text(item.description, style=desc_style),
            )

        panel = Panel(
            table,
            title="[bold]Main Menu[/bold]",
            subtitle="[dim]↑↓ Navigate • Enter Select • 1-5/A/S/D/I/Q Hotkeys[/dim]",
            border_style="blue",
        )
        self.console.print(panel)

    def _handle_main_menu_input(self) -> Optional[str]:
        """
        Handle input for the main menu.

        Returns:
            Optional[str]: Action to perform, or None to continue.
        """
        selectable_items = [item for item in self.MAIN_MENU_ITEMS
                          if item.item_type != MenuItemType.SEPARATOR]
        max_index = len(selectable_items) - 1

        key = get_key()

        if key == 'up':
            self.selected_index = max(0, self.selected_index - 1)
        elif key == 'down':
            self.selected_index = min(max_index, self.selected_index + 1)
        elif key == 'enter':
            item = selectable_items[self.selected_index]
            if item.stage_name:
                return item.stage_name
        elif key == 'q':
            return "quit"
        else:
            # Check for hotkey
            for item in selectable_items:
                if item.key.lower() == key:
                    if item.stage_name:
                        return item.stage_name

        return None

    def _draw_config_menu(self) -> None:
        """Render the stage configuration menu."""
        if not self.current_stage_config:
            return

        config = self.current_stage_config

        # Title
        title = Text(config.display_name, style="bold cyan")
        self.console.print(Panel(Align.center(title), border_style="cyan"))
        self.console.print()

        # Options
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Option", width=20)
        table.add_column("Value", width=40)

        option_keys = list(config.options.keys())
        for i, (option_name, values) in enumerate(config.options.items()):
            is_selected = i == self.config_selected_index
            if option_name == "Provinces":
                selected_value = self._format_province_selection()
            else:
                selected_value = values[config.selected_options.get(option_name, 0)]

            option_style = "bold reverse" if is_selected else "bold"
            value_style = "reverse" if is_selected else ""

            prefix = "► " if is_selected else "  "

            # Show arrow hints when row is selected
            if is_selected and option_name != "Provinces":
                value_text = f"◄ {selected_value} ►"
            elif is_selected and option_name == "Provinces":
                value_text = f"  {selected_value} (enter to edit)  "
            else:
                value_text = f"  {selected_value}  "

            table.add_row(
                Text(f"{prefix}{option_name}:", style=option_style),
                Text(value_text, style=value_style),
            )

        # For data_export stage, show sample counts info when Synthetic is selected
        if config.stage_name == "data_export":
            data_type_opt = config.selected_options.get("Data type", 0)
            if data_type_opt == 1:  # Synthetic data selected (index 1)
                table.add_row("", "")
                table.add_row(
                    Text("  Sample counts:", style="dim"),
                    Text("(auto-calculated from database)", style="dim italic"),
                )
                # Try to get actual counts (use cached value to avoid repeated DB queries)
                if not hasattr(self, '_cached_listing_counts'):
                    try:
                        # Suppress TensorFlow/NumExpr noise during import
                        import os
                        import sys
                        import logging
                        old_stderr = sys.stderr
                        sys.stderr = open(os.devnull, 'w')
                        old_level = logging.getLogger().level
                        logging.getLogger().setLevel(logging.CRITICAL)
                        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
                        try:
                            from italian_real_estate.synthetic_data.real_data_exporter import get_rounded_listing_counts
                            self._cached_listing_counts = get_rounded_listing_counts()
                        finally:
                            sys.stderr.close()
                            sys.stderr = old_stderr
                            logging.getLogger().setLevel(old_level)
                    except Exception:
                        self._cached_listing_counts = None

                if self._cached_listing_counts:
                    counts = self._cached_listing_counts
                    table.add_row(
                        Text("    Rent:", style="dim"),
                        Text(f"{counts['rent']:,}", style="dim"),
                    )
                    table.add_row(
                        Text("    Auction:", style="dim"),
                        Text(f"{counts['auction']:,}", style="dim"),
                    )
                    table.add_row(
                        Text("    Sale:", style="dim"),
                        Text(f"{counts['sale']:,}", style="dim"),
                    )
                else:
                    table.add_row(
                        Text("", style="dim"),
                        Text("(connect to database to see counts)", style="dim italic"),
                    )

        # Action buttons
        table.add_row("", "")
        run_selected = self.config_selected_index == len(option_keys)
        back_selected = self.config_selected_index == len(option_keys) + 1

        run_style = "bold green reverse" if run_selected else "bold green"
        back_style = "bold yellow reverse" if back_selected else "bold yellow"

        run_prefix = "► " if run_selected else "  "
        back_prefix = "► " if back_selected else "  "

        table.add_row(
            Text(f"{run_prefix}\\[R] RUN STAGE", style=run_style),
            "",
        )
        table.add_row(
            Text(f"{back_prefix}\\[B] BACK", style=back_style),
            "",
        )

        panel = Panel(
            table,
            title="[bold]Configuration[/bold]",
            subtitle="[dim]↑↓ Navigate • ←→ Change Value • Enter Select • B Back[/dim]",
            border_style="blue",
        )
        self.console.print(panel)

    def _handle_config_menu_input(self) -> Optional[str]:
        """
        Handle input for the configuration menu.

        Returns:
            Optional[str]: 'run', 'back', or None to continue.
        """
        if not self.current_stage_config:
            return "back"

        config = self.current_stage_config
        option_keys = list(config.options.keys())
        num_options = len(option_keys)
        max_index = num_options + 1  # +2 for Run and Back buttons, -1 for 0-index

        key = get_key()

        if key == 'up':
            self.config_selected_index = max(0, self.config_selected_index - 1)
        elif key == 'down':
            self.config_selected_index = min(max_index, self.config_selected_index + 1)
        elif key in ('left', 'right'):
            # Change option value
            if self.config_selected_index < num_options:
                option_name = option_keys[self.config_selected_index]
                if option_name != "Provinces":
                    values = config.options[option_name]
                    current = config.selected_options.get(option_name, 0)
                    if key == 'right':
                        config.selected_options[option_name] = (current + 1) % len(values)
                    else:
                        config.selected_options[option_name] = (current - 1) % len(values)
        elif key == 'enter':
            if self.config_selected_index < num_options:
                option_name = option_keys[self.config_selected_index]
                if option_name == "Provinces":
                    self._configure_provinces()
                    return None
            if self.config_selected_index == num_options:  # Run button
                return "run"
            elif self.config_selected_index == num_options + 1:  # Back button
                return "back"
        elif key == 'r':
            return "run"
        elif key == 'b':
            return "back"

        return None

    def _run_stage(self, config: StageConfig) -> None:
        """
        Execute a pipeline stage with the given configuration.

        Args:
            config: Stage configuration with selected options.
        """
        self._clear_screen()
        self.console.print(f"[bold cyan]Running {config.display_name}...[/bold cyan]\n")

        # Build arguments based on configuration
        args = self._build_stage_args(config)

        # Import and run the stage
        try:
            from italian_real_estate.config.logging_config import setup_logging, get_logger
            import logging
            setup_logging(level=logging.INFO)
            logger = get_logger(__name__)

            if config.stage_name == "scrape":
                self._execute_scrape(args, logger)
            elif config.stage_name == "etl":
                self._execute_etl(args, logger)
            elif config.stage_name == "migrate":
                self._execute_migrate(args, logger)
            elif config.stage_name == "data_export":
                self._execute_data_export(args, logger)
            elif config.stage_name == "train":
                self._execute_train(args, logger)

            self.console.print("\n[bold green]Stage completed![/bold green]")
        except Exception as e:
            self.console.print(f"\n[bold red]Error: {e}[/bold red]")

        self.console.print("\n[dim]Press any key to continue...[/dim]")
        get_key()

    def _build_stage_args(self, config: StageConfig) -> Dict[str, Any]:
        """Build stage arguments from configuration."""
        args = {}

        if config.stage_name == "scrape":
            args["provinces"] = list(self.scrape_selected_provinces)

            listing_opt = config.selected_options.get("Listing type", 0)
            listing_map = {0: "all", 1: "sale", 2: "rent", 3: "auction"}
            args["listing_type"] = listing_map.get(listing_opt, "all")

        elif config.stage_name == "etl":
            listing_opt = config.selected_options.get("Listing type", 0)
            args["all"] = listing_opt == 0
            if listing_opt > 0:
                listing_map = {1: "sale", 2: "rent", 3: "auction"}
                args["listing_type"] = listing_map.get(listing_opt)

            cleaning_opt = config.selected_options.get("Cleaning", 0)
            args["skip_cleaning"] = cleaning_opt == 1

        elif config.stage_name == "migrate":
            batch_opt = config.selected_options.get("Batch size", 0)
            batch_map = {0: 10000, 1: 5000, 2: 1000}
            args["batch_size"] = batch_map.get(batch_opt, 10000)

            trans_opt = config.selected_options.get("Translation", 0)
            args["skip_translation"] = trans_opt == 1

        elif config.stage_name == "data_export":
            data_type_opt = config.selected_options.get("Data type", 0)
            args["data_type"] = "real" if data_type_opt == 0 else "synthetic"

        elif config.stage_name == "train":
            test_opt = config.selected_options.get("Test size", 0)
            test_map = {0: 0.3, 1: 0.2, 2: 0.4}
            args["test_size"] = test_map.get(test_opt, 0.3)

        return args

    def _execute_scrape(self, args: Dict[str, Any], logger) -> None:
        """Execute the scrape stage."""
        provinces = args.get("provinces") or list(self._provinces_cache)
        if not provinces:
            raise RuntimeError("No provinces selected for scraping.")

        listing_type = args.get("listing_type", "all")
        listing_types = ["rent", "auction", "sale"] if listing_type == "all" else [listing_type]

        use_selenium_env = os.getenv("SCRAPING_USE_SELENIUM")
        use_selenium = True if use_selenium_env is None else use_selenium_env.lower() in ("1", "true", "yes")

        conf = {
            "provinces": provinces,
            "listing_types": listing_types,
            "use_selenium": use_selenium,
        }

        running_run = self._airflow_get_running_dag_run(AIRFLOW_SCRAPE_DAG_ID)
        if running_run:
            run_id = running_run.get("dag_run_id") or running_run.get("run_id")
            from rich.prompt import Confirm
            attach = Confirm.ask(
                "A scrape run is already running. Attach to it?",
                default=True,
            )
            if attach:
                run_details = self._airflow_get_dag_run(AIRFLOW_SCRAPE_DAG_ID, run_id)
                run_conf = run_details.get("conf") or {}
                provinces = run_conf.get("provinces") or list(self._provinces_cache)
                listing_types = run_conf.get("listing_types") or ["rent", "auction", "sale"]
                use_selenium = run_conf.get("use_selenium", use_selenium)
            else:
                run_id = self._airflow_trigger_dag_run(AIRFLOW_SCRAPE_DAG_ID, conf)
        else:
            run_id = self._airflow_trigger_dag_run(AIRFLOW_SCRAPE_DAG_ID, conf)

        if not run_id:
            raise RuntimeError("Airflow did not return a run id.")

        self._persist_scrape_run(run_id, provinces, listing_types, use_selenium=use_selenium)
        self._monitor_airflow_scrape(
            run_id,
            provinces,
            listing_types,
            use_selenium=use_selenium,
            refresh_seconds=2,
        )

    def _execute_etl(self, args: Dict[str, Any], logger) -> None:
        """Execute the ETL stage."""
        from italian_real_estate.etl.warehouse_loader import migrate, get_warehouse_stats
        from italian_real_estate.etl.transformers import fix_empty_child_listings

        if args.get("all"):
            listing_types = ["rent", "auction", "sale"]
        else:
            listing_types = [args.get("listing_type", "sale")]

        for lt in listing_types:
            logger.info(f"Running ETL for {lt} listings...")
            migrate(lt)

        if not args.get("skip_cleaning"):
            logger.info("Running data cleaning...")
            fix_empty_child_listings()

        get_warehouse_stats()

    def _execute_migrate(self, args: Dict[str, Any], logger) -> None:
        """Execute the migration stage."""
        from italian_real_estate.migration import (
            setup_batch_processing,
            process_batch,
            load_fact_table_for_batch,
            finalize_migration,
        )

        batch_size = args.get("batch_size", 10000)

        logger.info(f"Setting up migration (batch_size={batch_size})...")
        batch_info = setup_batch_processing(batch_size=batch_size)
        total_batches = batch_info.get("num_batches", 0)

        for batch_num in range(total_batches):
            logger.info(f"Processing batch {batch_num + 1}/{total_batches}...")
            dimension_mappings = process_batch(batch_num)
            load_fact_table_for_batch(batch_num, dimension_mappings)

        if not args.get("skip_translation"):
            logger.info("Running translation...")
            # Ensure scripts/ is in path for run_migration import
            import sys
            from pathlib import Path
            scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
            if scripts_dir.exists() and str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from run_migration import run_translation
            run_translation(logger)

        finalize_migration()

    def _execute_data_export(self, args: Dict[str, Any], logger) -> None:
        """Execute the data export stage (real or synthetic)."""
        data_type = args.get("data_type", "synthetic")

        if data_type == "real":
            self._execute_real_data_export(logger)
        else:
            self._execute_synthetic_data_export(logger)

    def _execute_real_data_export(self, logger) -> None:
        """Export real data from PostgreSQL to Parquet."""
        from italian_real_estate.synthetic_data import export_real_data

        output_path = "data/real_data.parquet"
        logger.info(f"Exporting real data to {output_path}...")
        df = export_real_data(output_path)
        logger.info(f"Exported {len(df):,} records to {output_path}")

    def _execute_synthetic_data_export(self, logger) -> None:
        """Generate synthetic data and export to Parquet."""
        from italian_real_estate.synthetic_data import (
            extract_data_from_postgres,
            extract_features_from_postgres,
            merge_features_with_data,
            preprocess_data,
            split_by_listing_type,
            process_dataset,
            generate_synthetic_data_in_chunks,
            postprocess_synthetic_data,
            combine_synthetic_data,
            configure_gpu_memory_growth,
            get_gpu_info,
            get_rounded_listing_counts,
        )

        # Auto-calculate sample counts from database
        logger.info("Calculating sample counts from database...")
        counts = get_rounded_listing_counts()
        num_rent = counts["rent"]
        num_auction = counts["auction"]
        num_sale = counts["sale"]
        logger.info(f"Sample counts: rent={num_rent:,}, auction={num_auction:,}, sale={num_sale:,}")

        gpu_info = get_gpu_info()
        if gpu_info['gpu_available']:
            configure_gpu_memory_growth()

        logger.info("Extracting data from PostgreSQL...")
        raw_data = extract_data_from_postgres()
        features_df = extract_features_from_postgres()
        data_with_features = merge_features_with_data(raw_data, features_df)

        logger.info("Preprocessing data...")
        preprocessed_data, numerical_columns, categorical_columns = preprocess_data(data_with_features)

        rent_data, auction_data, sale_data = split_by_listing_type(preprocessed_data)

        # Remove listing_type from categorical_columns since it's dropped after split
        categorical_columns = [c for c in categorical_columns if c != 'listing_type']

        distance_columns = ["price", "latitude", "longitude", "surface"]

        # Process and generate for each type
        logger.info("Processing datasets for KNN...")
        rent_transformed, rent_cat_idx, _, rent_num_trans, rent_cats = process_dataset(
            rent_data, "rent", numerical_columns, categorical_columns)
        auction_transformed, auction_cat_idx, _, auction_num_trans, auction_cats = process_dataset(
            auction_data, "auction", numerical_columns, categorical_columns)
        sale_transformed, sale_cat_idx, _, sale_num_trans, sale_cats = process_dataset(
            sale_data, "sale", numerical_columns, categorical_columns)

        logger.info(f"Generating {num_rent:,} synthetic rent samples...")
        synthetic_rent = generate_synthetic_data_in_chunks(
            rent_data, rent_transformed, rent_transformed[:, :len(distance_columns)],
            num_rent, rent_cat_idx, rent_cats, rent_num_trans, numerical_columns, categorical_columns,
            chunk_size=SYNTHETIC_DATA_CHUNK_SIZE, batch_size=SYNTHETIC_DATA_BATCH_SIZE)

        logger.info(f"Generating {num_auction:,} synthetic auction samples...")
        synthetic_auction = generate_synthetic_data_in_chunks(
            auction_data, auction_transformed, auction_transformed[:, :len(distance_columns)],
            num_auction, auction_cat_idx, auction_cats, auction_num_trans, numerical_columns, categorical_columns,
            chunk_size=SYNTHETIC_DATA_CHUNK_SIZE, batch_size=SYNTHETIC_DATA_BATCH_SIZE)

        logger.info(f"Generating {num_sale:,} synthetic sale samples...")
        synthetic_sale = generate_synthetic_data_in_chunks(
            sale_data, sale_transformed, sale_transformed[:, :len(distance_columns)],
            num_sale, sale_cat_idx, sale_cats, sale_num_trans, numerical_columns, categorical_columns,
            chunk_size=SYNTHETIC_DATA_CHUNK_SIZE, batch_size=SYNTHETIC_DATA_BATCH_SIZE)

        synthetic_rent = postprocess_synthetic_data(synthetic_rent)
        synthetic_auction = postprocess_synthetic_data(synthetic_auction)
        synthetic_sale = postprocess_synthetic_data(synthetic_sale)

        combined_data = combine_synthetic_data(synthetic_rent, synthetic_auction, synthetic_sale)

        output_path = "data/synthetic_data.parquet"
        logger.info(f"Saving {len(combined_data):,} records to {output_path}...")
        combined_data.to_parquet(output_path, index=False, engine='pyarrow')

    def _execute_train(self, args: Dict[str, Any], logger) -> None:
        """Execute the ML training stage."""
        import os
        import pandas as pd
        from italian_real_estate.ml import (
            engineer_all_features,
            prepare_rent_training_data,
            train_rent_model,
            evaluate_model,
            print_metrics,
            create_dashboard_data,
        )

        test_size = args.get("test_size", 0.3)
        output_dashboard = "data/dashboard_data.csv"

        # Auto-detect input file
        candidates = [
            "data/synthetic_data.parquet",
            "data/real_data.parquet",
            "data/synthetic_data.csv",
        ]
        input_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                input_path = candidate
                break

        if not input_path:
            raise FileNotFoundError("No data file found. Run DATA EXPORT stage first.")

        logger.info(f"Loading data from {input_path}...")
        if input_path.endswith('.parquet'):
            data = pd.read_parquet(input_path, engine='pyarrow')
        else:
            data = pd.read_csv(input_path)

        logger.info("Applying feature engineering...")
        engineered_data = engineer_all_features(data)

        logger.info("Preparing training data...")
        X, Y = prepare_rent_training_data(engineered_data)

        logger.info("Training RandomForest model...")
        model, X_train, X_test, Y_train, Y_test = train_rent_model(
            X, Y, test_size=test_size, random_state=2025)

        logger.info("Evaluating model...")
        Y_pred = model.predict(X_test)
        metrics = evaluate_model(Y_test, Y_pred)
        print_metrics(metrics)

        logger.info("Creating dashboard data...")
        dashboard_data = create_dashboard_data(model, engineered_data, output_path=output_dashboard)
        logger.info(f"Dashboard saved to {output_dashboard}")

    def _run_status(self) -> None:
        """Show pipeline status."""
        self._clear_screen()
        self.console.print("[bold cyan]Pipeline Status[/bold cyan]\n")

        try:
            from italian_real_estate.config.logging_config import setup_logging, get_logger
            import logging
            setup_logging(level=logging.INFO)
            logger = get_logger(__name__)

            # Import status function from pipeline_tui
            import importlib.util
            import sys
            from pathlib import Path

            # Get pipeline status
            status = self._get_pipeline_status()
            self._print_pipeline_status(status)

        except Exception as e:
            self.console.print(f"[red]Error getting status: {e}[/red]")

        self.console.print("\n[dim]Press any key to continue...[/dim]")
        get_key()

    def _get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status from all data stores."""
        import os
        status = {}

        # MongoDB Datalake
        try:
            from italian_real_estate.core.connections import get_mongodb_client
            from italian_real_estate.config.settings import (
                MONGODB_HOST, MONGODB_PORT, MONGODB_DATALAKE_NAME, COLLECTION_NAMES
            )
            client = get_mongodb_client(
                MONGODB_HOST,
                MONGODB_PORT,
                timeout_ms=2000,
            )
            db = client[MONGODB_DATALAKE_NAME]
            status["mongodb_datalake"] = {
                "connected": True,
                "collections": {name: db[name].count_documents({}) for name in COLLECTION_NAMES}
            }
            client.close()
        except Exception as e:
            status["mongodb_datalake"] = {"connected": False, "error": str(e)}

        # MongoDB Warehouse
        try:
            from italian_real_estate.core.connections import get_mongodb_client
            from italian_real_estate.config.settings import (
                MONGODB_HOST, MONGODB_PORT, MONGODB_WAREHOUSE_NAME, COLLECTION_NAMES
            )
            client = get_mongodb_client(
                MONGODB_HOST,
                MONGODB_PORT,
                timeout_ms=2000,
            )
            db = client[MONGODB_WAREHOUSE_NAME]
            status["mongodb_warehouse"] = {
                "connected": True,
                "collections": {name: db[name].count_documents({}) for name in COLLECTION_NAMES}
            }
            client.close()
        except Exception as e:
            status["mongodb_warehouse"] = {"connected": False, "error": str(e)}

        # PostgreSQL
        try:
            import psycopg2
            from italian_real_estate.config.settings import POSTGRES_CONNECTION_PARAMS
            conn = psycopg2.connect(**POSTGRES_CONNECTION_PARAMS, connect_timeout=2)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM fact_listing")
            fact_count = cursor.fetchone()[0]
            status["postgresql"] = {"connected": True, "fact_table_records": fact_count}
            cursor.close()
            conn.close()
        except Exception as e:
            status["postgresql"] = {"connected": False, "error": str(e)}

        # Data files
        data_files = [
            "data/real_data.parquet",
            "data/synthetic_data.parquet",
            "data/synthetic_data.csv",  # Legacy
            "data/dashboard_data.csv",
        ]
        status["data_files"] = {}
        for filepath in data_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                status["data_files"][filepath] = {"exists": True, "size_mb": size / (1024 * 1024)}
            else:
                status["data_files"][filepath] = {"exists": False}

        return status

    def _print_pipeline_status(self, status: Dict[str, Any]) -> None:
        """Print formatted pipeline status."""
        # MongoDB Datalake
        self.console.print("\n[bold][Stage 1: SCRAPE] MongoDB Datalake[/bold]")
        dl = status.get("mongodb_datalake", {})
        if dl.get("connected"):
            for coll, count in dl.get("collections", {}).items():
                self.console.print(f"  {coll}: {count:,} documents")
        else:
            self.console.print(f"  [red]Not connected: {dl.get('error', 'Unknown')}[/red]")

        # MongoDB Warehouse
        self.console.print("\n[bold][Stage 2: ETL] MongoDB Warehouse[/bold]")
        wh = status.get("mongodb_warehouse", {})
        if wh.get("connected"):
            for coll, count in wh.get("collections", {}).items():
                self.console.print(f"  {coll}: {count:,} documents")
        else:
            self.console.print(f"  [red]Not connected: {wh.get('error', 'Unknown')}[/red]")

        # PostgreSQL
        self.console.print("\n[bold][Stage 3: MIGRATE] PostgreSQL[/bold]")
        pg = status.get("postgresql", {})
        if pg.get("connected"):
            self.console.print(f"  fact_listing: {pg.get('fact_table_records', 0):,} records")
        else:
            self.console.print(f"  [red]Not connected: {pg.get('error', 'Unknown')}[/red]")

        # Data files
        self.console.print("\n[bold][Stages 4-5: DATA EXPORT/TRAIN] Data Files[/bold]")
        for filepath, info in status.get("data_files", {}).items():
            if info.get("exists"):
                self.console.print(f"  {filepath}: {info.get('size_mb', 0):.2f} MB")
            else:
                self.console.print(f"  {filepath}: [dim]Not found[/dim]")

    def _run_db_explorer(self) -> None:
        """Launch the database explorer."""
        self._clear_screen()
        try:
            from italian_real_estate.utils.db_explorer import run_explorer
            run_explorer()
        except Exception as e:
            self.console.print(f"[red]Error launching database explorer: {e}[/red]")
            self.console.print("\n[dim]Press any key to continue...[/dim]")
            get_key()

    def _run_init_databases(self) -> None:
        """Initialize database schemas."""
        self._clear_screen()
        self.console.print("[bold cyan]Database Initialization[/bold cyan]\n")

        try:
            # Import setup functions
            import sys
            from pathlib import Path
            scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
            if scripts_dir.exists() and str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from setup_databases import (
                setup_mongodb, setup_postgresql,
                check_mongodb_status, check_postgresql_status,
                get_mongodb_config, get_postgres_credentials,
                display_setup_results
            )

            # Show current status first
            self.console.print("[bold]Current Database Status:[/bold]\n")

            mongo_status = check_mongodb_status()
            if mongo_status.get("connected"):
                self.console.print("[green]MongoDB: Connected[/green]")
                for db_name, db_info in mongo_status.get("databases", {}).items():
                    total = sum(db_info.get("collections", {}).values())
                    self.console.print(f"  {db_name}: {total:,} documents")
            else:
                self.console.print(f"[red]MongoDB: Not connected - {mongo_status.get('error', 'Unknown')}[/red]")

            pg_status = check_postgresql_status()
            if pg_status.get("connected"):
                self.console.print("[green]PostgreSQL: Connected[/green]")
                total_tables = len(pg_status.get("tables", {}))
                total_rows = sum(pg_status.get("tables", {}).values())
                self.console.print(f"  {total_tables} tables, {total_rows:,} total rows")
            else:
                self.console.print(f"[red]PostgreSQL: Not connected - {pg_status.get('error', 'Unknown')}[/red]")

            self.console.print()

            # Ask what to initialize
            from rich.prompt import Prompt

            choice = Prompt.ask(
                "What would you like to initialize?",
                choices=["both", "mongodb", "postgres", "cancel"],
                default="both"
            )

            if choice == "cancel":
                return

            # Run initialization
            if choice in ["both", "mongodb"]:
                self.console.print("\n[bold]Setting up MongoDB...[/bold]")
                mongo_config = get_mongodb_config()
                mongo_results = setup_mongodb(mongo_config)
                if mongo_results.get("success"):
                    self.console.print("[green]MongoDB setup complete![/green]")
                    self.console.print(f"  Datalake collections: {', '.join(mongo_results['datalake']['collections'])}")
                    self.console.print(f"  Warehouse collections: {', '.join(mongo_results['warehouse']['collections'])}")
                else:
                    self.console.print(f"[red]MongoDB setup failed: {mongo_results.get('error')}[/red]")

            if choice in ["both", "postgres"]:
                self.console.print("\n[bold]Setting up PostgreSQL...[/bold]")
                # For PostgreSQL, use interactive credentials
                pg_credentials = get_postgres_credentials(interactive=True)
                pg_results = setup_postgresql(pg_credentials)
                if pg_results.get("success"):
                    self.console.print("[green]PostgreSQL setup complete![/green]")
                    if pg_results.get("database_created"):
                        self.console.print("  [cyan]Database created[/cyan]")
                    tables = pg_results.get("tables", {})
                    if tables.get("created"):
                        self.console.print(f"  Tables created/verified: {len(tables['created'])}")
                    if tables.get("errors"):
                        for table, error in tables["errors"]:
                            self.console.print(f"  [red]Error with {table}: {error}[/red]")
                else:
                    self.console.print(f"[red]PostgreSQL setup failed: {pg_results.get('error')}[/red]")

            self.console.print("\n[bold green]Initialization complete![/bold green]")

        except ImportError as e:
            self.console.print(f"[red]Error: Could not import setup modules: {e}[/red]")
            self.console.print("[yellow]Make sure setup_databases.py exists in scripts/[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error during initialization: {e}[/red]")

        self.console.print("\n[dim]Press any key to continue...[/dim]")
        get_key()

    def _run_all_stages(self) -> None:
        """Run all pipeline stages in sequence."""
        self._clear_screen()
        self.console.print("[bold cyan]Running ALL Pipeline Stages[/bold cyan]\n")

        from rich.prompt import Confirm

        if not Confirm.ask("This will run all 5 stages sequentially. Continue?"):
            return

        stages = ["scrape", "etl", "migrate", "data_export", "train"]

        for stage_name in stages:
            config = StageConfig(
                stage_name=self.STAGE_CONFIGS[stage_name].stage_name,
                display_name=self.STAGE_CONFIGS[stage_name].display_name,
                options=dict(self.STAGE_CONFIGS[stage_name].options),
                selected_options=dict(self.STAGE_CONFIGS[stage_name].selected_options),
            )

            self.console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
            self.console.print(f"[bold]Stage: {config.display_name}[/bold]")
            self.console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")

            try:
                args = self._build_stage_args(config)
                from italian_real_estate.config.logging_config import setup_logging, get_logger
                import logging
                setup_logging(level=logging.INFO)
                logger = get_logger(__name__)

                if stage_name == "scrape":
                    self._execute_scrape(args, logger)
                elif stage_name == "etl":
                    self._execute_etl(args, logger)
                elif stage_name == "migrate":
                    self._execute_migrate(args, logger)
                elif stage_name == "data_export":
                    self._execute_data_export(args, logger)
                elif stage_name == "train":
                    self._execute_train(args, logger)

                self.console.print(f"[green]✓ {config.display_name} completed[/green]")

            except Exception as e:
                self.console.print(f"[red]✗ {config.display_name} failed: {e}[/red]")
                if not Confirm.ask("Continue with remaining stages?"):
                    break

        self.console.print("\n[bold green]Pipeline execution finished![/bold green]")
        self.console.print("\n[dim]Press any key to continue...[/dim]")
        get_key()

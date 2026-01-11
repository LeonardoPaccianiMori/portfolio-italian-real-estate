"""
Progress tracking and visualization for parallel job execution.

This module provides Airflow-like progress visualization for CLI-based
parallel execution of pipeline tasks. It displays live progress bars,
job status tables, and timing information.

Components:
    - JobStatus: Enum for job states (PENDING, RUNNING, DONE, FAILED)
    - JobState: Dataclass tracking individual job state
    - ProgressTracker: Thread-safe tracker for all parallel jobs
    - ProgressDisplay: Rich-based terminal display
    - ParallelExecutor: Runs jobs in parallel with progress callbacks

Example:
    from italian_real_estate.utils.progress import ParallelExecutor

    def my_job(item):
        # Process item
        return result

    executor = ParallelExecutor(max_workers=8)
    results = executor.run(
        items=["Milano", "Roma", "Napoli"],
        job_func=my_job,
        job_name_func=lambda x: x
    )

Author: Leonardo Pacciani-Mori
License: MIT
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


T = TypeVar('T')
R = TypeVar('R')


class JobStatus(Enum):
    """Status of an individual job."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobState:
    """
    State of an individual job being tracked.

    Attributes:
        name: Display name for the job (e.g., province name).
        status: Current status of the job.
        start_time: When the job started (None if pending).
        end_time: When the job ended (None if not finished).
        result: Result returned by the job (None until done).
        error: Exception if the job failed.
        records_processed: Optional count of records processed.
    """
    name: str
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    records_processed: Optional[int] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate job duration."""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def duration_str(self) -> str:
        """Format duration as HH:MM:SS."""
        dur = self.duration
        if dur is None:
            return "-"
        total_seconds = int(dur.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressTracker:
    """
    Thread-safe tracker for parallel job progress.

    This class maintains the state of all jobs and provides methods
    for updating status and calculating statistics.

    Attributes:
        title: Title displayed in the progress header.
        jobs: Dictionary mapping job names to their states.
    """

    def __init__(self, title: str = "Progress"):
        """
        Initialize the progress tracker.

        Args:
            title: Title to display in progress header.
        """
        self.title = title
        self.jobs: Dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._start_time: Optional[datetime] = None

    def add_job(self, name: str) -> None:
        """
        Add a job to track.

        Args:
            name: Unique name for the job.
        """
        with self._lock:
            self.jobs[name] = JobState(name=name)

    def add_jobs(self, names: List[str]) -> None:
        """
        Add multiple jobs to track.

        Args:
            names: List of unique job names.
        """
        for name in names:
            self.add_job(name)

    def start_job(self, name: str) -> None:
        """
        Mark a job as started.

        Args:
            name: Name of the job to start.
        """
        with self._lock:
            if self._start_time is None:
                self._start_time = datetime.now()
            if name in self.jobs:
                self.jobs[name].status = JobStatus.RUNNING
                self.jobs[name].start_time = datetime.now()

    def complete_job(
        self,
        name: str,
        result: Any = None,
        records_processed: Optional[int] = None
    ) -> None:
        """
        Mark a job as completed successfully.

        Args:
            name: Name of the job.
            result: Result returned by the job.
            records_processed: Optional count of records processed.
        """
        with self._lock:
            if name in self.jobs:
                self.jobs[name].status = JobStatus.DONE
                self.jobs[name].end_time = datetime.now()
                self.jobs[name].result = result
                self.jobs[name].records_processed = records_processed

    def fail_job(self, name: str, error: Exception) -> None:
        """
        Mark a job as failed.

        Args:
            name: Name of the job.
            error: Exception that caused the failure.
        """
        with self._lock:
            if name in self.jobs:
                self.jobs[name].status = JobStatus.FAILED
                self.jobs[name].end_time = datetime.now()
                self.jobs[name].error = error

    def update_records(self, name: str, records: int) -> None:
        """
        Update the records processed count for a running job.

        Args:
            name: Name of the job.
            records: Number of records processed so far.
        """
        with self._lock:
            if name in self.jobs:
                self.jobs[name].records_processed = records

    @property
    def total_jobs(self) -> int:
        """Total number of jobs being tracked."""
        return len(self.jobs)

    @property
    def pending_count(self) -> int:
        """Number of pending jobs."""
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING)

    @property
    def running_count(self) -> int:
        """Number of currently running jobs."""
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.RUNNING)

    @property
    def completed_count(self) -> int:
        """Number of successfully completed jobs."""
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.DONE)

    @property
    def failed_count(self) -> int:
        """Number of failed jobs."""
        return sum(1 for j in self.jobs.values() if j.status == JobStatus.FAILED)

    @property
    def finished_count(self) -> int:
        """Number of finished jobs (completed + failed)."""
        return self.completed_count + self.failed_count

    @property
    def progress_percent(self) -> float:
        """Overall progress as a percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.finished_count / self.total_jobs) * 100

    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Time elapsed since tracking started."""
        if self._start_time is None:
            return None
        return datetime.now() - self._start_time

    @property
    def elapsed_str(self) -> str:
        """Format elapsed time as HH:MM:SS."""
        elapsed = self.elapsed_time
        if elapsed is None:
            return "00:00:00"
        total_seconds = int(elapsed.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @property
    def estimated_remaining(self) -> Optional[timedelta]:
        """Estimated time remaining based on current progress."""
        elapsed = self.elapsed_time
        if elapsed is None or self.finished_count == 0:
            return None
        avg_time_per_job = elapsed / self.finished_count
        remaining_jobs = self.total_jobs - self.finished_count
        return avg_time_per_job * remaining_jobs

    @property
    def estimated_remaining_str(self) -> str:
        """Format estimated remaining time as HH:MM:SS."""
        remaining = self.estimated_remaining
        if remaining is None:
            return "calculating..."
        total_seconds = int(remaining.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"~{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_jobs_by_status(self, status: JobStatus) -> List[JobState]:
        """
        Get all jobs with a specific status.

        Args:
            status: Status to filter by.

        Returns:
            List of job states with the specified status.
        """
        return [j for j in self.jobs.values() if j.status == status]

    def get_visible_jobs(self, max_display: int = 20) -> List[JobState]:
        """
        Get jobs to display, prioritizing running and recently finished.

        Args:
            max_display: Maximum number of jobs to return.

        Returns:
            List of job states to display.
        """
        # Priority: Running > Recently Completed > Recently Failed > Pending
        running = self.get_jobs_by_status(JobStatus.RUNNING)
        completed = sorted(
            self.get_jobs_by_status(JobStatus.DONE),
            key=lambda j: j.end_time or datetime.min,
            reverse=True
        )
        failed = sorted(
            self.get_jobs_by_status(JobStatus.FAILED),
            key=lambda j: j.end_time or datetime.min,
            reverse=True
        )
        pending = self.get_jobs_by_status(JobStatus.PENDING)

        result = []
        for jobs_list in [running, completed, failed, pending]:
            for job in jobs_list:
                if len(result) >= max_display:
                    break
                result.append(job)

        return result


class ProgressDisplay:
    """
    Rich-based terminal display for progress visualization.

    Creates an Airflow-like display showing overall progress,
    timing information, and individual job status.
    """

    STATUS_ICONS = {
        JobStatus.PENDING: ("○", "dim"),
        JobStatus.RUNNING: ("●", "yellow"),
        JobStatus.DONE: ("✓", "green"),
        JobStatus.FAILED: ("✗", "red"),
    }

    def __init__(
        self,
        tracker: ProgressTracker,
        max_display_jobs: int = 15,
        refresh_rate: float = 0.5
    ):
        """
        Initialize the progress display.

        Args:
            tracker: Progress tracker to visualize.
            max_display_jobs: Maximum jobs to show in the table.
            refresh_rate: Seconds between display updates.
        """
        self.tracker = tracker
        self.max_display_jobs = max_display_jobs
        self.refresh_rate = refresh_rate
        self.console = Console() if RICH_AVAILABLE else None
        self._live: Optional[Live] = None

    def _build_progress_bar(self) -> str:
        """Build ASCII progress bar."""
        width = 40
        filled = int(width * self.tracker.progress_percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {self.tracker.progress_percent:.1f}%"

    def _build_table(self) -> Table:
        """Build the job status table."""
        table = Table(show_header=True, header_style="bold", expand=True)
        table.add_column("Job", style="cyan", no_wrap=True, width=30)
        table.add_column("Status", justify="center", width=12)
        table.add_column("Records", justify="right", width=12)
        table.add_column("Time", justify="right", width=10)

        for job in self.tracker.get_visible_jobs(self.max_display_jobs):
            icon, style = self.STATUS_ICONS[job.status]
            status_text = Text(f"{icon} {job.status.value.title()}", style=style)

            records = "-"
            if job.records_processed is not None:
                records = f"{job.records_processed:,}"
                if job.status == JobStatus.RUNNING:
                    records += "..."

            table.add_row(
                job.name,
                status_text,
                records,
                job.duration_str
            )

        # Add summary row if there are hidden jobs
        hidden_count = self.tracker.total_jobs - len(self.tracker.get_visible_jobs(self.max_display_jobs))
        if hidden_count > 0:
            table.add_row(
                f"... and {hidden_count} more",
                "",
                "",
                "",
                style="dim"
            )

        return table

    def _build_panel(self) -> Panel:
        """Build the complete progress panel."""
        # Header with title and progress bar
        header_lines = [
            f"[bold]{self.tracker.title}[/bold]",
            "",
            f"Progress: {self._build_progress_bar()} ({self.tracker.finished_count}/{self.tracker.total_jobs})",
            f"Active: {self.tracker.running_count} | Completed: {self.tracker.completed_count} | Failed: {self.tracker.failed_count} | Remaining: {self.tracker.pending_count}",
            f"Time: {self.tracker.elapsed_str} elapsed | {self.tracker.estimated_remaining_str} remaining",
            "",
        ]
        header = "\n".join(header_lines)

        # Combine header and table
        from rich.console import Group
        content = Group(Text.from_markup(header), self._build_table())

        return Panel(content, border_style="blue")

    def start(self) -> None:
        """Start the live display."""
        if not RICH_AVAILABLE:
            print(f"Starting: {self.tracker.title}")
            print(f"Total jobs: {self.tracker.total_jobs}")
            return

        self._live = Live(
            self._build_panel(),
            console=self.console,
            refresh_per_second=1/self.refresh_rate,
            transient=False
        )
        self._live.start()

    def update(self) -> None:
        """Update the display."""
        if not RICH_AVAILABLE:
            # Fallback to simple progress output
            pct = self.tracker.progress_percent
            done = self.tracker.finished_count
            total = self.tracker.total_jobs
            print(f"\rProgress: {pct:.1f}% ({done}/{total})", end="", flush=True)
            return

        if self._live:
            self._live.update(self._build_panel())

    def stop(self) -> None:
        """Stop the live display."""
        if not RICH_AVAILABLE:
            print()  # Newline after progress
            return

        if self._live:
            self._live.stop()

    def print_summary(self) -> None:
        """Print final summary after completion."""
        if not RICH_AVAILABLE:
            print(f"\n{self.tracker.title} Complete!")
            print(f"  Total: {self.tracker.total_jobs}")
            print(f"  Completed: {self.tracker.completed_count}")
            print(f"  Failed: {self.tracker.failed_count}")
            print(f"  Time: {self.tracker.elapsed_str}")
            return

        summary_table = Table(show_header=False, box=None)
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Value")

        summary_table.add_row("Total Jobs", str(self.tracker.total_jobs))
        summary_table.add_row("Completed", f"[green]{self.tracker.completed_count}[/green]")
        if self.tracker.failed_count > 0:
            summary_table.add_row("Failed", f"[red]{self.tracker.failed_count}[/red]")
        summary_table.add_row("Total Time", self.tracker.elapsed_str)

        self.console.print(Panel(
            summary_table,
            title=f"[bold]{self.tracker.title} Complete[/bold]",
            border_style="green" if self.tracker.failed_count == 0 else "yellow"
        ))


class ParallelExecutor(Generic[T, R]):
    """
    Execute jobs in parallel with progress tracking.

    This class combines ThreadPoolExecutor with ProgressTracker and
    ProgressDisplay to provide Airflow-like parallel execution with
    live progress visualization.

    Example:
        executor = ParallelExecutor(max_workers=8, title="Scraping")
        results = executor.run(
            items=provinces,
            job_func=scrape_province,
            job_name_func=lambda p: p.name
        )
    """

    def __init__(
        self,
        max_workers: int = 8,
        title: str = "Processing",
        show_progress: bool = True,
        max_display_jobs: int = 15
    ):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of parallel workers.
            title: Title for the progress display.
            show_progress: Whether to show progress display.
            max_display_jobs: Maximum jobs to show in status table.
        """
        self.max_workers = max_workers
        self.title = title
        self.show_progress = show_progress
        self.max_display_jobs = max_display_jobs

    def run(
        self,
        items: List[T],
        job_func: Callable[[T], R],
        job_name_func: Optional[Callable[[T], str]] = None,
        on_complete: Optional[Callable[[T, R], Optional[int]]] = None
    ) -> Dict[str, R]:
        """
        Execute jobs in parallel with progress tracking.

        Args:
            items: List of items to process.
            job_func: Function to call for each item.
            job_name_func: Function to generate job name from item.
                          Defaults to str(item).
            on_complete: Optional callback when job completes.
                        Should return record count or None.

        Returns:
            Dictionary mapping job names to results.
        """
        if job_name_func is None:
            job_name_func = str

        # Create tracker and display
        tracker = ProgressTracker(title=self.title)
        job_names = [job_name_func(item) for item in items]
        tracker.add_jobs(job_names)

        display = ProgressDisplay(
            tracker,
            max_display_jobs=self.max_display_jobs
        ) if self.show_progress else None

        results: Dict[str, R] = {}
        item_by_name = {job_name_func(item): item for item in items}

        def execute_job(name: str, item: T) -> Tuple[str, R]:
            """Execute a single job and track progress."""
            tracker.start_job(name)
            if display:
                display.update()

            try:
                result = job_func(item)
                records = None
                if on_complete:
                    records = on_complete(item, result)
                tracker.complete_job(name, result, records)
                return name, result
            except Exception as e:
                tracker.fail_job(name, e)
                raise

        # Start display
        if display:
            display.start()

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(execute_job, name, item): name
                    for name, item in item_by_name.items()
                }

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        job_name, result = future.result()
                        results[job_name] = result
                    except Exception as e:
                        # Job already marked as failed in execute_job
                        pass

                    if display:
                        display.update()

        finally:
            if display:
                display.stop()
                display.print_summary()

        return results

    def run_with_args(
        self,
        items: List[Tuple],
        job_func: Callable[..., R],
        job_name_func: Optional[Callable[[Tuple], str]] = None,
        on_complete: Optional[Callable[[Tuple, R], Optional[int]]] = None
    ) -> Dict[str, R]:
        """
        Execute jobs with tuple arguments unpacked.

        This is useful when each job needs multiple arguments.

        Args:
            items: List of tuples, each containing args for one job.
            job_func: Function to call with unpacked tuple args.
            job_name_func: Function to generate job name from tuple.
            on_complete: Optional callback when job completes.

        Returns:
            Dictionary mapping job names to results.

        Example:
            items = [("Milano", "sale"), ("Roma", "rent")]
            executor.run_with_args(
                items,
                lambda prov, typ: scrape(prov, typ),
                lambda t: f"{t[0]}-{t[1]}"
            )
        """
        def wrapped_job_func(args_tuple: Tuple) -> R:
            """Invoke the provided job function with tuple-unpacked arguments."""
            return job_func(*args_tuple)

        return self.run(
            items=items,
            job_func=wrapped_job_func,
            job_name_func=job_name_func,
            on_complete=on_complete
        )


def run_jobs_with_progress(
    items: List[T],
    job_func: Callable[[T], R],
    title: str = "Processing",
    max_workers: int = 8,
    job_name_func: Optional[Callable[[T], str]] = None,
    show_progress: bool = True
) -> Dict[str, R]:
    """
    Convenience function for running parallel jobs with progress.

    This is a simpler interface to ParallelExecutor for common use cases.

    Args:
        items: List of items to process.
        job_func: Function to call for each item.
        title: Title for progress display.
        max_workers: Number of parallel workers.
        job_name_func: Function to generate job name from item.
        show_progress: Whether to show progress display.

    Returns:
        Dictionary mapping job names to results.

    Example:
        results = run_jobs_with_progress(
            items=provinces,
            job_func=scrape_province,
            title="Scraping provinces",
            max_workers=8
        )
    """
    executor = ParallelExecutor(
        max_workers=max_workers,
        title=title,
        show_progress=show_progress
    )
    return executor.run(items, job_func, job_name_func)

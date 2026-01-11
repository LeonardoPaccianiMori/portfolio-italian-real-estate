"""
Utility modules for the Italian Real Estate pipeline.

This package contains shared utilities for progress tracking, parallel
execution, database exploration, and other common functionality used
across the pipeline.

Author: Leonardo Pacciani-Mori
License: MIT
"""

from italian_real_estate.utils.progress import (
    JobStatus,
    JobState,
    ProgressTracker,
    ProgressDisplay,
    ParallelExecutor,
)
from italian_real_estate.utils.db_explorer import (
    DatabaseExplorer,
    MongoDBBackend,
    PostgreSQLBackend,
    run_explorer,
    run_single_command,
)
from italian_real_estate.utils.tui import (
    PipelineTUI,
)

__all__ = [
    # Progress tracking
    "JobStatus",
    "JobState",
    "ProgressTracker",
    "ProgressDisplay",
    "ParallelExecutor",
    # Database explorer
    "DatabaseExplorer",
    "MongoDBBackend",
    "PostgreSQLBackend",
    "run_explorer",
    "run_single_command",
    # TUI
    "PipelineTUI",
]

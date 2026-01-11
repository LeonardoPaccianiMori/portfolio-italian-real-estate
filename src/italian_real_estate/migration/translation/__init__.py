"""
Translation submodule for the migration pipeline.

This submodule provides Italian to English translation functionality
using LibreTranslate, with support for:
- Custom domain-specific translations for real estate terminology
- SQLite-based caching for efficient re-use of translations
- Batch and parallel translation for high-volume processing

Author: Leonardo Pacciani-Mori
License: MIT
"""

from .custom_translations import CUSTOM_TRANSLATIONS
from .cache import (
    initialize_translation_cache,
    get_from_cache,
    save_to_cache,
    get_cache_stats,
)
from .translator import (
    translate_text,
    start_libretranslate_service,
    stop_libretranslate_service,
)
from .batch_translator import (
    batch_translate,
    translate_values_parallel,
    resolve_conflicts_in_batch,
    bulk_update_translations,
)

__all__ = [
    # Custom translations dictionary
    "CUSTOM_TRANSLATIONS",
    # Cache functions
    "initialize_translation_cache",
    "get_from_cache",
    "save_to_cache",
    "get_cache_stats",
    # Single text translation
    "translate_text",
    "start_libretranslate_service",
    "stop_libretranslate_service",
    # Batch translation
    "batch_translate",
    "translate_values_parallel",
    "resolve_conflicts_in_batch",
    "bulk_update_translations",
]

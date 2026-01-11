"""
LibreTranslate integration for Italian to English translation.

This module provides functions for translating Italian text to English
using a local LibreTranslate instance. It includes custom translations
for domain-specific Italian real estate terminology and supports
placeholder-based translation to preserve custom terms.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import atexit
import re
import subprocess
import time
from typing import Optional

import requests
from lingua import Language, LanguageDetectorBuilder

from ...config.settings import LIBRETRANSLATE_URL
from ...config.logging_config import get_logger
from ...core.string_utils import preprocess_text
from .custom_translations import CUSTOM_TRANSLATIONS
from .cache import get_from_cache, save_to_cache

# Module-level logger for consistent logging.
logger = get_logger(__name__)

# Initializes the language detector for Italian and English detection.
# This is used to skip translation for text that's already in English.
LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.ITALIAN
).build()


def translate_text(
    text: str,
    source_lang: str = "it",
    target_lang: str = "en",
    max_retries: int = 10,
    cache_path: Optional[str] = None
) -> str:
    """
    Translate Italian text to English using LibreTranslate.

    This function translates text using a local LibreTranslate instance,
    with support for:
    - Custom translations for domain-specific terminology
    - Caching to avoid re-translating the same text
    - Language detection to skip already-English text
    - Automatic retries on failure

    The translation uses a placeholder approach:
    1. Domain-specific terms are replaced with unique placeholders
    2. The text with placeholders is sent to LibreTranslate
    3. After translation, placeholders are replaced with English terms

    Args:
        text: The Italian text to translate.
        source_lang: Source language code (default: "it" for Italian).
        target_lang: Target language code (default: "en" for English).
        max_retries: Maximum number of retry attempts on failure.
        cache_path: Path to SQLite cache database for storing translations.

    Returns:
        The translated English text, or the preprocessed original text
        if translation fails.

    Example:
        >>> translated = translate_text("Appartamento libero in via Roma")
        >>> print(translated)
        Apartment vacant on Roma street
    """
    # Handles None or non-string input by returning as-is.
    if not text or not isinstance(text, str):
        return text

    # Checks the cache first to avoid re-translating.
    cached_translation = get_from_cache(text, source_lang, target_lang, cache_path)
    if cached_translation:
        return cached_translation

    # Preprocesses the text to improve translation quality.
    preprocessed_text = preprocess_text(text)

    # Detects the language of the input text.
    detected_lang = LANGUAGE_DETECTOR.detect_language_of(preprocessed_text)

    # If the text is already in English, returns it without translation.
    if detected_lang == Language.ENGLISH:
        return preprocessed_text

    # Logs a warning if the language is neither Italian nor English.
    if detected_lang and detected_lang != Language.ITALIAN:
        logger.warning(
            f"Text detected as language other than Italian or English: {detected_lang}"
        )

    # Checks if the entire text exactly matches a custom translation pattern.
    # This handles cases where the whole string is a domain-specific term.
    for pattern, translation in CUSTOM_TRANSLATIONS.items():
        if re.match(f'^{pattern}$', preprocessed_text.lower(), re.IGNORECASE):
            # Saves to cache and returns the custom translation directly.
            save_to_cache(text, translation, source_lang, target_lang, cache_path)
            return translation

    # Creates a dictionary to track placeholders and their English translations.
    placeholders = {}
    placeholder_text = preprocessed_text

    # Replaces custom translation patterns with unique placeholders.
    # This preserves domain-specific terms during LibreTranslate processing.
    for pattern, translation in CUSTOM_TRANSLATIONS.items():
        # Adds word boundaries to patterns that don't already have them.
        if not (pattern.startswith(r'\b') or pattern.endswith(r'\b')):
            search_pattern = r'\b' + pattern + r'\b'
        else:
            search_pattern = pattern

        # Finds all matches of this pattern in the text.
        matches = list(re.finditer(search_pattern, placeholder_text, flags=re.IGNORECASE))

        # Processes matches in reverse order to avoid index shifting issues.
        for i, match in enumerate(reversed(matches)):
            # Creates a unique placeholder that LibreTranslate won't translate.
            placeholder = f"__CUSTOM_TRANS_{len(placeholders)}__"

            # Stores the mapping from placeholder to English translation.
            placeholders[placeholder] = translation

            # Replaces the matched text with the placeholder.
            start, end = match.span()
            placeholder_text = placeholder_text[:start] + placeholder + placeholder_text[end:]

    # Attempts translation with retries for resilience against transient failures.
    for attempt in range(max_retries):
        try:
            # Prepares the HTTP request payload for LibreTranslate.
            payload = {
                "q": placeholder_text,
                "source": source_lang,
                "target": target_lang
            }

            # Sends the translation request to the local LibreTranslate instance.
            response = requests.post(LIBRETRANSLATE_URL, json=payload, timeout=10)

            # Processes successful response.
            if response.status_code == 200:
                # Parses the JSON response to get the translated text.
                result = response.json()
                translated_text = result.get("translatedText", placeholder_text)

                # Replaces all placeholders with their English translations.
                final_text = translated_text
                for placeholder, custom_translation in placeholders.items():
                    final_text = final_text.replace(placeholder, custom_translation)

                # Cleans up any excessive underscores (common translation artifact).
                final_text = re.sub(r'_+', ' ', final_text).strip()

                # Saves the successful translation to cache.
                save_to_cache(text, final_text, source_lang, target_lang, cache_path)

                return final_text

            else:
                # Logs warning for non-200 responses and retries.
                logger.warning(
                    f"Translation attempt {attempt + 1} failed with status code "
                    f"{response.status_code}: {response.text}"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)

        except Exception as e:
            # Logs error for exceptions and retries.
            logger.error(f"Error during translation attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)

    # If all attempts fail, returns the preprocessed text without translation.
    return preprocessed_text


def start_libretranslate_service() -> subprocess.Popen:
    """
    Start a local LibreTranslate instance as a subprocess.

    This function launches LibreTranslate on localhost:5000 and waits
    for it to become available. It also registers a cleanup function
    to terminate the process when the script exits.

    Returns:
        The subprocess.Popen object representing the running
        LibreTranslate process.

    Raises:
        RuntimeError: If LibreTranslate fails to start after multiple
            attempts (30 attempts with 2-second intervals = 60 seconds).

    Example:
        >>> process = start_libretranslate_service()
        >>> # ... do translations ...
        >>> stop_libretranslate_service(process)
    """
    logger.info("Starting local LibreTranslate service...")

    # Starts LibreTranslate as a background subprocess.
    process = subprocess.Popen(
        ["libretranslate", "--host", "localhost", "--port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Defines a cleanup function to terminate the process on script exit.
    def cleanup():
        """Terminates the LibreTranslate process gracefully."""
        if process.poll() is None:  # Checks if process is still running.
            logger.info("Terminating LibreTranslate process...")
            process.terminate()
            try:
                # Waits up to 10 seconds for graceful termination.
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Forces termination if graceful shutdown fails.
                logger.warning(
                    "LibreTranslate process did not terminate gracefully, forcing..."
                )
                process.kill()

    # Registers the cleanup function to run when the script exits.
    atexit.register(cleanup)

    # Waits for the service to become available.
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            # Checks if LibreTranslate is responding to requests.
            response = requests.get("http://localhost:5000/languages")
            if response.status_code == 200:
                logger.info("LibreTranslate service is up and running!")
                return process
        except requests.exceptions.ConnectionError:
            # Service not ready yet, will retry.
            pass

        logger.info(
            f"Waiting for LibreTranslate to start (attempt {attempt + 1}/{max_attempts})..."
        )
        time.sleep(2)

    # If all attempts exhausted, cleans up and raises an error.
    logger.error("Failed to start LibreTranslate service")
    cleanup()
    raise RuntimeError("Failed to start LibreTranslate service after multiple attempts")


def stop_libretranslate_service(process: subprocess.Popen) -> None:
    """
    Stop a running LibreTranslate process.

    This function gracefully terminates the LibreTranslate subprocess,
    with a fallback to forceful termination if needed.

    Args:
        process: The subprocess.Popen object representing the
            LibreTranslate process to stop.

    Example:
        >>> process = start_libretranslate_service()
        >>> # ... do translations ...
        >>> stop_libretranslate_service(process)
    """
    # Checks if the process exists and is still running.
    if process and process.poll() is None:
        logger.info("Stopping LibreTranslate service...")

        # Sends a termination signal.
        process.terminate()

        try:
            # Waits up to 10 seconds for the process to terminate.
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # If graceful termination fails, forces the process to stop.
            logger.warning(
                "LibreTranslate process did not terminate gracefully, forcing..."
            )
            process.kill()

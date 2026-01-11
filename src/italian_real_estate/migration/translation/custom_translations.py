"""
Custom translation patterns for the Italian Real Estate pipeline.

This module contains regex patterns and their English translations for
domain-specific Italian terms that may not be correctly translated by
automated translation services.

These patterns are applied before sending text to LibreTranslate, using
a placeholder mechanism to preserve the custom translations.

Author: Leonardo Pacciani-Mori
License: MIT
"""

# =============================================================================
# CUSTOM TRANSLATIONS DICTIONARY
# =============================================================================

# Dictionary mapping Italian regex patterns to English translations.
# Format: 'regex_pattern': 'english_translation'
#
# The patterns use Python regex syntax. Common patterns:
#   - \b: Word boundary (ensures whole word matching)
#   - [oiae]: Character class for Italian word endings
#   - (?<!...): Negative lookbehind

CUSTOM_TRANSLATIONS = {
    # -------------------------------------------------------------------------
    # Occupancy Status Terms
    # -------------------------------------------------------------------------
    # These terms describe whether a property is currently occupied or vacant.
    # Common in auction and sale listings.

    # "occupato/a/i/e" -> "inhabited" (property is currently occupied)
    r'\boccupat[oiae]?\b': "inhabited",
    r'\bocupad[oiae]\b': "inhabited",  # Common misspelling

    # "libero/a/i/e" -> "vacant" (property is available/empty)
    r'\bliber[oiae]?\b': "vacant",
    r'\blibr[oiae]\b': "vacant",      # Common abbreviation
    r'\blibrr[oa]\b': "vacant",       # Typo variant
    r'\bvaci[oa]\b': "vacant",        # Spanish variant (some listings)

    # -------------------------------------------------------------------------
    # Building/System Terms
    # -------------------------------------------------------------------------

    # "autonomo/a" -> "independent" (for heating, access, etc.)
    r'\bautonom[oa]\b': "independent",

    # -------------------------------------------------------------------------
    # Address Abbreviations
    # -------------------------------------------------------------------------
    # Common abbreviations used in Italian addresses.

    r'\bv.\b': "via",                 # Street
    r'\bn.\b': "number",              # House number
    r'\bp.zza\b': "square",           # Piazza (square)
    r'\bimm.\b': "real estate",       # Real estate abbrev.
    r'\bpl.\b': "floor",              # Piano (floor)

    # -------------------------------------------------------------------------
    # Street Type Terms
    # -------------------------------------------------------------------------

    r'\bv.le\b': "avenue",            # Viale abbreviation
    r'\bviale\b': "avenue",           # Viale (avenue/boulevard)

    # "corso" -> "boulevard" (but not when preceded by "in corso" meaning "in progress")
    r'\b(?<!in )corso\b': "boulevard",
    r'\bc.so\b': "boulevard",         # Corso abbreviation

    # -------------------------------------------------------------------------
    # Property Terms
    # -------------------------------------------------------------------------

    # "telematico/a/che" -> "on line" (for online auctions)
    r'\btelematic[oia]\b': "on line",
    r'\btelematiche\b': "on line",

    # "mq" -> "squared meters" (common area abbreviation)
    r'\bmq\b': "squared meters",

    # "terratetto" -> "terraced house" (specific Italian property type)
    r'\bterratetto\b': "terraced house",
    r'\bvilletta a schiera\b': "terraced house",
    r'\bvilla a schiera\b': "terraced house",

    # -------------------------------------------------------------------------
    # Kitchen Types
    # -------------------------------------------------------------------------
    # These are specific Italian kitchen classifications that need exact translations.

    r'\bcucina abitabile\b': "eat-in kitchen",
    r'\bcucina cucinotto\b': "kitchenette",
    r'\bcucina a vista\b': "open kitchen",
    r'\bcucina semi abitabile\b': "semi eat-in kitchen",
    r'\bcucina angolo cottura\b': "cooking corner",

    # -------------------------------------------------------------------------
    # Building Features
    # -------------------------------------------------------------------------

    r'\bportiere\b': "doorman",           # Building doorman/concierge
    r'\bpasso carrabile\b': "driveway",   # Driveway access
    r'\bbox privato\b': "private parking spot",
    r'\barea accoglienza\b': "front desk",
    r'\binfissi\b': "window frames",

    # -------------------------------------------------------------------------
    # Edge Cases and Corrections
    # -------------------------------------------------------------------------
    # These handle specific translation errors or edge cases.

    # This corrects a specific garbled translation that occasionally appears
    r'\bdb_fullname_external_fixtures. ♪\b': "window frames outdoor glass / wood",
}


def get_all_patterns():
    """
    Get all custom translation patterns as a list.

    This function returns the pattern strings for inspection or modification.

    Returns:
        list: A list of regex pattern strings.

    Example:
        >>> patterns = get_all_patterns()
        >>> print(f"Total patterns: {len(patterns)}")
    """
    return list(CUSTOM_TRANSLATIONS.keys())


def get_translation(pattern: str) -> str:
    """
    Get the English translation for a specific pattern.

    Args:
        pattern: The regex pattern to look up.

    Returns:
        str: The English translation, or the pattern itself if not found.

    Example:
        >>> get_translation(r'\boccupat[oiae]?\b')
        'inhabited'
    """
    return CUSTOM_TRANSLATIONS.get(pattern, pattern)


def add_custom_translation(pattern: str, translation: str) -> None:
    """
    Add a new custom translation pattern.

    This function allows runtime addition of translation patterns.
    Changes are not persisted to the source file.

    Args:
        pattern: The regex pattern to match.
        translation: The English translation to use.

    Example:
        >>> add_custom_translation(r'\bnuovo termine\b', 'new term')
    """
    CUSTOM_TRANSLATIONS[pattern] = translation

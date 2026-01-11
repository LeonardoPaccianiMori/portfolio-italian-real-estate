"""
Feature engineering utilities for the ML rent prediction model.

This module provides mappings and extraction functions for transforming
raw real estate data into ML-ready features. It handles property type
simplification, heating/AC parsing, and window frame processing.

The key transformations are:
1. Property type mapping: Consolidates 40+ property types into 7 categories
2. Heating extraction: Parses heating_type string into type, delivery, and power
3. Air conditioning extraction: Parses air_conditioning into type, hot, and cold
4. Window frame processing: Consolidates 8 window columns into 2 features

Author: Leonardo Pacciani-Mori
License: MIT
"""

import re
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np

from ..config.logging_config import get_logger

# Module-level logger for consistent logging.
logger = get_logger(__name__)


# Maps detailed property types to simplified categories.
# This reduces the number of one-hot encoded features significantly.
PROPERTY_MAPPING: Dict[str, str] = {
    # Apartments and similar residential units.
    'apartment': 'apartment',
    'apartment in villa': 'apartment',
    'attic': 'apartment',
    'penthouse': 'apartment',
    'loft': 'apartment',
    'open space': 'apartment',
    'vacation apartment': 'vacation property',

    # Villas and larger standalone houses.
    'single-family villa': 'villa',
    'multi-family villa': 'villa',
    'semi-detached villa': 'villa',
    'villa': 'villa',
    'two-family villa': 'villa',
    'detached villa': 'villa',
    'vacation villa': 'vacation property',

    # Row houses and attached units.
    'terraced house': 'terraced house',
    'terraced house single-family': 'terraced house',
    'terraced house unfamiliar': 'terraced house',
    'terraced house multi-family': 'terraced house',

    # Independent/detached houses.
    'independent house': 'house',
    'detached house': 'house',
    'house': 'house',
    'household': 'house',
    'vacation house': 'vacation property',

    # Rural and agricultural properties.
    'rustic': 'rural property',
    'farmhouse': 'rural property',
    'farm': 'rural property',
    'rustic - farmhouse': 'rural property',
    'house farmhouse': 'rural property',
    'cascina': 'rural property',
    'agricultural land': 'land',
    'masseria': 'rural property',
    'vacation farmhouse': 'vacation_property',

    # Buildings and multi-unit structures.
    'building': 'building',
    'palace - building': 'building',
    'building - building': 'building',
    'building - stable': 'building',
    'palace - stable': 'building',
    'stable or palace': 'building',
    'stables': 'building',

    # Historical and special properties.
    'historical residence': 'special property',
    'trullo': 'special property',
    'dammuso': 'special property',
    'stone': 'special property',

    # Vacation properties.
    'vacation residence': 'vacation property',
    'cabin': 'vacation property',
    'refuge': 'vacation property',
    'maso': 'vacation property',
    'chalet': 'vacation property',

    # Commercial properties.
    'hotel': 'commercial',
    'hotel - accommodation': 'commercial',
    'office': 'commercial',
    'warehouse': 'commercial',
    'storage': 'commercial',
    'store': 'commercial',
    'shop': 'commercial',
    'garage - box': 'commercial',
    'parking place': 'commercial',
    'study': 'commercial',
    'bed & breakfast': 'commercial',

    # Land parcels.
    'residential land': 'land',
    'estate': 'land',

    # Small structures.
    'shed': 'small_structure',

    # Other/unusual properties.
    'project': 'other',
    'other': 'other',

    # Unknown kept as is for handling missing data.
    'unknown': 'unknown'
}


# Secondary mapping to further consolidate rare property types.
# This reduces the number of categories to 7 main types.
PROPERTY_SECOND_MAPPING: Dict[str, str] = {
    'apartment': 'apartment',
    'building': 'building',
    'villa': 'villa',
    'rural property': 'rural property',
    'terraced house': 'terraced house',
    'house': 'house',
    # All rare types mapped to 'other'.
    'unknown': 'other',
    'special property': 'other',
    'vacation property': 'other',
    'commercial': 'other',
    'land': 'other',
    'small structure': 'other',
    'other': 'other'
}


# Maps heating power sources to simplified categories.
# Combines synonyms and groups rare sources.
HEATING_POWER_MAPPING: Dict[str, str] = {
    # Synonyms grouped together.
    'photovoltaic': 'solar',
    'gpl': 'gas',

    # Rare sources grouped into 'other'.
    'diesel': 'other',
    'wood': 'other',
    'district_heating': 'other',
    'oil': 'other',
    'pellet': 'other',
    'pellets': 'other',

    # Common sources kept unchanged.
    'methane': 'methane',
    'power supply': 'power supply',
    'heat pump': 'heat pump',
    'gas': 'gas',
    'solar': 'solar',
    'unknown': 'unknown'
}


# Maps window glass types to simplified categories.
# Handles combinations when multiple glass types exist.
WINDOW_GLASS_MAPPING: Dict[str, str] = {
    'glass': 'glass',
    'double glass': 'double glass',
    'double glass, glass': 'double glass',
    'triple glass': 'triple glass',
    'double glass, triple glass': 'triple glass',
    'glass, triple glass': 'triple glass',
    'unknown': 'unknown'
}


# Column mappings for window frame processing.
# Each column maps to (glass_type, material).
WINDOW_COLUMN_MAPPINGS: Dict[str, Tuple[str, str]] = {
    'window frames double glass / metal exterior': ('double glass', 'metal'),
    'window frames exterior double glass / wood': ('double glass', 'wood'),
    'window frames exterior glass / metal': ('glass', 'metal'),
    'window frames exterior glass / wood': ('glass', 'wood'),
    'window frames exterior in triple glass / metal': ('triple glass', 'metal'),
    'window frames exterior in triple glass / pvc': ('triple glass', 'pvc'),
    'window frames exterior in triple glass / wood': ('triple glass', 'wood'),
    'window frames external double glass / pvc': ('double glass', 'pvc')
}


# List of all window frame columns in the dataset.
WINDOW_COLUMNS: List[str] = list(WINDOW_COLUMN_MAPPINGS.keys())


def extract_heating_components(value: str) -> Tuple[str, str, str]:
    """
    Parse heating_type string and extract type, delivery, and power.

    This function analyzes the heating_type field from real estate listings
    and extracts three components:
    - Type: centralized, independent, or unknown
    - Delivery: radiator, floor, stove, air, or unknown
    - Power: methane, gas, solar, heat pump, etc.

    The heating_type field in the original data is a complex string like:
    "independent, to radiators, methane powered"

    Args:
        value: The heating_type string to parse.

    Returns:
        Tuple of (heating_type, heating_delivery, heating_power).

    Example:
        >>> extract_heating_components("independent, to radiators, methane powered")
        ('independent', 'radiator', 'methane')
        >>> extract_heating_components("unknown")
        ('unknown', 'unknown', 'unknown')
    """
    # Default values when parsing fails or value is unknown.
    heating_type = 'unknown'
    heating_delivery = 'unknown'
    heating_power = 'unknown'

    # Handles the 'unknown' case immediately.
    if value == 'unknown':
        return heating_type, heating_delivery, heating_power

    # Extracts type based on prefix.
    if value.startswith('centralized'):
        heating_type = 'centralized'
    elif value.startswith('independent'):
        heating_type = 'independent'

    # Splits by comma to get individual components.
    parts = [part.strip() for part in value.split(',')]

    # Extracts delivery method using regex patterns.
    for part in parts:
        # Checks for floor-based heating.
        if re.search(r'\b(floor|flooring|floored|floor-mounted)\b', part, re.IGNORECASE):
            heating_delivery = 'floor'
            break
        # Checks for radiator-based heating.
        elif re.search(r'\bradiators?\b', part, re.IGNORECASE) or 'to radiators' in part.lower():
            heating_delivery = 'radiator'
            break
        # Checks for stove-based heating.
        elif re.search(r'\bstove\b', part, re.IGNORECASE):
            heating_delivery = 'stove'
            break
        # Checks for air-based heating (excluding power-related mentions).
        elif 'air' in part.lower() and not any(
            power in part.lower() for power in ['power', 'methane', 'pellet', 'solar']
        ):
            heating_delivery = 'air'
            break

    # Extracts power source using multiple patterns.
    power_found = False

    # Pattern 1: "powered by X".
    for part in parts:
        match = re.search(r'powered by ([\w\s-]+)', part, re.IGNORECASE)
        if match:
            heating_power = match.group(1).strip()
            power_found = True
            break

    # Pattern 2: "X powered".
    if not power_found:
        for part in parts:
            match = re.search(r'([\w\s-]+) powered', part, re.IGNORECASE)
            # Excludes "air powered" as it's usually a delivery method.
            if match and 'air' not in match.group(1).lower():
                heating_power = match.group(1).strip()
                power_found = True
                break

    # Pattern 3: Specific phrases.
    if not power_found:
        for part in parts:
            if 'fueled by oil' in part.lower():
                heating_power = 'oil'
                power_found = True
                break
            elif 'fed to gpl' in part.lower():
                heating_power = 'gpl'
                power_found = True
                break

    # Pattern 4: Direct mentions of power sources.
    if not power_found:
        power_sources = [
            'methane', 'power supply', 'heat pump', 'gpl', 'gas',
            'gasoline', 'solar', 'photovoltaic', 'diesel', 'wood',
            'oil', 'pellet', 'district heating'
        ]

        for part in parts:
            part_lower = part.lower()
            for source in power_sources:
                if source in part_lower:
                    heating_power = source
                    power_found = True
                    break
            if power_found:
                break

    return heating_type, heating_delivery, heating_power


def extract_air_conditioning_components(value: str) -> Tuple[str, int, int]:
    """
    Extract air conditioning type, hot indicator, and cold indicator.

    This function parses the air_conditioning field and extracts:
    - Type: centralized, independent, absent, predisposition, or unknown
    - Hot: 1 if provides hot air, 0 otherwise
    - Cold: 1 if provides cold air, 0 otherwise

    Args:
        value: The air_conditioning string to parse.

    Returns:
        Tuple of (ac_type, ac_hot, ac_cold).

    Example:
        >>> extract_air_conditioning_components("independent, hot and cold")
        ('independent', 1, 1)
        >>> extract_air_conditioning_components("absent")
        ('absent', 0, 0)
    """
    # Default values.
    ac_type = "unknown"
    ac_hot = 0
    ac_cold = 0

    # If value is unknown, returns defaults.
    if value == 'unknown':
        return ac_type, ac_hot, ac_cold

    value_lower = value.lower()

    # Extracts AC type.
    if 'centralized' in value_lower:
        ac_type = 'centralized'
    elif 'independent' in value_lower:
        ac_type = 'independent'
    elif 'absent' in value_lower:
        ac_type = 'absent'
    elif 'predisposition' in value_lower or 'plant predisposition' in value_lower:
        ac_type = 'predisposition'

    # Checks for hot capability.
    if any(term in value_lower for term in ['hot', 'warm']):
        ac_hot = 1

    # Checks for cold capability.
    if 'cold' in value_lower:
        ac_cold = 1

    return ac_type, ac_hot, ac_cold


def extract_window_info(row: pd.Series) -> Tuple[str, str]:
    """
    Extract window glass type and material from boolean columns.

    This function consolidates 8 window frame columns into 2 features:
    - Glass type: glass, double glass, or triple glass
    - Material: wood, metal, pvc, or combination

    Args:
        row: A pandas Series representing one row of the DataFrame.

    Returns:
        Tuple of (glass_type, material).

    Example:
        >>> # Row with 'window frames double glass / metal exterior' = 1
        >>> extract_window_info(row)
        ('double glass', 'metal')
    """
    # Gets columns that have a value of 1 (feature present).
    active_columns = [col for col in WINDOW_COLUMNS if col in row.index and row[col] == 1]

    # If no window features present, returns unknown.
    if not active_columns:
        return 'unknown', 'unknown'

    # Extracts glass types and materials from active columns.
    glass_types = set(WINDOW_COLUMN_MAPPINGS[col][0] for col in active_columns)
    materials = set(WINDOW_COLUMN_MAPPINGS[col][1] for col in active_columns)

    # Joins with commas if there are multiple values.
    glass_type_str = ', '.join(sorted(glass_types))
    material_str = ', '.join(sorted(materials))

    return glass_type_str, material_str


def apply_property_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply two-stage property type simplification to a DataFrame.

    This function applies PROPERTY_MAPPING first, then PROPERTY_SECOND_MAPPING
    to consolidate 40+ property types into 7 main categories.

    Args:
        df: DataFrame with 'type_of_property' column.

    Returns:
        DataFrame with simplified type_of_property values.

    Example:
        >>> df = apply_property_mapping(df)
        >>> df['type_of_property'].unique()
        ['apartment', 'villa', 'house', 'building', 'rural property',
         'terraced house', 'other']
    """
    result = df.copy()

    if 'type_of_property' not in result.columns:
        logger.warning("Column 'type_of_property' not found in DataFrame")
        return result

    # First mapping: detailed to intermediate categories.
    result['type_of_property'] = result['type_of_property'].map(PROPERTY_MAPPING)

    # Second mapping: intermediate to final categories.
    result['type_of_property'] = result['type_of_property'].map(PROPERTY_SECOND_MAPPING)

    logger.info("Applied property type mappings")
    return result


def apply_heating_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract heating components and replace heating_type column.

    This function extracts heating type, delivery, and power from the
    heating_type column and places the new columns in its position.

    Args:
        df: DataFrame with 'heating_type' column.

    Returns:
        DataFrame with heating_type replaced by heating_type, heating_delivery,
        and heating_power columns.

    Example:
        >>> df = apply_heating_extraction(df)
        >>> df.columns
        [..., 'heating_type', 'heating_delivery', 'heating_power', ...]
    """
    result = df.copy()

    if 'heating_type' not in result.columns:
        logger.warning("Column 'heating_type' not found in DataFrame")
        return result

    # Applies the extraction function to each row.
    extracted = pd.DataFrame(
        result['heating_type'].apply(extract_heating_components).tolist(),
        index=result.index,
        columns=['heating_type_new', 'heating_delivery', 'heating_power']
    )

    # Reorders columns to place new columns after original heating_type.
    cols = list(result.columns)
    heating_type_idx = cols.index('heating_type')

    # Inserts new columns after heating_type.
    result['heating_type_new'] = extracted['heating_type_new']
    result['heating_delivery'] = extracted['heating_delivery']
    result['heating_power'] = extracted['heating_power']

    cols.insert(heating_type_idx + 1, 'heating_type_new')
    cols.insert(heating_type_idx + 2, 'heating_delivery')
    cols.insert(heating_type_idx + 3, 'heating_power')

    # Removes old duplicates from column list.
    cols = [c for i, c in enumerate(cols) if c not in cols[:i]]
    result = result[cols]

    # Removes original and renames.
    result = result.drop('heating_type', axis=1)
    result = result.rename(columns={'heating_type_new': 'heating_type'})

    # Applies heating power mapping to consolidate rare sources.
    result['heating_power'] = result['heating_power'].map(HEATING_POWER_MAPPING)

    logger.info("Extracted heating components")
    return result


def apply_air_conditioning_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract air conditioning components and replace air_conditioning column.

    This function extracts AC type, hot, and cold indicators from the
    air_conditioning column and places the new columns in its position.

    Args:
        df: DataFrame with 'air_conditioning' column.

    Returns:
        DataFrame with air_conditioning replaced by air_conditioning_type,
        air_conditioning_hot, and air_conditioning_cold columns.
    """
    result = df.copy()

    if 'air_conditioning' not in result.columns:
        logger.warning("Column 'air_conditioning' not found in DataFrame")
        return result

    # Applies the extraction function to each row.
    extracted = pd.DataFrame(
        result['air_conditioning'].apply(extract_air_conditioning_components).tolist(),
        index=result.index,
        columns=['air_conditioning_type', 'air_conditioning_hot', 'air_conditioning_cold']
    )

    # Reorders columns to place new columns after original air_conditioning.
    cols = list(result.columns)
    ac_idx = cols.index('air_conditioning')

    # Inserts new columns after air_conditioning.
    result['air_conditioning_type'] = extracted['air_conditioning_type']
    result['air_conditioning_hot'] = extracted['air_conditioning_hot']
    result['air_conditioning_cold'] = extracted['air_conditioning_cold']

    cols.insert(ac_idx + 1, 'air_conditioning_type')
    cols.insert(ac_idx + 2, 'air_conditioning_hot')
    cols.insert(ac_idx + 3, 'air_conditioning_cold')

    # Removes old duplicates from column list.
    cols = [c for i, c in enumerate(cols) if c not in cols[:i]]
    result = result[cols]

    # Removes original column.
    result = result.drop('air_conditioning', axis=1)

    logger.info("Extracted air conditioning components")
    return result


def apply_window_frame_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate window frame columns into glass type and material.

    This function replaces 8 window frame boolean columns with 2 categorical
    columns: window_frames_glass and window_frames_material.

    Args:
        df: DataFrame with window frame boolean columns.

    Returns:
        DataFrame with window frame columns replaced by window_frames_glass
        and window_frames_material.
    """
    result = df.copy()

    # Checks if window columns exist.
    existing_window_cols = [col for col in WINDOW_COLUMNS if col in result.columns]
    if not existing_window_cols:
        logger.warning("No window frame columns found in DataFrame")
        return result

    # Applies the function to create the new columns.
    glass_material_info = result.apply(extract_window_info, axis=1, result_type='expand')
    result['window_frames_glass'] = glass_material_info[0]
    result['window_frames_material'] = glass_material_info[1]

    # Applies glass type mapping to consolidate.
    result['window_frames_glass'] = result['window_frames_glass'].map(WINDOW_GLASS_MAPPING)

    # Drops the original window frame columns.
    result = result.drop(columns=existing_window_cols)

    logger.info("Extracted window frame features")
    return result


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to a DataFrame.

    This function applies all feature engineering steps in order:
    1. Property type mapping
    2. Drop building_usage (mostly unknown values)
    3. Heating extraction
    4. Air conditioning extraction
    5. Window frame extraction
    6. Drop kitchen column (superfluous)
    7. Drop listing_id (irrelevant for prediction)

    Args:
        df: Raw DataFrame from synthetic data.

    Returns:
        DataFrame with all features engineered.

    Example:
        >>> engineered_df = engineer_all_features(raw_df)
    """
    result = df.copy()

    # Step 1: Apply property type mapping.
    result = apply_property_mapping(result)

    # Step 2: Drop building_usage (mostly unknown).
    if 'building_usage' in result.columns:
        result = result.drop('building_usage', axis=1)
        logger.info("Dropped building_usage column")

    # Step 3: Extract heating components.
    result = apply_heating_extraction(result)

    # Step 4: Extract air conditioning components.
    result = apply_air_conditioning_extraction(result)

    # Step 5: Extract window frame features.
    result = apply_window_frame_extraction(result)

    # Step 6: Drop kitchen column (superfluous).
    if 'kitchen' in result.columns:
        result = result.drop('kitchen', axis=1)
        logger.info("Dropped kitchen column")

    # Step 7: Drop listing_id (irrelevant for prediction).
    if 'listing_id' in result.columns:
        result = result.drop('listing_id', axis=1)
        logger.info("Dropped listing_id column")

    logger.info(f"Feature engineering complete. Shape: {result.shape}")
    return result

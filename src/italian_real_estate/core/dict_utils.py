"""
Dictionary and JSON utilities for the Italian Real Estate pipeline.

This module provides functions for converting dictionary keys between
different formats (tuples to strings and vice versa) to support JSON
serialization. It also includes custom JSON encoders for handling
datetime objects.

These utilities are essential for the migration pipeline which uses
Airflow's XCom mechanism for passing data between tasks. XCom requires
JSON-serializable data, but our dimension mappings use tuple keys for
efficiency.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import json
import datetime
from typing import Any, Dict, Union


class DateEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles datetime.date objects.

    The standard json module cannot serialize datetime objects. This encoder
    converts date objects to a dictionary format that can be deserialized
    back to dates using the decode_date_dict function.

    The encoded format is:
        {"__date__": True, "year": 2024, "month": 1, "day": 15}

    This class is used with Airflow's XCom mechanism to pass dimension
    mappings between tasks, where tuple keys may contain date objects.

    Example:
        >>> import json
        >>> from datetime import date
        >>> data = {"date": date(2024, 1, 15)}
        >>> json.dumps(data, cls=DateEncoder)
        '{"date": {"__date__": true, "year": 2024, "month": 1, "day": 15}}'
    """

    def default(self, obj: Any) -> Any:
        """
        Override the default method to handle datetime.date objects.

        This method is called by the JSON encoder when it encounters an
        object it doesn't know how to serialize. We handle datetime.date
        objects and delegate all other objects to the parent class.

        Args:
            obj: The object to serialize.

        Returns:
            A JSON-serializable representation of the object.

        Raises:
            TypeError: If the object is not JSON-serializable and not a date.
        """
        # Check if the object is a datetime.date instance
        if isinstance(obj, datetime.date):
            # Return a dictionary with a marker and the date components.
            # The "__date__" marker allows us to identify this as a date
            # during deserialization.
            return {
                "__date__": True,
                "year": obj.year,
                "month": obj.month,
                "day": obj.day
            }

        # For all other types, use the default behavior
        # This will raise TypeError for non-serializable objects
        return super().default(obj)


def decode_date_dict(obj: Dict[str, Any]) -> Union[datetime.date, Dict[str, Any]]:
    """
    Decode a dictionary back to a datetime.date object if it's a date encoding.

    This is the counterpart to DateEncoder, used during JSON deserialization
    to convert date dictionaries back to actual datetime.date objects.

    Args:
        obj: A dictionary that may be an encoded date or a regular dictionary.

    Returns:
        datetime.date if the dictionary is an encoded date (has "__date__": True),
        otherwise returns the dictionary unchanged.

    Example:
        >>> decode_date_dict({"__date__": True, "year": 2024, "month": 1, "day": 15})
        datetime.date(2024, 1, 15)
        >>> decode_date_dict({"name": "test"})
        {"name": "test"}
    """
    # Check if this dictionary is an encoded date
    if isinstance(obj, dict) and obj.get("__date__") is True:
        # Reconstruct the date object from its components
        return datetime.date(obj["year"], obj["month"], obj["day"])

    return obj


def convert_dict_keys_to_tuple(d: Dict[str, Any]) -> Dict[tuple, Any]:
    """
    Convert string keys that represent JSON arrays back to tuple keys.

    This function reverses the conversion done by convert_dict_keys_to_string.
    It parses string keys that look like JSON arrays (e.g., "[1, 2, 3]")
    and converts them back to tuples.

    This is necessary when receiving dimension mappings from Airflow XCom
    and needing to restore the original tuple keys for database operations.

    Args:
        d: A dictionary with string keys that may represent tuples.

    Returns:
        A new dictionary with tuple keys where applicable.

    Example:
        >>> data = {'["sale", "2024-01-15"]': 1, "regular_key": 2}
        >>> convert_dict_keys_to_tuple(data)
        {("sale", datetime.date(2024, 1, 15)): 1, "regular_key": 2}
    """
    # Return non-dict inputs unchanged
    if not isinstance(d, dict):
        return d

    result = {}

    for key, value in d.items():
        try:
            # Check if the key looks like a JSON array (starts with [ and ends with ])
            if isinstance(key, str) and key.startswith('[') and key.endswith(']'):
                # Parse the JSON array
                parsed = json.loads(key)

                if isinstance(parsed, list):
                    # Process the list items to convert any date dictionaries
                    processed_items = []
                    for item in parsed:
                        # Check if this item is an encoded date
                        if isinstance(item, dict) and item.get('__date__') is True:
                            # Convert back to date object
                            processed_items.append(
                                datetime.date(item['year'], item['month'], item['day'])
                            )
                        else:
                            processed_items.append(item)

                    # Convert the list to a tuple and use as the key
                    result[tuple(processed_items)] = value
                else:
                    # If parsing didn't produce a list, keep the original key
                    result[key] = value
            else:
                # Not a JSON array string, keep the original key
                result[key] = value

        except (json.JSONDecodeError, TypeError):
            # If there's an error parsing, keep the original key
            result[key] = value

    return result


def convert_dict_keys_to_string(d: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Convert tuple keys in a dictionary to string keys for JSON serialization.

    Tuple keys cannot be serialized to JSON directly. This function converts
    them to JSON array strings that can be serialized and later converted
    back to tuples using convert_dict_keys_to_tuple.

    This is used when storing dimension mappings in Airflow XCom, which
    requires JSON-serializable data.

    Args:
        d: A dictionary that may contain tuple keys.

    Returns:
        A new dictionary where all tuple keys have been converted to
        JSON array strings.

    Example:
        >>> from datetime import date
        >>> data = {("sale", date(2024, 1, 15)): 1}
        >>> convert_dict_keys_to_string(data)
        {'["sale", {"__date__": true, "year": 2024, "month": 1, "day": 15}]': 1}
    """
    # Return non-dict inputs unchanged
    if not isinstance(d, dict):
        return d

    result = {}

    for key, value in d.items():
        if isinstance(key, tuple):
            # Convert the tuple to a JSON string using our custom encoder
            # to handle any date objects within the tuple
            string_key = json.dumps(list(key), cls=DateEncoder)
            result[string_key] = value
        else:
            # Keep non-tuple keys unchanged
            result[key] = value

    return result


def convert_nested_dict_keys_to_tuple(d: Dict[str, Any]) -> Dict[Any, Any]:
    """
    Recursively convert string keys to tuples in nested dictionaries.

    This function handles nested dictionary structures where values may
    also be dictionaries containing string keys that should be converted
    to tuples.

    Args:
        d: A potentially nested dictionary with string keys.

    Returns:
        A new dictionary with tuple keys at all nesting levels.

    Example:
        >>> data = {"outer": {'["a", "b"]': {"inner": 1}}}
        >>> convert_nested_dict_keys_to_tuple(data)
        {"outer": {("a", "b"): {"inner": 1}}}
    """
    if not isinstance(d, dict):
        return d

    result = {}

    for key, value in d.items():
        # Recursively process nested dictionaries
        if isinstance(value, dict):
            result[key] = convert_nested_dict_keys_to_tuple(value)
        else:
            result[key] = value

    # Convert the keys at this level
    return convert_dict_keys_to_tuple(result)


def convert_nested_dict_keys_to_string(d: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Recursively convert tuple keys to strings in nested dictionaries.

    This is the inverse of convert_nested_dict_keys_to_tuple, used when
    preparing nested data structures for JSON serialization.

    Args:
        d: A potentially nested dictionary with tuple keys.

    Returns:
        A new dictionary with string keys at all nesting levels.

    Example:
        >>> data = {"outer": {("a", "b"): {"inner": 1}}}
        >>> convert_nested_dict_keys_to_string(data)
        {"outer": {'["a", "b"]': {"inner": 1}}}
    """
    if not isinstance(d, dict):
        return d

    result = {}

    for key, value in d.items():
        # Recursively process nested dictionaries
        if isinstance(value, dict):
            result[key] = convert_nested_dict_keys_to_string(value)
        else:
            result[key] = value

    # Convert the keys at this level
    return convert_dict_keys_to_string(result)


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely navigate through nested dictionaries.

    This utility function provides null-safe access to nested dictionary
    values, which is useful when processing MongoDB documents where fields
    may or may not be present.

    Args:
        d: The dictionary to navigate.
        *keys: The sequence of keys to follow.
        default: The value to return if any key is missing. Defaults to None.

    Returns:
        The value at the specified path, or the default if not found.

    Example:
        >>> data = {"a": {"b": {"c": 1}}}
        >>> safe_get(data, "a", "b", "c")
        1
        >>> safe_get(data, "a", "x", "y", default="missing")
        'missing'
    """
    current = d

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later dictionaries taking precedence.

    This is useful for combining default settings with user overrides or
    merging partial data updates.

    Args:
        *dicts: Two or more dictionaries to merge.

    Returns:
        A new dictionary containing all key-value pairs from the inputs.
        For duplicate keys, values from later dictionaries overwrite earlier ones.

    Example:
        >>> defaults = {"a": 1, "b": 2}
        >>> overrides = {"b": 3, "c": 4}
        >>> merge_dicts(defaults, overrides)
        {"a": 1, "b": 3, "c": 4}
    """
    result = {}

    for d in dicts:
        if d:
            result.update(d)

    return result


# Aliases for backward compatibility and alternative naming conventions.
# These aliases provide shorter or alternative names for the main functions.
convert_nested_dict_tuple_keys_to_str = convert_nested_dict_keys_to_string
convert_nested_dict_str_keys_to_tuple = convert_nested_dict_keys_to_tuple

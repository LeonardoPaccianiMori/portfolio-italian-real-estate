"""
Listing parsing utilities for the Italian Real Estate pipeline.

This module provides functions for parsing and validating real estate
listing pages scraped from listing.website. It handles detection of
sub-listings, identification of auction listings, and extraction of
metadata from the HTML source.

The parsing functions use BeautifulSoup for HTML parsing and extract
data from embedded JSON metadata in the pages.

Author: Leonardo Pacciani-Mori
License: MIT
"""

import json
from typing import Optional, Dict, Any, List

from bs4 import BeautifulSoup

from italian_real_estate.config.logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)

BLOCK_MARKERS = (
    "captcha-delivery.com",
    "geo.captcha-delivery.com",
    "datadome",
    "access blocked",
    "captcha",
)


def has_sublistings(html_source: str) -> bool:
    """
    Check if a listing page contains sub-listings.

    Some property listings on listing.website represent multiple properties
    (e.g., multiple apartments in the same building). This function checks
    the page metadata to determine if sub-listings exist.

    A listing has sub-listings when the number of properties shown is
    greater than one. Sub-listings need to be processed separately and
    linked to the parent listing.

    Args:
        html_source: The HTML source code of the listing page.

    Returns:
        bool: True if the listing has sub-listings (more than one property),
            False if it's a single property or if the check fails.

    Example:
        >>> html = fetch_listing_page(url)
        >>> if has_sublistings(html):
        ...     # Need to expand and process sub-listings
        ...     process_parent_with_children(html)
        >>> else:
        ...     # Simple single listing
        ...     process_single_listing(html)
    """
    try:
        # Parse the HTML source code with BeautifulSoup
        soup = BeautifulSoup(html_source, 'html.parser')

        # Find the script tag containing the page metadata.
        script = soup.find("script", id="__NEXT_DATA__")

        if not script or not script.contents:
            return False

        script_content = script.contents[0].replace("\n", "").replace("   ", "").replace("  ", "")

        # Parse the metadata JSON
        data_dict = json.loads(script_content)

        # Extract the number of properties from the metadata.
        number_of_properties = len(
            data_dict["props"]["pageProps"]["detailData"]["realEstate"]["properties"]
        )

        return number_of_properties > 1

    except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
        # If any error occurs during parsing, assume no sub-listings.
        # This is a safe default that prevents processing errors.
        logger.debug(f"Error checking for sub-listings: {str(e)}")
        return False


def is_auction_listing(html_source: str) -> bool:
    """
    Check if a listing is an auction listing.

    When scraping "sale" listings from listing.website, the search results
    can sometimes include auction listings mixed in. This function checks
    the page metadata to identify auction listings so they can be filtered
    out or processed separately.

    Auction listings have different data structures and should be stored
    in the auction collection, not the sale collection.

    Args:
        html_source: The HTML source code of the listing page.
            Can be None (returns False in this case).

    Returns:
        bool: True if the listing is an auction, False if it's a regular
            sale listing or if the check fails. Returns False for None input.

    Example:
        >>> html = fetch_listing_page(url)
        >>> if is_auction_listing(html):
        ...     # Skip or redirect to auction processing
        ...     logger.info("Skipping auction listing in sale scrape")
        ...     return
    """
    # Handle None input gracefully
    if html_source is None:
        return False

    try:
        # Parse the HTML source code with BeautifulSoup
        soup = BeautifulSoup(html_source, 'html.parser')

        # Find the script tag containing the page metadata.
        script = soup.find("script", id="__NEXT_DATA__")

        if not script or not script.contents:
            # Can't determine - default to True to be safe (skip uncertain listings)
            return True

        script_content = script.contents[0].replace("\n", "").replace("   ", "").replace("  ", "")

        # Parse the metadata JSON
        data_dict = json.loads(script_content)

        # Navigate to the property data.
        detail_data = data_dict.get('props', {}).get('pageProps', {}).get('detailData', {})
        real_estate = detail_data.get('realEstate', {})
        properties = real_estate.get('properties', [])

        # Check if the 'auction' key exists in the first property
        if properties and isinstance(properties, list) and len(properties) > 0:
            if isinstance(properties[0], dict):
                return 'auction' in properties[0].keys()

        # Default to True (safer to skip uncertain listings)
        return True

    except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
        # If any error occurs, default to True to be safe
        logger.debug(f"Error checking for auction listing: {str(e)}")
        return True


def has_valid_detail_data(html_source: str) -> bool:
    """
    Check if a listing page contains valid detail data.

    Sometimes a fetched page doesn't contain the expected listing data -
    this can happen when a listing is removed, when there's a server error,
    or when the URL doesn't point to a valid listing.

    This function validates that the page contains the expected metadata
    structure before attempting to process it.

    Args:
        html_source: The HTML source code of the listing page.

    Returns:
        bool: True if the page contains valid detail data, False otherwise.

    Example:
        >>> html = fetch_listing_page(url)
        >>> if not has_valid_detail_data(html):
        ...     logger.warning(f"Invalid listing page: {url}")
        ...     skip_listing(listing_id)
    """
    if not html_source:
        return False

    try:
        soup = BeautifulSoup(html_source, 'html.parser')

        # Find the metadata script tag (prefer Next.js payload).
        script = soup.find("script", id="__NEXT_DATA__")
        if not script:
            script = soup.find('script', {'type': 'application/json'})

        if not script:
            return False

        script_text = None
        if script.string:
            script_text = script.string
        elif script.contents:
            script_text = script.contents[0]

        if not script_text:
            return False

        # Parse and validate the JSON structure
        data_dict = json.loads(script_text)

        # Check for the expected data structure
        page_props = data_dict.get('props', {}).get('pageProps', {})

        if 'detailData' not in page_props:
            return False

        return True

    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        return False


def block_markers_in_html(html_source: Optional[str]) -> List[str]:
    if not html_source:
        return []
    lowered = html_source.lower()
    return [marker for marker in BLOCK_MARKERS if marker in lowered]


def has_next_data_script(html_source: Optional[str]) -> bool:
    if not html_source:
        return False
    try:
        soup = BeautifulSoup(html_source, 'html.parser')
        script = soup.find("script", id="__NEXT_DATA__")
        if not script:
            return False
        return bool(script.string or script.contents)
    except Exception:
        return False


def get_page_title(html_source: Optional[str]) -> Optional[str]:
    if not html_source:
        return None
    try:
        soup = BeautifulSoup(html_source, 'html.parser')
        if soup.title and soup.title.string:
            return " ".join(soup.title.string.split())
        return None
    except Exception:
        return None


def is_blocked_page(html_source: Optional[str]) -> bool:
    """
    Detect common anti-bot or access blocked pages.

    Args:
        html_source: HTML source code for a fetched page.

    Returns:
        bool: True if the HTML looks like a block/captcha page.
    """
    if not html_source:
        return False

    return bool(block_markers_in_html(html_source))


def extract_listing_metadata(html_source: str) -> Optional[Dict[str, Any]]:
    """
    Extract the metadata dictionary from a listing page.

    This function parses the embedded JSON metadata from a listing page
    and returns it as a Python dictionary. The metadata contains all the
    structured information about the property.

    Args:
        html_source: The HTML source code of the listing page.

    Returns:
        dict or None: The parsed metadata dictionary, or None if extraction
            fails. The dictionary structure matches the website's internal
            data format.

    Example:
        >>> html = fetch_listing_page(url)
        >>> metadata = extract_listing_metadata(html)
        >>> if metadata:
        ...     price = metadata['props']['pageProps']['detailData']['realEstate']['price']
    """
    if not html_source:
        return None

    try:
        soup = BeautifulSoup(html_source, 'html.parser')

        # Find the metadata script tag (prefer Next.js payload).
        script = soup.find("script", id="__NEXT_DATA__")
        if not script:
            script = soup.find('script', {'type': 'application/json'})

        if not script:
            return None

        script_text = None
        if script.string:
            script_text = script.string
        elif script.contents:
            script_text = script.contents[0]

        if not script_text:
            return None

        # Parse and return the JSON
        return json.loads(script_text)

    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.debug(f"Error extracting metadata: {str(e)}")
        return None


def extract_listing_id_from_url(url: str) -> Optional[int]:
    """
    Extract the listing ID from a listing URL.

    Listing URLs on listing.website follow a pattern where the listing ID
    is the second-to-last segment of the path. For example:
    https://www.listing.website/annunci/12345678/ -> 12345678

    Args:
        url: The listing URL to parse.

    Returns:
        int or None: The extracted listing ID, or None if extraction fails.

    Example:
        >>> extract_listing_id_from_url("https://www.listing.website/annunci/12345678/")
        12345678
    """
    if not url:
        return None

    try:
        # Split URL and get the second-to-last segment
        parts = url.rstrip('/').split('/')
        listing_id = int(parts[-1]) if parts[-1].isdigit() else int(parts[-2])
        return listing_id
    except (ValueError, IndexError):
        return None


def extract_sublisting_links(soup: BeautifulSoup) -> List[str]:
    """
    Extract sub-listing links from a parsed listing page.

    After expanding sub-listings using Selenium, this function extracts
    the individual property links from the page.

    Args:
        soup: A BeautifulSoup object of the expanded listing page.

    Returns:
        list: A list of URLs to sub-listings.

    Example:
        >>> html = expand_sublistings_with_selenium(url)
        >>> soup = BeautifulSoup(html, 'html.parser')
        >>> sublisting_urls = extract_sublisting_links(soup)
    """
    links = []

    # Find all sub-listing link elements
    link_elements = soup.find_all(
        "a",
        class_="re-realEstateUnitsOld__item re-realEstateUnitsOld__item--block"
    )

    for element in link_elements:
        href = element.get('href')
        if href:
            links.append(href)

    return links


def extract_sublisting_ids(links: List[str]) -> List[int]:
    """
    Extract listing IDs from a list of sub-listing URLs.

    This convenience function applies extract_listing_id_from_url to a
    list of URLs and filters out any that couldn't be parsed.

    Args:
        links: A list of sub-listing URLs.

    Returns:
        list: A list of integer listing IDs. Failed extractions are omitted.

    Example:
        >>> urls = ["https://example.com/12345/", "https://example.com/67890/"]
        >>> ids = extract_sublisting_ids(urls)
        >>> print(ids)
        [12345, 67890]
    """
    ids = []

    for link in links:
        listing_id = extract_listing_id_from_url(link)
        if listing_id is not None:
            ids.append(listing_id)

    return ids


def parse_page_count(html_source: str) -> int:
    """
    Parse the number of pages from a search results page.

    When scraping listings by price range, we need to know how many pages
    of results exist. This function extracts that count from the pagination
    elements on the search results page.

    The website caps results at 80 pages. If more pages would exist, we
    need to narrow the price range to get all results.

    Args:
        html_source: The HTML source of a search results page.

    Returns:
        int: The number of pages in the search results. Returns 1 if
            the page count cannot be determined.

    Example:
        >>> search_html = fetch_search_results(province, price_range)
        >>> page_count = parse_page_count(search_html)
        >>> if page_count >= 80:
        ...     # Too many results, need to narrow price range
        ...     split_and_retry(price_range)
    """
    try:
        soup = BeautifulSoup(html_source, 'html.parser')

        # Look for pagination elements
        last_page_data = soup.find_all(
            "a",
            class_="in-paginationItem nd-button nd-button--ghost is-mobileHidden"
        )
        page_number_data = soup.find_all(
            "div",
            class_="nd-button nd-button--ghost is-disabled in-paginationItem is-mobileHidden"
        )

        if len(last_page_data) == 0:
            return 1

        if len(page_number_data) == 1 and len(last_page_data) >= 2:
            return int(last_page_data[-2].text)

        return int(page_number_data[-1].text)

    except (ValueError, IndexError, AttributeError):
        # Default to 1 page if parsing fails
        return 1


def extract_listing_links_from_search(soup: BeautifulSoup) -> List[str]:
    """
    Extract listing links from a search results page.

    This function parses a search results page and extracts all the
    individual listing URLs from the results.

    Args:
        soup: A BeautifulSoup object of the search results page.

    Returns:
        list: A list of listing URLs from the search results.

    Example:
        >>> search_html = fetch_search_page(url)
        >>> soup = BeautifulSoup(search_html, 'html.parser')
        >>> listing_urls = extract_listing_links_from_search(soup)
    """
    links = []
    seen = set()

    def _normalize_url(href: str) -> Optional[str]:
        if not href:
            return None
        if href.startswith("//"):
            href = f"https:{href}"
        elif href.startswith("/"):
            href = f"https://www.listing.website{href}"
        return href

    def _add_link(href: str) -> None:
        url = _normalize_url(href)
        if not url:
            return
        if extract_listing_id_from_url(url) is None:
            return
        if url in seen:
            return
        seen.add(url)
        links.append(url)

    selectors = [
        "a.in-listingCardTitle",
        "a.in-listingCard__title",
        "a[data-cy='listing-card-link']",
        "a[data-cy='listing-item-link']",
        "a[data-cy='card-link']",
    ]

    for selector in selectors:
        for element in soup.select(selector):
            _add_link(element.get("href"))

    if not links:
        for element in soup.find_all("a", href=True):
            href = element.get("href")
            if "/annunci/" in href:
                _add_link(href)

    return links

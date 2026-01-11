"""
Asynchronous HTTP client for the Italian Real Estate pipeline.

This module provides high-performance asynchronous HTTP functionality
for fetching web pages during the scraping process. It uses aiohttp for
async requests with configurable concurrency limits, timeouts, and retry
logic.

The module is designed to handle the high volume of requests needed to
scrape property listings while being respectful of the target server's
resources through rate limiting.

Key features:
    - Concurrent request limiting via semaphore
    - Configurable timeouts
    - Automatic retry on transient failures
    - Batch processing for large URL lists

Author: Leonardo Pacciani-Mori
License: MIT
"""

import asyncio
import random
import time
from typing import List, Optional, Tuple, Dict, Any

import aiohttp

from italian_real_estate.config.settings import (
    HTTP_SEMAPHORE_LIMIT,
    HTTP_TIMEOUT_SECONDS,
    HTTP_BATCH_SIZE,
    HTTP_CONNECTION_LIMIT,
    SCRAPING_HEADERS,
    SCRAPING_POLITE_MODE,
    SCRAPING_POLITE_MIN_WAIT,
    SCRAPING_POLITE_MAX_WAIT,
    SCRAPING_HUMANIZE,
    SCRAPING_HUMANIZE_MIN_WAIT,
    SCRAPING_HUMANIZE_MAX_WAIT,
)
from italian_real_estate.config.logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)

# Create a module-level semaphore for controlling concurrent requests.
# This limits the number of simultaneous HTTP requests to prevent
# overwhelming the target server and local resources.
semaphore = asyncio.Semaphore(HTTP_SEMAPHORE_LIMIT)


async def get_single_url(
    url: str,
    session: aiohttp.ClientSession,
    timeout: int = HTTP_TIMEOUT_SECONDS
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Fetch a single URL asynchronously with error handling.

    This function makes a single HTTP GET request using the provided session.
    It implements timeout handling and returns both the response content and
    a status dictionary for error tracking.

    The function uses a semaphore to limit concurrent requests, which is
    shared across all calls to prevent overwhelming the server.

    Args:
        url: The URL to fetch. Should be a complete URL including protocol.
        session: An active aiohttp ClientSession to use for the request.
            Sessions should be reused for connection pooling benefits.
        timeout: Request timeout in seconds. Defaults to HTTP_TIMEOUT_SECONDS
            from settings.

    Returns:
        Tuple containing:
            - str or None: The HTML content as a UTF-8 decoded string,
              or None if the request failed.
            - dict: Status information with keys:
              - 'status': HTTP status code or error type string
              - 'message': Human-readable status/error message

    Example:
        >>> async with aiohttp.ClientSession() as session:
        ...     content, status = await get_single_url("https://example.com", session)
        ...     if content:
        ...         print(f"Fetched {len(content)} bytes")
        ...     else:
        ...         print(f"Error: {status['message']}")
    """
    # Acquire the semaphore to limit concurrent requests.
    # This blocks if the limit is reached until another request completes.
    async with semaphore:
        try:
            if SCRAPING_POLITE_MODE or SCRAPING_HUMANIZE:
                if SCRAPING_POLITE_MODE:
                    min_wait = max(0.0, SCRAPING_POLITE_MIN_WAIT)
                    max_wait = max(min_wait, SCRAPING_POLITE_MAX_WAIT)
                else:
                    min_wait = max(0.0, SCRAPING_HUMANIZE_MIN_WAIT)
                    max_wait = max(min_wait, SCRAPING_HUMANIZE_MAX_WAIT)
                await asyncio.sleep(random.uniform(min_wait, max_wait))

            # Make the HTTP request with timeout and redirect handling
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                allow_redirects=True
            ) as response:

                logger.info(f"Fetching data for url {url}")

                # Check for HTTP errors (4xx and 5xx status codes)
                if response.status >= 400:
                    logger.error(f"Error {response.status} when fetching {url}")
                    return None, {
                        'status': response.status,
                        'message': f'Error {response.status} when fetching URL'
                    }

                # Read and decode the response body
                resp = await response.read()
                resp_string = resp.decode("utf-8")

                logger.info(f"Data fetched successfully for url {url}")
                return resp_string, {
                    'status': 200,
                    'message': 'Success'
                }

        # Handle connection-related errors
        except (
            aiohttp.ClientOSError,
            aiohttp.ClientConnectionError,
            aiohttp.client_exceptions.ClientConnectorError,
            aiohttp.ServerDisconnectedError
        ) as e:
            logger.error(f"ERROR: Connection issue for {url}: {str(e)}")
            return None, {
                'status': 'connection_error',
                'message': f'Connection error: {str(e)}'
            }

        # Handle timeout errors
        except asyncio.exceptions.TimeoutError:
            logger.error(f"ERROR: Timeout error for {url}")
            return None, {
                'status': 'timeout_error',
                'message': 'Timeout error'
            }

        # Handle response payload errors
        except aiohttp.client_exceptions.ClientPayloadError:
            logger.error(f"ERROR: Payload error for {url}")
            return None, {
                'status': 'payload_error',
                'message': 'Payload error'
            }


async def get_multiple_urls(
    urls: List[str],
    batch_size: int = HTTP_BATCH_SIZE,
    timeout: int = HTTP_TIMEOUT_SECONDS,
    retry_delay: int = 5
) -> List[Optional[str]]:
    """
    Fetch multiple URLs concurrently with batching and retry logic.

    This function efficiently fetches many URLs by processing them in batches
    and making concurrent requests within each batch. It dramatically speeds
    up data extraction compared to sequential requests.

    The function maintains the order of URLs in the results - if a URL fails
    to fetch, its position in the result list will contain None.

    Args:
        urls: List of URLs to fetch. Can be any length.
        batch_size: Number of URLs to process concurrently in each batch.
            Larger batches are faster but use more memory and connections.
            Defaults to HTTP_BATCH_SIZE from settings.
        timeout: Request timeout in seconds for each individual request.
            Defaults to HTTP_TIMEOUT_SECONDS from settings.
        retry_delay: Seconds to wait before retrying on batch-level failure.
            Defaults to 5 seconds.

    Returns:
        List of HTML content strings in the same order as the input URLs.
        Failed requests are represented as None in the list.

    Example:
        >>> urls = ["https://example.com/1", "https://example.com/2"]
        >>> results = await get_multiple_urls(urls)
        >>> for url, content in zip(urls, results):
        ...     if content:
        ...         print(f"{url}: {len(content)} bytes")
    """
    if not urls:
        return []

    all_responses = []

    try:
        # Process URLs in batches to avoid overwhelming resources
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(urls) - 1) // batch_size + 1

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_urls)} URLs)"
            )

            # Create a TCP connector with connection limiting and cleanup.
            # force_close=True ensures connections are properly closed after use.
            connector = aiohttp.TCPConnector(
                force_close=True,
                limit=HTTP_CONNECTION_LIMIT
            )

            # Create a session for this batch
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers=SCRAPING_HEADERS,
            ) as session:
                # Create async tasks for all URLs in the batch
                tasks = [get_single_url(url, session, timeout) for url in batch_urls]

                # Execute all tasks concurrently and wait for completion
                results = await asyncio.gather(*tasks)

                # Process results, keeping track of which requests failed.
                # We need to maintain ordering to match URLs to their results.
                batch_responses = []
                for response, status in results:
                    batch_responses.append(response)

                all_responses.extend(batch_responses)

        return all_responses

    except Exception as error:
        # Log the error and retry after a delay
        logger.error(f"Error in get_multiple_urls: {str(error)}")
        logger.error(f"Retrying in {retry_delay}s...")
        time.sleep(retry_delay)

        # Recursive retry - in production, you might want to limit retries
        return await get_multiple_urls(urls, batch_size, timeout, retry_delay)


async def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    retry_delay: int = 2,
    timeout: int = HTTP_TIMEOUT_SECONDS
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Fetch a single URL with automatic retry on failure.

    This is a wrapper around get_single_url that adds retry logic for
    transient failures. It's useful for critical requests where you want
    to maximize the chance of success.

    Args:
        url: The URL to fetch.
        max_retries: Maximum number of retry attempts. Defaults to 3.
        retry_delay: Seconds to wait between retries. Defaults to 2.
        timeout: Request timeout in seconds.

    Returns:
        Same as get_single_url: tuple of (content, status_dict).

    Example:
        >>> content, status = await fetch_with_retry("https://example.com")
        >>> if content:
        ...     process(content)
    """
    last_error = None

    for attempt in range(max_retries):
        # Create a temporary connector and session for this attempt
        connector = aiohttp.TCPConnector(force_close=True, limit=25)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout),
            headers=SCRAPING_HEADERS,
        ) as session:
            content, status = await get_single_url(url, session, timeout)

            if content is not None:
                return content, status

            last_error = status

            # Only sleep if we're going to retry
            if attempt < max_retries - 1:
                logger.info(
                    f"Retry {attempt + 1}/{max_retries} for {url} after {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)

    return None, last_error


def fetch_url_sync(url: str, timeout: int = HTTP_TIMEOUT_SECONDS) -> Optional[str]:
    """
    Synchronous wrapper for fetching a single URL.

    This convenience function allows fetching a URL from synchronous code
    by running the async function in an event loop. Use this for simple
    cases where you need to fetch one URL without managing async context.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        The HTML content as a string, or None if the request failed.

    Example:
        >>> content = fetch_url_sync("https://example.com")
        >>> if content:
        ...     print(f"Got {len(content)} bytes")
    """
    async def _fetch():
        """Coroutine wrapper that proxies the retrying fetch for synchronous use."""
        content, _ = await fetch_with_retry(url, timeout=timeout)
        return content

    return asyncio.run(_fetch())

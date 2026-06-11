"""HTTP and pagination utilities for Property Finder result pages."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from src.cleaner import analysis_listings, clean_listings
from src.storage import (
    append_clean_rows,
    append_raw_jsonl,
    append_scrape_log,
    clear_data,
    load_clean_data,
    load_scrape_log,
    load_seen_keys,
)


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0 Safari/537.36"
)


@dataclass
class FetchResult:
    url: str
    status_code: int
    html: str
    page_number: int = 1


@dataclass
class ScrapeSummary:
    estimated_total_listings: int | None = None
    estimated_total_pages: int | None = None
    pages_requested: int = 0
    pages_scraped: int = 0
    unique_listings_collected: int = 0
    duplicate_listings_skipped: int = 0
    failed_pages: int = 0
    invalid_rows_removed: int = 0
    clean_rows_ready: int = 0
    saved_path: str = ""
    stop_reason: str = "Not started"


def fetch_search_page(
    url: str,
    delay_seconds: float = 1.5,
    timeout_seconds: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FetchResult:
    """Fetch a search page with simple rate limiting and browser-like headers."""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Please provide a valid http(s) Property Finder search URL.")

    time.sleep(max(delay_seconds, 0))
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    response = requests.get(url, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    return FetchResult(url=response.url, status_code=response.status_code, html=response.text)


def fetch_with_retries(
    url: str,
    page_number: int,
    delay_seconds: float = 1.5,
    timeout_seconds: int = 30,
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 2,
) -> FetchResult:
    """Fetch one page, retrying transient failures before raising the last error."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            result = fetch_search_page(
                url,
                delay_seconds=delay_seconds if attempt == 0 else delay_seconds + attempt,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
            )
            result.page_number = page_number
            return result
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(max(delay_seconds, 0) + attempt + 1)
    raise RuntimeError(f"Page {page_number} failed after {retries + 1} attempts: {last_error}")


def estimate_pagination(html: str, listings_per_page: int) -> tuple[int | None, int | None]:
    """Estimate total listings/pages from visible text such as '195 listed'."""
    total_listings = detect_total_listing_count(html)
    if total_listings and listings_per_page > 0:
        return total_listings, max(math.ceil(total_listings / listings_per_page), 1)
    return total_listings, None


def detect_total_listing_count(html: str) -> int | None:
    """Find total listing count from text in the page, without relying on pagination buttons."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    patterns = [
        r"([\d,]+)\s+(?:properties|property|listings|listing)\s+(?:found|listed|available)",
        r"([\d,]+)\s+(?:listed)",
        r"showing\s+\d+\s*-\s*\d+\s+of\s+([\d,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def build_page_url(url: str, page_number: int) -> str:
    """Add or replace the page query parameter while preserving all other params."""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Please provide a valid http(s) Property Finder search URL.")

    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["page"] = [str(max(int(page_number), 1))]
    encoded_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=encoded_query))


def first_page_url(url: str) -> str:
    """Preserve page-less search URLs, or normalize an existing page param to page 1."""
    return build_page_url(url, 1) if has_page_param(url) else url


def has_page_param(url: str) -> bool:
    return "page" in parse_qs(urlparse(url).query, keep_blank_values=True)


def scrape_paginated_listings(
    url: str,
    parse_callback: Callable[[str, str], list[dict]],
    max_pages: int = 5,
    max_listings: int = 500,
    delay_seconds: float = 1.5,
    stop_on_duplicate_page: bool = True,
    progress_callback: Callable[[int, int | None, int, str], None] | None = None,
) -> tuple[list[dict], ScrapeSummary]:
    """Scrape sequential pages while storing only parsed listing records."""
    summary = ScrapeSummary()
    records_by_key: dict[str, dict] = {}
    duplicate_skipped = 0
    failed_pages = 0
    consecutive_failures = 0
    pages_scraped = 0
    estimated_total_pages: int | None = None
    estimated_total_listings: int | None = None

    page_number = _resume_start_page() if resume else 1
    while page_number <= max(max_pages, 1):
        if estimated_total_pages and page_number > estimated_total_pages:
            summary.stop_reason = "Reached estimated total pages"
            break
        if len(records_by_key) >= max(max_listings, 1):
            summary.stop_reason = "Reached max listings"
            break

        page_url = first_page_url(url) if page_number == 1 else build_page_url(url, page_number)
        requested_total = estimated_total_pages or max_pages
        if progress_callback:
            progress_callback(page_number, requested_total, len(records_by_key), "Fetching")

        try:
            result = fetch_with_retries(page_url, page_number, delay_seconds=delay_seconds)
            page_records = parse_callback(result.html, result.url)
            consecutive_failures = 0
            pages_scraped += 1

            if page_number == 1:
                estimated_total_listings, estimated_total_pages = estimate_pagination(result.html, len(page_records))

            if not page_records:
                summary.stop_reason = "No valid listings found"
                break

            new_on_page = 0
            for record in page_records:
                key = _record_key(record)
                if not key:
                    key = f"page-{page_number}-row-{new_on_page}-{len(records_by_key)}"
                if key in records_by_key:
                    duplicate_skipped += 1
                    continue
                records_by_key[key] = record
                new_on_page += 1
                if len(records_by_key) >= max(max_listings, 1):
                    break

            if progress_callback:
                progress_callback(page_number, estimated_total_pages or max_pages, len(records_by_key), "Parsed")

            if new_on_page == 0 and stop_on_duplicate_page:
                summary.stop_reason = "Duplicate-only page found"
                break
        except Exception:
            failed_pages += 1
            consecutive_failures += 1
            if consecutive_failures >= 3:
                summary.stop_reason = "Stopped after 3 consecutive failures"
                break
        page_number += 1
    else:
        summary.stop_reason = "Reached max pages"

    summary.estimated_total_listings = estimated_total_listings
    summary.estimated_total_pages = estimated_total_pages
    summary.pages_requested = min(max_pages, estimated_total_pages or max_pages)
    summary.pages_scraped = pages_scraped
    summary.unique_listings_collected = len(records_by_key)
    summary.duplicate_listings_skipped = duplicate_skipped
    summary.failed_pages = failed_pages
    if not summary.stop_reason or summary.stop_reason == "Not started":
        summary.stop_reason = "Completed"
    return list(records_by_key.values()), summary


def _record_key(record: dict) -> str | None:
    for key in ("listing_id", "details_url"):
        value = record.get(key)
        if value:
            return str(value)
    return None


def scrape_paginated_to_disk(
    url: str,
    parse_callback: Callable[[str, str], list[dict]],
    max_pages: int = 5,
    max_listings: int = 500,
    batch_size: int = 500,
    delay_seconds: float = 1.5,
    stop_on_duplicate_page: bool = True,
    resume: bool = False,
    progress_callback: Callable[[int, int | None, int, str], None] | None = None,
) -> ScrapeSummary:
    """Scrape, parse, clean, dedupe, and save each page before fetching the next."""
    if not resume:
        clear_data()

    summary = ScrapeSummary()
    seen_keys = load_seen_keys() if resume else set()
    duplicate_skipped = 0
    failed_pages = 0
    consecutive_failures = 0
    pages_scraped = 0
    invalid_rows_removed = 0
    estimated_total_pages: int | None = None
    estimated_total_listings: int | None = None
    saved_path = ""

    page_number = 1
    while page_number <= max(max_pages, 1):
        if estimated_total_pages is not None and page_number > estimated_total_pages:
            summary.stop_reason = "Reached estimated total pages"
            break
        if len(seen_keys) >= max(max_listings, 1):
            summary.stop_reason = "Reached max listings"
            break

        page_url = first_page_url(url) if page_number == 1 else build_page_url(url, page_number)
        requested_total = estimated_total_pages or max_pages
        if progress_callback:
            progress_callback(page_number, requested_total, len(seen_keys), "Fetching")

        try:
            result = fetch_with_retries(page_url, page_number, delay_seconds=delay_seconds)
            raw_records = parse_callback(result.html, result.url)
            consecutive_failures = 0
            pages_scraped += 1

            if page_number == 1:
                estimated_total_listings, estimated_total_pages = estimate_pagination(result.html, len(raw_records))
                if estimated_total_pages is not None:
                    estimated_total_pages = min(estimated_total_pages, max_pages)

            if len(raw_records) == 0:
                summary.stop_reason = "No valid listings found"
                break

            append_raw_jsonl(raw_records)
            page_clean = clean_listings(raw_records)
            valid_page_clean = analysis_listings(page_clean, include_suspicious=True)
            invalid_rows_removed += max(len(page_clean) - len(valid_page_clean), 0)

            before_count = len(seen_keys)
            combined, new_count, saved = append_clean_rows(valid_page_clean, seen_keys)
            saved_path = str(saved)
            duplicate_skipped += max(len(valid_page_clean) - new_count, 0)

            append_scrape_log(
                {
                    "page": page_number,
                    "url": result.url,
                    "raw_records": len(raw_records),
                    "valid_records": len(valid_page_clean),
                    "new_records": new_count,
                    "unique_total": len(seen_keys),
                    "duplicate_records": max(len(valid_page_clean) - new_count, 0),
                    "saved_path": saved_path,
                }
            )

            if progress_callback:
                progress_callback(page_number, estimated_total_pages or max_pages, len(seen_keys), "Saved")

            if new_count == 0 and len(seen_keys) == before_count and stop_on_duplicate_page:
                summary.stop_reason = "Duplicate-only page found"
                break
            if len(seen_keys) >= max(max_listings, 1):
                summary.stop_reason = "Reached max listings"
                break
            if batch_size > 0 and len(seen_keys) % max(batch_size, 1) == 0:
                load_clean_data()
        except Exception as exc:
            failed_pages += 1
            consecutive_failures += 1
            append_scrape_log(
                {
                    "page": page_number,
                    "url": page_url,
                    "error": str(exc),
                    "unique_total": len(seen_keys),
                }
            )
            if consecutive_failures >= 3:
                summary.stop_reason = "Stopped after 3 consecutive failures"
                break
        page_number += 1
    else:
        summary.stop_reason = "Reached max pages"

    final_clean = load_clean_data()
    summary.estimated_total_listings = estimated_total_listings
    summary.estimated_total_pages = estimated_total_pages
    summary.pages_requested = min(max_pages, estimated_total_pages or max_pages)
    summary.pages_scraped = pages_scraped
    summary.unique_listings_collected = len(seen_keys)
    summary.duplicate_listings_skipped = duplicate_skipped
    summary.failed_pages = failed_pages
    summary.invalid_rows_removed = invalid_rows_removed
    summary.clean_rows_ready = len(final_clean)
    summary.saved_path = saved_path
    if not summary.stop_reason or summary.stop_reason == "Not started":
        summary.stop_reason = "Completed"
    return summary


def _resume_start_page() -> int:
    log_df = load_scrape_log()
    if log_df.empty or "page" not in log_df.columns:
        return 1
    pages = log_df["page"].dropna()
    if pages.empty:
        return 1
    return int(pages.max()) + 1

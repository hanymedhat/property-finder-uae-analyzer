"""Parse Property Finder listings from stable page data first, HTML second."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup


BASE_URL = "https://www.propertyfinder.ae"
LISTING_KEYS = ("listing", "property", "propertyCard", "propertyResult")


def parse_listings(html: str, base_url: str = BASE_URL) -> list[dict[str, Any]]:
    """Parse listings, preferring __NEXT_DATA__ and falling back to data-testid HTML."""
    soup = BeautifulSoup(html, "html.parser")
    listings = parse_next_data(soup, base_url=base_url)
    if listings:
        return listings
    return parse_html_fallback(soup, base_url=base_url)


def parse_next_data(soup: BeautifulSoup, base_url: str = BASE_URL) -> list[dict[str, Any]]:
    script = soup.find("script", id="__NEXT_DATA__", type="application/json")
    if not script or not script.string:
        return []

    try:
        payload = json.loads(script.string)
    except json.JSONDecodeError:
        return []

    raw_items = _find_listing_like_items(payload)
    listings = [_normalize_json_listing(item, base_url=base_url) for item in raw_items]
    return [item for item in listings if item.get("listing_id") or item.get("details_url")]


def parse_html_fallback(soup: BeautifulSoup, base_url: str = BASE_URL) -> list[dict[str, Any]]:
    cards = soup.select(
        "[data-testid*='property-card'], [data-testid*='listing-card'], "
        "[data-testid*='property-result'], article[data-testid]"
    )
    listings: list[dict[str, Any]] = []
    for card in cards:
        link = card.select_one("a[href*='/property/'], a[href*='/rent/'], a[href*='/buy/']")
        details_url = urljoin(base_url, link.get("href")) if link and link.get("href") else None
        listing = {
            "listing_id": _first_attr(card, ["data-id", "data-listing-id", "id"])
            or _id_from_url(details_url),
            "title": _text_first(
                card,
                [
                    "[data-testid*='title']",
                    "[data-testid*='property-card-title']",
                    "h2",
                    "h3",
                ],
            ),
            "price_aed": _text_first(card, ["[data-testid*='price']", "[aria-label*='price' i]"]),
            "property_type": _text_first(card, ["[data-testid*='property-type']"]),
            "bedrooms": _text_first(card, ["[data-testid*='bed']", "[aria-label*='bed' i]"]),
            "bathrooms": _text_first(card, ["[data-testid*='bath']", "[aria-label*='bath' i]"]),
            "area_sqft": _text_first(card, ["[data-testid*='area']", "[aria-label*='area' i]"]),
            "price_per_sqft": _text_first(card, ["[data-testid*='price-per']"]),
            "listed_date": _text_first(card, ["[data-testid*='date']", "time"]),
            "days_listed": None,
            "agent_name": _text_first(card, ["[data-testid*='agent-name']", "[data-testid*='agent']"]),
            "is_super_agent": bool(card.select_one("[data-testid*='super-agent'], [aria-label*='super agent' i]")),
            "broker_name": _text_first(card, ["[data-testid*='broker']", "[data-testid*='agency']"]),
            "developer_name": _text_first(card, ["[data-testid*='developer']"]),
            "location_full": _text_first(card, ["[data-testid*='location']", "[aria-label*='location' i]"]),
            "project_or_community": None,
            "details_url": details_url,
        }
        listing["project_or_community"] = _infer_project_or_community(listing["location_full"])
        if _has_value(listing.get("listing_id")) or _has_value(listing.get("details_url")):
            listings.append(listing)
    return listings


def _find_listing_like_items(payload: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if _looks_like_listing(node):
                candidates.append(node)
            for key in LISTING_KEYS:
                child = node.get(key)
                if isinstance(child, dict) and _looks_like_listing(child):
                    candidates.append(child)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            if node and all(isinstance(item, dict) for item in node):
                listing_count = sum(1 for item in node if _looks_like_listing(item))
                if listing_count >= 2:
                    candidates.extend(item for item in node if _looks_like_listing(item))
                    return
            for value in node:
                walk(value)

    walk(payload)
    unique: dict[str, dict[str, Any]] = {}
    for item in candidates:
        key = str(_deep_get(item, ["id", "listing_id", "property_id", "reference", "share_url"]) or id(item))
        unique[key] = item
    return list(unique.values())


def _looks_like_listing(item: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in item}
    has_id = bool(keys & {"id", "listing_id", "property_id", "reference", "reference_number"})
    has_price = bool(keys & {"price", "default_price", "price_value", "priceval"})
    has_url = bool(keys & {"url", "path", "share_url", "property_url"})
    has_location = bool(keys & {"location", "location_tree", "community", "city"})
    return (has_id and (has_price or has_url or has_location)) or (has_price and has_url)


def _normalize_json_listing(item: dict[str, Any], base_url: str = BASE_URL) -> dict[str, Any]:
    price = _deep_get(item, ["price.value", "price.amount", "price", "default_price", "price_value"])
    area = _deep_get(
        item,
        [
            "size.value",
            "size",
            "area.value",
            "area",
            "plot_size.value",
            "property_size.value",
        ],
    )
    location = _location_text(_deep_get(item, ["location", "location_tree", "locations"]))
    details_url = _deep_get(item, ["url", "path", "share_url", "property_url"])
    if details_url:
        details_url = urljoin(base_url, str(details_url))

    agent = _deep_get(item, ["agent", "listed_by", "contact", "broker.agent"])
    broker = _deep_get(item, ["broker", "agency", "company"])
    developer = _deep_get(item, ["developer", "developer_info", "project.developer"])
    listed_date = _deep_get(
        item,
        [
            "listed_date",
            "listedDate",
            "created_at",
            "createdAt",
            "published_at",
            "publishedAt",
            "date_insert",
            "listing_date",
        ],
    )

    return {
        "listing_id": _deep_get(item, ["listing_id", "id", "property_id", "reference", "reference_number"])
        or _id_from_url(details_url),
        "title": _deep_get(item, ["title", "name", "property_name", "heading"]),
        "price_aed": price,
        "property_type": _deep_get(item, ["property_type", "propertyType", "type.name", "category.name", "type"]),
        "bedrooms": _deep_get(item, ["bedrooms", "bedroom", "beds", "bed_count"]),
        "bathrooms": _deep_get(item, ["bathrooms", "bathroom", "baths", "bath_count"]),
        "area_sqft": area,
        "price_per_sqft": _deep_get(item, ["price_per_sqft", "pricePerSqft", "price_per_area"]),
        "listed_date": listed_date,
        "days_listed": _deep_get(item, ["days_listed", "daysListed"]),
        "agent_name": _person_name(agent) or _deep_get(item, ["agent_name", "agentName"]),
        "is_super_agent": bool(
            _deep_get(item, ["agent.is_super_agent", "agent.isSuperAgent", "is_super_agent", "isSuperAgent"])
        ),
        "broker_name": _person_name(broker) or _deep_get(item, ["broker_name", "brokerName", "agency_name"]),
        "developer_name": _person_name(developer)
        or _deep_get(
            item,
            [
                "developer_name",
                "developerName",
                "project.developer_name",
                "project.developerName",
                "developer.company_name",
            ],
        ),
        "location_full": location,
        "project_or_community": _deep_get(
            item,
            ["project.name", "project", "community.name", "community", "location.community"],
        )
        or _infer_project_or_community(location),
        "details_url": details_url,
    }


def _deep_get(data: Any, paths: Iterable[str]) -> Any:
    for path in paths:
        current = data
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                current = None
                break
        if _has_value(current):
            return current
    return None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _person_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _deep_get(value, ["name", "full_name", "fullName", "agent_name", "broker_name"])
    return None


def _location_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        name = _deep_get(value, ["full_name", "fullName", "name", "path_name"])
        if name:
            return str(name)
        parts = [_location_text(v) for v in value.values()]
        return ", ".join(dict.fromkeys(part for part in parts if part))
    if isinstance(value, list):
        parts = [_location_text(v) for v in value]
        return ", ".join(dict.fromkeys(part for part in parts if part))
    return None


def _text_first(card: Any, selectors: list[str]) -> str | None:
    for selector in selectors:
        element = card.select_one(selector)
        if element:
            return element.get_text(" ", strip=True)
    return None


def _first_attr(card: Any, attrs: list[str]) -> str | None:
    for attr in attrs:
        value = card.get(attr)
        if value:
            return str(value)
    return None


def _id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"(\d{5,})(?:[/?#-]|$)", url)
    return match.group(1) if match else None


def _infer_project_or_community(location: str | None) -> str | None:
    if not location:
        return None
    parts = [part.strip() for part in str(location).split(",") if part.strip()]
    return parts[0] if parts else None

"""Local incremental storage for large scrape jobs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils import safe_scalar_isna


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_JSONL = DATA_DIR / "raw_listings.jsonl"
CLEAN_PARQUET = DATA_DIR / "clean_listings.parquet"
CLEAN_CSV = DATA_DIR / "clean_listings.csv"
SCRAPE_LOG = DATA_DIR / "scrape_log.csv"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def clear_data() -> None:
    ensure_data_dir()
    for path in [RAW_JSONL, CLEAN_PARQUET, CLEAN_CSV, SCRAPE_LOG]:
        if path.exists():
            path.unlink()


def append_raw_jsonl(records: Iterable[dict]) -> None:
    ensure_data_dir()
    with RAW_JSONL.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_clean_data() -> pd.DataFrame:
    ensure_data_dir()
    if CLEAN_PARQUET.exists():
        try:
            return pd.read_parquet(CLEAN_PARQUET)
        except Exception:
            pass
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV)
    return pd.DataFrame()


def save_clean_data(df: pd.DataFrame) -> Path:
    ensure_data_dir()
    try:
        df.to_parquet(CLEAN_PARQUET, index=False)
        if CLEAN_CSV.exists():
            CLEAN_CSV.unlink()
        return CLEAN_PARQUET
    except Exception:
        df.to_csv(CLEAN_CSV, index=False)
        return CLEAN_CSV


def append_clean_rows(page_df: pd.DataFrame, seen_keys: set[str]) -> tuple[pd.DataFrame, int, Path]:
    existing = load_clean_data()
    if page_df.empty:
        saved_path = save_clean_data(existing) if not existing.empty else CLEAN_CSV
        return existing, 0, saved_path

    working = page_df.copy()
    working["_dedupe_key"] = working.apply(listing_key, axis=1)
    working = working[working["_dedupe_key"].notna()]
    working = working[~working["_dedupe_key"].isin(seen_keys)].copy()
    new_count = len(working)
    seen_keys.update(working["_dedupe_key"].astype(str).tolist())
    working = working.drop(columns=["_dedupe_key"], errors="ignore")
    combined = pd.concat([existing, working], ignore_index=True) if not existing.empty else working
    if not combined.empty:
        combined["_dedupe_key"] = combined.apply(listing_key, axis=1)
        combined = combined.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key"])
    saved_path = save_clean_data(combined)
    return combined, new_count, saved_path


def load_seen_keys() -> set[str]:
    df = load_clean_data()
    if df.empty:
        return set()
    return {key for key in df.apply(listing_key, axis=1).dropna().astype(str).tolist()}


def append_scrape_log(row: dict) -> None:
    ensure_data_dir()
    log_row = pd.DataFrame([row])
    if SCRAPE_LOG.exists():
        log_row.to_csv(SCRAPE_LOG, mode="a", header=False, index=False)
    else:
        log_row.to_csv(SCRAPE_LOG, index=False)


def load_scrape_log() -> pd.DataFrame:
    ensure_data_dir()
    if SCRAPE_LOG.exists():
        return pd.read_csv(SCRAPE_LOG)
    return pd.DataFrame()


def listing_key(row) -> str | None:
    for column in ["listing_id", "details_url"]:
        value = row.get(column) if hasattr(row, "get") else None
        if value is not None and not safe_scalar_isna(value) and str(value).strip():
            return str(value)
    return None


def summary_to_dict(summary) -> dict:
    return asdict(summary) if hasattr(summary, "__dataclass_fields__") else dict(summary or {})

"""Data cleaning, validity flags, and analysis-ready filtering."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.utils import safe_scalar_isna


FIELDS = [
    "listing_id",
    "title",
    "price_aed",
    "property_type",
    "bedrooms",
    "bathrooms",
    "area_sqft",
    "price_per_sqft",
    "listed_date",
    "days_listed",
    "agent_name",
    "is_super_agent",
    "broker_name",
    "developer_name",
    "location_full",
    "project_or_community",
    "details_url",
]

INVALID_PROPERTY_TYPES = {"subcommunity", "sub community", "community", "area", "district", "building"}


def clean_listings(records: list[dict]) -> pd.DataFrame:
    """Return cleaned raw data with quality flags; analysis can filter from this."""
    df = pd.DataFrame(records)
    for field in FIELDS:
        if field not in df.columns:
            df[field] = pd.NA
    df = df[FIELDS].copy()

    for column in ["price_aed", "area_sqft", "price_per_sqft", "bedrooms", "bathrooms"]:
        df[column] = df[column].apply(_to_number)

    missing_ppsf = df["price_per_sqft"].isna() & df["price_aed"].notna() & df["area_sqft"].gt(0)
    df.loc[missing_ppsf, "price_per_sqft"] = df.loc[missing_ppsf, "price_aed"] / df.loc[missing_ppsf, "area_sqft"]

    df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce", utc=True).dt.tz_convert(None)
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    missing_days = df["days_listed"].isna() & df["listed_date"].notna()
    df.loc[missing_days, "days_listed"] = (today - df.loc[missing_days, "listed_date"]).dt.days
    df["days_listed"] = df["days_listed"].apply(_to_number)

    df["listing_id"] = df["listing_id"].astype("string")
    df["is_super_agent"] = df["is_super_agent"].fillna(False).astype(bool)
    df["project_or_community"] = _clean_text_series(df["project_or_community"], "Unknown")
    df["property_type"] = _clean_text_series(df["property_type"], "Unknown")
    df["broker_name"] = _clean_text_series(df["broker_name"], "Unknown")
    df["developer_name"] = _clean_text_series(df["developer_name"], "").replace("", pd.NA)
    df["developer_name"] = df["developer_name"].fillna(df["broker_name"]).fillna("Unknown")
    df["agent_name"] = _clean_text_series(df["agent_name"], "Unknown")

    dedupe_key = df["listing_id"].fillna(df["details_url"].astype("string"))
    df = df.loc[~dedupe_key.duplicated()].copy()

    df["is_subcommunity_row"] = df["property_type"].str.lower().isin(INVALID_PROPERTY_TYPES)
    df["missing_price"] = df["price_aed"].isna() | df["price_aed"].le(0)
    df["missing_area"] = df["area_sqft"].isna() | df["area_sqft"].le(0)
    df["missing_price_per_sqft"] = df["price_per_sqft"].isna() | df["price_per_sqft"].le(0)
    df["invalid_property_row"] = (
        df["is_subcommunity_row"]
        | df["property_type"].isna()
        | df["property_type"].str.lower().eq("unknown")
        | df["bedrooms"].isna()
    )
    df["is_valid_listing"] = ~(
        df["missing_price"] | df["missing_area"] | df["missing_price_per_sqft"] | df["invalid_property_row"]
    )

    df = flag_outliers(df, "area_sqft", group_cols=["project_or_community", "property_type", "bedrooms"])
    df = flag_outliers(df, "price_per_sqft", group_cols=["project_or_community", "property_type", "bedrooms"])
    df["is_suspicious"] = df["area_sqft_outlier"] | df["price_per_sqft_outlier"]
    return df


def analysis_listings(df: pd.DataFrame, include_suspicious: bool = False) -> pd.DataFrame:
    """Rows suitable for market analysis."""
    if df.empty:
        return df.copy()
    if "is_valid_listing" not in df.columns:
        return df.iloc[0:0].copy()
    valid = df[df["is_valid_listing"].fillna(False).astype(bool)].copy()
    if not include_suspicious and "is_suspicious" in valid:
        valid = valid[~valid["is_suspicious"]].copy()
    return valid


def data_quality_report(raw_df: pd.DataFrame, analysis_df: pd.DataFrame | None = None) -> pd.DataFrame:
    analysis_count = len(analysis_df) if analysis_df is not None else len(analysis_listings(raw_df))
    valid_series = raw_df["is_valid_listing"].fillna(False).astype(bool) if "is_valid_listing" in raw_df.columns else pd.Series(False, index=raw_df.index)
    suspicious_series = raw_df["is_suspicious"].fillna(False).astype(bool) if "is_suspicious" in raw_df.columns else pd.Series(False, index=raw_df.index)
    metrics = {
        "total_scraped_rows": len(raw_df),
        "valid_listings": int(valid_series.sum()),
        "removed_rows": int((~valid_series).sum()),
        "suspicious_listings": int(suspicious_series.sum()),
        "final_analysis_rows": analysis_count,
    }
    return pd.DataFrame([metrics])


def flag_outliers(df: pd.DataFrame, column: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    outlier_col = f"{column}_outlier"
    df[outlier_col] = False
    if column not in df:
        return df

    if group_cols:
        for _, group in df.groupby(group_cols, dropna=False):
            df = _flag_outlier_index(df, group.index, column, outlier_col)
        return df
    return _flag_outlier_index(df, df.index, column, outlier_col)


def _flag_outlier_index(df: pd.DataFrame, index, column: str, outlier_col: str) -> pd.DataFrame:
    values = df.loc[index, column].dropna()
    if len(values) < 4:
        return df
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return df
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df.loc[index, outlier_col] = df.loc[index, column].lt(lower) | df.loc[index, column].gt(upper)
    return df


def _clean_text_series(series: pd.Series, default: str) -> pd.Series:
    return series.astype("string").str.strip().replace("", pd.NA).fillna(default)


def _to_number(value):
    if safe_scalar_isna(value):
        return pd.NA
    if isinstance(value, dict):
        value = value.get("value") or value.get("amount")
    if isinstance(value, (list, tuple, set)):
        value = next((item for item in value if not safe_scalar_isna(item)), pd.NA)
        if safe_scalar_isna(value):
            return pd.NA
    text = str(value).lower().replace(",", "").replace("aed", "").replace("sqft", "").strip()
    text = text.replace("studio", "0")
    multiplier = 1
    if text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    number = pd.to_numeric(text.replace("+", ""), errors="coerce")
    if pd.isna(number):
        return pd.NA
    return float(number) * multiplier

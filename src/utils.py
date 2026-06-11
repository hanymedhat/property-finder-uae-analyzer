"""Small defensive helpers for pandas-heavy app code."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def has_columns(df: pd.DataFrame | None, columns: Iterable[str]) -> bool:
    return isinstance(df, pd.DataFrame) and all(column in df.columns for column in columns)


def safe_numeric(series, default=None) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    if default is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(pd.Series(default), errors="coerce")


def safe_not_empty(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty


def safe_scalar_isna(value) -> bool:
    if isinstance(value, (dict, list, tuple, set)):
        return False
    result = pd.isna(value)
    if isinstance(result, (bool, type(pd.NA))):
        return bool(result)
    return False

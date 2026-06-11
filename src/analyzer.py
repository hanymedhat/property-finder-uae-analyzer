"""Segment-first analytics and Excel report generation."""

from __future__ import annotations

from io import BytesIO

import pandas as pd


SEGMENT_COLS = ["project_or_community", "property_type", "bedrooms"]
ROI_COLUMNS = [
    "segment_median_ppsf",
    "segment_sample_size",
    "fallback_median_ppsf",
    "fallback_sample_size",
    "valuation_median_ppsf",
    "fair_value_aed",
    "difference_from_fair_value_aed",
    "difference_from_fair_value_pct",
    "valuation_label",
    "opportunity_score",
    "opportunity_category",
    "roi_pct",
    "annualized_roi_pct",
    "investment_score",
    "investment_category",
    "reliability_label",
    "reliability_score",
    "fair_value_score",
    "future_value_aed",
    "capital_gain_aed",
    "total_rental_income_aed",
    "total_service_charges_aed",
    "total_maintenance_aed",
    "purchase_fees_aed",
    "selling_fees_aed",
    "net_profit_aed",
]
INVESTMENT_BASE_COLUMNS = [
    "price_aed",
    "area_sqft",
    "price_per_sqft",
    "project_or_community",
    "property_type",
    "bedrooms",
    "is_suspicious",
]
DEFAULT_INVESTMENT_ASSUMPTIONS = {
    "annual_appreciation_pct": 5.0,
    "rental_yield_pct": 6.0,
    "holding_years": 3,
    "annual_service_charges_aed": 0.0,
    "annual_maintenance_cost_aed": 0.0,
    "purchase_fees_pct": 4.0,
    "selling_fees_pct": 2.0,
}


def market_snapshot_table(df: pd.DataFrame) -> pd.DataFrame:
    return _segment_group(df, SEGMENT_COLS).sort_values(["project_or_community", "property_type", "bedrooms"])


def segment_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "listing_count": int(len(df)),
        "avg_price_aed": _safe_stat(df, "price_aed", "mean"),
        "median_price_aed": _safe_stat(df, "price_aed", "median"),
        "min_price_aed": _safe_stat(df, "price_aed", "min"),
        "max_price_aed": _safe_stat(df, "price_aed", "max"),
        "avg_area_sqft": _safe_stat(df, "area_sqft", "mean"),
        "median_area_sqft": _safe_stat(df, "area_sqft", "median"),
        "avg_price_per_sqft": _safe_stat(df, "price_per_sqft", "mean"),
        "median_price_per_sqft": _safe_stat(df, "price_per_sqft", "median"),
        "min_price_per_sqft": _safe_stat(df, "price_per_sqft", "min"),
        "max_price_per_sqft": _safe_stat(df, "price_per_sqft", "max"),
    }


def bedroom_analysis(df: pd.DataFrame) -> pd.DataFrame:
    columns = SEGMENT_COLS
    grouped = _segment_group(df, columns)
    return grouped[
        [
            "project_or_community",
            "property_type",
            "bedrooms",
            "listing_count",
            "avg_price_aed",
            "median_price_aed",
            "avg_area_sqft",
            "avg_price_per_sqft",
            "median_price_per_sqft",
        ]
    ].sort_values(columns)


def project_analysis(df: pd.DataFrame, property_type: str | None = None, bedrooms: float | None = None) -> pd.DataFrame:
    working = filter_segment(df, property_type=property_type, bedrooms=bedrooms)
    return (
        working.groupby("project_or_community", dropna=False)
        .agg(
            listing_count=("listing_id", "count"),
            avg_price_aed=("price_aed", "mean"),
            median_price_aed=("price_aed", "median"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
            avg_area_sqft=("area_sqft", "mean"),
        )
        .reset_index()
        .sort_values(["listing_count", "avg_price_per_sqft"], ascending=[False, False])
    )


def developer_analysis(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["developer"] = _developer_series(working)
    return (
        working.groupby(["developer", "project_or_community", "property_type", "bedrooms"], dropna=False)
        .agg(
            listing_count=("listing_id", "count"),
            avg_price_aed=("price_aed", "mean"),
            median_price_aed=("price_aed", "median"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
            avg_area_sqft=("area_sqft", "mean"),
        )
        .reset_index()
        .sort_values(["listing_count", "avg_price_per_sqft"], ascending=[False, False])
    )


def broker_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["broker_name", "property_type", "bedrooms"], dropna=False)
        .agg(
            listing_count=("listing_id", "count"),
            avg_price_aed=("price_aed", "mean"),
            median_price_aed=("price_aed", "median"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
        )
        .reset_index()
        .sort_values(["listing_count", "avg_price_per_sqft"], ascending=[False, False])
    )


def broker_rankings(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    analysis = broker_analysis(df)
    return {
        "most_active": analysis.sort_values("listing_count", ascending=False),
        "highest_avg_price_per_sqft": analysis.sort_values("avg_price_per_sqft", ascending=False),
        "largest_inventory": analysis.sort_values("listing_count", ascending=False),
    }


def agent_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["agent_name", "is_super_agent"], dropna=False)
        .agg(
            listings=("listing_id", "count"),
            avg_price_aed=("price_aed", "mean"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
        )
        .reset_index()
        .sort_values("listings", ascending=False)
    )


def super_agent_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if "is_super_agent" not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby("is_super_agent", dropna=False)
        .agg(
            listings=("listing_id", "count"),
            avg_price_aed=("price_aed", "mean"),
            median_price_aed=("price_aed", "median"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
            suspicious_rate=("is_suspicious", "mean"),
        )
        .reset_index()
    )


def suspicious_listings(df: pd.DataFrame) -> pd.DataFrame:
    if "is_suspicious" not in df.columns:
        return df.iloc[0:0]
    return df[df["is_suspicious"]].copy()


def add_roi_metrics(df: pd.DataFrame, assumptions: dict | None = None) -> pd.DataFrame:
    assumptions = normalize_investment_assumptions(assumptions)
    enriched = ensure_investment_base_columns(df.copy())
    enriched = ensure_roi_columns(enriched)

    if "project_or_community" not in enriched.columns:
        enriched["project_or_community"] = "Unknown"
    if "listing_id" not in enriched.columns:
        enriched["listing_id"] = enriched.index.astype(str)

    enriched = _merge_metric_if_missing(
        enriched,
        column="segment_median_ppsf",
        group_cols=SEGMENT_COLS,
        value_col="price_per_sqft",
        agg="median",
    )
    enriched = _merge_metric_if_missing(
        enriched,
        column="segment_sample_size",
        group_cols=SEGMENT_COLS,
        value_col="price_per_sqft",
        agg="count",
    )
    enriched = _merge_metric_if_missing(
        enriched,
        column="fallback_median_ppsf",
        group_cols=["property_type", "bedrooms"],
        value_col="price_per_sqft",
        agg="median",
    )
    enriched = _merge_metric_if_missing(
        enriched,
        column="fallback_sample_size",
        group_cols=["property_type", "bedrooms"],
        value_col="price_per_sqft",
        agg="count",
    )

    global_median_ppsf = pd.to_numeric(enriched.get("price_per_sqft"), errors="coerce").median()
    if pd.isna(global_median_ppsf):
        global_median_ppsf = 0.0
    use_fallback = pd.to_numeric(enriched.get("segment_sample_size"), errors="coerce").fillna(0).lt(5)
    enriched["valuation_median_ppsf"] = enriched.get("segment_median_ppsf").where(
        ~use_fallback,
        enriched.get("fallback_median_ppsf"),
    )
    enriched["valuation_median_ppsf"] = enriched["valuation_median_ppsf"].fillna(global_median_ppsf)
    enriched["valuation_sample_size"] = enriched.get("segment_sample_size").where(
        ~use_fallback,
        enriched.get("fallback_sample_size"),
    )
    enriched["uses_fallback_valuation"] = use_fallback
    price = pd.to_numeric(enriched.get("price_aed"), errors="coerce")
    area = pd.to_numeric(enriched.get("area_sqft"), errors="coerce")
    ppsf = pd.to_numeric(enriched.get("price_per_sqft"), errors="coerce")
    valuation_ppsf = pd.to_numeric(enriched.get("valuation_median_ppsf"), errors="coerce")

    enriched["fair_value_aed"] = area * valuation_ppsf
    enriched["difference_from_fair_value_aed"] = enriched["fair_value_aed"] - price
    enriched["difference_from_fair_value_pct"] = enriched["difference_from_fair_value_aed"] / price
    enriched["valuation_label"] = enriched["difference_from_fair_value_pct"].apply(fair_value_label)
    enriched["reliability_label"] = enriched["valuation_sample_size"].apply(reliability_label)
    enriched["reliability_score"] = enriched["reliability_label"].map(reliability_score).fillna(25)

    annual_appreciation = assumptions["annual_appreciation_pct"] / 100
    rental_yield = assumptions["rental_yield_pct"] / 100
    holding_years = max(int(assumptions["holding_years"]), 1)
    purchase_fees_pct = assumptions["purchase_fees_pct"] / 100
    selling_fees_pct = assumptions["selling_fees_pct"] / 100

    enriched["future_value_aed"] = price * ((1 + annual_appreciation) ** holding_years)
    enriched["capital_gain_aed"] = enriched["future_value_aed"] - price
    enriched["total_rental_income_aed"] = price * rental_yield * holding_years
    enriched["total_service_charges_aed"] = assumptions["annual_service_charges_aed"] * holding_years
    enriched["total_maintenance_aed"] = assumptions["annual_maintenance_cost_aed"] * holding_years
    enriched["purchase_fees_aed"] = price * purchase_fees_pct
    enriched["selling_fees_aed"] = enriched["future_value_aed"] * selling_fees_pct
    enriched["net_profit_aed"] = (
        enriched["capital_gain_aed"]
        + enriched["total_rental_income_aed"]
        - enriched["total_service_charges_aed"]
        - enriched["total_maintenance_aed"]
        - enriched["purchase_fees_aed"]
        - enriched["selling_fees_aed"]
    )
    enriched["roi_pct"] = enriched["net_profit_aed"] / price * 100
    enriched["annualized_roi_pct"] = (((price + enriched["net_profit_aed"]) / price) ** (1 / holding_years) - 1) * 100

    enriched = add_opportunity_scores(enriched)
    enriched["fair_value_score"] = enriched["valuation_label"].map(fair_value_score).fillna(30)
    enriched["normalized_roi_score"] = normalized_roi_score(enriched["roi_pct"])
    enriched["investment_score"] = (
        0.40 * enriched["normalized_roi_score"]
        + 0.30 * pd.to_numeric(enriched["opportunity_score"], errors="coerce").fillna(0)
        + 0.20 * pd.to_numeric(enriched["reliability_score"], errors="coerce").fillna(25)
        + 0.10 * pd.to_numeric(enriched["fair_value_score"], errors="coerce").fillna(30)
    ).clip(0, 100).round(1)
    enriched["investment_category"] = enriched["investment_score"].apply(investment_category)

    enriched["estimated_fair_value"] = enriched["fair_value_aed"]
    enriched["premium_discount_pct"] = -enriched["difference_from_fair_value_pct"]
    enriched["estimated_roi_pct"] = enriched["roi_pct"]
    enriched["market_position"] = enriched["valuation_label"]
    enriched = ensure_roi_columns(enriched)
    return enriched


def ensure_roi_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for column in ROI_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = None
    return enriched


def ensure_investment_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    defaults = {
        "project_or_community": "Unknown",
        "property_type": "Unknown",
        "bedrooms": None,
        "is_suspicious": False,
        "price_aed": None,
        "area_sqft": None,
        "price_per_sqft": None,
    }
    for column in INVESTMENT_BASE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = defaults.get(column)
    for column in ["price_aed", "area_sqft", "price_per_sqft", "bedrooms"]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    enriched["project_or_community"] = enriched["project_or_community"].fillna("Unknown")
    enriched["property_type"] = enriched["property_type"].fillna("Unknown")
    enriched["is_suspicious"] = enriched["is_suspicious"].fillna(False).astype(bool)
    return enriched


def investment_opportunities(df: pd.DataFrame, assumptions: dict | None = None) -> pd.DataFrame:
    roi = add_roi_metrics(df, assumptions=assumptions)
    sort_cols = [column for column in ["investment_score", "opportunity_score", "roi_pct"] if column in roi.columns]
    if not sort_cols:
        return roi
    return roi.sort_values(sort_cols, ascending=[False] * len(sort_cols))


def add_opportunity_scores(df: pd.DataFrame) -> pd.DataFrame:
    enriched = ensure_investment_base_columns(df.copy())
    if "project_or_community" not in enriched.columns:
        enriched["project_or_community"] = "Unknown"
    if "listing_id" not in enriched.columns:
        enriched["listing_id"] = enriched.index.astype(str)

    project_medians = enriched.groupby("project_or_community", dropna=False)["price_per_sqft"].median().rename(
        "project_median_ppsf_calc"
    )
    segment_medians = enriched.groupby(SEGMENT_COLS, dropna=False)["price_per_sqft"].median().rename(
        "segment_median_ppsf_calc"
    )
    bedroom_counts = (
        enriched.groupby(["property_type", "bedrooms"], dropna=False)["listing_id"].count().rename("bedroom_demand")
    )

    enriched = enriched.join(project_medians, on="project_or_community")
    if "project_median_ppsf" not in enriched.columns:
        enriched["project_median_ppsf"] = enriched["project_median_ppsf_calc"]
    else:
        enriched["project_median_ppsf"] = enriched["project_median_ppsf"].fillna(enriched["project_median_ppsf_calc"])
    enriched = enriched.join(segment_medians, on=SEGMENT_COLS)
    if "segment_median_ppsf" not in enriched.columns:
        enriched["segment_median_ppsf"] = enriched["segment_median_ppsf_calc"]
    else:
        enriched["segment_median_ppsf"] = enriched["segment_median_ppsf"].fillna(enriched["segment_median_ppsf_calc"])
    enriched = enriched.join(bedroom_counts, on=["property_type", "bedrooms"])
    if "segment_sample_size" not in enriched.columns:
        enriched["segment_sample_size"] = enriched["bedroom_demand"]
    segment_area_median = enriched.groupby(SEGMENT_COLS, dropna=False)["area_sqft"].median().rename("segment_median_area")
    enriched = enriched.join(segment_area_median, on=SEGMENT_COLS)
    if "fair_value_aed" not in enriched.columns:
        enriched["fair_value_aed"] = None
    if "valuation_sample_size" not in enriched.columns:
        enriched["valuation_sample_size"] = enriched["segment_sample_size"]

    price = pd.to_numeric(enriched.get("price_aed"), errors="coerce")
    fair_value = pd.to_numeric(enriched.get("fair_value_aed"), errors="coerce")
    ppsf = pd.to_numeric(enriched.get("price_per_sqft"), errors="coerce")
    segment_ppsf = pd.to_numeric(enriched.get("segment_median_ppsf"), errors="coerce")
    area = pd.to_numeric(enriched.get("area_sqft"), errors="coerce")
    median_area = pd.to_numeric(enriched.get("segment_median_area"), errors="coerce")
    if "fair_value_aed" not in enriched.columns or fair_value.isna().all():
        enriched["fair_value_aed"] = area * segment_ppsf
        fair_value = pd.to_numeric(enriched.get("fair_value_aed"), errors="coerce")
    sample_size = pd.to_numeric(enriched.get("valuation_sample_size"), errors="coerce").fillna(
        pd.to_numeric(enriched.get("segment_sample_size"), errors="coerce")
    )

    discount_from_fair_value_pct = ((fair_value - price) / price).fillna(0)
    ppsf_discount_pct = ((segment_ppsf - ppsf) / segment_ppsf).fillna(0)
    area_value_pct = ((area - median_area) / median_area).fillna(0)

    score = pd.Series(50.0, index=enriched.index)
    score += discount_from_fair_value_pct.clip(lower=0, upper=0.25).div(0.25).mul(25)
    score += ppsf_discount_pct.clip(lower=0, upper=0.15).div(0.15).mul(15)
    score += area_value_pct.clip(lower=0, upper=0.15).div(0.15).mul(10)
    suspicious = enriched["is_suspicious"].fillna(False).astype(bool)
    score -= suspicious.map({True: 20, False: 0})
    score -= sample_size.fillna(0).lt(3).map({True: 10, False: 0})

    enriched["opportunity_score"] = score.clip(0, 100).round(1)
    enriched["opportunity_category"] = enriched["opportunity_score"].apply(opportunity_category)
    enriched = enriched.drop(
        columns=["project_median_ppsf_calc", "segment_median_ppsf_calc", "segment_median_area"],
        errors="ignore",
    )
    return enriched


def selected_listing_roi(
    listing: pd.Series,
    market_df: pd.DataFrame,
    assumptions: dict | None = None,
) -> dict[str, float | str | bool]:
    roi_df = add_roi_metrics(market_df, assumptions=assumptions)
    if "listing_id" in roi_df.columns and listing.get("listing_id") is not None:
        selected = roi_df[roi_df["listing_id"].astype(str) == str(listing.get("listing_id"))]
    else:
        selected = roi_df.iloc[[listing.name]] if listing.name in roi_df.index else roi_df.iloc[0:0]
    if selected.empty:
        return {}
    row = selected.iloc[0]
    return {
        "current_asking_price": row.get("price_aed", 0),
        "estimated_fair_value": row.get("fair_value_aed", 0),
        "valuation_median_ppsf": row.get("valuation_median_ppsf", 0),
        "sample_size": row.get("valuation_sample_size", 0),
        "uses_fallback": bool(row.get("uses_fallback_valuation", False)),
        "premium_discount_pct": row.get("premium_discount_pct", 0),
        "estimated_roi_pct": row.get("roi_pct", 0),
        "annualized_roi_pct": row.get("annualized_roi_pct", 0),
        "market_position": row.get("valuation_label", ""),
        "low_sample_warning": row.get("reliability_label") in {"Low", "Very low"},
        "opportunity_score": row.get("opportunity_score", 0),
        "investment_score": row.get("investment_score", 0),
    }


def fair_value_label(premium_discount_pct: float) -> str:
    if pd.isna(premium_discount_pct):
        return "Unknown"
    if premium_discount_pct >= 0.10:
        return "Undervalued"
    if premium_discount_pct <= -0.10:
        return "Overpriced"
    return "Fairly Priced"


def opportunity_category(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 80:
        return "Excellent Opportunity"
    if score >= 60:
        return "Good Opportunity"
    if score >= 40:
        return "Fair"
    return "Weak Opportunity"


def reliability_label(sample_size: float | int | None) -> str:
    if pd.isna(sample_size) or sample_size < 3:
        return "Very low"
    if sample_size < 5:
        return "Low"
    if sample_size < 10:
        return "Medium"
    return "High"


def reliability_score(label: str) -> int:
    return {"High": 100, "Medium": 75, "Low": 50, "Very low": 25}.get(str(label), 25)


def fair_value_score(label: str) -> int:
    return {"Undervalued": 100, "Fairly Priced": 70, "Overpriced": 30}.get(str(label), 30)


def normalized_roi_score(roi_pct: pd.Series) -> pd.Series:
    values = pd.to_numeric(roi_pct, errors="coerce").fillna(0)
    return values.clip(lower=-20, upper=40).add(20).div(60).mul(100).clip(0, 100)


def investment_category(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 80:
        return "Strong Investment"
    if score >= 60:
        return "Good Investment"
    if score >= 40:
        return "Average Investment"
    return "Weak Investment"


def normalize_investment_assumptions(assumptions: dict | None = None) -> dict:
    merged = DEFAULT_INVESTMENT_ASSUMPTIONS.copy()
    if assumptions:
        merged.update({key: value for key, value in assumptions.items() if value is not None})
    numeric_keys = [
        "annual_appreciation_pct",
        "rental_yield_pct",
        "holding_years",
        "annual_service_charges_aed",
        "annual_maintenance_cost_aed",
        "purchase_fees_pct",
        "selling_fees_pct",
    ]
    for key in numeric_keys:
        merged[key] = float(merged.get(key, DEFAULT_INVESTMENT_ASSUMPTIONS[key]))
    merged["holding_years"] = max(int(merged["holding_years"]), 1)
    return merged


def project_investment_ranking(df: pd.DataFrame, assumptions: dict | None = None) -> pd.DataFrame:
    roi = add_roi_metrics(df, assumptions=assumptions)
    if roi.empty:
        return pd.DataFrame()
    ranking = (
        roi.groupby(SEGMENT_COLS, dropna=False)
        .agg(
            listing_count=("listing_id", "count"),
            median_price_aed=("price_aed", "median"),
            median_price_per_sqft=("price_per_sqft", "median"),
            median_roi_pct=("roi_pct", "median"),
            avg_roi_pct=("roi_pct", "mean"),
            best_roi_pct=("roi_pct", "max"),
            avg_opportunity_score=("opportunity_score", "mean"),
            avg_investment_score=("investment_score", "mean"),
            median_sample_size=("valuation_sample_size", "median"),
        )
        .reset_index()
    )
    ranking["reliability_label"] = ranking["median_sample_size"].apply(reliability_label)
    return ranking.sort_values("avg_investment_score", ascending=False)


TOP_LISTING_COLUMNS = [
    "title",
    "project_or_community",
    "property_type",
    "bedrooms",
    "price_aed",
    "area_sqft",
    "price_per_sqft",
    "fair_value_aed",
    "difference_from_fair_value_pct",
    "valuation_label",
    "opportunity_score",
    "roi_pct",
    "annualized_roi_pct",
    "investment_score",
    "reliability_label",
    "broker_name",
    "details_url",
]


def top_opportunity_listings(df: pd.DataFrame, assumptions: dict | None = None, limit: int = 20) -> pd.DataFrame:
    return _top_listing_table(add_roi_metrics(df, assumptions=assumptions), "opportunity_score", limit)


def top_roi_listings(df: pd.DataFrame, assumptions: dict | None = None, limit: int = 20) -> pd.DataFrame:
    return _top_listing_table(add_roi_metrics(df, assumptions=assumptions), "roi_pct", limit)


def top_investment_listings(df: pd.DataFrame, assumptions: dict | None = None, limit: int = 20) -> pd.DataFrame:
    return _top_listing_table(add_roi_metrics(df, assumptions=assumptions), "investment_score", limit)


def _top_listing_table(df: pd.DataFrame, sort_col: str, limit: int) -> pd.DataFrame:
    for column in TOP_LISTING_COLUMNS:
        if column not in df.columns:
            df[column] = None
    if sort_col not in df.columns:
        return df[TOP_LISTING_COLUMNS].head(0)
    return df.sort_values(sort_col, ascending=False)[TOP_LISTING_COLUMNS].head(limit)


def has_repeated_group(df: pd.DataFrame, group_cols: list[str]) -> bool:
    if df.empty:
        return False
    counts = df.groupby(group_cols, dropna=False).size()
    return bool((counts > 1).any())


def filter_segment(
    df: pd.DataFrame,
    property_type: str | None = None,
    bedrooms: float | int | None = None,
) -> pd.DataFrame:
    working = df.copy()
    if property_type:
        working = working[working["property_type"].astype(str) == str(property_type)]
    if bedrooms is not None:
        working = working[working["bedrooms"] == bedrooms]
    return working


def build_summary_report_excel(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    data_quality_df: pd.DataFrame,
    assumptions: dict | None = None,
) -> bytes:
    output = BytesIO()
    roi_df = add_roi_metrics(clean_df.copy(), assumptions=assumptions)
    sheets = {
        "Clean Listings": clean_df,
        "Fair Value Analysis": roi_df,
        "Opportunity Score": roi_df,
        "ROI Analysis": roi_df,
        "Investment Score": roi_df,
        "Project Investment Ranking": project_investment_ranking(clean_df, assumptions=assumptions),
        "Top Opportunities": top_opportunity_listings(clean_df, assumptions=assumptions),
        "Top ROI Listings": top_roi_listings(clean_df, assumptions=assumptions),
        "Top Investment Listings": top_investment_listings(clean_df, assumptions=assumptions),
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


def _segment_group(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        df.groupby(columns, dropna=False)
        .agg(
            listing_count=("listing_id", "count"),
            avg_price_aed=("price_aed", "mean"),
            median_price_aed=("price_aed", "median"),
            min_price_aed=("price_aed", "min"),
            max_price_aed=("price_aed", "max"),
            avg_area_sqft=("area_sqft", "mean"),
            median_area_sqft=("area_sqft", "median"),
            avg_price_per_sqft=("price_per_sqft", "mean"),
            median_price_per_sqft=("price_per_sqft", "median"),
            min_price_per_sqft=("price_per_sqft", "min"),
            max_price_per_sqft=("price_per_sqft", "max"),
        )
        .reset_index()
    )


def _developer_series(df: pd.DataFrame) -> pd.Series:
    developer = df.get("developer_name", pd.Series(index=df.index, dtype="string")).astype("string")
    broker = df.get("broker_name", pd.Series(index=df.index, dtype="string")).astype("string")
    return developer.fillna("").replace("", pd.NA).fillna(broker).fillna("Unknown")


def _merge_metric_if_missing(
    df: pd.DataFrame,
    column: str,
    group_cols: list[str],
    value_col: str,
    agg: str,
) -> pd.DataFrame:
    if column in df.columns and df[column].notna().any():
        return df
    if value_col not in df.columns or any(group_col not in df.columns for group_col in group_cols):
        if column not in df.columns:
            df[column] = None
        return df

    metric = df.groupby(group_cols, dropna=False)[value_col].agg(agg).rename(f"{column}_calc")
    df = df.join(metric, on=group_cols)
    if column not in df.columns:
        df[column] = df[f"{column}_calc"]
    else:
        df[column] = df[column].fillna(df[f"{column}_calc"])
    return df.drop(columns=[f"{column}_calc"], errors="ignore")


def _percentile_score(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True).fillna(0.5).clip(0, 1)


def _safe_stat(df: pd.DataFrame, column: str, method: str) -> float:
    if column not in df.columns or df[column].dropna().empty:
        return 0.0
    return float(getattr(df[column], method)())

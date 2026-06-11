from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.analyzer import (  # noqa: E402
    add_roi_metrics,
    agent_analysis,
    bedroom_analysis,
    broker_analysis,
    broker_rankings,
    build_summary_report_excel,
    developer_analysis,
    has_repeated_group,
    investment_opportunities,
    market_snapshot_table,
    project_investment_ranking,
    project_analysis,
    segment_kpis,
    super_agent_comparison,
    suspicious_listings,
    top_investment_listings,
    top_opportunity_listings,
    top_roi_listings,
)
from src.cleaner import analysis_listings, clean_listings, data_quality_report  # noqa: E402
from src.parser import parse_listings  # noqa: E402
from src.scraper import scrape_paginated_to_disk  # noqa: E402
from src.storage import clear_data, load_clean_data, load_scrape_log, save_clean_data  # noqa: E402
from src.utils import has_columns, safe_not_empty  # noqa: E402
from src.visualizer import (  # noqa: E402
    area_vs_ppsf,
    area_vs_price,
    listing_count_by_bedrooms_within_property_type,
    listing_count_by_property_type,
    investment_score_distribution,
    median_ppsf_by_project_property_type,
    opportunity_score_distribution,
    ppsf_box_by_bedrooms,
    ppsf_box_by_property_type,
    ppsf_distribution_by_property_type,
    roi_distribution,
)


st.set_page_config(page_title="Property Finder UAE Analyzer", layout="wide")


def main() -> None:
    st.title("Property Finder UAE Listing Analyzer")
    st.caption("Segment-first UAE listing scraper and investment analysis dashboard.")

    with st.sidebar:
        st.header("Scrape")
        url = st.text_input("Property Finder search URL", placeholder="https://www.propertyfinder.ae/...")
        max_pages = st.number_input("Max Pages", min_value=1, max_value=500, value=5, step=1)
        max_listings = st.number_input("Max Listings", min_value=1, max_value=10000, value=500, step=50)
        batch_size = st.number_input("Batch size", min_value=50, max_value=5000, value=500, step=50)
        delay = st.slider("Rate Delay", 0.0, 10.0, 1.5, 0.5)
        resume_previous = st.checkbox("Resume previous scrape", value=False)
        stop_on_duplicate = st.checkbox("Stop On Duplicate Page", value=True)
        scrape_clicked = st.button("Fetch and analyze", type="primary", use_container_width=True)
        if st.button("Clear previous data", use_container_width=True):
            clear_data()
            load_saved_clean_data.clear()
            load_saved_scrape_log.clear()
            st.session_state.pop("raw_df", None)
            st.session_state.pop("scrape_summary", None)
            st.session_state.pop("excel_report_bytes", None)
            st.success("Previous local scrape data cleared.")
        st.divider()
        uploaded = st.file_uploader("Or load a cleaned/raw CSV", type=["csv"])

    if scrape_clicked:
        run_scrape(url, int(max_pages), int(max_listings), int(batch_size), delay, stop_on_duplicate, resume_previous)

    if uploaded is not None:
        uploaded_df = normalize_uploaded_data(pd.read_csv(uploaded))
        save_clean_data(uploaded_df)
        load_saved_clean_data.clear()
        load_saved_scrape_log.clear()
        st.session_state["raw_df"] = uploaded_df
        st.session_state["scrape_summary"] = None
        st.session_state["source_url"] = "Uploaded CSV"

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        raw_df = load_saved_clean_data()
    raw_df = ensure_app_columns(raw_df)
    if raw_df is None or raw_df.empty:
        st.info("Enter a Property Finder search URL and click Fetch and analyze.")
        return

    filtered_df, include_suspicious = apply_sidebar_filters(raw_df)
    assumptions = investment_assumptions_sidebar()
    quality_df = build_data_quality(raw_df, filtered_df)
    render_exports(raw_df, filtered_df, quality_df, assumptions)
    render_scrape_summary()
    render_pages(raw_df, filtered_df, quality_df, include_suspicious, assumptions)


@st.cache_data(show_spinner=False)
def load_saved_clean_data() -> pd.DataFrame:
    return load_clean_data()


@st.cache_data(show_spinner=False)
def load_saved_scrape_log() -> pd.DataFrame:
    return load_scrape_log()


def run_scrape(
    url: str,
    max_pages: int,
    max_listings: int,
    batch_size: int,
    delay: float,
    stop_on_duplicate: bool,
    resume: bool,
) -> None:
    if not url:
        st.warning("Paste a Property Finder search URL first.")
        return

    progress = st.progress(0, text="Starting scraper...")
    status = st.empty()

    def update_progress(page_number: int, estimated_total_pages: int | None, unique_count: int, phase: str) -> None:
        denominator = max(estimated_total_pages or max_pages, 1)
        progress.progress(
            min(page_number / denominator, 1.0),
            text=f"Scraping page {page_number} of estimated {denominator}",
        )
        status.caption(f"{phase}: collected {unique_count:,} unique listings")

    try:
        summary = scrape_paginated_to_disk(
            url,
            parse_callback=parse_listings,
            max_pages=max_pages,
            max_listings=max_listings,
            batch_size=batch_size,
            delay_seconds=delay,
            stop_on_duplicate_page=stop_on_duplicate,
            resume=resume,
            progress_callback=update_progress,
        )
        load_saved_clean_data.clear()
        load_saved_scrape_log.clear()
        raw_df = load_saved_clean_data()
        if raw_df.empty:
            st.error("No valid listings were saved. The page may require JavaScript, bot checks, or a changed data shape.")
            return
        st.session_state["raw_df"] = raw_df
        st.session_state["scrape_summary"] = summary
        st.session_state["source_url"] = url
        progress.empty()
        status.empty()
        st.success(f"Saved {len(raw_df):,} clean listings from {summary.pages_scraped:,} pages.")
    except Exception as exc:
        st.error(f"Scrape failed: {exc}")


def normalize_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    return clean_listings(df.to_dict("records"))


def build_data_quality(clean_df: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    log_df = load_saved_scrape_log()
    if not log_df.empty and "raw_records" in log_df.columns:
        total_scraped = int(pd.to_numeric(log_df.get("raw_records"), errors="coerce").fillna(0).sum())
        valid_records = int(pd.to_numeric(log_df.get("valid_records"), errors="coerce").fillna(0).sum())
        metrics = {
            "total_scraped_rows": total_scraped,
            "valid_listings": len(clean_df),
            "removed_rows": max(total_scraped - valid_records, 0),
            "suspicious_listings": int(clean_df.get("is_suspicious", pd.Series(False, index=clean_df.index)).fillna(False).sum()),
            "final_analysis_rows": len(analysis_df),
        }
        return pd.DataFrame([metrics])
    return data_quality_report(clean_df, analysis_df)


def ensure_app_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    defaults = {
        "project_or_community": "Unknown",
        "property_type": "Unknown",
        "bedrooms": pd.NA,
        "broker_name": "Unknown",
        "price_aed": pd.NA,
        "area_sqft": pd.NA,
        "price_per_sqft": pd.NA,
        "is_valid_listing": True,
        "is_suspicious": False,
    }
    working = df.copy()
    for column, default in defaults.items():
        if column not in working.columns:
            working[column] = default
    return working


@st.cache_data(show_spinner=False)
def cached_market_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    return market_snapshot_table(df)


@st.cache_data(show_spinner=False)
def cached_bedroom_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return bedroom_analysis(df)


@st.cache_data(show_spinner=False)
def cached_project_investment_ranking(df: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    return project_investment_ranking(df, assumptions=assumptions)


@st.cache_data(show_spinner=False)
def cached_roi_metrics(df: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    return add_roi_metrics(df, assumptions=assumptions)


def apply_sidebar_filters(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    with st.sidebar:
        st.header("Filters")
        include_suspicious = st.checkbox("Include suspicious listings", value=False)

        base = analysis_listings(raw_df, include_suspicious=include_suspicious)
        projects = sorted(base["project_or_community"].dropna().astype(str).unique())
        selected_projects = st.multiselect("Project/Community", projects)

        working = base[base["project_or_community"].astype(str).isin(selected_projects)] if selected_projects else base
        property_types = sorted(working["property_type"].dropna().astype(str).unique())
        selected_types = st.multiselect("Property Type", property_types)

        working = working[working["property_type"].astype(str).isin(selected_types)] if selected_types else working
        bedrooms = sorted(working["bedrooms"].dropna().unique())
        selected_bedrooms = st.multiselect("Bedrooms", bedrooms)

        working = working[working["bedrooms"].isin(selected_bedrooms)] if selected_bedrooms else working
        brokers = sorted(working["broker_name"].dropna().astype(str).unique())
        selected_brokers = st.multiselect("Broker", brokers)

        area_min, area_max = numeric_bounds(base, "area_sqft")
        price_min, price_max = numeric_bounds(base, "price_aed")
        area_range = st.slider("Area Range", area_min, area_max, (area_min, area_max))
        price_range = st.slider("Price Range", price_min, price_max, (price_min, price_max))

    filtered = base.copy()
    if selected_projects:
        filtered = filtered[filtered["project_or_community"].astype(str).isin(selected_projects)]
    if selected_types:
        filtered = filtered[filtered["property_type"].astype(str).isin(selected_types)]
    if selected_bedrooms:
        filtered = filtered[filtered["bedrooms"].isin(selected_bedrooms)]
    if selected_brokers:
        filtered = filtered[filtered["broker_name"].astype(str).isin(selected_brokers)]
    filtered = filtered[filtered["area_sqft"].between(area_range[0], area_range[1])]
    filtered = filtered[filtered["price_aed"].between(price_range[0], price_range[1])]
    return filtered, include_suspicious


def investment_assumptions_sidebar() -> dict:
    with st.sidebar:
        st.header("Investment Assumptions")
        annual_appreciation_pct = st.number_input("Expected annual appreciation %", value=5.0, step=0.5)
        rental_yield_pct = st.number_input("Expected rental yield %", value=6.0, step=0.5)
        holding_years = st.number_input("Holding period years", min_value=1, max_value=50, value=3, step=1)
        annual_service_charges_aed = st.number_input("Annual service charges AED", min_value=0.0, value=0.0, step=1000.0)
        annual_maintenance_cost_aed = st.number_input("Annual maintenance cost AED", min_value=0.0, value=0.0, step=1000.0)
        purchase_fees_pct = st.number_input("Purchase fees %", min_value=0.0, value=4.0, step=0.25)
        selling_fees_pct = st.number_input("Selling fees %", min_value=0.0, value=2.0, step=0.25)
    return {
        "annual_appreciation_pct": annual_appreciation_pct,
        "rental_yield_pct": rental_yield_pct,
        "holding_years": holding_years,
        "annual_service_charges_aed": annual_service_charges_aed,
        "annual_maintenance_cost_aed": annual_maintenance_cost_aed,
        "purchase_fees_pct": purchase_fees_pct,
        "selling_fees_pct": selling_fees_pct,
    }


def render_exports(raw_df: pd.DataFrame, clean_df: pd.DataFrame, quality_df: pd.DataFrame, assumptions: dict) -> None:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.download_button(
            "Download cleaned CSV",
            data=clean_df.to_csv(index=False).encode("utf-8"),
            file_name="property_finder_clean_listings.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if st.button("Generate Excel Report", use_container_width=True):
            try:
                with st.spinner("Building Excel report..."):
                    st.session_state["excel_report_bytes"] = build_summary_report_excel(
                        raw_df,
                        clean_df,
                        quality_df,
                        assumptions=assumptions,
                    )
            except Exception as exc:
                st.warning(f"Excel Export failed: {exc}")
        if "excel_report_bytes" in st.session_state:
            st.download_button(
                "Download Excel report",
                data=st.session_state["excel_report_bytes"],
                file_name="property_finder_market_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with col3:
        st.caption(f"Source: {st.session_state.get('source_url', 'Unknown')}")


def render_scrape_summary() -> None:
    summary = st.session_state.get("scrape_summary")
    if not summary:
        return
    with st.expander("Final scrape summary", expanded=False):
        st.json(
            {
                "Estimated total listings": summary.estimated_total_listings,
                "Estimated total pages": summary.estimated_total_pages,
                "Pages requested": summary.pages_requested,
                "Pages scraped": summary.pages_scraped,
                "Unique listings collected": summary.unique_listings_collected,
                "Duplicate listings skipped": summary.duplicate_listings_skipped,
                "Invalid rows removed": summary.invalid_rows_removed,
                "Clean rows ready for analysis": summary.clean_rows_ready,
                "Failed pages": summary.failed_pages,
                "File saved path": summary.saved_path,
                "Stop reason": summary.stop_reason,
            }
        )


def render_pages(
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    quality_df: pd.DataFrame,
    include_suspicious: bool,
    assumptions: dict,
) -> None:
    tabs = st.tabs(
        [
            "Data Table",
            "Data Quality",
            "Market Snapshot",
            "Bedroom Analysis",
            "Project Analysis",
            "Developer Analysis",
            "Broker Analysis",
            "Outlier Detection",
            "Visualizations",
            "ROI",
            "Investment Analysis",
            "Project Investment Ranking",
        ]
    )

    with tabs[0]:
        safe_section("Data Table", render_data_table, df)

    with tabs[1]:
        safe_section("Data Quality", render_data_quality, quality_df)

    with tabs[2]:
        safe_section("Market Snapshot", render_market_snapshot, df)

    with tabs[3]:
        safe_section("Bedroom Analysis", render_bedroom_analysis, df)

    with tabs[4]:
        safe_section("Project Benchmark", render_project_analysis, df)

    with tabs[5]:
        safe_section("Developer Analysis", render_developer_analysis, df)

    with tabs[6]:
        safe_section("Broker Analysis", render_broker_analysis, df)

    with tabs[7]:
        safe_section("Outlier Detection", render_outliers, raw_df)

    with tabs[8]:
        safe_section("Visualizations", render_visualizations, df, assumptions)

    with tabs[9]:
        safe_section("ROI", render_roi_analysis, df, assumptions)

    with tabs[10]:
        safe_section("Opportunity Score", render_investment_analysis, df, assumptions)

    with tabs[11]:
        safe_section("Project Investment Ranking", render_project_investment_ranking, df, assumptions)


def safe_section(name: str, fn, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:
        st.warning(f"{name} failed: {exc}")


def render_data_table(df: pd.DataFrame) -> None:
    st.subheader("Clean Analysis Data")
    st.caption("This table reflects the current filters and excludes invalid rows. Suspicious rows are hidden by default.")
    if len(df) > 1000:
        st.info(f"Dataset has {len(df):,} rows. Showing a preview of the first 500 rows for performance.")
        st.dataframe(df.head(500), use_container_width=True, height=620)
    else:
        st.dataframe(df, use_container_width=True, height=620)
    st.download_button(
        "Download full filtered data",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="property_finder_filtered_clean_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_outliers(raw_df: pd.DataFrame) -> None:
    outliers = suspicious_listings(raw_df)
    st.subheader("Suspicious Listings")
    st.caption("These rows are flagged by segment-level IQR checks for area or AED/sqft.")
    st.metric("Flagged listings", f"{len(outliers):,}")
    preview = outliers.head(500) if len(outliers) > 1000 else outliers
    if len(outliers) > 1000:
        st.info("Showing the first 500 flagged listings. Use CSV export for the full dataset.")
    st.dataframe(preview, use_container_width=True, height=520)


def render_data_quality(quality_df: pd.DataFrame) -> None:
    st.subheader("Data Quality")
    st.caption("Invalid rows are kept in raw exports but excluded from market calculations.")
    row = quality_df.iloc[0].to_dict()
    cols = st.columns(5)
    for col, key in zip(cols, row):
        col.metric(key.replace("_", " ").title(), f"{int(row[key]):,}")
    st.dataframe(quality_df, use_container_width=True)


def render_market_snapshot(df: pd.DataFrame) -> None:
    st.subheader("Grouped Market Snapshot")
    st.caption("Market values are grouped by project, property type, and bedrooms so unlike unit types are not mixed.")
    if df.empty:
        st.warning("No listings match the current filters.")
        return

    single_segment = (
        df["project_or_community"].nunique() == 1 and df["property_type"].nunique() == 1 and df["bedrooms"].nunique() == 1
    )
    if single_segment:
        render_kpis(segment_kpis(df))
    else:
        st.info("Multiple unit types selected. Review grouped analysis below.")
    st.dataframe(cached_market_snapshot(df), use_container_width=True)


def render_bedroom_analysis(df: pd.DataFrame) -> None:
    st.subheader("Bedroom Analysis")
    st.caption("Bedrooms are compared only inside the same project and property type hierarchy.")
    analysis = cached_bedroom_analysis(df)
    st.dataframe(analysis, use_container_width=True)
    if analysis.empty:
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(
            px.bar(
                analysis,
                x="bedrooms",
                y="avg_price_aed",
                color="project_or_community",
                facet_col="property_type",
                facet_col_wrap=2,
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.bar(
                analysis,
                x="bedrooms",
                y="avg_price_per_sqft",
                color="project_or_community",
                facet_col="property_type",
                facet_col_wrap=2,
            ),
            use_container_width=True,
        )
    with col3:
        st.plotly_chart(
            px.bar(
                analysis,
                x="bedrooms",
                y="listing_count",
                color="project_or_community",
                facet_col="property_type",
                facet_col_wrap=2,
            ),
            use_container_width=True,
        )


def render_project_analysis(df: pd.DataFrame) -> None:
    st.subheader("Project Analysis")
    st.caption("Choose one property type and bedroom count to compare matching segments across projects.")
    if df.empty:
        st.warning("No listings match the current filters.")
        return
    col1, col2 = st.columns(2)
    property_type = col1.selectbox("Property Type selector", sorted(df["property_type"].dropna().astype(str).unique()))
    bedroom_options = sorted(df[df["property_type"].astype(str) == property_type]["bedrooms"].dropna().unique())
    bedrooms = col2.selectbox("Bedroom selector", bedroom_options)
    analysis = project_analysis(df, property_type=property_type, bedrooms=bedrooms)
    st.dataframe(analysis, use_container_width=True)


def render_developer_analysis(df: pd.DataFrame) -> None:
    st.subheader("Developer Analysis")
    st.caption("Developer comparisons are segmented by developer, project, property type, and bedrooms.")
    analysis = developer_analysis(df)
    st.dataframe(analysis, use_container_width=True)
    developer_source = df.copy()
    if "developer_name" not in developer_source.columns:
        developer_source["developer_name"] = pd.NA
    if "broker_name" not in developer_source.columns:
        developer_source["broker_name"] = "Unknown"
    developer_source = developer_source.assign(
        developer=developer_source["developer_name"].fillna(developer_source["broker_name"])
    )
    if not has_repeated_group(developer_source, ["developer", "property_type", "bedrooms"]):
        st.warning("Not enough repeated broker/developer data for meaningful comparison.")
        return
    chart_data = analysis[analysis["listing_count"] > 1].head(10)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(px.bar(chart_data, x="developer", y="listing_count"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(chart_data, x="developer", y="avg_price_per_sqft"), use_container_width=True)
    with col3:
        st.plotly_chart(px.bar(chart_data, x="developer", y="avg_price_aed"), use_container_width=True)


def render_broker_analysis(df: pd.DataFrame) -> None:
    st.subheader("Broker Analysis")
    st.caption("Broker rankings stay segmented by property type and bedrooms to avoid mixed inventory comparisons.")
    analysis = broker_analysis(df)
    st.dataframe(analysis, use_container_width=True)
    if not has_repeated_group(df, ["broker_name", "property_type", "bedrooms"]):
        st.warning("Not enough repeated broker/developer data for meaningful comparison.")
    rankings = broker_rankings(df)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Most Active Brokers")
        st.dataframe(rankings["most_active"].head(10), use_container_width=True)
    with col2:
        st.subheader("Highest Avg AED/sqft")
        st.dataframe(rankings["highest_avg_price_per_sqft"].head(10), use_container_width=True)
    with col3:
        st.subheader("Largest Inventory")
        st.dataframe(rankings["largest_inventory"].head(10), use_container_width=True)
    st.subheader("Agent and SuperAgent Context")
    st.dataframe(agent_analysis(df), use_container_width=True)
    st.dataframe(super_agent_comparison(df), use_container_width=True)


def render_visualizations(df: pd.DataFrame, assumptions: dict) -> None:
    st.subheader("Visualizations")
    st.caption("All charts use the filtered, analysis-ready dataset.")
    if df.empty:
        st.warning("No listings match the current filters.")
        return
    charts = [
        ("Area vs Price", area_vs_price(df)),
        ("Area vs AED/sqft", area_vs_ppsf(df)),
        ("AED/sqft Distribution by Property Type", ppsf_distribution_by_property_type(df)),
        ("AED/sqft by Bedrooms", ppsf_box_by_bedrooms(df)),
        ("AED/sqft by Property Type", ppsf_box_by_property_type(df)),
        ("Listing Count by Property Type", listing_count_by_property_type(df)),
        ("Listing Count by Bedrooms within Property Type", listing_count_by_bedrooms_within_property_type(df)),
        ("Median AED/sqft by Project + Property Type", median_ppsf_by_project_property_type(df)),
        ("Estimated ROI Distribution - Based on Assumptions", roi_distribution(df, assumptions=assumptions)),
        (
            "Opportunity Score Distribution - Pricing Attractiveness",
            opportunity_score_distribution(df, assumptions=assumptions),
        ),
        ("Investment Score Distribution", investment_score_distribution(df, assumptions=assumptions)),
    ]
    for title, fig in charts:
        st.subheader(title)
        if title.startswith("Estimated ROI"):
            st.caption("ROI is estimated from user assumptions such as appreciation, rental yield, fees, and holding period. It is not guaranteed.")
        elif title.startswith("Opportunity Score"):
            st.caption("Opportunity Score is based on relative pricing and data quality. It is not ROI.")
        elif title.startswith("Investment Score"):
            st.caption("Investment Score blends estimated ROI, pricing attractiveness, reliability, and fair-value position.")
        st.plotly_chart(fig, use_container_width=True)


def render_roi_analysis(df: pd.DataFrame, assumptions: dict) -> None:
    st.subheader("ROI")
    st.caption("ROI is estimated from user assumptions such as appreciation, rental yield, fees, and holding period. It is not guaranteed.")
    warn_missing_investment_columns(df)
    valid = cached_roi_metrics(df, assumptions)
    valid = valid[
        valid.get("price_aed").notna()
        & valid.get("area_sqft").notna()
        & valid.get("price_per_sqft").notna()
    ].copy()
    if valid.empty:
        st.warning("No listings have enough price and area data for ROI analysis.")
        return
    valid["selector_label"] = valid.apply(
        lambda row: f"{row.get('listing_id', 'No ID')} | {row.get('title', 'Untitled')} | {format_aed(row['price_aed'])}",
        axis=1,
    )
    selected_label = st.selectbox("Select listing", valid["selector_label"].tolist())
    listing = valid.loc[valid["selector_label"] == selected_label].iloc[0]
    roi = {
        "current_asking_price": listing.get("price_aed", 0),
        "estimated_fair_value": listing.get("fair_value_aed", 0),
        "uses_fallback": bool(listing.get("uses_fallback_valuation", False)),
        "low_sample_warning": listing.get("reliability_label") in {"Low", "Very low"},
        "premium_discount_pct": listing.get("premium_discount_pct", 0),
        "estimated_roi_pct": listing.get("roi_pct", 0),
        "annualized_roi_pct": listing.get("annualized_roi_pct", 0),
        "market_position": listing.get("valuation_label", "Unknown"),
    }
    if roi["low_sample_warning"]:
        st.warning("Low sample size. ROI estimate may be unreliable.")
    if roi["uses_fallback"]:
        st.info("Using fallback valuation from the same property type and bedrooms across selected projects.")

    cols = st.columns(5)
    cols[0].metric("Current Asking Price", format_aed(roi["current_asking_price"]))
    cols[1].metric("Estimated Fair Value", format_aed(roi["estimated_fair_value"]))
    cols[2].metric("Premium/Discount %", f"{float(roi['premium_discount_pct']):.1%}")
    cols[3].metric("Estimated ROI %", f"{float(roi['estimated_roi_pct']):.1f}%")
    cols[4].metric("Market Position", str(roi["market_position"]))
    st.metric("Annualized ROI %", f"{float(roi.get('annualized_roi_pct', 0)):.1f}%")


def render_investment_analysis(df: pd.DataFrame, assumptions: dict) -> None:
    st.subheader("Investment Analysis")
    st.caption("Opportunity Score is pricing attractiveness, ROI is assumption-based return, and Investment Score blends both with reliability.")
    warn_missing_investment_columns(df)
    if df.empty:
        st.warning("No listings match the current filters.")
        return
    roi_df = cached_roi_metrics(df, assumptions)
    cols = st.columns(2)
    cols[0].metric("Cheapest AED/sqft", format_number(roi_df["price_per_sqft"].min()))
    cols[1].metric("Most expensive AED/sqft", format_number(roi_df["price_per_sqft"].max()))
    st.subheader("Top 20 Opportunity Listings")
    st.caption("Sorted by pricing attractiveness and data quality. This is not ROI.")
    st.dataframe(top_opportunity_listings(df, assumptions=assumptions), use_container_width=True)
    st.subheader("Top 20 ROI Listings")
    st.caption("Sorted by assumption-based ROI using appreciation, rental yield, costs, and fees.")
    st.dataframe(top_roi_listings(df, assumptions=assumptions), use_container_width=True)
    st.subheader("Top 20 Investment Listings")
    st.caption("Sorted by blended Investment Score: ROI, Opportunity Score, reliability, and fair-value position.")
    st.dataframe(top_investment_listings(df, assumptions=assumptions), use_container_width=True)


def render_project_investment_ranking(df: pd.DataFrame, assumptions: dict) -> None:
    st.subheader("Project Investment Ranking")
    st.caption("Projects are ranked only within matching property type and bedroom segments. Unit types are not mixed.")
    warn_missing_investment_columns(df)
    ranking = cached_project_investment_ranking(df, assumptions)
    st.dataframe(ranking, use_container_width=True)
    if ranking.empty:
        return
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average Investment Score by project")
        st.plotly_chart(
            px.bar(ranking, x="project_or_community", y="avg_investment_score", color="property_type"),
            use_container_width=True,
        )
        st.subheader("Average Opportunity Score by project")
        st.plotly_chart(
            px.bar(ranking, x="project_or_community", y="avg_opportunity_score", color="property_type"),
            use_container_width=True,
        )
    with col2:
        st.subheader("Median ROI by project")
        st.plotly_chart(
            px.bar(ranking, x="project_or_community", y="median_roi_pct", color="property_type"),
            use_container_width=True,
        )
        st.subheader("Median AED/sqft by project")
        st.plotly_chart(
            px.bar(ranking, x="project_or_community", y="median_price_per_sqft", color="property_type"),
            use_container_width=True,
        )


def render_kpis(kpis: dict[str, float | int]) -> None:
    labels = [
        ("Listings", kpis["listing_count"]),
        ("Average Price", format_aed(kpis["avg_price_aed"])),
        ("Median Price", format_aed(kpis["median_price_aed"])),
        ("Minimum Price", format_aed(kpis["min_price_aed"])),
        ("Maximum Price", format_aed(kpis["max_price_aed"])),
        ("Average Area", f"{format_number(kpis['avg_area_sqft'])} sqft"),
        ("Median Area", f"{format_number(kpis['median_area_sqft'])} sqft"),
        ("Average AED/sqft", format_number(kpis["avg_price_per_sqft"])),
        ("Median AED/sqft", format_number(kpis["median_price_per_sqft"])),
        ("Lowest AED/sqft", format_number(kpis["min_price_per_sqft"])),
        ("Highest AED/sqft", format_number(kpis["max_price_per_sqft"])),
    ]
    for start in range(0, len(labels), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, labels[start : start + 4]):
            col.metric(label, value)


def warn_missing_investment_columns(df: pd.DataFrame) -> None:
    required = [
        "price_aed",
        "area_sqft",
        "price_per_sqft",
        "project_or_community",
        "property_type",
        "bedrooms",
        "is_suspicious",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        st.warning(f"Missing investment columns were filled with defaults where possible: {', '.join(missing)}")


def numeric_bounds(df: pd.DataFrame, column: str) -> tuple[float, float]:
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return 0.0, 1.0
    low = float(values.min())
    high = float(values.max())
    if low == high:
        high = low + 1.0
    return low, high


def format_aed(value: float | int) -> str:
    return f"AED {float(value):,.0f}"


def format_number(value: float | int) -> str:
    return f"{float(value):,.0f}"


if __name__ == "__main__":
    main()

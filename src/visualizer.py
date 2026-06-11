"""Plotly chart builders for segment-aware market analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from src.analyzer import add_roi_metrics


TEMPLATE = "plotly_white"


def area_vs_price(df: pd.DataFrame):
    return px.scatter(
        df,
        x="area_sqft",
        y="price_aed",
        color="project_or_community",
        hover_data=["title", "property_type", "bedrooms", "price_per_sqft"],
        template=TEMPLATE,
        labels={"area_sqft": "Area (sqft)", "price_aed": "Price (AED)", "project_or_community": "Project"},
    )


def area_vs_ppsf(df: pd.DataFrame):
    return px.scatter(
        df,
        x="area_sqft",
        y="price_per_sqft",
        color="property_type",
        hover_data=["title", "project_or_community", "bedrooms", "broker_name"],
        template=TEMPLATE,
        labels={"area_sqft": "Area (sqft)", "price_per_sqft": "AED/sqft", "property_type": "Property Type"},
    )


def ppsf_distribution_by_property_type(df: pd.DataFrame):
    return px.histogram(
        df,
        x="price_per_sqft",
        color="property_type",
        nbins=40,
        barmode="overlay",
        template=TEMPLATE,
        labels={"price_per_sqft": "AED/sqft", "property_type": "Property Type"},
    )


def ppsf_box_by_bedrooms(df: pd.DataFrame):
    return px.box(
        df,
        x="bedrooms",
        y="price_per_sqft",
        color="property_type",
        points="outliers",
        template=TEMPLATE,
        labels={"bedrooms": "Bedrooms", "price_per_sqft": "AED/sqft"},
    )


def ppsf_box_by_property_type(df: pd.DataFrame):
    return px.box(
        df,
        x="property_type",
        y="price_per_sqft",
        points="outliers",
        template=TEMPLATE,
        labels={"property_type": "Property Type", "price_per_sqft": "AED/sqft"},
    )


def listing_count_by_property_type(df: pd.DataFrame):
    grouped = df.groupby("property_type", dropna=False, as_index=False).size().rename(columns={"size": "listing_count"})
    return px.bar(
        grouped,
        x="property_type",
        y="listing_count",
        template=TEMPLATE,
        labels={"property_type": "Property Type", "listing_count": "Listing Count"},
    )


def listing_count_by_bedrooms_within_property_type(df: pd.DataFrame):
    grouped = (
        df.groupby(["property_type", "bedrooms"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "listing_count"})
    )
    return px.bar(
        grouped,
        x="bedrooms",
        y="listing_count",
        color="property_type",
        barmode="group",
        template=TEMPLATE,
        labels={"bedrooms": "Bedrooms", "listing_count": "Listing Count", "property_type": "Property Type"},
    )


def median_ppsf_by_project_property_type(df: pd.DataFrame):
    grouped = (
        df.groupby(["project_or_community", "property_type"], dropna=False, as_index=False)["price_per_sqft"]
        .median()
        .sort_values("price_per_sqft", ascending=False)
    )
    return px.bar(
        grouped,
        x="project_or_community",
        y="price_per_sqft",
        color="property_type",
        barmode="group",
        template=TEMPLATE,
        labels={"project_or_community": "Project", "price_per_sqft": "Median AED/sqft"},
    )


def roi_distribution(df: pd.DataFrame, assumptions: dict | None = None):
    roi_df = add_roi_metrics(df, assumptions=assumptions)
    return px.histogram(
        roi_df,
        x="roi_pct",
        nbins=40,
        template=TEMPLATE,
        labels={"roi_pct": "Estimated ROI %"},
    )


def opportunity_score_distribution(df: pd.DataFrame, assumptions: dict | None = None):
    roi_df = add_roi_metrics(df, assumptions=assumptions)
    return px.histogram(
        roi_df,
        x="opportunity_score",
        color="opportunity_category",
        nbins=20,
        template=TEMPLATE,
        labels={"opportunity_score": "Opportunity Score", "opportunity_category": "Category"},
    )


def investment_score_distribution(df: pd.DataFrame, assumptions: dict | None = None):
    roi_df = add_roi_metrics(df, assumptions=assumptions)
    return px.histogram(
        roi_df,
        x="investment_score",
        color="investment_category",
        nbins=20,
        template=TEMPLATE,
        labels={"investment_score": "Investment Score", "investment_category": "Category"},
    )


def avg_price_by_broker(df: pd.DataFrame, top_n: int = 20):
    grouped = (
        df.groupby("broker_name", dropna=False, as_index=False)["price_aed"]
        .mean()
        .sort_values("price_aed", ascending=False)
        .head(top_n)
    )
    return px.bar(
        grouped,
        x="broker_name",
        y="price_aed",
        template=TEMPLATE,
        labels={"broker_name": "Broker", "price_aed": "Avg price (AED)"},
    )

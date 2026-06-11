# Property Finder UAE Listing Analyzer

Streamlit dashboard for a real estate broker to scrape and analyze Property Finder UAE search listings.

## Features

- Accepts a Property Finder search URL.
- Estimates pagination from total listing count and listings found on page 1, with sequential fallback scraping when counts are unavailable.
- Adds controls for `Max Pages`, `Max Listings`, `Rate Delay`, and `Stop On Duplicate Page`.
- Parses listing data from the `__NEXT_DATA__` JSON script first.
- Falls back to selectors based on `data-testid` attributes when embedded JSON is unavailable.
- Keeps raw scraped rows, then excludes subcommunity, missing price, missing area, invalid property, and suspicious rows from analysis by default.
- Uses the primary hierarchy: Project/Community -> Property Type -> Bedrooms -> Area -> Non-Suspicious Listings.
- Provides dashboard pages for data, data quality, market snapshot, bedroom analysis, project analysis, developer analysis, broker analysis, outliers, visualizations, ROI, and investment analysis.
- Uses `developer_name` when available and falls back to `broker_name` for developer/broker comparison.
- Connects scraped segment median AED/sqft data to fair value, ROI, market position, and opportunity scoring.
- Exports cleaned listings as CSV and a multi-sheet Excel report with raw data, clean listings, data quality, market snapshot, bedroom/project/developer/broker analysis, ROI analysis, and investment opportunities.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

Property Finder can change its Next.js data structure and may use bot protection. This project avoids random CSS class names and uses a defensive JSON traversal, then a `data-testid` fallback parser.

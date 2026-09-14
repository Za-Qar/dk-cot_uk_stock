import re
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from google.cloud import bigquery

from uk_dkcot.config import END_DATE, RAW_DATA_DIR, START_DATE, Company, load_companies

DRY_RUN = False
GDELT_QUERY = """
    SELECT
        date,
        url,
        lang,
        title
    FROM `gdelt-bq.gdeltv2.gsg_docembed`
    WHERE DATE(date) BETWEEN @start_date AND @end_date
    AND UPPER(lang) = 'ENGLISH'
    AND title IS NOT NULL
    AND REGEXP_CONTAINS(LOWER(title), @alias_pattern)
    """

def build_alias_pattern(companies: list[Company]):
    """Combine all company aliases into one BigQuery search pattern."""

    aliases = set()

    for company in companies:
        for alias in company.aliases:
            aliases.add(alias.lower())

    # Longest first, with ties sorted alphabetically so the pattern is identical on every run.
    ordered_aliases = sorted(aliases, key=lambda alias: (-len(alias), alias))
    joined_aliases = "|".join(ordered_aliases)

    return f"(^|[^a-z0-9])({joined_aliases})([^a-z0-9]|$)"

def find_matching_company(title: str, companies: list[Company]):
    """Return the one company mentioned in the headline."""

    matches = []

    for company in companies:
        for alias in company.aliases:
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"

            if re.search(pattern, title, re.IGNORECASE):
                matches.append(company)
                break

    # A multi-company headline may have different sentiment implications for each company, so retain only headlines that map clearly to one company.
    if len(matches) == 1:
        return matches[0]

    return None

def collect_prices(companies):
    """Collect and save daily prices for all companies."""

    output_path = RAW_DATA_DIR / "yfinance_prices.csv"

    if output_path.exists():
        print(f"Prices already exist; skipping: {output_path}")
        return

    frames = []
    # Include the first 2026 trading day needed to execute 31 December headlines.
    # yfinance treats its end date as exclusive.
    final_date = date.fromisoformat(END_DATE) + timedelta(days=3)

    for company in companies:
        print(f"Downloading prices for {company.ticker}")

        company_prices = yf.download(
            company.ticker,
            start=START_DATE,
            end=final_date.isoformat(),
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )

        if company_prices.empty:
            print(f"No prices returned for {company.ticker}")
            continue

        company_prices = company_prices.reset_index()
        company_prices = company_prices.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        company_prices["ticker"] = company.ticker
        company_prices["date"] = pd.to_datetime(
            company_prices["date"]
        ).dt.strftime("%Y-%m-%d")

        frames.append(
            company_prices[
                [
                    "date",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj_close",
                    "volume",
                ]
            ]
        )

    prices_df = pd.concat(frames, ignore_index=True)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices_df.to_csv(output_path, index=False)

    print(f"Saved {len(prices_df)} price rows to {output_path}")

def collect_headlines(companies):
    """Collect and save GDELT headlines."""
    alias_pattern = build_alias_pattern(companies)

    bigquery_client = bigquery.Client(project="uk-dkcot")

    query_config = bigquery.QueryJobConfig(
        dry_run=DRY_RUN,
        use_query_cache=not DRY_RUN,
        maximum_bytes_billed=30_000_000_000,
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "start_date", "DATE", date.fromisoformat(START_DATE)
            ),
            bigquery.ScalarQueryParameter(
                "end_date", "DATE", date.fromisoformat(END_DATE)
            ),
            bigquery.ScalarQueryParameter(
                "alias_pattern", "STRING", alias_pattern
            ),
        ],
    )

    query_job = bigquery_client.query(
        GDELT_QUERY,
        job_config=query_config,
    )

    if DRY_RUN:
        gigabytes = query_job.total_bytes_processed / 1_000_000_000
        print(f"Dry run: {gigabytes:.2f} GB will be processed")
        return

    rows = query_job.result()
    print(f"Query returned {rows.total_rows} rows")
    headlines = []

    for row in rows:
        title = row.title.strip()
        company = find_matching_company(title, companies)

        # Exclude headlines matching zero or multiple companies.
        if company is None:
            continue

        headlines.append(
            {
                "published_at_utc": row.date.isoformat(),
                "url": row.url,
                "ticker": company.ticker,
                "company_name": company.company_name,
                "headline_text": title,
            }
        )

    headlines_df = pd.DataFrame(headlines)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / "gdelt_headlines.csv"
    headlines_df.to_csv(output_path, index=False)

    print(f"Saved {len(headlines_df)} headlines to {output_path}")


def main():
    """Collect the raw headline and price datasets."""

    companies = load_companies()
    headlines_path = RAW_DATA_DIR / "gdelt_headlines.csv"

    if headlines_path.exists():
        print(f"Headlines already exist; skipping: {headlines_path}")
    else:
        collect_headlines(companies)

    collect_prices(companies)


# __name__ is a special variable automatically created by Python
# This means that, if this file was run directly, call main().
if __name__ == "__main__":
    main()


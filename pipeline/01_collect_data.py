import re
import pandas as pd
import yfinance as yf

from google.cloud import bigquery
from datetime import date, timedelta
from uk_dkcot.config import (
    PD_DATE_FORMAT,
    END_DATE,
    RAW_DATA_DIR,
    START_DATE,
    GOOGLE_CLOUD_PROJECT_ID,
    GDELT_HEADLINES_CSV_FILE_NAME,
    Company,
    load_companies,
)

# This was added initially to test whether the Google BigQuery would be able to process the 20.49 GB data before running it. This is because BigQuery had to search a large amount of GDELT data before being able to return the small subset of relevant headline data for the defined ten companies.
DRY_RUN = False

# - https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/string_functions#regexp_contains
# - The `gdeltv2.gsg_docembed` is the table provided and maintained by GDELT and hosted on Google BigQuery and holds a large amount of records about global news articles:
    # - https://blog.gdeltproject.org/announcing-the-global-similarity-graph-document-embeddings-using-the-universal-sentence-encoder/
    # - https://console.cloud.google.com/bigquery?project=gdelt-bq&ws=&p=gdelt-bq&d=gdeltv2&t=gsg_docembed&page=table&pli=1
# The SELECT clause retrieves article metadata and titles.
# REGEXP_CONTAINS keeps titles that mention one of the configured company aliases.
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


def build_alias_pattern(companies: list[Company]) -> str:
    """Combine all company aliases into one BigQuery search pattern."""

    # Using a `set()` to ensure there are no duplicate aliases
    aliasesSet = set()

    for company in companies:
        for alias in company.aliases:
            aliasesSet.add(alias.lower())

    ordered_aliases = sorted(aliasesSet, key=len, reverse=True)
    joined_aliases = "|".join(ordered_aliases)

    return f"(^|[^a-z0-9])({joined_aliases})([^a-z0-9]|$)"


def find_matching_company(title: str, companies: list[Company]) -> Company | None:
    """Return the one company mentioned in the headline."""

    matches = []

    for company in companies:
        for alias in company.aliases:
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"

            if re.search(pattern, title, re.IGNORECASE):
                matches.append(company)
                break

    # Ignore headlines that match no company or more than one company. This is to only retain headlines that can be mapped unambiguously to one company which is supported by Sinha et al. (2022) when discussing the entity-aware financial sentiment research, which shows that one headline can mention several companies with conflicting sentiments (doi: https://doi.org/10.1002/asi.24634).
    if len(matches) != 1:
        return None

    return matches[0]


def collect_prices(companies) -> None:
    """Collect and save daily prices for all companies from yfinance."""

    output_path = RAW_DATA_DIR / "yfinance_prices.csv"

    if output_path.exists():
        print(f"Prices already exist; skipping: {output_path}")
        return

    frames = []
    
    # Extend the range (`timedelta(days=3)`) to include 2 January 2026, the first trading day after 31 December 2025.
    # Since yfinance excludes the specified end date, 3 January is passed as the boundary.
    final_date = date.fromisoformat(END_DATE) + timedelta(days=3)

    for company in companies:
        print(f"Downloading prices for {company.ticker}")

        # https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
        company_prices = yf.download(
            company.ticker,
            start=START_DATE,
            end=final_date.isoformat(),
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )

        if company_prices.empty:
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
        ).dt.strftime(PD_DATE_FORMAT)

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

    # Save the `prices_df (the dataframe)
    prices_df.to_csv(output_path, index=False)


def collect_headlines(companies) -> None:
    """Collect and save GDELT headlines from Google BigQuery."""
    alias_pattern = build_alias_pattern(companies)

    bigquery_client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT_ID)

    # https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.job.QueryJobConfig
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

    # BigQuery reports the estimated amount of processed data in bytes. The `1_000_000_000` is used to display the amount in gigabytes
    if DRY_RUN:
        gigabytes = query_job.total_bytes_processed / 1_000_000_000
        print(f"Dry run: {gigabytes:.2f} GB will be processed")
        return

    rows = query_job.result()
    headlines = []

    for row in rows:
        title = row.title.strip()
        company = find_matching_company(title, companies)

        # Skip headlines that could not be mapped to one company
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
    output_path = RAW_DATA_DIR / GDELT_HEADLINES_CSV_FILE_NAME
    headlines_df.to_csv(output_path, index=False)

    print(f"Saved {len(headlines_df)} headlines to {output_path}")


def main():
    """Collect the raw headline and price datasets."""

    companies = load_companies()
    headlines_path = RAW_DATA_DIR / GDELT_HEADLINES_CSV_FILE_NAME

    if headlines_path.exists():
        print(f"Headlines already exist; skipping: {headlines_path}")
    else:
        collect_headlines(companies)

    collect_prices(companies)


if __name__ == "__main__":
    main()


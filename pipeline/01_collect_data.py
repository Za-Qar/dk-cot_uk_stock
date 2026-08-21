from uk_dkcot.config import END_DATE, START_DATE, load_companies, Company, RAW_DATA_DIR
from google.cloud import bigquery
from datetime import date
import re
import pandas as pd

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

    # The longest first so more specific names match before shorter ones. This helps avoid short aliases accidentally matching inside longer text first.
    ordered_aliases = sorted(aliases, key=len, reverse=True)
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

    if len(matches) == 1:
        return matches[0]

    return None


def main():
    """Load and display the configured companies."""
    companies = load_companies()
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
    headlines = []

    print("---------------------Here:")
    print(rows)

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


# __name__ is a special variable automatically created by Python
# This means that, if this file was run directly, call main().
if __name__ == "__main__":
    main()

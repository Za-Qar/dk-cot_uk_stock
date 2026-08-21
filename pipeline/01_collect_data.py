from uk_dkcot.config import END_DATE, START_DATE, load_companies, Company
from google.cloud import bigquery
from datetime import date

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
    print("-------------------------here:")
    print(ordered_aliases)
    joined_aliases = "|".join(ordered_aliases)

    return f"(^|[^a-z0-9])({joined_aliases})([^a-z0-9]|$)"

def main():
    """Load and display the configured companies."""
    companies = load_companies()
    alias_pattern = build_alias_pattern(companies)

    bigquery_client = bigquery.Client(project="uk-dkcot")

    query_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
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

    gigabytes = query_job.total_bytes_processed / 1_000_000_000
    print(f"Dry run: {gigabytes:.2f} GB will be processed")


# __name__ is a special variable automatically created by Python
# This means that, if this file was run directly, call main().
if __name__ == "__main__":
    main()

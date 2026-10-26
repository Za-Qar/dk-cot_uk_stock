import hashlib
import pandas as pd

from uk_dkcot.config import (
    GDELT_HEADLINES_CLEAN_CSV_FILE_NAME,
    HEADLINE_ID_COLUMN,
    HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME,
    PD_DATE_FORMAT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    YFINANCE_PRICES_CSV_FILE_NAME,
)

TICKER_COLUMN = "ticker"
PUBLISHED_AT_UTC_COLUMN = "published_at_utc"
HEADLINE_TEXT_COLUMN = "headline_text"
PUBLISHED_AT_LONDON_COLUMN = "published_at_london"
HEADLINE_DATE_LONDON_COLUMN = "headline_date_london"
LONDON_TIMEZONE = "Europe/London"
DATE_COLUMN = "date"
TRADE_DATE_COLUMN = "trade_date"


def create_headline_id(row):
    """Create a stable identifier from the company, timestamp and headline."""

    identifying_text = "|".join(
        (
            row[TICKER_COLUMN],
            row[PUBLISHED_AT_UTC_COLUMN],
            row[HEADLINE_TEXT_COLUMN],
        )
    )
    return hashlib.sha256(identifying_text.encode("utf-8")).hexdigest()


def align_company_headlines(headlines, prices):
    """Assign each headline to the company's next available trading-day open."""

    # https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html
    return pd.merge_asof(
        headlines.sort_values(HEADLINE_DATE_LONDON_COLUMN),
        prices.sort_values(TRADE_DATE_COLUMN),
        left_on=HEADLINE_DATE_LONDON_COLUMN,
        right_on=TRADE_DATE_COLUMN,
        direction="forward",
        allow_exact_matches=False,
    )


def main():
    """Align cleaned headlines with next-trading-day opening prices."""

    headlines_path = PROCESSED_DATA_DIR / GDELT_HEADLINES_CLEAN_CSV_FILE_NAME
    prices_path = RAW_DATA_DIR / YFINANCE_PRICES_CSV_FILE_NAME
    output_path = PROCESSED_DATA_DIR / HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME

    headlines_df = pd.read_csv(headlines_path)
    prices_df = pd.read_csv(prices_path)

    published_at_utc = pd.to_datetime(
        headlines_df[PUBLISHED_AT_UTC_COLUMN],
        utc=True,
        errors="raise",
    )
    # Convert each UTC publication timestamp into UK local time
    headlines_df[PUBLISHED_AT_LONDON_COLUMN] = published_at_utc.dt.tz_convert(
        LONDON_TIMEZONE
    )
    # Remove the timezone and time so the local date can be matched with daily prices
    headlines_df[HEADLINE_DATE_LONDON_COLUMN] = (
        headlines_df[PUBLISHED_AT_LONDON_COLUMN].dt.tz_localize(None).dt.normalize()
    )

    prices_df[DATE_COLUMN] = pd.to_datetime(prices_df[DATE_COLUMN], errors="raise")

    aligned_companies = []

    # Process each company separately so its headlines only use its own respective prices
    for ticker in sorted(headlines_df[TICKER_COLUMN].unique()):
        # Select only the current company's headlines
        company_headlines = headlines_df.loc[
            headlines_df[TICKER_COLUMN] == ticker
        ].copy()
        # Select the current company's dates and opening prices
        company_prices = prices_df.loc[
            prices_df[TICKER_COLUMN] == ticker,
            [DATE_COLUMN, "open"],
        ].rename(columns={DATE_COLUMN: TRADE_DATE_COLUMN, "open": "trade_open"})

        # Match each headline to this company's next trading-day opening price
        aligned_companies.append(
            align_company_headlines(company_headlines, company_prices)
        )

    aligned_df = pd.concat(aligned_companies, ignore_index=True)

    # Find headlines for which no later trading-day price was available
    missing_prices = aligned_df[TRADE_DATE_COLUMN].isna()
    if missing_prices.any():
        missing_count = int(missing_prices.sum())
        raise ValueError(
            f"No later trading-day price exists for {missing_count} headlines. "
            "Extend the price data beyond the final headline date."
        )

    # Confirm that the alignment did not add or remove headline rows
    assert len(aligned_df) == len(headlines_df)
    assert (
        aligned_df[TRADE_DATE_COLUMN] > aligned_df[HEADLINE_DATE_LONDON_COLUMN]
    ).all()

    aligned_df = aligned_df.sort_values(
        [PUBLISHED_AT_UTC_COLUMN, TICKER_COLUMN, HEADLINE_TEXT_COLUMN]
    ).reset_index(drop=True)

    aligned_df.insert(
        0,
        HEADLINE_ID_COLUMN,
        aligned_df.apply(create_headline_id, axis=1),
    )
    # Confirm that every generated headline identifier is unique
    assert aligned_df[HEADLINE_ID_COLUMN].is_unique

    # Save both date columns in a consistent year-month-day format
    aligned_df[HEADLINE_DATE_LONDON_COLUMN] = aligned_df[
        HEADLINE_DATE_LONDON_COLUMN
    ].dt.strftime(PD_DATE_FORMAT)
    aligned_df[TRADE_DATE_COLUMN] = aligned_df[TRADE_DATE_COLUMN].dt.strftime(
        PD_DATE_FORMAT
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(output_path, index=False)

    print(f"Aligned {len(aligned_df):,} headlines.")
    print(f"Saved aligned data to {output_path}")


if __name__ == "__main__":
    main()

import hashlib

import pandas as pd

from uk_dkcot.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def create_headline_id(row):
    """Create a stable identifier from the company, timestamp and headline."""

    identifying_text = (
        f"{row['ticker']}|{row['published_at_utc']}|{row['headline_text']}"
    )
    return hashlib.sha256(identifying_text.encode("utf-8")).hexdigest()


def align_company_headlines(headlines, prices):
    """Assign each headline to the company's next available trading-day open."""

    return pd.merge_asof(
        headlines.sort_values("headline_date_london"),
        prices.sort_values("trade_date"),
        left_on="headline_date_london",
        right_on="trade_date",
        direction="forward",
        allow_exact_matches=False,
    )


def main():
    """Align cleaned headlines with next-trading-day opening prices."""

    headlines_path = PROCESSED_DATA_DIR / "gdelt_headlines_clean.csv"
    prices_path = RAW_DATA_DIR / "yfinance_prices.csv"
    output_path = PROCESSED_DATA_DIR / "headlines_aligned_prices.csv"

    headlines_df = pd.read_csv(headlines_path)
    prices_df = pd.read_csv(prices_path)

    published_at_utc = pd.to_datetime(
        headlines_df["published_at_utc"],
        utc=True,
        errors="raise",
    )
    headlines_df["published_at_london"] = published_at_utc.dt.tz_convert(
        "Europe/London"
    )
    headlines_df["headline_date_london"] = (
        headlines_df["published_at_london"].dt.tz_localize(None).dt.normalize()
    )

    prices_df["date"] = pd.to_datetime(prices_df["date"], errors="raise")

    aligned_companies = []

    for ticker in sorted(headlines_df["ticker"].unique()):
        company_headlines = headlines_df.loc[
            headlines_df["ticker"] == ticker
        ].copy()
        company_prices = prices_df.loc[
            prices_df["ticker"] == ticker,
            ["date", "open"],
        ].rename(columns={"date": "trade_date", "open": "trade_open"})

        aligned_companies.append(
            align_company_headlines(company_headlines, company_prices)
        )

    aligned_df = pd.concat(aligned_companies, ignore_index=True)

    missing_prices = aligned_df["trade_date"].isna()
    if missing_prices.any():
        missing_count = int(missing_prices.sum())
        raise ValueError(
            f"No later trading-day price exists for {missing_count} headlines. "
            "Extend the price data beyond the final headline date."
        )

    assert len(aligned_df) == len(headlines_df)
    assert (
        aligned_df["trade_date"] > aligned_df["headline_date_london"]
    ).all()

    aligned_df = aligned_df.sort_values(
        ["published_at_utc", "ticker", "headline_text"]
    ).reset_index(drop=True)

    aligned_df.insert(
        0,
        "headline_id",
        aligned_df.apply(create_headline_id, axis=1),
    )
    assert aligned_df["headline_id"].is_unique

    aligned_df["published_at_london"] = aligned_df[
        "published_at_london"
    ].astype(str)
    aligned_df["headline_date_london"] = aligned_df[
        "headline_date_london"
    ].dt.strftime("%Y-%m-%d")
    aligned_df["trade_date"] = aligned_df["trade_date"].dt.strftime("%Y-%m-%d")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(output_path, index=False)

    print(f"Aligned {len(aligned_df):,} headlines.")
    print(f"Saved aligned data to {output_path}")


if __name__ == "__main__":
    main()

import argparse
import hashlib
import importlib.util
import re
import unicodedata

import numpy as np
import pandas as pd

from uk_dkcot.config import (
    DKCOT_MODEL_ID,
    END_DATE,
    FINBERT_MODEL_ID,
    KNOWLEDGE_LEVELS,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    START_DATE,
    load_companies,
)


VALID_LABELS = {"positive", "neutral", "negative"}
GOLD_COLUMNS = ["headline_id", "label_gold", "annotator_pass", "qa_flag"]
HEADLINE_FIELDS = [
    "headline_id",
    "source",
    "published_at_utc",
    "published_at_london",
    "ticker",
    "company_name",
    "headline_text",
    "mapping_confidence",
]
SAMPLE_PER_COMPANY = 30
MAX_HEADLINE_GAP_DAYS = 7
TABLES_DIRECTORY = PROJECT_ROOT / "results" / "tables"
RERUN_CHECKS_PATH = TABLES_DIRECTORY / "model_rerun_checks.csv"


def load_stage(file_name):
    """Import a numbered pipeline script so its functions can be reused."""

    path = PROJECT_ROOT / "pipeline" / file_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_headline(value):
    """Apply the same duplicate-detection normalisation as notebook 02."""

    normalized = unicodedata.normalize("NFKC", value)
    without_formatting_marks = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    return re.sub(r"\s+", " ", without_formatting_marks).strip().casefold()


def check(name, passed, detail, warn_only=False):
    """Return one row of the checks table."""

    if passed:
        status = "pass"
    else:
        status = "warn" if warn_only else "fail"
    return {"check": name, "status": status, "detail": detail}


def check_headlines(clean_df, aligned_df, companies):
    """Check duplicates, company mapping and the headline schema."""

    normalized = clean_df["headline_text"].map(normalize_headline)
    repeated = int(
        pd.DataFrame({"ticker": clean_df["ticker"], "headline": normalized})
        .duplicated()
        .sum()
    )

    collect = load_stage("01_collect_data.py")
    matched_tickers = clean_df["headline_text"].map(
        lambda title: getattr(
            collect.find_matching_company(title, companies), "ticker", None
        )
    )
    wrong_company = int((matched_tickers != clean_df["ticker"]).sum())

    missing_fields = sorted(set(HEADLINE_FIELDS) - set(aligned_df.columns))
    empty_fields = (
        int(aligned_df[HEADLINE_FIELDS].isna().any(axis=1).sum())
        if not missing_fields
        else 0
    )

    return [
        check(
            "no duplicate headlines",
            repeated == 0 and aligned_df["headline_id"].is_unique,
            f"{repeated} repeated (ticker, normalised headline) pairs",
        ),
        check(
            "each headline matches only its own company",
            wrong_company == 0,
            f"{wrong_company} of {len(clean_df):,} headlines fail the alias match",
        ),
        check(
            "headline table has the specified fields",
            not missing_fields and empty_fields == 0,
            f"missing: {missing_fields or 'none'}; rows with empty fields: {empty_fields}",
        ),
    ]


def check_coverage(aligned_df):
    """Find long periods of the study window with no headlines at all."""

    headline_dates = pd.to_datetime(aligned_df["published_at_utc"], utc=True)
    dates = pd.Series(
        sorted(
            {
                pd.Timestamp(START_DATE),
                *headline_dates.dt.tz_localize(None).dt.normalize(),
                pd.Timestamp(END_DATE),
            }
        )
    )
    gap_days = dates.diff().dt.days
    long_gaps = [
        f"{(date - pd.Timedelta(days=days)).date()} to {date.date()} ({days} days)"
        for date, days in zip(dates, gap_days.fillna(0).astype(int))
        if days > MAX_HEADLINE_GAP_DAYS
    ]

    return [
        check(
            "headlines cover the whole study period",
            not long_gaps,
            "; ".join(long_gaps) or f"no gap longer than {MAX_HEADLINE_GAP_DAYS} days",
            warn_only=True,
        )
    ]


def check_prices(prices_df, companies):
    """Check the price file for gaps, duplicates and invalid values."""

    price_columns = ["open", "high", "low", "close", "adj_close"]
    invalid_rows = int(
        (prices_df[price_columns].isna() | (prices_df[price_columns] <= 0))
        .any(axis=1)
        .sum()
    )
    duplicates = int(prices_df.duplicated(["ticker", "date"]).sum())
    tickers_match = set(prices_df["ticker"]) == {c.ticker for c in companies}

    all_dates = set(prices_df["date"])
    gaps = {
        ticker: sorted(all_dates - set(ticker_prices["date"]))
        for ticker, ticker_prices in prices_df.groupby("ticker")
    }
    gaps = {ticker: dates for ticker, dates in gaps.items() if dates}
    gap_detail = "; ".join(
        f"{ticker} missing {', '.join(dates)}" for ticker, dates in gaps.items()
    )

    return [
        check(
            "price file is valid",
            tickers_match and duplicates == 0 and invalid_rows == 0,
            f"{prices_df['ticker'].nunique()} tickers, {duplicates} duplicate rows, "
            f"{invalid_rows} rows with missing or non-positive prices",
        ),
        check(
            "no missing trading dates",
            not gaps,
            gap_detail or "every ticker has every trading date",
            warn_only=True,
        ),
    ]


def check_alignment(aligned_df, prices_df):
    """Recalculate the London dates and trade dates independently."""

    london_time = pd.to_datetime(
        aligned_df["published_at_utc"], utc=True
    ).dt.tz_convert("Europe/London")
    london_matches = (
        london_time.astype(str).eq(aligned_df["published_at_london"]).all()
        and london_time.dt.strftime("%Y-%m-%d")
        .eq(aligned_df["headline_date_london"])
        .all()
    )

    price_dates = {
        ticker: np.array(sorted(ticker_prices["date"]))
        for ticker, ticker_prices in prices_df.groupby("ticker")
    }
    expected_trade_dates = []
    for ticker, headline_date in zip(
        aligned_df["ticker"], aligned_df["headline_date_london"]
    ):
        dates = price_dates[ticker]
        position = np.searchsorted(dates, headline_date, side="right")
        expected_trade_dates.append(
            dates[position] if position < len(dates) else None
        )
    wrong_trade_date = int(
        (aligned_df["trade_date"] != pd.Series(expected_trade_dates)).sum()
    )
    look_ahead = int(
        (aligned_df["trade_date"] <= aligned_df["headline_date_london"]).sum()
    )

    open_prices = prices_df.set_index(["ticker", "date"])["open"]
    saved_opens = open_prices.reindex(
        list(zip(aligned_df["ticker"], aligned_df["trade_date"]))
    ).to_numpy()
    wrong_open = int((~np.isclose(saved_opens, aligned_df["trade_open"])).sum())

    return [
        check(
            "London time conversion",
            london_matches,
            "London timestamps and dates recalculated from UTC",
        ),
        check(
            "next trading day rule",
            wrong_trade_date == 0 and wrong_open == 0,
            f"{wrong_trade_date} wrong trade dates, {wrong_open} wrong opening prices",
        ),
        check(
            "no look-ahead",
            look_ahead == 0,
            f"{look_ahead} headlines trade on or before their publication date",
        ),
    ]


def check_predictions(aligned_df, finbert_df, dkcot_df):
    """Check that every model variant has one valid label per headline."""

    headline_ids = set(aligned_df["headline_id"])
    finbert_complete = (
        len(finbert_df) == len(aligned_df)
        and set(finbert_df["headline_id"]) == headline_ids
        and set(finbert_df["model_id"]) == {FINBERT_MODEL_ID}
    )
    dkcot_complete = (
        len(dkcot_df) == len(aligned_df) * len(KNOWLEDGE_LEVELS)
        and not dkcot_df.duplicated(["headline_id", "knowledge_level"]).any()
        and set(dkcot_df["headline_id"]) == headline_ids
        and set(dkcot_df["knowledge_level"]) == set(KNOWLEDGE_LEVELS)
        and set(dkcot_df["model_id"]) == {DKCOT_MODEL_ID}
    )
    labels_valid = (
        set(finbert_df["label_pred"]) | set(dkcot_df["label_pred"])
    ) <= VALID_LABELS

    unparsed = dkcot_df.loc[
        ~dkcot_df["parse_ok"].astype(str).str.lower().eq("true")
    ]
    unparsed_by_level = (
        unparsed["knowledge_level"]
        .value_counts()
        .reindex(KNOWLEDGE_LEVELS, fill_value=0)
    )

    return [
        check(
            "one prediction per headline and variant",
            finbert_complete and dkcot_complete and labels_valid,
            f"FinBERT {len(finbert_df):,} rows, DK-CoT {len(dkcot_df):,} rows",
        ),
        check(
            "DK-CoT responses parsed",
            unparsed.empty,
            f"{len(unparsed)} unparsed responses set to neutral ("
            + ", ".join(f"{level} {count}" for level, count in unparsed_by_level.items())
            + ")",
            warn_only=True,
        ),
    ]


def check_gold_labels(aligned_df, gold_df):
    """Check the gold-label schema, balance and annotation passes."""

    tickers = gold_df.merge(
        aligned_df[["headline_id", "ticker"]], on="headline_id", how="left"
    )["ticker"]
    per_company = tickers.value_counts()
    complete = (
        gold_df.columns.tolist() == GOLD_COLUMNS
        and gold_df["headline_id"].is_unique
        and tickers.notna().all()
        and len(per_company) == 10
        and per_company.eq(SAMPLE_PER_COMPANY).all()
        and set(gold_df["label_gold"]) <= VALID_LABELS
    )

    # Every sampled headline is either labelled or replaced by one from the same company.
    sample_ids = set(
        aligned_df.groupby("ticker", group_keys=False)
        .sample(n=SAMPLE_PER_COMPANY, random_state=RANDOM_SEED)["headline_id"]
    )
    gold_ids = set(gold_df["headline_id"])
    replaced = len(sample_ids - gold_ids)
    replacements = len(gold_ids - sample_ids)

    second_pass = int(gold_df["annotator_pass"].eq(2).sum()) if complete else 0
    changed = int(gold_df["qa_flag"].sum()) if complete else 0

    return [
        check(
            "gold labels are complete",
            complete,
            f"{len(gold_df)} labels, {per_company.min()} to {per_company.max()} per company",
        ),
        check(
            "gold sample reproduces from the seed",
            replaced == replacements,
            f"{len(sample_ids) - replaced} sampled headlines kept, "
            f"{replaced} replaced as wrong-company",
        ),
        check(
            "second annotation pass",
            second_pass == len(gold_df),
            f"{second_pass} of {len(gold_df)} labels checked twice, {changed} changed",
            warn_only=True,
        ),
    ]


def check_model_reruns(aligned_df, gold_df, finbert_df, dkcot_df):
    """Rerun both models on the gold headlines and compare with saved labels."""

    import torch

    gold_headlines = aligned_df.loc[
        aligned_df["headline_id"].isin(gold_df["headline_id"])
    ].reset_index(drop=True)

    finbert = load_stage("05_finbert.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = finbert.load_model(device)
    finbert_rerun = finbert.predict_headlines(gold_headlines, tokenizer, model, device)
    finbert_compared = finbert_rerun.merge(
        finbert_df, on="headline_id", suffixes=("_rerun", "_saved")
    )
    finbert_same = int(
        (finbert_compared["label_pred_rerun"] == finbert_compared["label_pred_saved"]).sum()
    )
    del model
    torch.cuda.empty_cache()

    dkcot = load_stage("06_dkcot.py")
    companies = {company.ticker: company for company in load_companies()}
    tokenizer, model = dkcot.load_model()
    rerun_rows = []

    for knowledge_level in KNOWLEDGE_LEVELS:
        for start_index in range(0, len(gold_headlines), dkcot.BATCH_SIZE):
            batch = gold_headlines.iloc[start_index:start_index + dkcot.BATCH_SIZE]
            prompts = [
                dkcot.build_prompt(row.headline_text, companies[row.ticker], knowledge_level)
                for row in batch.itertuples(index=False)
            ]
            responses = dkcot.generate_responses(prompts, tokenizer, model)
            rerun_rows.extend(
                (row.headline_id, knowledge_level, dkcot.parse_sentiment(response)[0])
                for row, response in zip(batch.itertuples(index=False), responses)
            )
        print(f"Rerun DK-CoT ({knowledge_level}) on {len(gold_headlines)} headlines")

    dkcot_rerun = pd.DataFrame(
        rerun_rows, columns=["headline_id", "knowledge_level", "label_rerun"]
    )
    dkcot_compared = dkcot_rerun.merge(
        dkcot_df, on=["headline_id", "knowledge_level"]
    )
    dkcot_same = int(
        (dkcot_compared["label_rerun"] == dkcot_compared["label_pred"]).sum()
    )

    return [
        check(
            "FinBERT predictions reproduce",
            finbert_same == len(finbert_compared),
            f"{finbert_same} of {len(finbert_compared)} gold-headline labels identical",
            warn_only=True,
        ),
        check(
            "DK-CoT predictions reproduce",
            dkcot_same == len(dkcot_compared),
            f"{dkcot_same} of {len(dkcot_compared)} gold-headline labels identical "
            "(batches differ from the original run)",
            warn_only=True,
        ),
    ]


def write_manifest(output_path):
    """Record the row count and SHA-256 hash of every data file and table."""

    skipped = {"data_manifest.csv", "technical_checks.csv", RERUN_CHECKS_PATH.name}
    paths = sorted(
        path
        for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIRECTORY]
        for path in directory.glob("*.csv")
        if path.name not in skipped
    )
    manifest_df = pd.DataFrame(
        {
            "file": [path.relative_to(PROJECT_ROOT).as_posix() for path in paths],
            "rows": [len(pd.read_csv(path)) for path in paths],
            "sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        }
    )
    manifest_df.to_csv(output_path, index=False)
    return len(manifest_df)


def main():
    """Run the technical-correctness checks and save the results."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--rerun-models",
        action="store_true",
        help="rerun FinBERT and DK-CoT on the gold headlines (DK-CoT needs a GPU)",
    )
    arguments = parser.parse_args()

    companies = load_companies()
    prices_df = pd.read_csv(RAW_DATA_DIR / "yfinance_prices.csv")
    clean_df = pd.read_csv(PROCESSED_DATA_DIR / "gdelt_headlines_clean.csv")
    aligned_df = pd.read_csv(PROCESSED_DATA_DIR / "headlines_aligned_prices.csv")
    gold_df = pd.read_csv(
        PROCESSED_DATA_DIR / "gold_labels.csv", keep_default_na=False
    )
    finbert_df = pd.read_csv(PROCESSED_DATA_DIR / "finbert_predictions.csv")
    dkcot_df = pd.read_csv(PROCESSED_DATA_DIR / "dkcot_predictions.csv")

    results = [
        *check_headlines(clean_df, aligned_df, companies),
        *check_coverage(aligned_df),
        *check_prices(prices_df, companies),
        *check_alignment(aligned_df, prices_df),
        *check_predictions(aligned_df, finbert_df, dkcot_df),
        *check_gold_labels(aligned_df, gold_df),
    ]

    # The slow model reruns are saved separately and reused on later runs.
    TABLES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if arguments.rerun_models:
        pd.DataFrame(
            check_model_reruns(aligned_df, gold_df, finbert_df, dkcot_df)
        ).to_csv(RERUN_CHECKS_PATH, index=False)
    if RERUN_CHECKS_PATH.exists():
        results.extend(pd.read_csv(RERUN_CHECKS_PATH).to_dict("records"))

    file_count = write_manifest(TABLES_DIRECTORY / "data_manifest.csv")
    results.append(
        check("output manifest", True, f"SHA-256 recorded for {file_count} files")
    )

    results_df = pd.DataFrame(results)
    results_path = TABLES_DIRECTORY / "technical_checks.csv"
    results_df.to_csv(results_path, index=False)

    print(results_df.to_string(index=False))
    print(f"\nSaved checks to {results_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv

from dataclasses import dataclass
from pathlib import Path
from enum import StrEnum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Shared input and output filenames
COMPANIES_CSV_FILE_NAME = "companies.csv"
GDELT_HEADLINES_CSV_FILE_NAME = "gdelt_headlines.csv"
YFINANCE_PRICES_CSV_FILE_NAME = "yfinance_prices.csv"
GDELT_HEADLINES_DEDUPLICATED_CSV_FILE_NAME = "gdelt_headlines_deduplicated.csv"
HEADLINE_MAPPING_REVIEW_CSV_FILE_NAME = "headline_mapping_review.csv"
HEADLINE_MAPPING_EXCLUSIONS_CSV_FILE_NAME = "headline_mapping_exclusions.csv"
GDELT_HEADLINES_CLEAN_CSV_FILE_NAME = "gdelt_headlines_clean.csv"
HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME = "headlines_aligned_prices.csv"
GOLD_LABELS_CSV_FILE_NAME = "gold_labels.csv"
FINBERT_PREDICTIONS_CSV_FILE_NAME = "finbert_predictions.csv"
DKCOT_PREDICTIONS_CSV_FILE_NAME = "dkcot_predictions.csv"
CLASSIFICATION_RESULTS_CSV_FILE_NAME = "classification_results.csv"
PAIRED_SIGNIFICANCE_TESTS_CSV_FILE_NAME = "paired_significance_tests.csv"
DAILY_SIGNALS_CSV_FILE_NAME = "daily_signals.csv"
BACKTEST_RESULTS_CSV_FILE_NAME = "backtest_results.csv"
BACKTEST_SUMMARY_CSV_FILE_NAME = "backtest_summary.csv"
COMPANIES_PATH = PROJECT_ROOT / "config" / COMPANIES_CSV_FILE_NAME


# Figures created and their respective names
CLASSIFICATION_METRICS_PNG_FILE_NAME = "classification_metrics.png"
CLASSIFICATION_CONFUSION_MATRICES_PNG_FILE_NAME = (
    "classification_confusion_matrices.png"
)
BACKTEST_CUMULATIVE_RETURNS_PNG_FILE_NAME = "backtest_cumulative_returns.png"
BACKTEST_METRICS_PNG_FILE_NAME = "backtest_metrics.png"
BACKTEST_SUMMARY_TABLE_PNG_FILE_NAME = "backtest_summary_table.png"


# Shared figure colours. This is to maintain consistency with the colours used for the graphics in the reports and poster and the graphics generated in the codebase
FIGURE_PRIMARY_TEAL = "#006363"
FIGURE_SECONDARY_TEAL = "#0A6A70"
FIGURE_DARK_TEAL = "#09424C"
FIGURE_DEEP_BLUE = "#122E52"
FIGURE_NAVY = "#0D1427"
FIGURE_FONT_FAMILY = "DejaVu Serif"


# Shared sentiment labels used by manual labelling and model evaluation
POSITIVE_LABEL = "positive"
NEUTRAL_LABEL = "neutral"
NEGATIVE_LABEL = "negative"
VALID_SENTIMENT_LABELS = {POSITIVE_LABEL, NEUTRAL_LABEL, NEGATIVE_LABEL}


# Shared DataFrame and output-table column names
HEADLINE_ID_COLUMN = "headline_id"
LABEL_GOLD_COLUMN = "label_gold"
LABEL_PRED_COLUMN = "label_pred"
KNOWLEDGE_LEVEL_COLUMN = "knowledge_level"
MODEL_COLUMN = "model"
MODEL_ID_COLUMN = "model_id"
ACCURACY_COLUMN = "accuracy"
MACRO_F1_COLUMN = "macro_f1"


# Shared DK-CoT experiment configuration
DKCOT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NO_KNOWLEDGE_LEVEL = "none"
SECTOR_KNOWLEDGE_LEVEL = "sector"
FIRM_KNOWLEDGE_LEVEL = "firm"
DKCOT_KNOWLEDGE_LEVELS = (
    NO_KNOWLEDGE_LEVEL,
    SECTOR_KNOWLEDGE_LEVEL,
    FIRM_KNOWLEDGE_LEVEL,
)


# Shared FinBERT experiment configuration
FINBERT_DISPLAY_NAME = "FinBERT"
FINBERT_MODEL_NAME = "ProsusAI/finbert"


# The date format string `%Y-%m-%d` (https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html) removes the time and keeps the year-month-day. This is done to ensure a consistent format to match with headline trading dates later.
PD_DATE_FORMAT = "%Y-%m-%d"


# Other config-relevant constants
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
GOOGLE_CLOUD_PROJECT_ID = "uk-dkcot"


@dataclass(frozen=True)
class Company:
    ticker: str
    company_name: str
    aliases: tuple[str, ...]
    sector: str
    products: str
    risks: str


def load_companies(path: str | Path = COMPANIES_PATH) -> list[Company]:
    """Load the fixed ten-company research universe."""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            Company(
                ticker=row["ticker"].strip(),
                company_name=row["company_name"].strip(),
                aliases=tuple(
                    alias.strip()
                    for alias in row["aliases"].split("|")
                    if alias.strip()
                ),
                sector=row["sector"].strip(),
                products=row["products"].strip(),
                risks=row["risks"].strip(),
            )
            for row in reader
        ]


class Ticker(StrEnum):
    ASTRAZENECA = "AZN.L"
    HSBC = "HSBA.L"
    GSK = "GSK.L"
    RIO_TINTO = "RIO.L"
    VODAFONE = "VOD.L"
    NATIONAL_GRID = "NG.L"
    RELX = "REL.L"
    HALMA = "HLMA.L"
    SMITH_AND_NEPHEW = "SN.L"
    INTERTEK = "ITRK.L"


def get_figure_font_settings(font_size):
    """Return the shared figure font family with the requested font size."""

    return {
        "font.family": FIGURE_FONT_FAMILY,
        "font.size": font_size,
    }

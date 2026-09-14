from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
COMPANIES_PATH = PROJECT_ROOT / "config" / "companies.csv"

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
RANDOM_SEED = 2025

FINBERT_MODEL_ID = "ProsusAI/finbert"
DKCOT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
KNOWLEDGE_LEVELS = ("none", "sector", "firm")

# Display name, model ID and knowledge level of every evaluated variant.
MODEL_VARIANTS = [
    ("FinBERT", FINBERT_MODEL_ID, "none"),
    ("DK-CoT (none)", DKCOT_MODEL_ID, "none"),
    ("DK-CoT (sector)", DKCOT_MODEL_ID, "sector"),
    ("DK-CoT (firm)", DKCOT_MODEL_ID, "firm"),
]


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

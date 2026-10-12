from __future__ import annotations

import csv

from dataclasses import dataclass
from pathlib import Path
from enum import StrEnum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
COMPANIES_PATH = PROJECT_ROOT / "config" / "companies.csv"
GOOGLE_CLOUD_PROJECT_ID = "uk-dkcot"

GDELT_HEADLINES_CSV_FILE_NAME = "gdelt_headlines.csv"

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"


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

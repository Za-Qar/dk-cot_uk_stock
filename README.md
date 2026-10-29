# DK-CoT evaluated using UK news article headline data

This project evaluates DK-CoT using English news article headlines for ten UK-listed companies. It compares FinBERT with `Qwen/Qwen2.5-3B-Instruct` under three DK-CoT knowledge conditions:

- **No knowledge:** the headline and target company name are supplied without sector or firm-specific information.
- **Sector knowledge:** the company's sector is added.
- **Firm-specific knowledge:** the company's products, services and business risks are added.

The same Qwen model and common prompt structure are used for all three DK-CoT conditions. The experiment evaluates classification performance against 300 manually checked reference labels and then compares long-only and long-short trading simulations.

## Requirements

The reported run used:

- Windows 11 and Git Bash.
- Python 3.12.10.
- An NVIDIA GeForce RTX 3060.
- PyTorch 2.11.0 with CUDA 12.8.
- Transformers 4.57.6.

## Setup

Run the following commands from the project root in Git Bash:

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

The separate PyTorch command installs the CUDA 12.8 build used for the reported run. A different computer may require a different PyTorch installation.

## Google Cloud setup

These commands are only needed when collecting the GDELT headlines again:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project uk-dkcot
```

Another user must replace `uk-dkcot` with their own Google Cloud project in both the command above and `GOOGLE_CLOUD_PROJECT_ID` in `uk_dkcot/config.py`.

## Running the pipeline

Run the stages in numeric order:

1. `./.venv/Scripts/python.exe pipeline/01_collect_data.py`
2. Open `pipeline/02_clean_headlines.ipynb` and run all cells using the `.venv` kernel.
3. `./.venv/Scripts/python.exe pipeline/03_align_prices.py`
4. Open `pipeline/04_label_sentiment.ipynb` and run all cells using the `.venv` kernel.
5. `./.venv/Scripts/python.exe pipeline/05_finbert.py`
6. `./.venv/Scripts/python.exe pipeline/06_dkcot.py`
7. `./.venv/Scripts/python.exe pipeline/07_evaluate.py`
8. `./.venv/Scripts/python.exe pipeline/08_backtest.py`

Stage 1 skips the headline or price collection when the corresponding raw CSV already exists. Stage 6 saves each completed batch to `dkcot_predictions.csv`, allowing an interrupted DK-CoT run to continue from the missing predictions.

## Main outputs

- Raw data: `data/raw/gdelt_headlines.csv` and `data/raw/yfinance_prices.csv`.
- Cleaned and aligned data: `data/processed/gdelt_headlines_clean.csv` and `data/processed/headlines_aligned_prices.csv`.
- Reference labels: `data/processed/gold_labels.csv`.
- Predictions: `data/processed/finbert_predictions.csv` and `data/processed/dkcot_predictions.csv`.
- Classification tables: `results/tables/classification_results.csv` and `results/tables/paired_significance_tests.csv`.
- Trading outputs: `data/processed/backtest_results.csv`, `data/processed/daily_signals.csv` and `results/tables/backtest_summary.csv`.
- Figures: `results/figures/`.

Files under `data/raw/` and `data/processed/` are excluded from Git. They should be retained with the reported run so that the results can be inspected without repeating data collection or model inference.

## Data coverage

The collected GDELT data contain an 89-day interval with no headline records from 15 June to 11 September 2025. This gap comes from the available source data and should be considered when interpreting the classification and backtest results.

## Formatting

```bash
./.venv/Scripts/python.exe -m black .
```

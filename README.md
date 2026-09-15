# DK-CoT UK Stocks

Master's capstone project comparing financial-news sentiment methods for ten UK-listed companies.

## Project scope

- Collect company headlines from GDELT.
- Collect daily share prices.
- Clean and map headlines to companies.
- Compare FinBERT with three DK-CoT knowledge treatments.
- Evaluate classification and risk-adjusted trading performance.

## Requirements

- Python 3.11 or later.
- Google Cloud Application Default Credentials and access to the `uk-dkcot` project for GDELT collection.
- PyCharm notebook support or another Jupyter-compatible environment.
- An NVIDIA GPU with CUDA-enabled PyTorch for the DK-CoT stage. FinBERT also uses CUDA when available.

Configure Google authentication before collecting headlines:

```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project uk-dkcot
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Generated datasets are stored in `data/raw` and `data/processed` and are not committed to Git.

## Pipeline

Run the stages in numeric order from the project root:

1. `python pipeline/01_collect_data.py`
2. Open `pipeline/02_clean_headlines.ipynb` and run all cells.
3. `python pipeline/03_align_prices.py`
4. Open `pipeline/04_label_sentiment.ipynb` and run all cells.
5. `python pipeline/05_finbert.py`
6. `python pipeline/06_dkcot.py`
7. `python pipeline/07_evaluate.py`
8. `python pipeline/08_backtest.py`

Existing raw data and completed DK-CoT checkpoints prevent unnecessary repeated collection or generation.

## Outputs

Classification tables are written to `results/tables`. Backtest results and daily signals are written to `data/processed`. Report-ready figures are written to `results/figures`.

## Data coverage

The collected 2025 GDELT extract contains no headline records from 15 June to 11 September 2025. This source-coverage limitation should be considered when interpreting the classification and backtest results.

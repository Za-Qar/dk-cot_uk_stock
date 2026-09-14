# DK-CoT UK Stocks

Master's capstone project comparing financial-news sentiment methods for ten UK-listed companies.

## Initial scope

- Collect company headlines from GDELT.
- Collect daily share prices.
- Clean and map headlines to companies.
- Compare sentiment-classification approaches.
- Evaluate sentiment against subsequent stock movement.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Generated datasets belong in `data/raw` and `data/processed` and are not committed to Git.

## Run the completed pipeline

Run each stage from the project root, in this order. Stages 02 and 04 are the
interactive cleaning and labelling notebooks.

1. `python pipeline/01_collect_data.py`
2. `pipeline/02_clean_headlines.ipynb`
3. `python pipeline/03_align_prices.py`
4. `pipeline/04_label_sentiment.ipynb`
5. `python pipeline/05_finbert.py`
6. `python pipeline/06_dkcot.py` (needs a CUDA GPU)
7. `python pipeline/07_evaluate.py`
8. `python pipeline/08_backtest.py`
9. `python pipeline/09_technical_checks.py`

Add `--rerun-models` to stage 09 to repeat the FinBERT and DK-CoT predictions for
the gold headlines and compare them with the saved ones. The result is kept in
`results/tables/model_rerun_checks.csv` and reused by later runs.

Final numerical tables are written to `results/tables` and
`data/processed/backtest_results.csv`. Report-ready figures are written to
`results/figures`.

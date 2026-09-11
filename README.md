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

Run each stage from the project root:

```powershell
python pipeline/01_collect_data.py
python pipeline/03_align_prices.py
python pipeline/05_finbert.py
python pipeline/06_dkcot.py
python pipeline/07_evaluate.py
python pipeline/08_backtest.py
```

The two notebooks contain the interactive cleaning and labelling stages:

- `pipeline/02_clean_headlines.ipynb`
- `pipeline/04_label_sentiment.ipynb`

Final numerical tables are written to `results/tables` and
`data/processed/backtest_results.csv`. Report-ready figures are written to
`results/figures`.

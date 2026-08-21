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

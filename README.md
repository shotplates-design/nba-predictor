# NBA Game Predictor (2026-27)

Local / Streamlit Community Cloud app showing **leak-free logistic** home-win probabilities for the 2026-27 NBA season.

## Analysis only

This is research / analysis. **Not betting advice.** No odds. No bet buttons. No recommended wagers.

On the 2025-26 holdout the same model scored ~68.5% accuracy / log loss 0.604 and **did not beat** closing no-vig market probabilities (market LL ~0.570).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Main file: `app.py`  
Data: `data/processed/*.parquet`, `reports/metrics_2026_27.json` (precomputed snapshot).

## Streamlit Community Cloud

1. Deploy from this repo.
2. Main file path: `app.py`
3. Python version: 3.11+ recommended.

## Caveats

- Early-season finished-game metrics use a small sample and are unstable.
- Upcoming predictions freeze Elo/form after the last played game in the snapshot; only rest/schedule walk forward.
- Snapshot is static until the repo data files are refreshed.

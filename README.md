# NBA Predictor (analysis only)

Leak-free logistic model for 2026–27 NBA games. **Not betting advice.**

## App

Streamlit Community Cloud reads `app.py` plus parquet under `data/processed/`.

## Daily refresh (self-sufficient)

GitHub Actions workflow `.github/workflows/daily-refresh.yml` runs daily at **11:00 UTC** (and on `workflow_dispatch`):

1. Ingest current-season games (`nba_api` when reachable; ESPN public site API fallback)
2. Rebuild prior-only features (Elo / rest / rolling PPG)
3. Retrain on 2023–24 … 2025–26 + finished 2026–27
4. Walk-forward log-loss gate — ship predictions only if not worse than the current model
5. Commit updated parquet + `reports/metrics_2026_27.json` + `reports/daily_history.json` to `main` (triggers Streamlit redeploy)

No secrets required. Uses `GITHUB_TOKEN` with `contents: write`.

## Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/daily_refresh.py
streamlit run app.py
```

"""NBA Game Predictor — 2026-27 leak-free logistic win probabilities (analysis only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="NBA Game Predictor 2026-27",
    page_icon="🏀",
    layout="wide",
)

st.title("NBA Game Predictor — 2026-27")
st.caption(
    "Leak-free logistic model (16 pre-tip features). "
    "Trained on 2023-24 → 2025-26. Analysis only — no betting picks, odds, or wager buttons."
)

st.info(
    "**Analysis only.** These are model win probabilities for research. "
    "Not betting advice. No odds. No bet buttons. The model did not beat closing markets "
    "on the 2025-26 holdout (market LL 0.570 vs model LL 0.604)."
)


@st.cache_data(ttl=300)
def load_finished() -> pd.DataFrame:
    path = PROC / "predictions_2026_27_finished.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["GAME_DATE", "GAME_ID"], ascending=[False, True])


@st.cache_data(ttl=300)
def load_upcoming() -> pd.DataFrame:
    path = PROC / "predictions_2026_27_upcoming.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df.sort_values(["GAME_DATE", "GAME_ID"])


@st.cache_data(ttl=300)
def load_metrics() -> dict:
    import json

    path = REPORTS / "metrics_2026_27.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


finished = load_finished()
upcoming = load_upcoming()
metrics = load_metrics()

st.header("2026-27 season summary")
fm = metrics.get("finished_metrics") or {}
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Finished games (n)", fm.get("n", len(finished)))
c2.metric("Accuracy", f"{fm.get('accuracy', float('nan'))*100:.1f}%" if fm else "—")
c3.metric("Log loss", f"{fm.get('log_loss', float('nan')):.4f}" if fm else "—")
c4.metric("Brier", f"{fm.get('brier', float('nan')):.4f}" if fm else "—")
c5.metric("Upcoming predicted", metrics.get("upcoming_n", len(upcoming)))

st.warning(
    f"Early-season sample is small (n={fm.get('n', '?')}). "
    "For reference, the same leak-free model scored **68.46% / LL 0.6044** on the full "
    "2025-26 holdout (n=1,230). Do not over-read early accuracy."
)

st.header("Upcoming / in-progress games")
horizon = st.slider("Show next N days", min_value=1, max_value=60, value=14)
if upcoming.empty:
    st.write("No upcoming predictions yet.")
else:
    start = upcoming["GAME_DATE"].min()
    end = start + pd.Timedelta(days=horizon)
    view = upcoming[upcoming["GAME_DATE"] <= end].copy()
    view["Date"] = view["GAME_DATE"].dt.strftime("%Y-%m-%d")
    view["Matchup"] = view["AWAY_ABBR"] + " @ " + view["HOME_ABBR"]
    view["Home win %"] = (view["PRED_HOME_WIN_PROB"] * 100).round(1)
    view["Away win %"] = (view["PRED_AWAY_WIN_PROB"] * 100).round(1)
    view["Pred. margin (home)"] = view["PRED_MARGIN"].round(1)
    view["Tip / status"] = view["GAME_STATUS_TEXT"]
    show_cols = ["Date", "Tip / status", "Matchup", "Home win %", "Away win %", "Pred. margin (home)"]
    if "GAME_LABEL" in view.columns:
        view["Label"] = view["GAME_LABEL"].replace("", "Regular Season")
        show_cols.insert(2, "Label")
    st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(view)} games through {end.date()} (of {len(upcoming)} remaining).")

st.header("Finished 2026-27 games vs model")
if finished.empty:
    st.write("No finished predictions.")
else:
    f = finished.copy()
    f["Date"] = f["GAME_DATE"].dt.strftime("%Y-%m-%d")
    f["Matchup"] = f["AWAY_ABBR"] + " @ " + f["HOME_ABBR"]
    f["Score"] = f["AWAY_PTS"].astype(int).astype(str) + "-" + f["HOME_PTS"].astype(int).astype(str)
    f["Home win %"] = (f["PRED_HOME_WIN_PROB"] * 100).round(1)
    f["Model pick"] = f.apply(
        lambda r: r["HOME_ABBR"] if r["PRED_HOME_WIN"] == 1 else r["AWAY_ABBR"], axis=1
    )
    f["Actual winner"] = f.apply(
        lambda r: r["HOME_ABBR"] if r["HOME_WIN"] == 1 else r["AWAY_ABBR"], axis=1
    )
    f["Hit"] = f["CORRECT"].map({1: "✓", 0: "✗"})
    st.dataframe(
        f[["Date", "Matchup", "Score", "Home win %", "Model pick", "Actual winner", "Hit"]],
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    chron = finished.sort_values("GAME_DATE")
    chron["cum_acc"] = chron["CORRECT"].expanding().mean()
    st.subheader("Expanding accuracy through the season")
    chart = chron.set_index("GAME_DATE")[["cum_acc"]].rename(columns={"cum_acc": "Expanding accuracy"})
    st.line_chart(chart)

with st.expander("Model details & caveats"):
    st.markdown(
        f"""
**Primary model:** leak-free logistic regression (C=0.5), 16 features  
**Train:** {', '.join(metrics.get('train_seasons', ['2023-24','2024-25','2025-26']))} (n={metrics.get('train_n', 3690)})  
**Features:** Elo (pre-game), rest, rolling prior PPG / opp PPG / net rating, implied margin  
**Not used:** dressed-roster / injury-report player aggregates (roster leak)  

**2025-26 holdout reference:** 68.46% accuracy, LL 0.6044, Brier 0.2082 (n=1,230).  
Closing no-vig market on that holdout: 69.44% / LL 0.570 — market was better.

**Caveats**
- Early 2026-27 sample is small; metrics bounce.
- Upcoming probs freeze team strength at last tip; only rest/schedule walk forward.
- Injury-report pre-tip roster rebuild was skipped for this snapshot.
- Analysis only — no betting.
"""
    )

st.divider()
st.caption(
    "Data: stats.nba.com via nba_api (precomputed snapshot). "
    f"Generated: {metrics.get('generated_et', '—')}. Analysis only — no betting."
)

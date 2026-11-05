"""NBA Game Predictor — 2026-27 leak-free logistic (analysis only). Mobile-first dark UI."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

NEON = ["#7CFFB2", "#5CE1FF", "#B794FF", "#FF6AD5", "#FFD166", "#FF8A5B", "#4DFFDF", "#A0E7E5"]
BG = "rgba(11,15,20,0)"
CARD = "#151B24"
GRID = "rgba(232,238,247,0.08)"
TEXT = "#E8EEF7"
MUTED = "#8B98A8"

st.set_page_config(
    page_title="NBA Predictor 2026-27",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px; }
  h1.app-title {
    font-size: clamp(1.6rem, 5vw, 2.4rem);
    font-weight: 800; letter-spacing: -0.02em; margin: 0 0 0.15rem 0;
    background: linear-gradient(90deg, #7CFFB2 0%, #5CE1FF 45%, #B794FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .subhead { color: #8B98A8; font-size: 0.92rem; margin-bottom: 0.8rem; }
  .banner {
    border: 1px solid rgba(124,255,178,0.25);
    background: linear-gradient(120deg, rgba(124,255,178,0.08), rgba(92,225,255,0.06));
    border-radius: 14px; padding: 0.75rem 1rem; margin: 0.4rem 0 1rem 0;
    color: #E8EEF7; font-size: 0.9rem;
  }
  .metric-grid {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.65rem; margin-bottom: 1rem;
  }
  @media (min-width: 720px) {
    .metric-grid { grid-template-columns: repeat(4, 1fr); }
  }
  .metric-card {
    background: #151B24; border: 1px solid rgba(232,238,247,0.08);
    border-radius: 16px; padding: 0.85rem 1rem;
    box-shadow: 0 0 24px rgba(124,255,178,0.04);
  }
  .metric-card .label { color: #8B98A8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-card .value {
    font-size: 1.55rem; font-weight: 700; margin-top: 0.15rem;
    color: #7CFFB2;
  }
  .game-card {
    background: #151B24; border: 1px solid rgba(232,238,247,0.08);
    border-radius: 14px; padding: 0.75rem 0.9rem; margin-bottom: 0.55rem;
  }
  .game-top { display: flex; justify-content: space-between; gap: 0.5rem; align-items: baseline; }
  .matchup { font-weight: 700; font-size: 1.05rem; color: #E8EEF7; }
  .gdate { color: #8B98A8; font-size: 0.8rem; white-space: nowrap; }
  .bar-wrap {
    margin-top: 0.45rem; height: 10px; border-radius: 999px; overflow: hidden;
    background: rgba(232,238,247,0.08); display: flex;
  }
  .bar-home { height: 100%; background: linear-gradient(90deg, #5CE1FF, #7CFFB2); }
  .bar-away { height: 100%; background: linear-gradient(90deg, #B794FF, #FF6AD5); }
  .probs {
    display: flex; justify-content: space-between; margin-top: 0.35rem;
    font-size: 0.82rem; color: #8B98A8;
  }
  .probs strong { color: #E8EEF7; }
  .chip {
    display: inline-block; font-size: 0.72rem; padding: 0.15rem 0.45rem;
    border-radius: 999px; margin-top: 0.35rem; font-weight: 600;
  }
  .chip-hi { background: rgba(124,255,178,0.15); color: #7CFFB2; }
  .chip-mid { background: rgba(255,209,102,0.12); color: #FFD166; }
  .chip-lo { background: rgba(139,152,168,0.18); color: #8B98A8; }
  .chip-hit { background: rgba(124,255,178,0.18); color: #7CFFB2; }
  .chip-miss { background: rgba(255,106,213,0.15); color: #FF6AD5; }
  .scoreline { color: #8B98A8; font-size: 0.85rem; margin-top: 0.2rem; }
  div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }
  /* tighter mobile filters */
  @media (max-width: 640px) {
    .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  }
</style>
""",
    unsafe_allow_html=True,
)


def _plotly_layout(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=40, r=20, t=48 if title else 24, b=40),
        height=height,
        title=dict(text=title or "", font=dict(size=15, color=TEXT), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, showline=False)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, showline=False)
    return fig


@st.cache_data(ttl=300)
def load_finished() -> pd.DataFrame:
    path = PROC / "predictions_2026_27_finished.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    if "P_HOME_WIN" in df.columns and "PRED_HOME_WIN" not in df.columns:
        df["PRED_HOME_WIN"] = (df["P_HOME_WIN"] >= 0.5).astype(int)
    if "HOME_WIN" in df.columns and "P_HOME_WIN" in df.columns:
        df["CORRECT"] = (df["PRED_HOME_WIN"] == df["HOME_WIN"]).astype(int)
    return df.sort_values(["GAME_DATE", "GAME_ID"])


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
    path = REPORTS / "metrics_2026_27.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def confidence_label(p: float) -> tuple[str, str]:
    conf = abs(p - 0.5) * 2  # 0..1
    if conf >= 0.35:
        return "High confidence", "chip-hi"
    if conf >= 0.15:
        return "Medium", "chip-mid"
    return "Toss-up", "chip-lo"


def game_card_html(
    *,
    away: str,
    home: str,
    p_home: float,
    gdate: str,
    score: str | None = None,
    hit: int | None = None,
) -> str:
    p_away = 1.0 - p_home
    label, cls = confidence_label(p_home)
    hit_html = ""
    if hit is not None:
        hit_html = (
            f'<span class="chip {"chip-hit" if hit else "chip-miss"}">'
            f'{"Hit" if hit else "Miss"}</span>'
        )
    score_html = f'<div class="scoreline">{score}</div>' if score else ""
    return f"""
    <div class="game-card">
      <div class="game-top">
        <div class="matchup">{away} <span style="color:#8B98A8;font-weight:500">@</span> {home}</div>
        <div class="gdate">{gdate}</div>
      </div>
      {score_html}
      <div class="bar-wrap">
        <div class="bar-away" style="width:{p_away*100:.1f}%"></div>
        <div class="bar-home" style="width:{p_home*100:.1f}%"></div>
      </div>
      <div class="probs">
        <span>{away} <strong>{p_away*100:.0f}%</strong></span>
        <span>{home} <strong>{p_home*100:.0f}%</strong></span>
      </div>
      <span class="chip {cls}">{label}</span> {hit_html}
    </div>
    """


def chart_confidence_ring(today: pd.DataFrame) -> go.Figure:
    """Radial bars for today's (or nearest slate) home win probs."""
    df = today.copy()
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No games on this slate", showarrow=False, font=dict(color=MUTED))
        return _plotly_layout(fig, height=360, title="Confidence ring")
    df = df.sort_values("P_HOME_WIN")
    labels = [f"{a}@{h}" for a, h in zip(df["AWAY_ABBR"], df["HOME_ABBR"])]
    # polar bar: r = confidence (distance from 0.5), theta = game
    conf = (df["P_HOME_WIN"] - 0.5).abs() * 2
    favorite = np.where(df["P_HOME_WIN"] >= 0.5, df["HOME_ABBR"], df["AWAY_ABBR"])
    colors = [NEON[i % len(NEON)] for i in range(len(df))]
    fig = go.Figure(
        go.Barpolar(
            r=conf.values,
            theta=labels,
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.4)", width=1)),
            hovertemplate="<b>%{theta}</b><br>Favorite: %{customdata[0]}"
            "<br>Home P(win)=%{customdata[1]:.0%}<br>Confidence=%{r:.0%}<extra></extra>",
            customdata=np.stack([favorite, df["P_HOME_WIN"].values], axis=1),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(21,27,36,0.6)",
            radialaxis=dict(
                range=[0, 1], showticklabels=True, ticks="",
                gridcolor=GRID, tickfont=dict(color=MUTED, size=10),
            ),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10, color=TEXT)),
        ),
        showlegend=False,
    )
    return _plotly_layout(fig, height=400, title="Today's confidence ring")


def chart_rolling_accuracy(finished: pd.DataFrame) -> go.Figure:
    df = finished.sort_values("GAME_DATE").copy()
    if df.empty or "CORRECT" not in df.columns:
        fig = go.Figure()
        return _plotly_layout(fig, title="Rolling accuracy")
    df["expanding_acc"] = df["CORRECT"].expanding().mean()
    df["roll10"] = df["CORRECT"].rolling(10, min_periods=3).mean()
    x = df["GAME_DATE"]
    fig = go.Figure()
    # glow under expanding
    for w, op in [(10, 0.08), (6, 0.12), (3, 0.18)]:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["expanding_acc"], mode="lines",
                line=dict(width=w, color=f"rgba(124,255,178,{op})"),
                hoverinfo="skip", showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x, y=df["expanding_acc"], mode="lines",
            name="Expanding",
            line=dict(width=2.5, color="#7CFFB2"),
            hovertemplate="Expanding %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=df["roll10"], mode="lines",
            name="Roll 10",
            line=dict(width=2, color="#5CE1FF", dash="dot"),
            hovertemplate="Roll10 %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line=dict(color=MUTED, width=1, dash="dash"))
    fig.update_yaxes(tickformat=".0%", range=[0.35, 0.85])
    return _plotly_layout(fig, height=320, title="Rolling accuracy")


def chart_calibration(finished: pd.DataFrame) -> go.Figure:
    df = finished.dropna(subset=["P_HOME_WIN", "HOME_WIN"]).copy()
    if df.empty:
        return _plotly_layout(go.Figure(), title="Calibration")
    # bin predicted probs
    df["bin"] = pd.cut(df["P_HOME_WIN"], bins=np.linspace(0, 1, 11), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(
        pred=("P_HOME_WIN", "mean"),
        actual=("HOME_WIN", "mean"),
        n=("HOME_WIN", "size"),
    ).dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=MUTED, dash="dash", width=1),
            name="Perfect", hoverinfo="skip",
        )
    )
    sizes = 18 + (g["n"] / max(g["n"].max(), 1) * 42)
    fig.add_trace(
        go.Scatter(
            x=g["pred"], y=g["actual"], mode="markers+text",
            text=[str(int(n)) for n in g["n"]],
            textposition="top center",
            textfont=dict(size=10, color=MUTED),
            marker=dict(
                size=sizes,
                color=g["pred"],
                colorscale=[[0, "#B794FF"], [0.5, "#5CE1FF"], [1, "#7CFFB2"]],
                line=dict(width=1, color="rgba(255,255,255,0.25)"),
                opacity=0.9,
            ),
            name="Bins",
            hovertemplate="Pred %{x:.0%}<br>Actual %{y:.0%}<br>n=%{text}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Predicted home win %", tickformat=".0%", range=[-0.02, 1.02])
    fig.update_yaxes(title="Actual home win %", tickformat=".0%", range=[-0.02, 1.02])
    return _plotly_layout(fig, height=360, title="Calibration bubbles")


def chart_constellation(finished: pd.DataFrame, upcoming: pd.DataFrame) -> go.Figure:
    """Team strength (Elo) vs model home win prob — neon scatter."""
    parts = []
    if not finished.empty:
        f = finished.copy()
        f["slate"] = "Finished"
        parts.append(f)
    if not upcoming.empty:
        u = upcoming.head(120).copy()
        u["slate"] = "Upcoming"
        parts.append(u)
    if not parts:
        return _plotly_layout(go.Figure(), title="Team strength constellation")
    df = pd.concat(parts, ignore_index=True)
    # x = home elo (strength), y = p_home, size = |net diff|, color by slate
    fig = go.Figure()
    for slate, color, opacity in [
        ("Finished", "#5CE1FF", 0.55),
        ("Upcoming", "#FF6AD5", 0.75),
    ]:
        s = df[df["slate"] == slate]
        if s.empty:
            continue
        size = 8 + (s["NET_DIFF"].abs().fillna(0) / 20.0).clip(0, 1) * 16
        fig.add_trace(
            go.Scatter(
                x=s["HOME_ELO"],
                y=s["P_HOME_WIN"],
                mode="markers",
                name=slate,
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity,
                    line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
                ),
                customdata=np.stack(
                    [s["AWAY_ABBR"], s["HOME_ABBR"], s["AWAY_ELO"], s.get("NET_DIFF", 0)],
                    axis=1,
                ),
                hovertemplate="%{customdata[0]} @ %{customdata[1]}"
                "<br>Home Elo %{x:.0f} · Away Elo %{customdata[2]:.0f}"
                "<br>P(home) %{y:.0%}<extra>" + slate + "</extra>",
            )
        )
    # faint constellation lines among upcoming high-elo games
    up = df[df["slate"] == "Upcoming"].nlargest(12, "HOME_ELO")
    if len(up) >= 2:
        fig.add_trace(
            go.Scatter(
                x=up["HOME_ELO"], y=up["P_HOME_WIN"], mode="lines",
                line=dict(color="rgba(183,148,255,0.25)", width=1),
                hoverinfo="skip", showlegend=False,
            )
        )
    fig.update_yaxes(title="P(home win)", tickformat=".0%", range=[0.15, 0.9])
    fig.update_xaxes(title="Home Elo (pre-tip)")
    return _plotly_layout(fig, height=380, title="Strength constellation")


def chart_prob_ridge(upcoming: pd.DataFrame) -> go.Figure:
    df = upcoming.copy()
    if df.empty:
        return _plotly_layout(go.Figure(), title="Upcoming probability field")
    p = df["P_HOME_WIN"].clip(0.01, 0.99)
    # histogram + KDE-ish smooth via numpy
    hist, edges = np.histogram(p, bins=24, range=(0, 1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # simple smooth
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    smooth = np.convolve(hist, kernel, mode="same")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([centers, centers[::-1]]),
            y=np.concatenate([smooth, np.zeros_like(smooth)]),
            fill="toself",
            fillcolor="rgba(124,255,178,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=centers, y=smooth, mode="lines",
            line=dict(color="#7CFFB2", width=2.5),
            name="Density",
            hovertemplate="P(home) %{x:.0%}<br>density %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Histogram(
            x=p, nbinsx=24, histnorm="probability density",
            marker=dict(color="rgba(92,225,255,0.35)", line=dict(width=0)),
            name="Games",
            opacity=0.7,
        )
    )
    fig.add_vline(x=0.5, line=dict(color=MUTED, dash="dot", width=1))
    fig.update_xaxes(title="Predicted home win probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Density", showticklabels=False)
    return _plotly_layout(fig, height=300, title="Upcoming probability field")


# ─── Load ────────────────────────────────────────────────────────────────────
finished = load_finished()
upcoming = load_upcoming()
metrics = load_metrics()
fm = metrics.get("finished_metrics") or {}

st.markdown('<h1 class="app-title">NBA Predictor 26/27</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="subhead">Leak-free logistic · prior-only features · analysis research view</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="banner"><b>Analysis only.</b> Model win probabilities for research — '
    "not betting advice. No odds. No bet buttons. "
    "2025-26 holdout: model LL 0.604 vs closing market LL 0.570.</div>",
    unsafe_allow_html=True,
)

# Metric cards
acc = fm.get("accuracy")
ll = fm.get("log_loss")
n_fin = fm.get("n", len(finished))
n_up = metrics.get("upcoming_n", len(upcoming))
acc_s = f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else "—"
ll_s = f"{ll:.3f}" if isinstance(ll, (int, float)) else "—"
st.markdown(
    f"""
<div class="metric-grid">
  <div class="metric-card"><div class="label">Season accuracy</div><div class="value">{acc_s}</div></div>
  <div class="metric-card"><div class="label">Log loss</div><div class="value">{ll_s}</div></div>
  <div class="metric-card"><div class="label">Games scored</div><div class="value">{int(n_fin)}</div></div>
  <div class="metric-card"><div class="label">Upcoming</div><div class="value">{int(n_up)}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# Filters
all_teams = sorted(
    set(finished["HOME_ABBR"]).union(finished["AWAY_ABBR"])
    .union(upcoming["HOME_ABBR"] if not upcoming.empty else [])
    .union(upcoming["AWAY_ABBR"] if not upcoming.empty else [])
) if not finished.empty or not upcoming.empty else []

fc1, fc2 = st.columns(2)
with fc1:
    team = st.selectbox("Team filter", ["All teams"] + all_teams, index=0)
with fc2:
    # date picker defaults to "today" relative to data: max(finished)+1 or min upcoming
    if not upcoming.empty:
        default_day = upcoming["GAME_DATE"].min().date()
        min_d = min(
            finished["GAME_DATE"].min().date() if not finished.empty else default_day,
            upcoming["GAME_DATE"].min().date(),
        )
        max_d = upcoming["GAME_DATE"].max().date()
    elif not finished.empty:
        default_day = finished["GAME_DATE"].max().date()
        min_d = finished["GAME_DATE"].min().date()
        max_d = default_day
    else:
        default_day = date.today()
        min_d = max_d = default_day
    pick = st.date_input("Date", value=default_day, min_value=min_d, max_value=max_d)


def _team_mask(df: pd.DataFrame) -> pd.Series:
    if team == "All teams" or df.empty:
        return pd.Series(True, index=df.index)
    return (df["HOME_ABBR"] == team) | (df["AWAY_ABBR"] == team)


fin_f = finished[_team_mask(finished)] if not finished.empty else finished
up_f = upcoming[_team_mask(upcoming)] if not upcoming.empty else upcoming

# Slate for ring: picked date, else nearest upcoming day
pick_ts = pd.Timestamp(pick)
slate = up_f[up_f["GAME_DATE"].dt.normalize() == pick_ts]
if slate.empty:
    slate = fin_f[fin_f["GAME_DATE"].dt.normalize() == pick_ts]
if slate.empty and not up_f.empty:
    # nearest upcoming date
    nearest = up_f.iloc[(up_f["GAME_DATE"] - pick_ts).abs().argsort()[:1]]["GAME_DATE"].iloc[0]
    slate = up_f[up_f["GAME_DATE"] == nearest]

st.subheader("Abstract charts")
c_ring, c_roll = st.columns(2)
with c_ring:
    st.plotly_chart(chart_confidence_ring(slate), width='stretch', config={"displayModeBar": False})
with c_roll:
    st.plotly_chart(chart_rolling_accuracy(fin_f), width='stretch', config={"displayModeBar": False})

c_cal, c_con = st.columns(2)
with c_cal:
    st.plotly_chart(chart_calibration(fin_f), width='stretch', config={"displayModeBar": False})
with c_con:
    st.plotly_chart(
        chart_constellation(fin_f, up_f),
        width='stretch',
        config={"displayModeBar": False},
    )

st.plotly_chart(chart_prob_ridge(up_f), width='stretch', config={"displayModeBar": False})

# Game lists
st.subheader("Games")
tab_up, tab_fin = st.tabs(["Upcoming", "Finished"])

with tab_up:
    if up_f.empty:
        st.caption("No upcoming games for this filter.")
    else:
        # show from pick forward ~21 days (mobile-friendly cards)
        window = up_f[up_f["GAME_DATE"] >= pick_ts].head(40)
        if window.empty:
            window = up_f.head(40)
        for _, r in window.iterrows():
            st.markdown(
                game_card_html(
                    away=r["AWAY_ABBR"],
                    home=r["HOME_ABBR"],
                    p_home=float(r["P_HOME_WIN"]),
                    gdate=pd.Timestamp(r["GAME_DATE"]).strftime("%b %d"),
                ),
                unsafe_allow_html=True,
            )
        st.caption(f"Showing {len(window)} of {len(up_f)} upcoming (filtered).")

with tab_fin:
    if fin_f.empty:
        st.caption("No finished games for this filter.")
    else:
        view = fin_f.sort_values(["GAME_DATE", "GAME_ID"], ascending=[False, True]).head(50)
        for _, r in view.iterrows():
            score = f"{int(r['AWAY_PTS'])}–{int(r['HOME_PTS'])}"
            st.markdown(
                game_card_html(
                    away=r["AWAY_ABBR"],
                    home=r["HOME_ABBR"],
                    p_home=float(r["P_HOME_WIN"]),
                    gdate=pd.Timestamp(r["GAME_DATE"]).strftime("%b %d"),
                    score=score,
                    hit=int(r["CORRECT"]) if "CORRECT" in r and pd.notna(r["CORRECT"]) else None,
                ),
                unsafe_allow_html=True,
            )
        st.caption(f"Showing latest {len(view)} of {len(fin_f)} finished (filtered).")

with st.expander("Model details & caveats"):
    st.markdown(
        f"""
**Primary model:** leak-free logistic regression, 16 prior-only features  
**Train:** {', '.join(metrics.get('train_seasons', ['2023-24','2024-25','2025-26']))}  
(n={metrics.get('train_n', '—')}) · data source: `{metrics.get('data_source', '—')}`  
**Walk-forward LL:** {metrics.get('walk_forward_log_loss', '—')}  
**Features:** Elo, rest, rolling prior PPG / opp PPG / net, implied margin  
**Not used:** injury-report / dressed-roster aggregates  

**2025-26 holdout reference:** 68.46% / LL 0.6044 / Brier 0.2082 (n=1,230).  
Closing no-vig market on that holdout beat the model (LL 0.570).

Early 2026-27 sample is still small — don't over-read short-run accuracy.  
Analysis only — no betting.
"""
    )

st.caption(
    f"Updated {metrics.get('updated_at_utc') or metrics.get('generated_et', '—')} · "
    "Analysis only — no betting."
)

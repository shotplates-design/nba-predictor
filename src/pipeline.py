"""Daily refresh: ingest → features → retrain → gate → write artifacts."""
from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.constants import FEATURE_COLS, LIVE_SEASON, TRAIN_SEASONS
from src.features import build_features
from src.fetch_espn import fetch_season_games
from src.fetch_nba import try_fetch_finished
from src.model import decide_ship, load_metrics, metrics_finished, predict_df, train, walk_forward_log_loss

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
HIST_PATH = DATA / "games_history.parquet"
PRED_FIN = DATA / "predictions_2026_27_finished.parquet"
PRED_UP = DATA / "predictions_2026_27_upcoming.parquet"
METRICS_PATH = REPORTS / "metrics_2026_27.json"
HISTORY_PATH = REPORTS / "daily_history.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_history_log() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())
    except json.JSONDecodeError:
        return []


def _append_history(entry: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    log = _load_history_log()
    log.append(entry)
    HISTORY_PATH.write_text(json.dumps(log, indent=2) + "\n")


def ingest_live() -> tuple[pd.DataFrame, str]:
    """
    Try nba_api once for finished games; always use ESPN for full schedule
    (upcoming + finished). Prefer nba_api scores when available for finished ids.
    """
    source_notes = []
    espn = fetch_season_games(LIVE_SEASON)
    source_notes.append("espn_schedule")

    nba = try_fetch_finished(LIVE_SEASON)
    if nba is not None and len(nba) > 0:
        source_notes.append(f"nba_api_finished_n={len(nba)}")
        # merge scores: match on date+home+away when GAME_IDs differ
        espn_f = espn.copy()
        nba_key = nba.assign(
            _k=nba["GAME_DATE"].astype(str) + "|" + nba["HOME_ABBR"] + "|" + nba["AWAY_ABBR"]
        )
        espn_f["_k"] = (
            espn_f["GAME_DATE"].astype(str) + "|" + espn_f["HOME_ABBR"] + "|" + espn_f["AWAY_ABBR"]
        )
        score_map = nba_key.set_index("_k")[["HOME_PTS", "AWAY_PTS", "HOME_WIN", "MARGIN"]]
        for col in ["HOME_PTS", "AWAY_PTS"]:
            mapped = espn_f["_k"].map(score_map[col])
            espn_f[col] = mapped.fillna(espn_f[col])
        # mark completed if we have scores
        has = espn_f["HOME_PTS"].notna() & espn_f["AWAY_PTS"].notna()
        espn_f.loc[has, "COMPLETED"] = 1
        espn_f.loc[has, "STATUS_STATE"] = "post"
        espn = espn_f.drop(columns=["_k"])
        primary = "espn+nba_api"
    else:
        source_notes.append("nba_api_unavailable")
        primary = "espn"

    print("ingest source:", primary, source_notes)
    return espn, primary


def run() -> dict[str, Any]:
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    hist = pd.read_parquet(HIST_PATH)
    hist["GAME_DATE"] = pd.to_datetime(hist["GAME_DATE"])
    hist["COMPLETED"] = 1
    if "HOME_WIN" not in hist.columns:
        hist["HOME_WIN"] = (hist["HOME_PTS"] > hist["AWAY_PTS"]).astype(int)
    if "MARGIN" not in hist.columns:
        hist["MARGIN"] = hist["HOME_PTS"] - hist["AWAY_PTS"]
    # history already has features; keep as-is for training rows
    hist_feat = hist.copy()
    hist_feat["SOURCE"] = "frozen_history"

    live_raw, source = ingest_live()
    # Build features for LIVE season only starting from blank Elo would be wrong —
    # carry Elo/rest/rolling by chaining history game outcomes then live.
    # Rebuild features from raw history scores + live schedule.
    hist_raw = hist[["GAME_ID", "GAME_DATE", "SEASON", "HOME_ABBR", "AWAY_ABBR",
                     "HOME_PTS", "AWAY_PTS"]].copy()
    hist_raw["COMPLETED"] = 1
    hist_raw["SOURCE"] = "frozen_history"

    live_cols = ["GAME_ID", "GAME_DATE", "SEASON", "HOME_ABBR", "AWAY_ABBR",
                 "HOME_PTS", "AWAY_PTS", "COMPLETED", "SOURCE"]
    for c in live_cols:
        if c not in live_raw.columns:
            live_raw[c] = None if c.endswith("PTS") else (0 if c == "COMPLETED" else "")

    chained = pd.concat([hist_raw[live_cols], live_raw[live_cols]], ignore_index=True)
    chained["GAME_DATE"] = pd.to_datetime(chained["GAME_DATE"])
    chained = chained.drop_duplicates("GAME_ID", keep="last")
    feat_all = build_features(chained)

    live_feat = feat_all[feat_all["SEASON"] == LIVE_SEASON].copy()
    # training set: frozen hist features (already leak-free) + finished live
    # Prefer rebuilt features for hist too so Elo is continuous into live.
    train_pool = feat_all[
        (feat_all["SEASON"].isin(TRAIN_SEASONS))
        | ((feat_all["SEASON"] == LIVE_SEASON) & (feat_all["COMPLETED"] == 1))
    ].copy()
    train_pool = train_pool.dropna(subset=FEATURE_COLS + ["HOME_WIN"])

    old_metrics = load_metrics(METRICS_PATH)
    wf_ll = walk_forward_log_loss(feat_all, recent_n=40)
    pipe = train(train_pool)

    finished = live_feat[live_feat["COMPLETED"] == 1].copy()
    upcoming = live_feat[live_feat["COMPLETED"] == 0].copy()
    scored_f = predict_df(pipe, finished) if len(finished) else finished
    scored_u = predict_df(pipe, upcoming) if len(upcoming) else upcoming

    fm = metrics_finished(scored_f) if len(scored_f) else {
        "n": 0, "accuracy": None, "log_loss": None, "brier": None, "margin_mae": None,
    }
    new_metrics = {
        "generated_et": _utcnow().replace("Z", " UTC"),
        "label": "2026-27_live_leak_free_logistic",
        "leak_free": True,
        "train_seasons": TRAIN_SEASONS,
        "train_n": int(len(train_pool)),
        "test_season": LIVE_SEASON,
        "finished_metrics": fm,
        "finished_n": int(fm.get("n") or 0),
        "upcoming_n": int(len(scored_u)),
        "feature_cols": FEATURE_COLS,
        "walk_forward_log_loss": None if wf_ll is None else round(wf_ll, 6),
        "log_loss": fm.get("log_loss"),
        "accuracy": fm.get("accuracy"),
        "brier": fm.get("brier"),
        "updated_at_utc": _utcnow(),
        "data_source": source,
        "analysis_only_no_betting": True,
        "caveats": [
            "Early-season sample is small; metrics are unstable.",
            "Upcoming features use current Elo/form after last played game.",
            "No dressed-roster / injury-report features.",
            "Analysis only — no betting picks, odds, or wager recommendations.",
        ],
    }

    ship, reason = decide_ship(wf_ll, old_metrics)
    new_metrics["shipped"] = ship
    new_metrics["gate_reason"] = reason

    entry = {
        "ts_utc": _utcnow(),
        "source": source,
        "shipped": ship,
        "gate_reason": reason,
        "walk_forward_log_loss": new_metrics["walk_forward_log_loss"],
        "metrics": {
            "n_finished": new_metrics["finished_n"],
            "n_upcoming": new_metrics["upcoming_n"],
            "n_train": new_metrics["train_n"],
            "accuracy": fm.get("accuracy"),
            "log_loss": fm.get("log_loss"),
            "brier": fm.get("brier"),
            "mae_margin": fm.get("margin_mae"),
        },
    }

    if ship:
        # write predictions + metrics
        keep = [
            "GAME_ID", "GAME_DATE", "SEASON", "HOME_ABBR", "AWAY_ABBR",
            "HOME_PTS", "AWAY_PTS", "HOME_WIN", "MARGIN",
            *FEATURE_COLS, "P_HOME_WIN", "PRED_HOME_WIN", "PRED_MARGIN", "COMPLETED",
        ]
        if len(scored_f):
            scored_f[keep].to_parquet(PRED_FIN, index=False, compression="zstd")
        if len(scored_u):
            scored_u[[c for c in keep if c in scored_u.columns]].to_parquet(
                PRED_UP, index=False, compression="zstd"
            )
        METRICS_PATH.write_text(json.dumps(new_metrics, indent=2) + "\n")
        print("SHIPPED:", reason)
    else:
        # still log; leave prior artifacts
        print("HELD:", reason)
        # write a hold marker into metrics history only
        hold_path = REPORTS / "last_hold.json"
        hold_path.write_text(json.dumps(new_metrics, indent=2) + "\n")

    _append_history(entry)
    return entry


def main() -> int:
    try:
        entry = run()
        print(json.dumps(entry, indent=2))
        return 0
    except Exception:
        traceback.print_exc()
        try:
            _append_history({
                "ts_utc": _utcnow(),
                "shipped": False,
                "gate_reason": "pipeline_error",
                "error": traceback.format_exc()[-2000:],
            })
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

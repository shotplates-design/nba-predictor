"""Optional nba_api ingest (often blocked from GitHub-hosted runners)."""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.constants import LIVE_SEASON


def fetch_season_team_logs(season: str = LIVE_SEASON) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    print(f"  nba_api LeagueGameLog team {season}…")
    lg = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
        timeout=60,
    )
    df = lg.get_data_frames()[0]
    time.sleep(0.9)
    return df


def team_logs_to_games(team_df: pd.DataFrame, season: str = LIVE_SEASON) -> pd.DataFrame:
    """Convert team gamelog rows into one row per finished game."""
    df = team_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # MATCHUP: 'AAA vs. BBB' home, 'AAA @ BBB' away
    def is_home(m: str) -> bool:
        return " vs." in str(m) or " vs " in str(m)

    df["HOME"] = df["MATCHUP"].map(is_home).astype(int)
    home = df[df["HOME"] == 1].copy()
    away = df[df["HOME"] == 0].copy()
    home = home.rename(columns={
        "TEAM_ABBREVIATION": "HOME_ABBR", "PTS": "HOME_PTS", "WL": "HOME_WL",
    })
    away = away.rename(columns={
        "TEAM_ABBREVIATION": "AWAY_ABBR", "PTS": "AWAY_PTS", "WL": "AWAY_WL",
    })
    keys = ["GAME_ID", "GAME_DATE"]
    g = home[keys + ["HOME_ABBR", "HOME_PTS", "HOME_WL"]].merge(
        away[keys + ["AWAY_ABBR", "AWAY_PTS", "AWAY_WL"]], on=keys, how="inner"
    )
    g["SEASON"] = season
    g["HOME_WIN"] = (g["HOME_WL"] == "W").astype(int)
    g["MARGIN"] = g["HOME_PTS"] - g["AWAY_PTS"]
    g["COMPLETED"] = 1
    g["STATUS_STATE"] = "post"
    g["STATUS_DETAIL"] = "Final"
    g["SOURCE"] = "nba_api"
    g["GAME_ID"] = g["GAME_ID"].astype(str)
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def try_fetch_finished(season: str = LIVE_SEASON) -> pd.DataFrame | None:
    try:
        logs = fetch_season_team_logs(season)
        if logs is None or len(logs) == 0:
            return None
        return team_logs_to_games(logs, season)
    except Exception as e:  # noqa: BLE001
        print(f"nba_api failed: {e}")
        return None

"""Fetch NBA schedule + scores from ESPN public site API (Actions-friendly)."""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.constants import ESPN_TO_NBA, LIVE_SEASON

UA = (
    "Mozilla/5.0 (compatible; nba-predictor/1.0; research; "
    "+https://github.com/shotplates-design/nba-predictor)"
)
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SLEEP = 0.25


def _get(url: str, retries: int = 3) -> dict[str, Any]:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"ESPN fetch failed for {url}: {last}")


def _norm_abbr(a: str) -> str:
    a = (a or "").upper().strip()
    return ESPN_TO_NBA.get(a, a)


def _score_val(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        raw = raw.get("value", raw.get("displayValue"))
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def list_team_ids() -> list[tuple[str, str]]:
    d = _get(f"{BASE}/teams")
    out = []
    for t in d["sports"][0]["leagues"][0]["teams"]:
        tm = t["team"]
        out.append((_norm_abbr(tm["abbreviation"]), str(tm["id"])))
    return out


def _event_row(ev: dict[str, Any], season: str) -> dict[str, Any] | None:
    comps = ev.get("competitions") or []
    if not comps:
        return None
    c0 = comps[0]
    competitors = c0.get("competitors") or []
    if len(competitors) != 2:
        return None
    home = next((x for x in competitors if x.get("homeAway") == "home"), None)
    away = next((x for x in competitors if x.get("homeAway") == "away"), None)
    if not home or not away:
        return None
    status = (ev.get("status") or {}).get("type") or {}
    state = status.get("state") or ""
    completed_flag = bool(status.get("completed")) or state == "post"
    h_score = _score_val(home.get("score"))
    a_score = _score_val(away.get("score"))
    # team schedule often omits status.type; treat both scores as finished
    if h_score is not None and a_score is not None:
        completed_flag = True
        state = state or "post"
    dt = pd.to_datetime(ev.get("date"), utc=True).tz_convert("America/New_York").tz_localize(None)
    return {
        "GAME_ID": str(ev.get("id")),
        "GAME_DATE": dt.normalize(),
        "TIP_ET": dt,
        "SEASON": season,
        "HOME_ABBR": _norm_abbr(home["team"]["abbreviation"]),
        "AWAY_ABBR": _norm_abbr(away["team"]["abbreviation"]),
        "HOME_PTS": h_score,
        "AWAY_PTS": a_score,
        "COMPLETED": int(completed_flag and h_score is not None and a_score is not None),
        "STATUS_STATE": state,
        "STATUS_DETAIL": status.get("detail") or status.get("description") or "",
        "SOURCE": "espn",
    }


def events_to_games(events: list[dict[str, Any]], season: str = LIVE_SEASON) -> pd.DataFrame:
    rows = []
    for ev in events:
        row = _event_row(ev, season)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("GAME_ID").sort_values(
        ["GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)


def fetch_team_schedules(season: str = LIVE_SEASON) -> pd.DataFrame:
    teams = list_team_ids()
    frames = []
    for abbr, tid in teams:
        d = _get(f"{BASE}/teams/{tid}/schedule")
        time.sleep(SLEEP)
        g = events_to_games(list(d.get("events") or []), season=season)
        if not g.empty:
            frames.append(g)
        print(f"  ESPN schedule {abbr}: {len(g)}")
    if not frames:
        raise RuntimeError("ESPN team schedules returned no games")
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.sort_values(["GAME_ID", "COMPLETED"]).drop_duplicates("GAME_ID", keep="last")
    return allg.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def fetch_scoreboard_range(
    start: date, end: date, season: str = LIVE_SEASON
) -> pd.DataFrame:
    """Day-by-day scoreboard for finished + in-progress games."""
    frames = []
    d = start
    while d <= end:
        key = d.strftime("%Y%m%d")
        try:
            sb = _get(f"{BASE}/scoreboard?dates={key}")
            time.sleep(SLEEP)
            g = events_to_games(list(sb.get("events") or []), season=season)
            if not g.empty:
                frames.append(g)
                print(f"  ESPN scoreboard {key}: {len(g)} (completed={int(g['COMPLETED'].sum())})")
        except Exception as e:  # noqa: BLE001
            print(f"  ESPN scoreboard {key} failed: {e}")
        d += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    allg = pd.concat(frames, ignore_index=True)
    allg = allg.sort_values(["GAME_ID", "COMPLETED"]).drop_duplicates("GAME_ID", keep="last")
    return allg.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def fetch_season_games(season: str = LIVE_SEASON) -> pd.DataFrame:
    """
    Full season via team schedules (upcoming + many finished), enriched with
    scoreboard dates from ~season open through tomorrow (ET).
    """
    print("ESPN: team schedules…")
    sched = fetch_team_schedules(season)
    # Regular season tipoff ~ mid-Oct; use min schedule date or Oct 20
    today_et = datetime.now(timezone.utc).astimezone().date()  # box is America/New_York
    start = date(2026, 10, 20)
    if not sched.empty:
        start = min(start, pd.to_datetime(sched["GAME_DATE"]).min().date())
    end = today_et + timedelta(days=1)
    print(f"ESPN: scoreboards {start} → {end}…")
    boards = fetch_scoreboard_range(start, end, season=season)

    parts = [sched]
    if not boards.empty:
        parts.append(boards)
    allg = pd.concat(parts, ignore_index=True)
    # prefer completed + scores
    allg["_rank"] = allg["COMPLETED"] * 2 + allg["HOME_PTS"].notna().astype(int)
    allg = allg.sort_values(["GAME_ID", "_rank"]).drop_duplicates("GAME_ID", keep="last")
    allg = allg.drop(columns=["_rank"]).sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    finished = int(allg["COMPLETED"].sum())
    upcoming = int((allg["COMPLETED"] == 0).sum())
    print(f"ESPN season merge: total={len(allg)} finished={finished} upcoming={upcoming}")
    return allg

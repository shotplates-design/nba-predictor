"""Prior-only Elo / rest / rolling PPG features (leak-free)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.constants import ELO_HOME, ELO_K, ELO_START, ROLL


def elo_expect(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    games columns needed: GAME_ID, GAME_DATE, SEASON, HOME_ABBR, AWAY_ABBR,
    HOME_PTS, AWAY_PTS (NaN ok for upcoming), COMPLETED (1/0).
    Finished rows must have scores; upcoming may lack them.
    Features use only prior games for each team.
    """
    g = games.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    g = g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    elo: dict[str, float] = defaultdict(lambda: ELO_START)
    last_date: dict[str, pd.Timestamp] = {}
    # rolling lists of own pts / opp pts (prior only — update after emit)
    own_hist: dict[str, list[float]] = defaultdict(list)
    opp_hist: dict[str, list[float]] = defaultdict(list)

    rows = []
    for _, r in g.iterrows():
        h, a = r["HOME_ABBR"], r["AWAY_ABBR"]
        he, ae = elo[h], elo[a]
        # prior rest
        gd = r["GAME_DATE"]
        h_rest = (gd - last_date[h]).days if h in last_date else 7
        a_rest = (gd - last_date[a]).days if a in last_date else 7
        h_rest = int(min(max(h_rest, 0), 14))
        a_rest = int(min(max(a_rest, 0), 14))

        def prior_avg(hist: list[float]) -> float:
            if not hist:
                return 112.0  # league-ish prior
            return float(np.mean(hist[-ROLL:]))

        h_ppg = prior_avg(own_hist[h])
        a_ppg = prior_avg(own_hist[a])
        h_opp = prior_avg(opp_hist[h])
        a_opp = prior_avg(opp_hist[a])
        h_net = h_ppg - h_opp
        a_net = a_ppg - a_opp
        elo_home_adj = he + ELO_HOME
        p_home = elo_expect(elo_home_adj, ae)

        out = {
            "GAME_ID": str(r["GAME_ID"]),
            "GAME_DATE": gd,
            "SEASON": r["SEASON"],
            "HOME_ABBR": h,
            "AWAY_ABBR": a,
            "HOME_PTS": r.get("HOME_PTS"),
            "AWAY_PTS": r.get("AWAY_PTS"),
            "COMPLETED": int(r.get("COMPLETED") or 0),
            "HOME_ELO": he,
            "AWAY_ELO": ae,
            "ELO_DIFF": he - ae,
            "ELO_HOME_WIN_PROB": p_home,
            "HOME_REST": h_rest,
            "AWAY_REST": a_rest,
            "REST_DIFF": h_rest - a_rest,
            "HOME_PRIOR_PPG": h_ppg,
            "AWAY_PRIOR_PPG": a_ppg,
            "HOME_PRIOR_OPP_PPG": h_opp,
            "AWAY_PRIOR_OPP_PPG": a_opp,
            "HOME_PRIOR_NET": h_net,
            "AWAY_PRIOR_NET": a_net,
            "PPG_DIFF": h_ppg - a_ppg,
            "NET_DIFF": h_net - a_net,
            "IMPLIED_MARGIN": (he - ae + ELO_HOME) / 28.0,
            "SOURCE": r.get("SOURCE", ""),
        }
        if out["COMPLETED"] and pd.notna(out["HOME_PTS"]) and pd.notna(out["AWAY_PTS"]):
            hp, ap = float(out["HOME_PTS"]), float(out["AWAY_PTS"])
            out["HOME_WIN"] = int(hp > ap)
            out["MARGIN"] = hp - ap
        else:
            out["HOME_WIN"] = np.nan
            out["MARGIN"] = np.nan
        rows.append(out)

        # update state only after finished games
        if out["COMPLETED"] and pd.notna(out["HOME_PTS"]) and pd.notna(out["AWAY_PTS"]):
            hp, ap = float(out["HOME_PTS"]), float(out["AWAY_PTS"])
            home_won = 1 if hp > ap else 0
            eh = elo_expect(he + ELO_HOME, ae)
            elo[h] = he + ELO_K * (home_won - eh)
            elo[a] = ae + ELO_K * ((1 - home_won) - (1 - eh))
            own_hist[h].append(hp)
            opp_hist[h].append(ap)
            own_hist[a].append(ap)
            opp_hist[a].append(hp)
            last_date[h] = gd
            last_date[a] = gd

    return pd.DataFrame(rows)

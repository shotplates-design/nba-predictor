"""Leak-free logistic regression train / score / gate."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import FEATURE_COLS, LIVE_SEASON, TRAIN_SEASONS


def _pipe() -> Pipeline:
    return Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
    ])


def train(df: pd.DataFrame) -> Pipeline:
    m = df.dropna(subset=FEATURE_COLS + ["HOME_WIN"]).copy()
    X = m[FEATURE_COLS].astype(float)
    y = m["HOME_WIN"].astype(int)
    pipe = _pipe()
    pipe.fit(X, y)
    return pipe


def predict_df(pipe: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    X = out[FEATURE_COLS].astype(float)
    proba = pipe.predict_proba(X)[:, 1]
    out["P_HOME_WIN"] = proba
    out["PRED_HOME_WIN"] = (proba >= 0.5).astype(int)
    out["PRED_MARGIN"] = (proba - 0.5) * 28.0 + out["IMPLIED_MARGIN"] * 0.35
    return out


def metrics_finished(scored: pd.DataFrame) -> dict[str, Any]:
    m = scored.dropna(subset=["HOME_WIN", "P_HOME_WIN"]).copy()
    if not len(m):
        return {
            "n": 0, "accuracy": None, "log_loss": None, "brier": None, "margin_mae": None,
        }
    y = m["HOME_WIN"].astype(int)
    p = m["P_HOME_WIN"].astype(float).clip(1e-6, 1 - 1e-6)
    return {
        "n": int(len(m)),
        "accuracy": float((m["PRED_HOME_WIN"] == y).mean()),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "margin_mae": float(np.mean(np.abs(m["MARGIN"] - m["PRED_MARGIN"]))),
    }


def walk_forward_log_loss(feat: pd.DataFrame, recent_n: int = 40) -> float | None:
    finished = feat[(feat["COMPLETED"] == 1) & feat["HOME_WIN"].notna()].copy()
    finished = finished.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    live = finished[finished["SEASON"] == LIVE_SEASON]
    if len(live) < 10:
        return None
    recent = live.tail(min(recent_n, len(live)))
    idxs = recent.index.tolist()
    ys, ps = [], []
    for i in idxs:
        train_df = finished.iloc[:i]
        if len(train_df) < 200:
            continue
        pipe = train(train_df)
        row = finished.iloc[[i]]
        p = float(pipe.predict_proba(row[FEATURE_COLS].astype(float))[0, 1])
        p = min(max(p, 1e-6), 1 - 1e-6)
        ys.append(int(row["HOME_WIN"].iloc[0]))
        ps.append(p)
    if len(ys) < 8:
        return None
    return float(log_loss(ys, ps))


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def baseline_ll(old: dict[str, Any] | None) -> float | None:
    if not old:
        return None
    if old.get("walk_forward_log_loss") is not None:
        return float(old["walk_forward_log_loss"])
    if old.get("log_loss") is not None:
        return float(old["log_loss"])
    fm = old.get("finished_metrics") or {}
    if fm.get("log_loss") is not None:
        return float(fm["log_loss"])
    return None


def decide_ship(new_wf_ll: float | None, old_metrics: dict[str, Any] | None) -> tuple[bool, str]:
    """Compare walk-forward LL only to a prior walk-forward LL (same metric)."""
    if new_wf_ll is None:
        return True, "insufficient finished games for walk-forward; ship baseline"
    old_wf = None
    if old_metrics and old_metrics.get("walk_forward_log_loss") is not None:
        old_wf = float(old_metrics["walk_forward_log_loss"])
    if old_wf is None:
        return True, f"no prior walk-forward baseline; ship (wf_ll={new_wf_ll:.6f})"
    if new_wf_ll <= old_wf + 1e-6:
        return True, f"wf_ll {new_wf_ll:.6f} <= baseline {old_wf:.6f}; ship"
    return False, f"wf_ll {new_wf_ll:.6f} worse than baseline {old_wf:.6f}; keep prior"

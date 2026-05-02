"""
features.py
===========
Build per-event feature matrix for the NB04 event-study classifiers.

Three ablation arms:
  A — BuyHold baseline:   no features (model is the train-set base rate)
  B — Tech + controls:    RSI-7, MACD-hist, 5d momentum, 10d realized vol,
                          log average dollar volume, sector one-hot
  C — Tech + LLM:         B + sentiment_num, confidence, certainty, evasion,
                          guidance_num, tone_num

All technicals are computed using prices STRICTLY BEFORE the event date to
avoid look-ahead leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import sys

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

LLM_FEATURES = [
    "sentiment_num", "confidence", "certainty",
    "evasion", "guidance_num", "tone_num",
]
TECH_FEATURES = ["rsi_7", "macd_hist", "mom_5d", "vol_10d", "log_dollar_vol"]


def _technicals_at_event(price_t: pd.DataFrame, event_date) -> dict | None:
    """Compute pre-event technicals using only data with date < event_date."""
    sub = price_t[price_t.date < pd.to_datetime(event_date)].sort_values("date")
    if len(sub) < 35:
        return None
    close = sub["adj_close"].astype(float).reset_index(drop=True)
    vol = sub["volume"].astype(float).reset_index(drop=True)
    ret = close.pct_change()

    try:
        rsi_7 = float(RSIIndicator(close, window=7).rsi().iloc[-1])
        macd_hist = float(MACD(close).macd_diff().iloc[-1])
    except Exception:
        return None
    if not np.isfinite(rsi_7) or not np.isfinite(macd_hist):
        return None

    mom_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else None
    vol_10d = float(ret.iloc[-10:].std() * np.sqrt(252)) if len(ret) >= 11 else None
    dollar_vol = (close * vol).iloc[-30:-10]
    log_dollar_vol = float(np.log(dollar_vol.mean())) if (dollar_vol > 0).all() else None

    return {
        "rsi_7": rsi_7,
        "macd_hist": macd_hist,
        "mom_5d": mom_5d,
        "vol_10d": vol_10d,
        "log_dollar_vol": log_dollar_vol,
    }


def build_technicals(sessions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Per-session technical features (one row per session)."""
    sessions = sessions.drop_duplicates(
        subset=["ticker", "fiscal_year", "fiscal_quarter"], keep="first"
    ).copy()
    sessions["event_date"] = pd.to_datetime(sessions["mostimportantdate"])
    cache = {t: prices[prices.ticker == t] for t in sessions.ticker.unique()}

    rows = []
    for r in sessions.itertuples(index=False):
        feats = _technicals_at_event(cache[r.ticker], r.event_date) or {
            k: None for k in TECH_FEATURES
        }
        rows.append({
            "session_id": f"{r.ticker}_{r.fiscal_year}_Q{r.fiscal_quarter}",
            **feats,
        })
    return pd.DataFrame(rows)


def _sector_dummies(tickers: pd.Series) -> pd.DataFrame:
    sec = tickers.map(config.SECTOR_MAP)
    dummies = pd.get_dummies(sec, prefix="sec").reindex(
        columns=[f"sec_{s}" for s in config.SECTOR_LIST], fill_value=0
    )
    return dummies.astype(int)


def build_feature_matrix(
    sessions: pd.DataFrame,
    prices: pd.DataFrame,
    llm_signals: pd.DataFrame | None,
    arm: Literal["A", "B", "C"],
) -> pd.DataFrame:
    """Assemble feature matrix indexed by session_id for the given ablation arm.

    Returns a DataFrame with `session_id`, `ticker`, `fiscal_year`, `event_date`
    plus the feature columns appropriate to the arm. NaN rows are kept; the
    caller handles dropna after merging with labels.
    """
    base = sessions.drop_duplicates(
        subset=["ticker", "fiscal_year", "fiscal_quarter"], keep="first"
    ).copy()
    base["event_date"] = pd.to_datetime(base["mostimportantdate"])
    base["session_id"] = (
        base.ticker + "_" + base.fiscal_year.astype(str)
        + "_Q" + base.fiscal_quarter.astype(str)
    )
    base = base[["session_id", "ticker", "fiscal_year", "fiscal_quarter", "event_date"]]

    if arm == "A":
        base["intercept"] = 1.0
        return base

    tech = build_technicals(sessions, prices)
    out = base.merge(tech, on="session_id", how="left")
    sec = _sector_dummies(out["ticker"])
    out = pd.concat([out.reset_index(drop=True), sec.reset_index(drop=True)], axis=1)

    if arm == "B":
        return out

    if llm_signals is None:
        raise ValueError("arm='C' requires llm_signals")
    llm = llm_signals[["session_id", *LLM_FEATURES]]
    out = out.merge(llm, on="session_id", how="left")
    return out


def feature_columns(arm: Literal["A", "B", "C"]) -> list[str]:
    """Return the list of model-input columns for an arm (excludes id columns)."""
    if arm == "A":
        return ["intercept"]
    sector_cols = [f"sec_{s}" for s in config.SECTOR_LIST]
    base = TECH_FEATURES + sector_cols
    if arm == "B":
        return base
    return base + LLM_FEATURES

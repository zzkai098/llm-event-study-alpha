"""
data_loader.py
==============
Single entry point for reading project data. All downstream code
(notebooks, models, backtest) goes through here so paths are not
hard-coded in multiple places.
"""

import json
from pathlib import Path

import pandas as pd

# Make the project root importable so we can pick up config.py
import sys
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
import config  # noqa: E402


def load_sessions() -> pd.DataFrame:
    """Session-level table (one row per earnings call)."""
    fp = config.CLEAN_DIR / "earnings_sessions.parquet"
    if not fp.exists():
        fp_csv = config.CLEAN_DIR / "earnings_sessions.csv"
        if fp_csv.exists():
            return pd.read_csv(fp_csv)
        raise FileNotFoundError(
            "Sessions table missing. Run scripts/02_build_datasets.py first."
        )
    return pd.read_parquet(fp)


def load_segments() -> pd.DataFrame:
    """Segment-level table (one row per speaker turn). Use this for NLP."""
    fp = config.CLEAN_DIR / "earnings_segments.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            "Segments table missing. Run scripts/02_build_datasets.py first."
        )
    return pd.read_parquet(fp)


def load_prices() -> pd.DataFrame:
    """Long-format prices: date, ticker, open, high, low, close, volume."""
    fp = config.CLEAN_DIR / "prices.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            "Prices missing. Run scripts/03_fetch_prices.py first."
        )
    return pd.read_parquet(fp)


def load_labels() -> pd.DataFrame:
    """Per-event labels (10-day forward excess return + binary label)."""
    fp = config.CLEAN_DIR / "labels.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            "Labels missing. Run scripts/04_build_labels.py first."
        )
    return pd.read_parquet(fp)


def load_llm_signals() -> pd.DataFrame:
    """Claude-extracted structured signals (produced by NB2)."""
    fp = config.SIGNALS_DIR / "llm_signals.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            "LLM signals missing. Run notebooks/02_rag_llm_agent.ipynb first."
        )
    return pd.read_parquet(fp)


def load_single_transcript(ticker: str, year: int, quarter: int) -> dict:
    """Load the raw JSON for one earnings call."""
    pattern = f"{ticker}_{year}_Q{quarter}__*.json"
    candidates = list(config.TRANSCRIPTS_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No transcript found for {ticker} {year} Q{quarter}"
        )
    with open(candidates[0], "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Split helpers ----------
def split_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Slice a DataFrame by train / val / test_2024 / test_2025.

    Requires a `fiscal_year` column.
    """
    period_map = {
        "train": config.TRAIN_YEARS,
        "val": config.VAL_YEARS,
        "test_2024": config.TEST_YEARS_1,
        "test_2025": config.TEST_YEARS_2,
    }
    if period not in period_map:
        raise ValueError(f"period must be one of {list(period_map.keys())}")
    return df[df.fiscal_year.isin(period_map[period])].copy()

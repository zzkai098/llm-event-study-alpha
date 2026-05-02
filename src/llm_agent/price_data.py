"""
price_data.py
=============
Thin wrappers over yfinance plus the forward / excess return helpers
used by label construction and the backtest.
"""

from typing import Iterable
import pandas as pd


def fetch_prices(tickers: Iterable[str],
                 start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for many tickers and return a long-format DataFrame:
    (date, ticker, open, high, low, close, adj_close, volume).
    """
    import yfinance as yf
    frames = []
    tickers = list(tickers)
    for i, t in enumerate(tickers, 1):
        try:
            df = yf.download(t, start=start, end=end,
                             progress=False, auto_adjust=False)
            if df.empty:
                print(f"  [{i}/{len(tickers)}] {t}: no data")
                continue

            # Newer yfinance returns a MultiIndex on columns; flatten it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [
                str(c).lower().replace(" ", "_") for c in df.columns
            ]
            df["ticker"] = t
            frames.append(df)
            if i % 10 == 0:
                print(f"  [{i}/{len(tickers)}] fetched {i} tickers so far")
        except Exception as e:
            print(f"  [{i}/{len(tickers)}] {t} failed: {e}")

    if not frames:
        raise RuntimeError("No price data was fetched")

    result = pd.concat(frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


def compute_forward_return(prices: pd.DataFrame,
                           ticker: str,
                           from_date,
                           holding_days: int) -> float:
    """Return the holding-period return of `ticker` starting from the first
    available trading day on or after `from_date`, held for `holding_days`
    trading days.
    """
    from_date = pd.to_datetime(from_date)
    sub = prices[prices.ticker == ticker].sort_values("date").reset_index(drop=True)
    mask = sub.date >= from_date
    if not mask.any():
        return None
    idx_start = mask.idxmax()  # first True position
    idx_end = idx_start + holding_days
    if idx_end >= len(sub):
        return None

    col = "adj_close" if "adj_close" in sub.columns else "close"
    start_price = sub.loc[idx_start, col]
    end_price = sub.loc[idx_end, col]
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return None
    return float(end_price / start_price - 1)


def compute_excess_return(prices: pd.DataFrame,
                          ticker: str, benchmark: str,
                          from_date, holding_days: int) -> float:
    """Return excess of the benchmark (e.g. SPY)."""
    r_stock = compute_forward_return(prices, ticker, from_date, holding_days)
    r_bench = compute_forward_return(prices, benchmark, from_date, holding_days)
    if r_stock is None or r_bench is None:
        return None
    return r_stock - r_bench

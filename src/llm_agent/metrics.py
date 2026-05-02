"""
metrics.py
==========
Performance metrics: Sharpe, annualised return, max drawdown,
information ratio, AUC.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, rf: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio of a return series."""
    excess = returns - rf / periods_per_year
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def annualized_return(returns: pd.Series,
                      periods_per_year: int = 252) -> float:
    """CAGR of a return series."""
    total = (1 + returns).prod() - 1
    n = len(returns)
    return (1 + total) ** (periods_per_year / n) - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown of the cumulative return curve."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def information_ratio(returns: pd.Series,
                      benchmark_returns: pd.Series,
                      periods_per_year: int = 252) -> float:
    """Annualised information ratio against a benchmark."""
    active = returns - benchmark_returns
    if active.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * active.mean() / active.std()


def compute_all_metrics(returns: pd.Series,
                        benchmark_returns: pd.Series = None) -> dict:
    """Compute the standard metric bundle in one call."""
    out = {
        "sharpe": sharpe_ratio(returns),
        "annual_return": annualized_return(returns),
        "max_drawdown": max_drawdown(returns),
        "volatility": returns.std() * np.sqrt(252),
        "n_periods": len(returns),
    }
    if benchmark_returns is not None:
        out["information_ratio"] = information_ratio(
            returns, benchmark_returns
        )
        out["excess_return"] = (
            annualized_return(returns) - annualized_return(benchmark_returns)
        )
    return out

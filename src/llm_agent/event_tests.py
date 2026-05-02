"""
event_tests.py
==============
Event-level rank tests for the event-study backtest.

The motivation is power. The daily-PnL Newey-West and Memmel ΔSharpe
tests in NB05 fail to reject at 5% in our universe (50 large caps,
~165 trading days per OOS year). One reason is that those tests treat
each trading day as an observation and lose information about
event-level signal strength. A complementary test that pools
*events* directly across the two test years recovers ~220 observations
in one number, which is power-friendlier in this small-sample regime.

Two tests, both one-sided in the direction of the modelling
hypothesis (higher predicted probability → higher realised CAR):

1. Spearman rank correlation between predicted P(positive CAR) and
   realised CAR over a long evaluation window (default CAR_2_21).
   The model is trained on CAR_2_11, but a Spearman test on the
   longer window is a stricter check: if the signal genuinely
   captures post-earnings drift, it should show up over +2 to +21
   as well, not only the trained window.

2. Mann-Whitney U test between the realised-CAR distributions of
   long-bucket events (predicted prob in the top quintile of the
   2023 validation set) and short-bucket events (bottom quintile).
   The thresholds are frozen on val 2023 to match NB05.

Conventions:
    - p-values are reported one-sided.
    - Both tests pool 2024 and 2025 OOS events.
    - Spearman ρ uses scipy.stats.spearmanr with alternative='greater'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline


@dataclass
class EventTestResult:
    """Bundle of test statistics for one OOS evaluation."""
    n_events: int
    long_n: int
    short_n: int
    spearman_rho: float
    spearman_p_one_sided: float
    long_mean_car: float
    short_mean_car: float
    car_diff: float                # long_mean - short_mean
    mw_u: float
    mw_p_one_sided: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def quintile_thresholds(probs: np.ndarray, q: float = 0.2) -> tuple[float, float]:
    """Return (short_cutoff, long_cutoff) at the q-th and (1-q)-th
    quantiles of `probs`. Matches NB05's frozen thresholds when called
    on val 2023 predictions."""
    short_cut = float(np.quantile(probs, q))
    long_cut  = float(np.quantile(probs, 1.0 - q))
    return short_cut, long_cut


def event_level_tests(
    pipe: Pipeline,
    feats: list[str],
    df_test: pd.DataFrame,
    long_cut: float,
    short_cut: float,
    car_col: str = "car_2_21",
    label_col: str = "y",
) -> EventTestResult:
    """Run both event-level tests on a pooled OOS DataFrame.

    Args:
        pipe:        fitted classifier from event_study_best.pkl.
        feats:       feature column list (must match pipe).
        df_test:     OOS rows; must contain `feats`, `car_col`, and
                     `label_col` columns. Rows with NaN are dropped.
        long_cut:    upper threshold for long-bucket assignment
                     (typically the val 2023 80th-percentile prob).
        short_cut:   lower threshold for short bucket
                     (typically val 2023 20th-percentile).
        car_col:     which realised-CAR column to test against;
                     CAR_2_21 is the literature-standard PEAD window.
        label_col:   binary label; only used for reporting positive rate.
    """
    df = df_test.dropna(subset=feats + [car_col]).copy()
    X = df[feats].astype(float).values
    df["prob"] = pipe.predict_proba(X)[:, 1]

    # Spearman: monotonic relationship between prob and realised CAR
    rho, p_two = stats.spearmanr(df["prob"], df[car_col])
    # Convert two-sided to one-sided in the hypothesised direction
    p_one_sided = (p_two / 2) if rho > 0 else (1 - p_two / 2)

    # Mann-Whitney U on the bucket extremes
    long_car  = df.loc[df["prob"] >= long_cut,  car_col].values
    short_car = df.loc[df["prob"] <= short_cut, car_col].values
    if len(long_car) == 0 or len(short_car) == 0:
        u_stat, mw_p = float("nan"), float("nan")
    else:
        u_stat, mw_p = stats.mannwhitneyu(long_car, short_car,
                                          alternative="greater")

    return EventTestResult(
        n_events=len(df),
        long_n=int(len(long_car)),
        short_n=int(len(short_car)),
        spearman_rho=float(rho),
        spearman_p_one_sided=float(p_one_sided),
        long_mean_car=float(long_car.mean()) if len(long_car) else float("nan"),
        short_mean_car=float(short_car.mean()) if len(short_car) else float("nan"),
        car_diff=float(long_car.mean() - short_car.mean())
                 if (len(long_car) and len(short_car)) else float("nan"),
        mw_u=float(u_stat),
        mw_p_one_sided=float(mw_p),
    )

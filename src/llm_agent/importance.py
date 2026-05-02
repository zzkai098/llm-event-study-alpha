"""
importance.py
=============
Model-interpretation utilities for the trained event-study classifier.

Currently provides permutation importance (sklearn's
permutation_importance with AUC scoring), block-level aggregation across
the three feature blocks defined in features.py (Tech / Sector / LLM),
and a side-by-side comparison against the model's impurity-based
feature_importances_.

Why permutation, not impurity:
    Impurity-based importance (the default for RF) is structurally biased
    toward high-cardinality continuous features — `log_dollar_vol` admits
    thousands of candidate split points per tree, while `guidance_num`
    ∈ {-1,0,+1} admits at most two. Permutation importance is invariant
    to feature type; it asks the empirical question "does the model use
    this feature at prediction time?" and is therefore the more honest
    metric for comparing the Tech, Sector, and LLM blocks against each
    other.

Inputs are always:
    pipe   — the fitted sklearn Pipeline from event_study_best.pkl
    feats  — the feature column list from the same pkl (order matters)
    df     — a feature matrix produced by features.build_feature_matrix
             with a binary 'y' column added by the caller

This module does NOT build the feature matrix; that is features.py's
job. The caller is expected to merge labels and assemble splits before
calling run_permutation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .features import LLM_FEATURES, TECH_FEATURES


def run_permutation(
    pipe: Pipeline,
    feats: list[str],
    df: pd.DataFrame,
    label_col: str = "y",
    n_repeats: int = 30,
    random_state: int = 0,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Permutation importance for each feature in `feats`.

    Args:
        pipe:        fitted sklearn Pipeline (e.g. from event_study_best.pkl).
        feats:       feature column names; must match what `pipe` was trained on.
        df:          DataFrame containing all `feats` columns plus `label_col`.
                     Rows with NaN in feats or label are dropped.
        label_col:   name of the binary target column (0/1).
        n_repeats:   number of independent shuffles per feature (default 30).
        random_state: passed to sklearn's permutation_importance.
        scoring:     sklearn scorer name. AUC is the model-native metric
                     here because thresholding happens downstream in the
                     backtest, not at the model.

    Returns:
        DataFrame with columns
            feature, delta_metric_mean, delta_metric_std
        sorted descending by mean. Δ is "score before shuffle minus score
        after shuffle", so larger = more important.
    """
    df = df.dropna(subset=list(feats) + [label_col])
    X = df[feats].astype(float).values
    y = df[label_col].astype(int).values
    res = permutation_importance(
        pipe, X, y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame({
            "feature": list(feats),
            "delta_metric_mean": res.importances_mean,
            "delta_metric_std":  res.importances_std,
        })
        .sort_values("delta_metric_mean", ascending=False)
        .reset_index(drop=True)
    )


def block_summary(
    perm_df: pd.DataFrame,
    feats: list[str],
    pipe: Optional[Pipeline] = None,
) -> pd.DataFrame:
    """Aggregate per-feature importance into Tech / Sector / LLM blocks.

    If `pipe` is provided, also reports the impurity-based importance
    share for each block (read from `pipe.named_steps['clf']`),
    enabling the "permutation vs impurity" comparison that motivates
    using permutation in the first place.

    Args:
        perm_df: output of run_permutation.
        feats:   the feature list (used to identify sector dummies by
                 their `sec_` prefix).
        pipe:    optional fitted pipeline; if its final step exposes
                 `feature_importances_`, an `impurity_pct` column is added.

    Returns:
        DataFrame with one row per block:
            block, perm_delta_metric, perm_pct[, impurity_pct]
        where perm_pct normalizes positive-only contributions to 100%
        (negative contributions are noise, not "share of importance").
    """
    sector_cols = [c for c in feats if c.startswith("sec_")]
    blocks = {
        "Tech":   list(TECH_FEATURES),
        "Sector": sector_cols,
        "LLM":    list(LLM_FEATURES),
    }

    rows = []
    for name, cols in blocks.items():
        rows.append({
            "block": name,
            "perm_delta_metric": float(
                perm_df.loc[perm_df.feature.isin(cols), "delta_metric_mean"].sum()
            ),
        })
    out = pd.DataFrame(rows)

    pos_total = out.loc[out.perm_delta_metric > 0, "perm_delta_metric"].sum()
    out["perm_pct"] = out["perm_delta_metric"] / pos_total * 100 if pos_total > 0 else np.nan

    if pipe is not None:
        clf = pipe.named_steps.get("clf")
        if clf is not None and hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=feats)
            for name, cols in blocks.items():
                share = imp[imp.index.isin(cols)].sum()
                out.loc[out.block == name, "impurity_share"] = share
            out["impurity_pct"] = out["impurity_share"] / out["impurity_share"].sum() * 100

    return out


def build_per_feature_table(
    perm_df: pd.DataFrame,
    feats: list[str],
    pipe: Optional[Pipeline] = None,
) -> pd.DataFrame:
    """Per-feature view with block tag and (optionally) impurity comparison.

    Convenience for paper appendix tables: tags each feature with its
    block, sorts by permutation importance, and joins impurity share
    when the pipeline is provided.
    """
    sector_cols = [c for c in feats if c.startswith("sec_")]
    block_of = {f: ("Tech"   if f in TECH_FEATURES else
                    "LLM"    if f in LLM_FEATURES else
                    "Sector" if f in sector_cols   else "Other")
                for f in feats}

    out = perm_df.copy()
    out["block"] = out["feature"].map(block_of)

    if pipe is not None:
        clf = pipe.named_steps.get("clf")
        if clf is not None and hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=feats, name="impurity")
            out = out.merge(imp.reset_index().rename(columns={"index": "feature"}),
                            on="feature", how="left")
            out["impurity_pct"] = out["impurity"] / out["impurity"].sum() * 100

    return out.sort_values("delta_metric_mean", ascending=False).reset_index(drop=True)

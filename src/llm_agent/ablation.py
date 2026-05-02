"""
ablation.py
===========
Feature-set ablations for the trained Random Forest classifier.

Two analyses, both using the pkl's Optuna-tuned hyperparameters as
the fixed model, varying only which columns are passed in:

    D.1 (block_ablation):
        full / drop-Tech / drop-Sector / drop-LLM / LLM-only.
        Quantifies the marginal val-AUC contribution of each of the
        three feature blocks.

    D.2 (drop_one_llm):
        full / drop one of the six LLM features at a time.
        Identifies which LLM feature is doing the actual work.

The hyperparameters are held constant at the pkl's `best_params`
(Optuna result), so any AUC difference is attributable to the
feature subset, not to retuning. This is the cleanest possible
ablation: one variable changes per row.

Train / val regime mirrors NB04: train 2018–2022, val 2023, tail
filter `|car_2_11| > 0.5·σ_e·√10` on both. Test sets are not used
here — these are val-AUC ablations, not OOS scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import LLM_FEATURES, TECH_FEATURES


@dataclass
class AblationResult:
    """One row of an ablation table."""
    config: str
    n_features: int
    feature_names: list[str]
    val_auc: float
    val_top_decile_precision: float
    delta_vs_full: float = 0.0     # filled in by run_*_ablation


def _split(base: pd.DataFrame, feats: list[str],
           train_years: list[int], val_years: list[int]) -> tuple:
    """Train/val split with NB04's tail filter on both."""
    cols = list(feats) + ["y", "car_2_11", "sigma_e", "fiscal_year"]
    df = base.dropna(subset=cols).copy()
    df = df[df["car_2_11"].abs() >= 0.5 * df["sigma_e"] * np.sqrt(10)]
    train = df[df.fiscal_year.isin(train_years)]
    val   = df[df.fiscal_year.isin(val_years)]
    return train, val


def fit_with_subset(base: pd.DataFrame, feature_subset: list[str],
                    rf_params: dict, train_years: list[int],
                    val_years: list[int]) -> AblationResult:
    """Refit the RF on a column subset and report val metrics.

    The Pipeline (StandardScaler → RandomForestClassifier) is
    rebuilt here rather than reusing the pickled one, because the
    pickle expects a fixed 22-column input. Hyperparameters are
    copied verbatim from the pkl.
    """
    train, val = _split(base, feature_subset, train_years, val_years)
    X_tr = train[feature_subset].astype(float).values
    y_tr = train["y"].astype(int).values
    X_vl = val[feature_subset].astype(float).values
    y_vl = val["y"].astype(int).values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, random_state=0, n_jobs=-1, **rf_params,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    p_val = pipe.predict_proba(X_vl)[:, 1]

    auc = float(roc_auc_score(y_vl, p_val))
    k = max(1, int(0.1 * len(p_val)))
    top_idx = np.argsort(-p_val)[:k]
    top_prec = float(y_vl[top_idx].mean())

    return AblationResult(
        config="",
        n_features=len(feature_subset),
        feature_names=list(feature_subset),
        val_auc=auc,
        val_top_decile_precision=top_prec,
    )


def run_block_ablation(base: pd.DataFrame, full_feats: list[str],
                       rf_params: dict,
                       train_years: list[int], val_years: list[int]
                       ) -> pd.DataFrame:
    """D.1 — drop one block at a time."""
    sector = [c for c in full_feats if c.startswith("sec_")]
    tech   = [c for c in full_feats if c in TECH_FEATURES]
    llm    = [c for c in full_feats if c in LLM_FEATURES]

    configs = {
        "full (Tech + Sector + LLM)": full_feats,
        "drop Tech":   sector + llm,
        "drop Sector": tech + llm,
        "drop LLM":    tech + sector,
        "LLM only":    llm,
        "Tech only":   tech,
    }

    rows = []
    full_auc = None
    for name, sub in configs.items():
        res = fit_with_subset(base, sub, rf_params, train_years, val_years)
        res.config = name
        if name.startswith("full"):
            full_auc = res.val_auc
        rows.append(res)

    for r in rows:
        r.delta_vs_full = r.val_auc - full_auc

    return pd.DataFrame([{
        "config": r.config, "n_features": r.n_features,
        "val_auc": r.val_auc, "delta_vs_full": r.delta_vs_full,
        "val_top_decile_precision": r.val_top_decile_precision,
    } for r in rows])


def run_drop_one_llm(base: pd.DataFrame, full_feats: list[str],
                     rf_params: dict,
                     train_years: list[int], val_years: list[int]
                     ) -> pd.DataFrame:
    """D.2 — drop one LLM feature at a time."""
    rows = []

    # Anchor: full model
    full_res = fit_with_subset(base, full_feats, rf_params,
                                train_years, val_years)
    full_res.config = "full (all 6 LLM features)"
    rows.append(full_res)

    for dropped in LLM_FEATURES:
        sub = [c for c in full_feats if c != dropped]
        res = fit_with_subset(base, sub, rf_params, train_years, val_years)
        res.config = f"drop {dropped}"
        rows.append(res)

    full_auc = full_res.val_auc
    for r in rows:
        r.delta_vs_full = r.val_auc - full_auc

    return pd.DataFrame([{
        "config": r.config, "n_features": r.n_features,
        "val_auc": r.val_auc, "delta_vs_full": r.delta_vs_full,
        "val_top_decile_precision": r.val_top_decile_precision,
    } for r in rows])

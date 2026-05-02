"""
bakeoff.py
==========
Cross-validated model comparison for the NB04 / appendix-C bake-off.

Reproduces the NB04 training regime that produced
`models/event_study_best.pkl` and adds the three other classifier
families (logistic regression, SVM, XGBoost) plus a prior-strategy
DummyClassifier baseline. Reporting metrics: ROC AUC (mean ± std
across CV folds), held-out val AUC, top-decile precision on val.

Design constraints carried over from NB04:
    - target  = sign(car_2_11), binary
    - tail filter applies to train and val only
    - features = arm 'C' (22 features: 5 tech + 11 sector + 6 LLM)
    - CV       = TimeSeriesSplit(n_splits=5) on train only
    - tuning   = GridSearchCV with scoring='roc_auc', refit=True
    - val      = held out; never used in CV or tuning
    - all classifiers wrapped in Pipeline([StandardScaler, clf])

The Pipeline is uniform across families even though tree-based models
(RF, XGB) do not need scaling — keeping a single Pipeline interface
simplifies the training loop at zero performance cost.

This module does not redefine the pipelines or grids; it imports them
from `models.make_model_grid` to stay consistent with NB04.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .models import make_model_grid, eval_metrics, predict_proba_safe


@dataclass
class BakeoffResult:
    """One row of the bake-off summary table."""
    model: str
    cv_auc_mean: float
    cv_auc_std: float
    cv_auc_folds: list[float]
    val_auc: float
    val_top_decile_precision: float
    val_acc: float
    val_f1: float
    best_params: dict = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0


def _split_by_period(base: pd.DataFrame, train_years: list[int],
                     val_years: list[int],
                     feats: list[str], label_col: str = "y",
                     tail_filter: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the base DataFrame into train / val with the NB04 regime.

    Tail filter (|car_2_11| > 0.5·σ_e·√10) is applied to both train and
    val. Test sets are never filtered — they are not produced here.
    """
    cols = list(feats) + [label_col, "car_2_11", "sigma_e", "fiscal_year"]
    df = base.dropna(subset=cols).copy()
    if tail_filter:
        df = df[df["car_2_11"].abs() >= 0.5 * df["sigma_e"] * np.sqrt(10)]
    train = df[df.fiscal_year.isin(train_years)].copy()
    val   = df[df.fiscal_year.isin(val_years)].copy()
    return train, val


def fit_one_model(name: str, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  n_splits: int = 5,
                  random_state: int = 0) -> BakeoffResult:
    """Fit one named model with TimeSeriesSplit CV grid search and
    score the best estimator on the held-out val set.
    """
    factory, grid = make_model_grid()[name]
    base_pipe = factory()
    cv = TimeSeriesSplit(n_splits=n_splits)

    if grid:
        gs = GridSearchCV(base_pipe, grid, scoring="roc_auc",
                          cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = dict(gs.best_params_)
        idx = gs.best_index_
        fold_scores = [
            float(gs.cv_results_[f"split{i}_test_score"][idx])
            for i in range(n_splits)
        ]
    else:
        # No grid (e.g. baseline) → just CV-score and refit on full train
        from sklearn.model_selection import cross_val_score
        fold_scores = list(map(float, cross_val_score(
            base_pipe, X_train, y_train, cv=cv,
            scoring="roc_auc", n_jobs=-1,
        )))
        model = base_pipe.fit(X_train, y_train)
        best_params = {}

    # Val metrics
    p_val = predict_proba_safe(model, X_val)
    val_metrics = eval_metrics(y_val, p_val)

    return BakeoffResult(
        model=name,
        cv_auc_mean=float(np.mean(fold_scores)),
        cv_auc_std=float(np.std(fold_scores)),
        cv_auc_folds=fold_scores,
        val_auc=val_metrics["auc"],
        val_top_decile_precision=val_metrics["precision_top10"],
        val_acc=val_metrics["acc"],
        val_f1=val_metrics["f1"],
        best_params=best_params,
        n_train=len(X_train),
        n_val=len(X_val),
    )


def run_bakeoff(base: pd.DataFrame, feats: list[str],
                train_years: list[int], val_years: list[int],
                model_names: Optional[list[str]] = None,
                n_splits: int = 5,
                random_state: int = 0,
                tail_filter: bool = True) -> pd.DataFrame:
    """Run the bake-off across `model_names` and return a summary table.

    Args:
        base:        feature matrix from features.build_feature_matrix
                     plus 'y', 'car_2_11', 'sigma_e', 'fiscal_year' cols.
        feats:       feature column names (must match NB04's feature list).
        train_years: list of fiscal_years for training (e.g. [2018..2022]).
        val_years:   list of fiscal_years for validation (e.g. [2023]).
        model_names: which models to run. Defaults to
                     ['baseline', 'logreg', 'svm', 'rf', 'xgb'] minus
                     any not available in models.make_model_grid().
        n_splits:    TimeSeriesSplit folds. NB04 used 5.
        random_state: reserved (RF and XGB random_state are set inside
                     models.py).
        tail_filter: apply NB04's tail filter to train/val.

    Returns:
        DataFrame with one row per model, columns:
            model, n_train, n_val, cv_auc_mean, cv_auc_std,
            val_auc, val_top_decile_precision, val_acc, val_f1, best_params
    """
    train, val = _split_by_period(base, train_years, val_years, feats,
                                   tail_filter=tail_filter)
    X_train = train[feats].astype(float).values
    y_train = train["y"].astype(int).values
    X_val   = val[feats].astype(float).values
    y_val   = val["y"].astype(int).values

    available = list(make_model_grid().keys())
    names = model_names or ["baseline", "logreg", "svm", "rf", "xgb"]
    names = [n for n in names if n in available]

    rows = []
    for name in names:
        print(f"  fitting {name}...")
        res = fit_one_model(name, X_train, y_train, X_val, y_val,
                            n_splits=n_splits, random_state=random_state)
        rows.append({
            "model": res.model,
            "n_train": res.n_train,
            "n_val": res.n_val,
            "cv_auc_mean": res.cv_auc_mean,
            "cv_auc_std": res.cv_auc_std,
            "val_auc": res.val_auc,
            "val_top_decile_precision": res.val_top_decile_precision,
            "val_acc": res.val_acc,
            "val_f1": res.val_f1,
            "best_params": res.best_params,
        })

    return pd.DataFrame(rows)

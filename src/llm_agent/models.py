"""
models.py
=========
Thin wrappers around the classifiers used in NB04: logistic regression,
linear SVM, RBF SVM, random forest, XGBoost, plus a constant baseline
for the BuyHold (arm A) comparison.

All models are wrapped in a sklearn Pipeline that StandardScales features
first; tree-based models technically don't need scaling but keeping a
single Pipeline interface makes the training loop simpler.

Tuning uses sklearn TimeSeriesSplit on the train set only — the val and
test sets are held out for reporting.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except Exception as _xgb_err:  # libomp missing on some macs, etc.
    XGBClassifier = None
    _XGB_OK = False
    _XGB_IMPORT_ERROR = _xgb_err


def _pipe(est) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def make_model_grid() -> Dict[str, Tuple[Callable[[], Pipeline], dict]]:
    """name -> (pipeline factory, param grid keyed on 'clf__*')."""
    grid: Dict[str, Tuple[Callable[[], Pipeline], dict]] = {
        "baseline": (
            lambda: Pipeline([("clf", DummyClassifier(strategy="prior"))]),
            {},
        ),
        "logreg": (
            lambda: _pipe(LogisticRegression(max_iter=2000, solver="liblinear")),
            {"clf__C": [0.1, 1.0, 10.0], "clf__penalty": ["l2"]},
        ),
        "svm": (
            lambda: _pipe(SVC(probability=True)),
            {"clf__kernel": ["linear", "rbf"],
             "clf__C": [0.1, 1.0, 10.0],
             "clf__gamma": ["scale", 0.1]},
        ),
        "rf": (
            lambda: _pipe(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
            {"clf__max_depth": [3, 6, None], "clf__min_samples_leaf": [1, 5]},
        ),
    }
    if _XGB_OK:
        grid["xgb"] = (
            lambda: _pipe(XGBClassifier(
                n_estimators=300, learning_rate=0.05, eval_metric="logloss",
                random_state=0, n_jobs=-1, verbosity=0,
            )),
            {"clf__max_depth": [3, 5], "clf__subsample": [0.7, 1.0]},
        )
    return grid


def fit_with_cv(name: str, X_train: pd.DataFrame, y_train: np.ndarray,
                n_splits: int = 4) -> Pipeline:
    """Fit a model with TimeSeriesSplit CV on the training set."""
    factory, grid = make_model_grid()[name]
    base = factory()
    if not grid:
        return base.fit(X_train, y_train)
    cv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def train_eval(name: str,
               X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               n_splits: int = 5) -> dict:
    """Train one model with k-fold TimeSeriesSplit CV on train + score on held-out val.

    Returns:
        {
          'model': fitted_pipeline (refit on full train),
          'cv_auc_mean': float,
          'cv_auc_std': float,
          'cv_auc_folds': list[float],
          'best_params': dict,
          'val_metrics': {auc, acc, f1, precision_top10},
        }
    """
    from sklearn.model_selection import cross_val_score
    factory, grid = make_model_grid()[name]
    base = factory()
    cv = TimeSeriesSplit(n_splits=n_splits)

    if grid:
        gs = GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
        # CV scores at the chosen hyperparams come from gs.cv_results_
        idx = gs.best_index_
        fold_scores = [
            float(gs.cv_results_[f"split{i}_test_score"][idx]) for i in range(n_splits)
        ]
    else:
        fold_scores = list(map(float, cross_val_score(
            base, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )))
        model = base.fit(X_train, y_train)
        best_params = {}

    p_val = predict_proba_safe(model, X_val)
    return {
        "model": model,
        "cv_auc_mean": float(np.mean(fold_scores)),
        "cv_auc_std": float(np.std(fold_scores)),
        "cv_auc_folds": fold_scores,
        "best_params": best_params,
        "val_metrics": eval_metrics(y_val, p_val),
    }


def eval_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """Binary classification metrics. p_pred is the positive-class probability."""
    y_pred = (p_pred >= 0.5).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, p_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    # precision @ top decile
    k = max(1, int(0.1 * len(p_pred)))
    top_idx = np.argsort(-p_pred)[:k]
    out["precision_top10"] = float(y_true[top_idx].mean())
    return out


def predict_proba_safe(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """predict_proba positive class; for DummyClassifier returns prior."""
    if hasattr(model.named_steps["clf"], "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

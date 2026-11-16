"""Challenger models, tuned on the validation months with a small grid each."""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from credit_scorecard_service.config import RANDOM_SEED
from credit_scorecard_service.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
)
from credit_scorecard_service.training.metrics import classification_metrics

log = logging.getLogger(__name__)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(build_features(df))


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[Any] = [SimpleImputer(strategy="median")]
    if scale_numeric:
        # Income and loan size are heavily right-skewed; log them before scaling.
        numeric_steps += [FunctionTransformer(np.log1p), StandardScaler()]
    return ColumnTransformer(
        [
            ("num", make_pipeline(*numeric_steps), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.001),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[..., ClassifierMixin]
    grid: dict[str, list[Any]]
    scale_numeric: bool = False

    def pipeline(self, **params: Any) -> Pipeline:
        return Pipeline(
            [("prep", _preprocessor(self.scale_numeric)), ("model", self.build(**params))]
        )

    def param_sets(self) -> list[dict[str, Any]]:
        keys = list(self.grid)
        return [
            dict(zip(keys, combo, strict=True)) for combo in itertools.product(*self.grid.values())
        ]


CANDIDATES = (
    Candidate(
        "logreg_onehot",
        lambda **p: LogisticRegression(max_iter=2000, **p),
        {"C": [0.01, 1.0]},
        scale_numeric=True,
    ),
    Candidate(
        "decision_tree",
        lambda **p: DecisionTreeClassifier(random_state=RANDOM_SEED, **p),
        {"max_depth": [4, 6, 8], "min_samples_leaf": [500, 2000]},
    ),
    Candidate(
        "random_forest",
        lambda **p: RandomForestClassifier(
            n_estimators=200, max_samples=0.2, n_jobs=-1, random_state=RANDOM_SEED, **p
        ),
        {"max_depth": [8, 12], "min_samples_leaf": [200, 1000]},
    ),
    Candidate(
        "xgboost",
        lambda **p: XGBClassifier(
            n_estimators=400,
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            **p,
        ),
        {"max_depth": [3, 5], "learning_rate": [0.05, 0.1], "min_child_weight": [50, 200]},
    ),
)


@dataclass
class ChallengerResult:
    name: str
    params: dict[str, Any]
    valid: dict[str, float]
    test: dict[str, float]
    fit_seconds: float
    model: Pipeline


def tune_and_evaluate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    candidates: tuple[Candidate, ...] = CANDIDATES,
) -> list[ChallengerResult]:
    X_train, y_train = feature_frame(train), train[TARGET].to_numpy()
    X_valid, y_valid = feature_frame(valid), valid[TARGET].to_numpy()
    X_test, y_test = feature_frame(test), test[TARGET].to_numpy()

    results = []
    for cand in candidates:
        best: tuple[float, dict[str, Any], Pipeline, float] | None = None
        for params in cand.param_sets():
            start = time.perf_counter()
            model = cand.pipeline(**params).fit(X_train, y_train)
            elapsed = time.perf_counter() - start
            auc = classification_metrics(y_valid, model.predict_proba(X_valid)[:, 1])["auc"]
            log.info("%-14s %s valid AUC=%.4f (%.1fs)", cand.name, params, auc, elapsed)
            if best is None or auc > best[0]:
                best = (auc, params, model, elapsed)

        assert best is not None
        _, params, model, elapsed = best
        results.append(
            ChallengerResult(
                name=cand.name,
                params=params,
                valid=classification_metrics(y_valid, model.predict_proba(X_valid)[:, 1]),
                test=classification_metrics(y_test, model.predict_proba(X_test)[:, 1]),
                fit_seconds=round(elapsed, 1),
                model=model,
            )
        )
    return results

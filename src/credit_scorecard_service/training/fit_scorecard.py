"""Fit the champion WoE logistic-regression scorecard and export it as a points table."""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.linear_model import LogisticRegression

from credit_scorecard_service.config import (
    BINNING,
    RISK_BAND_PD_LIMITS,
    SCALING,
    BinningConfig,
    Scaling,
)
from credit_scorecard_service.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    TARGET,
    build_features,
)
from credit_scorecard_service.scorecard import (
    CategoricalBins,
    FeatureCard,
    NumericBins,
    RiskBand,
    Scorecard,
)

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    scorecard: Scorecard
    information_values: dict[str, float]
    selected: list[str]
    dropped: dict[str, str]  # feature -> reason


def _fit_binning(
    name: str, values: np.ndarray, y: np.ndarray, cfg: BinningConfig
) -> OptimalBinning:
    categorical = name in CATEGORICAL_FEATURES
    ob = OptimalBinning(
        name=name,
        dtype="categorical" if categorical else "numerical",
        solver="cp",
        monotonic_trend="auto_asc_desc" if not categorical else None,
        max_n_bins=cfg.max_n_bins,
        min_bin_size=cfg.min_bin_size,
    )
    ob.fit(values, y)
    return ob


def _woe(ob: OptimalBinning, values: np.ndarray) -> np.ndarray:
    return np.asarray(ob.transform(values, metric="woe"), dtype=float)


def _feature_card(
    name: str, ob: OptimalBinning, coef: float, iv: float, factor: float
) -> FeatureCard:
    """Points per bin = -factor * coef * WoE, evaluated on one representative per bin
    so the exported table reproduces optbinning's transform exactly."""

    def points_for(values: np.ndarray) -> list[float]:
        return [float(-factor * coef * w) for w in _woe(ob, values)]

    if name in CATEGORICAL_FEATURES:
        groups = [[str(c) for c in group] for group in ob.splits]
        bins: NumericBins | CategoricalBins = CategoricalBins(
            categories=groups,
            points=points_for(np.array([g[0] for g in groups], dtype=object)),
            unknown_points=points_for(np.array(["__unseen__"], dtype=object))[0],
        )
        kind = "categorical"
    else:
        splits = [float(s) for s in ob.splits]
        reps = np.array([splits[0] - 1.0, *splits]) if splits else np.array([0.0])
        edges = ["-inf", *(f"{s:g}" for s in splits), "inf"]
        bins = NumericBins(
            splits=splits,
            points=points_for(reps),
            missing_points=points_for(np.array([np.nan]))[0],
            labels=[f"[{lo}, {hi})" for lo, hi in itertools.pairwise(edges)],
        )
        kind = "numeric"
    return FeatureCard(name=name, kind=kind, coefficient=coef, information_value=iv, bins=bins)


def fit_scorecard(
    train: pd.DataFrame,
    version: str,
    scaling: Scaling = SCALING,
    cfg: BinningConfig = BINNING,
) -> FitResult:
    y = train[TARGET].to_numpy()
    feats = build_features(train)

    binnings: dict[str, OptimalBinning] = {}
    ivs: dict[str, float] = {}
    for name in MODEL_FEATURES:
        ob = _fit_binning(name, feats[name], y, cfg)
        ob.binning_table.build()
        binnings[name] = ob
        ivs[name] = round(float(ob.binning_table.iv), 4)
        log.info("Binned %-18s IV=%.4f bins=%d", name, ivs[name], len(ob.splits) + 1)

    dropped = {n: f"IV {ivs[n]:.4f} < {cfg.min_iv}" for n in MODEL_FEATURES if ivs[n] < cfg.min_iv}
    selected = [n for n in MODEL_FEATURES if n not in dropped]
    woe = {n: _woe(binnings[n], feats[n]) for n in selected}

    # WoE is ln(%good / %bad), so every coefficient on PD should be negative.
    # A positive one signals collinearity; drop the weakest such feature and refit.
    while True:
        X = np.column_stack([woe[n] for n in selected])
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y)
        coefs = dict(zip(selected, lr.coef_[0], strict=True))
        wrong = [n for n in selected if coefs[n] >= 0]
        if not wrong:
            break
        worst = min(wrong, key=lambda n: ivs[n])
        dropped[worst] = f"coefficient sign flipped ({coefs[worst]:+.3f})"
        selected.remove(worst)
        log.warning("Dropping %s: %s", worst, dropped[worst])

    factor = scaling.pdo / math.log(2)
    offset = scaling.base_score - factor * math.log(scaling.base_odds)
    cards = [_feature_card(n, binnings[n], float(coefs[n]), ivs[n], factor) for n in selected]

    scorecard = Scorecard(
        version=version,
        base_score=scaling.base_score,
        base_odds=scaling.base_odds,
        pdo=scaling.pdo,
        base_points=float(offset - factor * lr.intercept_[0]),
        features=cards,
        risk_bands=[],
        cutoff_score=0.0,
    )
    scorecard.risk_bands = risk_bands_from_pd(scorecard)

    # The points table must reproduce the fitted model.
    expected = lr.predict_proba(X)[:, 1]
    actual = scorecard.predict_pd(train)
    max_diff = float(np.max(np.abs(expected - actual)))
    if max_diff > 1e-6:
        raise AssertionError(f"Points table disagrees with logistic regression by {max_diff}")

    return FitResult(
        scorecard=scorecard, information_values=ivs, selected=selected, dropped=dropped
    )


def risk_bands_from_pd(scorecard: Scorecard) -> list[RiskBand]:
    """Translate the configured PD limits into score thresholds."""
    lowest = math.floor(scorecard.score_range[0]) - 1
    bands = []
    for name, max_pd in RISK_BAND_PD_LIMITS:
        min_score = float(scorecard.pd_to_score(max_pd)) if max_pd < 1.0 else lowest
        bands.append(RiskBand(name=name, min_score=round(min_score, 2)))
    return bands

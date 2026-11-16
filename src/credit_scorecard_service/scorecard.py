"""Serving-time scorecard: a points table applied with numpy.

The fitted WoE logistic regression is exported as points per bin (see
`training.fit_scorecard`), so the artifact is plain JSON that a reviewer can
read and the service can apply without any ML library.

Scaling: score = offset + factor * ln(odds_good), with
factor = pdo / ln(2) and offset = base_score - factor * ln(base_odds).
The score is the sum of `base_points` and each feature's bin points, and the
probability of default is recovered exactly by inverting that formula.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from credit_scorecard_service.features import FEATURE_LABELS, build_features


@dataclass(frozen=True)
class NumericBins:
    """Bins are left-closed: bin i covers [splits[i-1], splits[i])."""

    splits: list[float]
    points: list[float]  # len(splits) + 1
    missing_points: float
    labels: list[str]

    def apply(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        idx = np.searchsorted(np.asarray(self.splits, dtype=float), values, side="right")
        missing = np.isnan(values)
        points = np.where(missing, self.missing_points, np.asarray(self.points)[idx])
        return points, np.where(missing, -1, idx)

    def max_points(self) -> float:
        return max(self.points)


@dataclass(frozen=True)
class CategoricalBins:
    """Each bin groups one or more categories. Unseen categories get `unknown_points`."""

    categories: list[list[str]]
    points: list[float]
    unknown_points: float

    def apply(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lookup = {cat: i for i, group in enumerate(self.categories) for cat in group}
        idx = np.array([lookup.get(v, -1) for v in values], dtype=int)
        table = np.asarray(self.points, dtype=float)
        points = np.where(idx >= 0, table[np.maximum(idx, 0)], self.unknown_points)
        return points, idx

    def max_points(self) -> float:
        return max(self.points)


@dataclass(frozen=True)
class FeatureCard:
    name: str
    kind: str  # "numeric" | "categorical"
    coefficient: float
    information_value: float
    bins: NumericBins | CategoricalBins

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "coefficient": self.coefficient,
            "information_value": self.information_value,
            "bins": asdict(self.bins),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FeatureCard:
        bins_cls = NumericBins if d["kind"] == "numeric" else CategoricalBins
        return cls(
            name=d["name"],
            kind=d["kind"],
            coefficient=d["coefficient"],
            information_value=d["information_value"],
            bins=bins_cls(**d["bins"]),
        )


@dataclass(frozen=True)
class RiskBand:
    name: str
    min_score: float  # inclusive lower bound


@dataclass(frozen=True)
class ReasonCode:
    feature: str
    description: str
    value: Any
    points: float
    points_below_max: float


@dataclass(frozen=True)
class ScoreResult:
    probability_of_default: float
    score: int
    risk_band: str
    decision: str
    reason_codes: list[ReasonCode]


@dataclass
class Scorecard:
    version: str
    base_score: float
    base_odds: float
    pdo: float
    base_points: float
    features: list[FeatureCard]
    risk_bands: list[RiskBand]  # sorted by min_score, highest first
    cutoff_score: float  # approve when score >= cutoff_score
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- scaling -----------------------------------------------------------
    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def score_to_pd(self, score: np.ndarray | float) -> np.ndarray:
        ln_odds_good = (np.asarray(score, dtype=float) - self.offset) / self.factor
        return 1.0 / (1.0 + np.exp(ln_odds_good))

    def pd_to_score(self, pd: np.ndarray | float) -> np.ndarray:
        pd = np.asarray(pd, dtype=float)
        return self.offset + self.factor * np.log((1.0 - pd) / pd)

    # -- scoring -----------------------------------------------------------
    def points(self, raw: Mapping[str, Sequence[Any]]) -> dict[str, np.ndarray]:
        """Points per feature for each row of `raw` (column layout)."""
        feats = build_features(raw)
        return {card.name: card.bins.apply(feats[card.name])[0] for card in self.features}

    def total_score(self, raw: Mapping[str, Sequence[Any]]) -> np.ndarray:
        per_feature = self.points(raw)
        return self.base_points + np.sum(list(per_feature.values()), axis=0)

    def predict_pd(self, raw: Mapping[str, Sequence[Any]]) -> np.ndarray:
        return self.score_to_pd(self.total_score(raw))

    def risk_band(self, score: float) -> str:
        for band in self.risk_bands:
            if score >= band.min_score:
                return band.name
        return self.risk_bands[-1].name

    def score_records(
        self, raw: Mapping[str, Sequence[Any]], n_reasons: int = 3
    ) -> list[ScoreResult]:
        """Full scoring output (PD, score, band, decision, reason codes) per row."""
        per_feature = self.points(raw)
        names = [card.name for card in self.features]
        point_matrix = np.column_stack([per_feature[n] for n in names])
        max_points = np.array([card.bins.max_points() for card in self.features])
        totals = self.base_points + point_matrix.sum(axis=1)
        pds = self.score_to_pd(totals)
        feats = build_features(raw)

        results = []
        for i, total in enumerate(totals):
            shortfall = max_points - point_matrix[i]
            order = np.argsort(-shortfall, kind="stable")
            reasons = [
                ReasonCode(
                    feature=names[j],
                    description=FEATURE_LABELS.get(names[j], names[j]),
                    value=_to_builtin(feats[names[j]][i]),
                    points=round(float(point_matrix[i, j]), 1),
                    points_below_max=round(float(shortfall[j]), 1),
                )
                for j in order[:n_reasons]
                if shortfall[j] > 1e-9
            ]
            results.append(
                ScoreResult(
                    probability_of_default=float(pds[i]),
                    score=round(total),
                    risk_band=self.risk_band(total),
                    decision="approve" if total >= self.cutoff_score else "decline",
                    reason_codes=reasons,
                )
            )
        return results

    @property
    def score_range(self) -> tuple[float, float]:
        lo = self.base_points + sum(min(c.bins.points) for c in self.features)
        hi = self.base_points + sum(max(c.bins.points) for c in self.features)
        return lo, hi

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scaling": {
                "base_score": self.base_score,
                "base_odds": self.base_odds,
                "pdo": self.pdo,
            },
            "base_points": self.base_points,
            "cutoff_score": self.cutoff_score,
            "risk_bands": [asdict(b) for b in self.risk_bands],
            "features": [c.to_dict() for c in self.features],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Scorecard:
        return cls(
            version=d["version"],
            base_points=d["base_points"],
            cutoff_score=d["cutoff_score"],
            risk_bands=[RiskBand(**b) for b in d["risk_bands"]],
            features=[FeatureCard.from_dict(c) for c in d["features"]],
            metadata=d.get("metadata", {}),
            **d["scaling"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Scorecard:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

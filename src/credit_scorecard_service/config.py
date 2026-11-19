"""Central configuration: paths, data source, split dates, scaling and business rules.

Anything a deployment may want to change is read from an environment variable;
modelling constants live here so training runs are reproducible from the code alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("CS_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("CS_MODELS_DIR", PROJECT_ROOT / "models"))

RANDOM_SEED = 42


@dataclass(frozen=True)
class DataSource:
    """Lending Club granting dataset on Zenodo (CC BY 4.0)."""

    record_id: str = "11295916"
    filename: str = "LC_loans_granting_model_dataset.csv"
    md5: str = "b019384d6bc65bf2a3e839362e4ff502"

    @property
    def url(self) -> str:
        return f"https://zenodo.org/api/records/{self.record_id}/files/{self.filename}/content"

    @property
    def local_path(self) -> Path:
        return DATA_DIR / "raw" / self.filename


DATA_SOURCE = DataSource()


@dataclass(frozen=True)
class TimeSplit:
    """Out-of-time split on issue date (inclusive month boundaries).

    The gap between validation and test leaves time for loans to reach an
    outcome; recent vintages only contain loans that resolved early, which
    biases their default rate.
    """

    train_end: str = "2015-12"
    valid_start: str = "2016-01"
    valid_end: str = "2016-06"
    test_start: str = "2016-10"
    test_end: str = "2017-03"


TIME_SPLIT = TimeSplit()


@dataclass(frozen=True)
class Scaling:
    """Points scaling: `base_score` points at good:bad odds of `base_odds`,
    and every `pdo` points doubles the odds."""

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0


SCALING = Scaling()


@dataclass(frozen=True)
class BinningConfig:
    # Fixed by hand, not tuned; `cs-sensitivity` checks each one (README, Results).
    min_iv: float = 0.02  # features below this information value are dropped
    max_n_bins: int = 10
    min_bin_size: float = 0.02  # never binds: optbinning's pre-bins already hold >= 5%


BINNING = BinningConfig()


@dataclass(frozen=True)
class DecisionCosts:
    """Relative cost of the two decision errors used to pick the cutoff.

    Approving a loan that defaults loses most of the principal; declining a
    good borrower loses the interest margin. A 5:1 ratio is a common rule of
    thumb for unsecured consumer lending.
    """

    approve_bad: float = 5.0
    decline_good: float = 1.0


DECISION_COSTS = DecisionCosts()

# Risk bands as upper PD bounds, best band first. The last band takes the rest.
RISK_BAND_PD_LIMITS: tuple[tuple[str, float], ...] = (
    ("A", 0.06),
    ("B", 0.12),
    ("C", 0.20),
    ("D", 0.30),
    ("E", 1.0),
)

N_REASON_CODES = 3

# Regression guard: training fails rather than ship a model below this.
MIN_VALID_AUC = 0.65


@dataclass(frozen=True)
class ServiceSettings:
    model_path: Path
    prediction_log_path: Path
    max_batch_size: int

    @classmethod
    def from_env(cls) -> ServiceSettings:
        return cls(
            model_path=Path(os.getenv("CS_MODEL_PATH", MODELS_DIR / "champion.json")),
            prediction_log_path=Path(
                os.getenv("CS_PREDICTION_LOG", PROJECT_ROOT / "logs" / "predictions.db")
            ),
            max_batch_size=int(os.getenv("CS_MAX_BATCH_SIZE", "1000")),
        )

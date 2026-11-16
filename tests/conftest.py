"""Shared fixtures: a synthetic Lending Club-like dataset and a scorecard fitted on it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scorecard_service.features import (
    EMP_LENGTH_VALUES,
    HOME_OWNERSHIP_VALUES,
    PURPOSE_VALUES,
)


def make_loans(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Synthetic loans whose default risk rises with DTI and loan-to-income
    and falls with FICO, roughly like the real data."""
    rng = np.random.default_rng(seed)
    revenue = np.exp(rng.normal(11.1, 0.5, n)).round(-2)
    loan_amnt = rng.integers(10, 400, n) * 100
    fico = rng.normal(695, 30, n).clip(610, 850).round()
    dti = rng.gamma(4, 4.5, n).clip(0, 60).round(2)
    home = rng.choice(HOME_OWNERSHIP_VALUES[:3], n, p=[0.5, 0.4, 0.1])
    logit = (
        -1.4
        - 0.025 * (fico - 695)
        + 0.03 * (dti - 18)
        + 2.0 * (loan_amnt / revenue - 0.2)
        + np.where(home == "RENT", 0.25, 0.0)
    )
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    months = pd.period_range("2014-01", "2017-06", freq="M")
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "issue_d": rng.choice(months, n),
            "revenue": revenue,
            "dti_n": dti,
            "loan_amnt": loan_amnt,
            "fico_n": fico,
            "emp_length": rng.choice(EMP_LENGTH_VALUES, n),
            "purpose": rng.choice(PURPOSE_VALUES, n),
            "home_ownership_n": home,
            "Default": default,
        }
    )


@pytest.fixture(scope="session")
def loans() -> pd.DataFrame:
    return make_loans()


@pytest.fixture(scope="session")
def fitted(loans):
    from credit_scorecard_service.training.fit_scorecard import fit_scorecard

    result = fit_scorecard(loans, version="test")
    result.scorecard.cutoff_score = float(result.scorecard.pd_to_score(0.2))
    return result


@pytest.fixture(scope="session")
def scorecard(fitted):
    return fitted.scorecard


@pytest.fixture(scope="session")
def artifact_path(scorecard, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "champion.json"
    scorecard.save(path)
    return path

import numpy as np
import pandas as pd
import pytest

from credit_scorecard_service.config import TimeSplit
from credit_scorecard_service.features import build_features
from credit_scorecard_service.training.data import time_split
from credit_scorecard_service.training.metrics import (
    calibration_table,
    classification_metrics,
    cost_optimal_threshold,
    psi,
)

rng = np.random.default_rng(1)


def test_perfect_and_random_ranking():
    y = np.array([0, 0, 1, 1])
    perfect = classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["auc"] == 1.0 and perfect["gini"] == 1.0 and perfect["ks"] == 1.0

    y = rng.integers(0, 2, 50_000)
    noise = classification_metrics(y, rng.random(50_000))
    assert abs(noise["gini"]) < 0.03


def test_psi_detects_shift():
    base = rng.normal(0, 1, 50_000)
    assert psi(base, rng.normal(0, 1, 50_000)) < 0.01
    assert psi(base, rng.normal(1, 1, 50_000)) > 0.25


def test_cost_threshold_matches_theory_when_calibrated():
    # With calibrated PDs the optimal threshold is c_good / (c_bad + c_good).
    pd_ = rng.random(200_000)
    y = (rng.random(200_000) < pd_).astype(int)
    res = cost_optimal_threshold(y, pd_, cost_approve_bad=4.0, cost_decline_good=1.0)
    assert res["pd_threshold"] == pytest.approx(0.2, abs=0.02)


def test_calibration_table_covers_all_rows():
    pd_ = rng.random(1000)
    table = calibration_table((rng.random(1000) < pd_).astype(int), pd_)
    assert sum(r["count"] for r in table) == 1000


def test_time_split_is_ordered_and_disjoint(loans):
    split = TimeSplit(
        train_end="2015-12",
        valid_start="2016-01",
        valid_end="2016-06",
        test_start="2016-10",
        test_end="2017-03",
    )
    s = time_split(loans, split)
    assert s.train["issue_d"].max() < s.valid["issue_d"].min()
    assert s.valid["issue_d"].max() < s.test["issue_d"].min()
    assert set(s.train["id"]).isdisjoint(s.test["id"])


def test_build_features_same_for_frame_and_dict(loans):
    head = loans.head(5)
    from_frame = build_features(head)
    from_dict = build_features({k: head[k].tolist() for k in head.columns})
    for name in from_frame:
        assert list(from_frame[name]) == list(from_dict[name])
    np.testing.assert_allclose(from_frame["loan_to_income"], head["loan_amnt"] / head["revenue"])


def test_zero_income_gives_missing_ratio():
    feats = build_features(
        pd.DataFrame(
            {
                "revenue": [0.0],
                "dti_n": [1.0],
                "loan_amnt": [1000.0],
                "fico_n": [700.0],
                "emp_length": ["NI"],
                "purpose": ["car"],
                "home_ownership_n": ["RENT"],
            }
        )
    )
    assert np.isnan(feats["loan_to_income"][0])

import numpy as np
import pytest

from credit_scorecard_service.config import MIN_VALID_AUC, MODELS_DIR
from credit_scorecard_service.features import records_to_columns
from credit_scorecard_service.scorecard import CategoricalBins, NumericBins, Scorecard

GOOD = {
    "revenue": 120000,
    "dti_n": 5.0,
    "loan_amnt": 5000,
    "fico_n": 790,
    "emp_length": "10+ years",
    "purpose": "credit_card",
    "home_ownership_n": "MORTGAGE",
}
RISKY = {
    **GOOD,
    "revenue": 30000,
    "dti_n": 40.0,
    "loan_amnt": 30000,
    "fico_n": 640,
    "home_ownership_n": "RENT",
}


def test_weak_features_are_dropped(fitted):
    # emp_length and purpose carry no signal in the synthetic data.
    assert "emp_length" in fitted.dropped
    assert "fico_n" in fitted.selected
    assert all(card.coefficient < 0 for card in fitted.scorecard.features)


def test_pd_and_score_are_consistent(scorecard, loans):
    pd_ = scorecard.predict_pd(loans)
    assert np.all((pd_ > 0) & (pd_ < 1))
    lo, hi = scorecard.score_range
    scores = scorecard.total_score(loans)
    assert np.all((scores >= lo - 1e-9) & (scores <= hi + 1e-9))
    np.testing.assert_allclose(scorecard.score_to_pd(scores), pd_)
    np.testing.assert_allclose(scorecard.pd_to_score(pd_), scores)


def test_scaling_anchor(scorecard):
    # base_score points correspond to good:bad odds of base_odds; +pdo doubles them.
    pd_base = float(scorecard.score_to_pd(scorecard.base_score))
    pd_double = float(scorecard.score_to_pd(scorecard.base_score + scorecard.pdo))
    assert (1 - pd_base) / pd_base == pytest.approx(scorecard.base_odds)
    assert (1 - pd_double) / pd_double == pytest.approx(2 * scorecard.base_odds)


def test_risky_applicant_scores_lower(scorecard):
    good, risky = scorecard.score_records(records_to_columns([GOOD, RISKY]))
    assert good.score > risky.score
    assert good.probability_of_default < risky.probability_of_default
    assert good.risk_band <= risky.risk_band  # bands are ordered A (best) to E


def test_reason_codes_ranked_by_points_lost(scorecard):
    (result,) = scorecard.score_records(records_to_columns([RISKY]), n_reasons=3)
    lost = [r.points_below_max for r in result.reason_codes]
    assert 0 < len(lost) <= 3
    assert lost == sorted(lost, reverse=True)
    assert all(v > 0 for v in lost)


def test_categorical_bins_map_groups_and_unknowns():
    bins = CategoricalBins(
        categories=[["RENT"], ["OWN", "MORTGAGE"]], points=[-5.0, 7.0], unknown_points=0.0
    )
    points, idx = bins.apply(np.array(["OWN", "RENT", "boat"], dtype=object))
    assert list(idx) == [1, 0, -1]
    assert list(points) == [7.0, -5.0, 0.0]


def test_numeric_bins_are_left_closed():
    bins = NumericBins(splits=[10.0, 20.0], points=[1.0, 2.0, 3.0], missing_points=-1.0, labels=[])
    points, _ = bins.apply(np.array([9.99, 10.0, 20.0, np.nan]))
    assert list(points) == [1.0, 2.0, 3.0, -1.0]


def test_save_load_roundtrip(scorecard, artifact_path, loans):
    loaded = Scorecard.load(artifact_path)
    assert loaded.to_dict() == scorecard.to_dict()
    np.testing.assert_allclose(loaded.predict_pd(loans), scorecard.predict_pd(loans))


def test_risk_bands_descending(scorecard):
    mins = [b.min_score for b in scorecard.risk_bands]
    assert mins == sorted(mins, reverse=True)
    assert scorecard.risk_band(scorecard.score_range[0]) == scorecard.risk_bands[-1].name


@pytest.mark.skipif(not (MODELS_DIR / "champion.json").exists(), reason="no trained champion")
def test_champion_meets_auc_guard():
    champion = Scorecard.load(MODELS_DIR / "champion.json")
    assert champion.metadata["metrics"]["valid"]["auc"] >= MIN_VALID_AUC

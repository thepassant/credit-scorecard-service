from credit_scorecard_service.config import BINNING, TimeSplit
from credit_scorecard_service.training.data import time_split
from credit_scorecard_service.training.sensitivity import sweep


def test_sweep_varies_one_setting_at_a_time(loans):
    split = TimeSplit(
        train_end="2015-12",
        valid_start="2016-01",
        valid_end="2016-06",
        test_start="2016-10",
        test_end="2017-03",
    )
    rows = sweep(time_split(loans, split), sweeps={"max_n_bins": [3, 10], "min_iv": [0.0, 0.05]})

    assert [(r["param"], r["value"]) for r in rows] == [
        ("max_n_bins", 3),
        ("max_n_bins", 10),
        ("min_iv", 0.0),
        ("min_iv", 0.05),
    ]
    assert [r["configured"] for r in rows] == [False, BINNING.max_n_bins == 10, False, False]
    by_value = {(r["param"], r["value"]): r for r in rows}
    assert by_value[("max_n_bins", 3)]["total_bins"] < by_value[("max_n_bins", 10)]["total_bins"]
    # Noise features (emp_length, purpose) pass a zero cutoff but not a 0.05 one.
    assert set(by_value[("min_iv", 0.05)]["features"]) < set(by_value[("min_iv", 0.0)]["features"])
    assert all(0.5 < r["valid_auc"] <= 1.0 for r in rows)

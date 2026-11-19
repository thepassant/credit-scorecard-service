"""Sensitivity check of the scorecard's hand-set binning settings.

The challengers are tuned with grids; the scorecard's settings in `config.BINNING` are
fixed. This varies one setting at a time around its configured value, holding the
others fixed, and reports validation and test AUC so the choice is backed by numbers.
Settings are judged on validation; test is shown for reference only.

uv run cs-sensitivity                     # full data, writes models/sensitivity.json
uv run cs-sensitivity --sample-frac 0.1   # quick run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from credit_scorecard_service.config import BINNING, MODELS_DIR, RANDOM_SEED, BinningConfig
from credit_scorecard_service.features import TARGET
from credit_scorecard_service.training import data
from credit_scorecard_service.training.fit_scorecard import fit_scorecard
from credit_scorecard_service.training.metrics import classification_metrics

log = logging.getLogger("sensitivity")

SWEEPS: dict[str, list[Any]] = {
    "max_n_bins": [5, 10, 20],
    "min_bin_size": [0.01, 0.02, 0.05],
    "min_iv": [0.0, 0.01, 0.02, 0.03],
}


def sweep(
    splits: data.Splits,
    base: BinningConfig = BINNING,
    sweeps: dict[str, list[Any]] = SWEEPS,
) -> list[dict[str, Any]]:
    rows = []
    for param, values in sweeps.items():
        for value in values:
            fit = fit_scorecard(
                splits.train, version="sensitivity", cfg=dataclasses.replace(base, **{param: value})
            )
            row: dict[str, Any] = {
                "param": param,
                "value": value,
                "configured": value == getattr(base, param),
                "features": fit.selected,
                "total_bins": sum(len(card.bins.points) for card in fit.scorecard.features),
            }
            for name in ("valid", "test"):
                part = getattr(splits, name)
                pd_ = fit.scorecard.predict_pd(part)
                row[f"{name}_auc"] = classification_metrics(part[TARGET].to_numpy(), pd_)["auc"]
            rows.append(row)
            log.info(
                "%-12s %-5g valid AUC=%.4f test AUC=%.4f bins=%d features=%d%s",
                param,
                value,
                row["valid_auc"],
                row["test_auc"],
                row["total_bins"],
                len(row["features"]),
                "  (configured)" if row["configured"] else "",
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=None, help="CSV path (default: downloaded)")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "sensitivity.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.data is None:
        data.download()
    df = data.load(args.data)
    if args.sample_frac:
        df = df.sample(frac=args.sample_frac, random_state=RANDOM_SEED).sort_index()

    rows = sweep(data.time_split(df))
    report = {"configured": dataclasses.asdict(BINNING), "sample_frac": args.sample_frac}
    args.output.write_text(json.dumps({**report, "results": rows}, indent=2))
    log.info("Wrote %s", args.output)

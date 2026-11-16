"""Train the scorecard, compare challengers and write a versioned artifact.

uv run cs-train                     # full run
uv run cs-train --sample-frac 0.1   # quick run on a 10% sample
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from credit_scorecard_service.config import (
    DATA_SOURCE,
    DECISION_COSTS,
    MIN_VALID_AUC,
    MODELS_DIR,
    RANDOM_SEED,
    TIME_SPLIT,
)
from credit_scorecard_service.features import NUMERIC_FEATURES, TARGET, build_features
from credit_scorecard_service.scorecard import Scorecard
from credit_scorecard_service.training import data
from credit_scorecard_service.training.challengers import tune_and_evaluate
from credit_scorecard_service.training.fit_scorecard import fit_scorecard
from credit_scorecard_service.training.metrics import (
    calibration_table,
    classification_metrics,
    cost_optimal_threshold,
    psi,
)

log = logging.getLogger("train")

CHAMPION_FILE = "champion.json"


def next_version(models_dir: Path) -> str:
    existing = [
        int(m.group(1))
        for p in models_dir.glob("scorecard-v*.json")
        if (m := re.fullmatch(r"scorecard-v(\d+)\.json", p.name))
    ]
    return f"v{max(existing, default=0) + 1}"


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate_scorecard(scorecard: Scorecard, splits: data.Splits) -> dict[str, Any]:
    pd_by_split = {name: scorecard.predict_pd(part) for name, part in splits.__dict__.items()}
    y_by_split = {name: part[TARGET].to_numpy() for name, part in splits.__dict__.items()}
    scores = {name: scorecard.pd_to_score(p) for name, p in pd_by_split.items()}

    train_feats = build_features(splits.train)
    stability = {
        name: {
            "score": round(psi(scores["train"], scores[name]), 4),
            **{f: round(psi(train_feats[f], build_features(part)[f]), 4) for f in NUMERIC_FEATURES},
        }
        for name, part in (("valid", splits.valid), ("test", splits.test))
    }
    return {
        "metrics": {n: classification_metrics(y_by_split[n], p) for n, p in pd_by_split.items()},
        "calibration_test": calibration_table(y_by_split["test"], pd_by_split["test"]),
        "psi_vs_train": stability,
        "_valid_pd": pd_by_split["valid"],
        "_valid_y": y_by_split["valid"],
    }


def run(
    models_dir: Path,
    data_path: Path | None,
    sample_frac: float | None,
    with_challengers: bool,
    promote: bool,
) -> Path:
    df = data.load(data_path)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=RANDOM_SEED).sort_index()
    splits = data.time_split(df)
    for name, info in splits.summary().items():
        log.info("%-5s %s", name, info)

    models_dir.mkdir(parents=True, exist_ok=True)
    version = next_version(models_dir)

    fit = fit_scorecard(splits.train, version=version)
    scorecard = fit.scorecard
    log.info("Selected features: %s; dropped: %s", fit.selected, fit.dropped)

    evaluation = evaluate_scorecard(scorecard, splits)
    cutoff = cost_optimal_threshold(
        evaluation.pop("_valid_y"),
        evaluation.pop("_valid_pd"),
        DECISION_COSTS.approve_bad,
        DECISION_COSTS.decline_good,
    )
    scorecard.cutoff_score = round(float(scorecard.pd_to_score(cutoff["pd_threshold"])), 2)

    valid_auc = evaluation["metrics"]["valid"]["auc"]
    if valid_auc < MIN_VALID_AUC:
        raise RuntimeError(f"Validation AUC {valid_auc} below guard {MIN_VALID_AUC}")

    comparison = {"woe_scorecard": {k: evaluation["metrics"][k] for k in ("valid", "test")}}
    if with_challengers:
        for res in tune_and_evaluate(splits.train, splits.valid, splits.test):
            comparison[res.name] = {
                "params": res.params,
                "valid": res.valid,
                "test": res.test,
                "fit_seconds": res.fit_seconds,
            }

    data_info = {
        "source": DATA_SOURCE.url,
        "md5": DATA_SOURCE.md5 if data_path is None else None,
        "sample_frac": sample_frac,
        "split": TIME_SPLIT.__dict__,
        "rows": splits.summary(),
    }
    scorecard.metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "random_seed": RANDOM_SEED,
        "data": data_info,
        "metrics": evaluation["metrics"],
        "cutoff": {**cutoff, "costs": DECISION_COSTS.__dict__},
        "score_range": [round(s, 1) for s in scorecard.score_range],
    }
    metrics_report = {
        "version": version,
        **scorecard.metadata,
        "information_values": fit.information_values,
        "selected_features": fit.selected,
        "dropped_features": fit.dropped,
        "calibration_test": evaluation["calibration_test"],
        "psi_vs_train": evaluation["psi_vs_train"],
        "model_comparison": comparison,
    }

    artifact = models_dir / f"scorecard-{version}.json"
    scorecard.save(artifact)
    (models_dir / f"metrics-{version}.json").write_text(json.dumps(metrics_report, indent=2))
    log.info("Wrote %s", artifact)
    _log_comparison(comparison)

    if promote:
        shutil.copyfile(artifact, models_dir / CHAMPION_FILE)
        log.info("Promoted %s to %s", version, models_dir / CHAMPION_FILE)
    return artifact


def _log_comparison(comparison: dict[str, Any]) -> None:
    log.info("%-16s %8s %8s %8s %8s", "model", "val_auc", "val_gini", "val_ks", "test_auc")
    for name, res in comparison.items():
        v, t = res["valid"], res["test"]
        log.info("%-16s %8.4f %8.4f %8.4f %8.4f", name, v["auc"], v["gini"], v["ks"], t["auc"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--data", type=Path, default=None, help="CSV path (default: downloaded)")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--skip-challengers", action="store_true")
    parser.add_argument("--no-promote", action="store_true", help="do not update champion.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    np.random.seed(RANDOM_SEED)
    if args.data is None:
        data.download()
    run(
        models_dir=args.models_dir,
        data_path=args.data,
        sample_frac=args.sample_frac,
        with_challengers=not args.skip_challengers,
        promote=not args.no_promote,
    )


if __name__ == "__main__":
    main()

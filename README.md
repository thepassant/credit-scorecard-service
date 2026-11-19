# Credit Scorecard Service

Predicts a Lending Club borrower's **probability of default (PD)** from information
available at application time, converts it into a **credit score**, **risk band**,
**approve/decline decision** and **reason codes**, and serves it with FastAPI.

ML Zoomcamp midterm project.

## Problem

A lender must decide whether to grant a loan using only what it knows when the
application arrives. The champion model is a classic **WoE logistic-regression
scorecard**: regulators and credit officers can read every point it assigns. It is
benchmarked against tree-based challengers (decision tree, random forest, XGBoost)
to measure what that transparency costs in accuracy.

**Why the AUC is about 0.66.** This is a limit of the data, not of the model. Only
seven fields are known at application time: income, DTI, loan amount, FICO, employment
length, purpose and home ownership. The dataset has no credit-bureau detail
(utilisation, delinquencies, inquiries), and the lender's own interest rate and grade
are removed because they would leak its risk view. Tuned XGBoost, given the same
fields, reaches only 0.67 AUC, so a more flexible model does not find much more signal.
The goal is the best *transparent* model on these inputs, plus an honest measurement of
what that transparency costs (see [Results](#results)).

## Data

[Lending Club loan dataset for granting models](https://zenodo.org/records/11295916)
(Zenodo, **CC BY 4.0**): 1,347,681 loans issued 2007–2018 that ended either *fully
paid* (`Default = 0`) or *charged off* (`Default = 1`). Loans still running are not
included. Fields set by the lender's own risk view (interest rate, grade) are already
removed.

```bash
uv run cs-download   # fetches data/raw/LC_loans_granting_model_dataset.csv and verifies its MD5
```

| Used as features | Excluded, and why |
|---|---|
| `revenue` (annual income), `dti_n`, `loan_amnt`, `fico_n`, `emp_length`, `purpose`, `home_ownership_n`, plus derived `loan_to_income` | `addr_state`, `zip_code`: geographic proxies for protected attributes · `title`, `desc`: free text · `experience_c`: constant (25 exceptions in 1.3M rows) |

**Out-of-time split** on issue month (see `config.py`):

| Split | Months | Purpose |
|---|---|---|
| train | 2007-06 … 2015-12 | fit binning and models |
| valid | 2016-01 … 2016-06 | tune challengers, choose the cutoff |
| *gap* | 2016-07 … 2016-09 | loans need time to reach an outcome |
| test | 2016-10 … 2017-03 | final, one-time comparison |

Vintages after 2017 are left out of evaluation. The dataset keeps only *resolved*
loans, so recent cohorts over-represent early defaults and early payoffs. Their
default rate drops to 2% for 2018Q4.

## Method

1. **Binning:** each feature is binned with [optbinning](https://gnpalencia.org/optbinning/)
   (monotonic trend for numeric features, at most 10 bins). Each bin's weight
   of evidence is `WoE = ln(%good / %bad)`, and its information value (IV) is computed.
2. **Selection:** features with IV < 0.005 are dropped. The logistic regression is refit
   until every coefficient has the expected negative sign, since a flipped sign means
   collinearity. These settings are fixed by hand, not tuned, and are checked with a
   sensitivity run instead (see *Scorecard settings* under Results).
3. **Scaling:** `score = offset + factor · ln(odds_good)`, with 600 points at 50:1 good:bad
   odds and 20 points to double the odds (PDO). Each feature's bin gets
   `−factor · β · WoE` points, so the score is the sum of a base and per-feature points.
4. **Export:** the fitted model is written as a plain **JSON points table**
   (`models/scorecard-vN.json`). Training checks that the table reproduces the logistic
   regression's PD to 1e-6. The service applies the table with numpy and needs no ML
   libraries.
5. **Risk bands** A–E are fixed PD limits (6%, 12%, 20%, 30%) translated into scores.
6. **Cutoff:** the score that minimises expected cost on the validation months, where
   approving a bad loan costs 5× declining a good one.
7. **Reason codes:** for each applicant, the three features with the most points lost
   relative to that feature's best bin.

## Results

Full output: `models/metrics-vN.json` (metrics, IV per feature, calibration, PSI, challenger params).
Plots, the scorecard points table and the SHAP comparison are in
[`notebooks/01_eda_and_model_comparison.ipynb`](notebooks/01_eda_and_model_comparison.ipynb).

<!-- RESULTS:START -->
Champion `v2`: 829k training loans, 175k validation, 104k test.

| Model | Valid AUC | Valid Gini | Valid KS | Test AUC | Test Gini |
|---|---|---|---|---|---|
| **WoE scorecard v2 (champion)** | 0.662 | 0.323 | 0.230 | 0.661 | 0.321 |
| WoE scorecard v1 (IV cutoff 0.02) | 0.658 | 0.317 | 0.227 | 0.658 | 0.317 |
| One-hot logistic regression | 0.663 | 0.327 | 0.233 | 0.663 | 0.326 |
| Decision tree (depth 8) | 0.654 | 0.309 | 0.223 | 0.650 | 0.300 |
| Random forest (depth 12) | 0.664 | 0.329 | 0.234 | 0.665 | 0.330 |
| XGBoost (depth 5) | 0.669 | 0.339 | 0.240 | 0.668 | 0.337 |

**v1 → v2.** v2 lowers the IV cutoff from 0.02 to 0.005, so `purpose` and `emp_length`
are back in the scorecard (see *Scorecard settings* below). Validation AUC rises from
0.658 to 0.662, and test AUC from 0.658 to 0.661. Both versions stay in `models/`.

**Trade-off.** XGBoost gains under 1 AUC point (1.5 Gini points) over the scorecard.
With only seven application-time fields, the remaining signal is mostly the shape of
FICO, DTI and loan-to-income, which binning already captures. The scorecard keeps
every decision explainable as points per bin, gives monotonic, defensible reason
codes and needs no ML runtime, so it stays the champion. SHAP shows the two models
agree on the main drivers (rank correlation 0.93, 84% overlap in approvals) and now
give loan purpose similar weight. XGBoost's remaining edge comes from the FICO tail
above 760 and from interactions.

**Information value.** loan_to_income 0.126 · fico_n 0.122 · dti_n 0.075 ·
loan_amnt 0.036 · revenue 0.032 · home_ownership_n 0.021 · purpose 0.018 ·
emp_length 0.007. All eight pass the 0.005 cutoff.

**Scorecard settings.** The challengers are tuned with grids. The scorecard's settings
are fixed in `config.py`, so `make sensitivity` refits it with one setting changed at a
time and records the result in `models/sensitivity.json`. Settings are judged on
validation; test is for reference. Configured values are in bold.

| Setting | Value | Valid AUC | Test AUC | Bins | Features |
|---|---|---|---|---|---|
| max bins per feature | 5 | 0.6597 | 0.6586 | 37 | 8 |
| | **10** | **0.6616** | **0.6607** | 65 | 8 |
| | 20 | 0.6619 | 0.6608 | 77 | 8 |
| min loans per bin | 1%, **2%**, 5% | 0.6616 | 0.6607 | 65 | 8 |
| IV cutoff | 0, **0.005** | **0.6616** | **0.6607** | 65 | 8 |
| | 0.01 | 0.6598 | 0.6578 | 55 | 7 |
| | 0.02 (v1) | 0.6584 | 0.6584 | 51 | 6 |
| | 0.03 | 0.6531 | 0.6512 | 48 | 5 |

- **Max bins:** going from 5 to 10 bins gains 0.002 AUC. Going to 20 adds only 0.0003
  for 12 more bins to read and explain, so 10 is kept.
- **Min loans per bin:** has no effect. optbinning first splits each feature into at most
  20 pre-bins of at least 5% of loans each, so a minimum of 5% or less never binds.
- **IV cutoff:** the one setting that matters. The conventional 0.02 drops `purpose`
  (IV 0.018) and `emp_length` (0.007), which costs 0.003 AUC on both validation and
  test. Both get coefficients with the expected sign when kept. v2 uses 0.005, which
  keeps them and still screens out features with no signal.

**Stability.** PSI of the score against training is 0.0005 (valid) and 0.011 (test).
All features are below 0.06, so the inputs are stable.

**Calibration.** Ranking is monotonic across PD deciles, but the test months default
more than predicted: 23.5% observed against 17.6% predicted on average. Default rates
rose for the 2016 vintages, and the dataset keeps only resolved loans. The cutoff is
set on observed validation outcomes, so decisions account for this, but quoted PDs
should be recalibrated. That step is planned for the retraining loop.

**Decision.** At a 5:1 cost ratio the cost-optimal cutoff (PD < 13.0%, score ≥ 542)
approves 29% of validation applicants with an 11.3% bad rate among approvals,
against 22% overall.
<!-- RESULTS:END -->

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and, for the container, Docker.

```bash
make install      # uv sync --extra train + pre-commit hooks
make train        # download data, train, compare challengers, promote champion (~5 min)
make test
make serve        # http://localhost:8000/docs
make sensitivity  # refit the scorecard with each binning setting varied (~1 min)
make notebook     # re-execute the EDA / model-comparison notebook in place
```

Container (uses the committed `models/champion.json`):

```bash
docker build -t credit-scorecard .
docker run --rm -p 8000:8000 credit-scorecard
# or: docker compose up --build
```

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | One applicant → PD, score, risk band, decision, reason codes |
| `POST` | `/predict/batch` | `{"applicants": [...]}` (max 1000) → one prediction each |
| `POST` | `/predict/batch/csv` | Multipart CSV upload with a header row |
| `GET` | `/health` | Liveness/readiness, used by the Docker `HEALTHCHECK` |
| `GET` | `/model-info` | Version, training data range, metrics, bands, cutoff |

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d @examples/applicant.json
curl -s localhost:8000/predict/batch/csv -F file=@examples/applicants.csv
```

```json
{
  "probability_of_default": 0.194,
  "score": 528,
  "risk_band": "C",
  "decision": "decline",
  "reason_codes": [
    {"feature": "fico_n", "description": "FICO score", "value": 702.0, "points": 3.7, "points_below_max": 26.7},
    {"feature": "loan_to_income", "description": "Loan amount relative to income", "value": 0.231, "points": 0.7, "points_below_max": 9.8},
    {"feature": "loan_amnt", "description": "Loan amount", "value": 15000.0, "points": -1.5, "points_below_max": 8.9}
  ],
  "model_version": "v2",
  "request_id": "ca67c5fc2ea94b94aec34db918823eeb"
}
```

**Input rules**
- Missing fields, wrong types and out-of-range values return **422** with the offending field.
- **Unknown categories are rejected** (422), not mapped to a default. A typo in
  `purpose` should fail loudly rather than receive a neutral score. Allowed values are
  listed in `features.py` and in the OpenAPI schema at `/docs`.
- Unexpected fields are rejected, including `zip_code` and `addr_state`, so proxies for
  protected attributes cannot reach the model by accident. CSV uploads ignore extra columns.
- `dti_n = 999` is Lending Club's "not available" value and is accepted.

**Prediction log:** every scored applicant is written to SQLite (`CS_PREDICTION_LOG`)
with its request ID, inputs, outputs and model version, ready to be joined with real
outcomes for monitoring and retraining. Every request also emits one JSON log line
with its `X-Request-ID`.

## Configuration

| Variable | Default | |
|---|---|---|
| `CS_MODEL_PATH` | `models/champion.json` | scorecard artifact the service loads |
| `CS_PREDICTION_LOG` | `logs/predictions.db` | SQLite prediction log |
| `CS_MAX_BATCH_SIZE` | `1000` | larger batches get 413 |
| `CS_DATA_DIR`, `CS_MODELS_DIR` | `data/`, `models/` | training paths |

Modelling constants (split dates, scaling, IV threshold, costs, risk bands) live in
`src/credit_scorecard_service/config.py`.

## Model versioning

`cs-train` writes `models/scorecard-vN.json` and `models/metrics-vN.json` (git commit,
data checksum, date ranges, seed, metrics), then copies the new scorecard to
`models/champion.json` (use `--no-promote` to skip that step). The artifacts are a few
KB of readable JSON, so they are **committed to git**: model changes show up as
reviewable diffs in pull requests, and rolling back is a revert.

## Project layout

```
src/credit_scorecard_service/
  config.py            paths, split dates, scaling, costs, bands (env-overridable where it matters)
  features.py          feature definitions shared by training and serving
  scorecard.py         serving-time points table: score, PD, band, reason codes (numpy only)
  schemas.py           Pydantic request/response models
  api.py               FastAPI app
  prediction_log.py    SQLite prediction log
  training/            needs the `train` extra; never imported by the service
    data.py            download + checksum, load, out-of-time split
    fit_scorecard.py   optbinning WoE + logistic regression → points table
    challengers.py     one-hot logreg, decision tree, random forest, XGBoost
    metrics.py         AUC/Gini/KS, calibration, cost-optimal cutoff, PSI
    train.py           `cs-train` entry point
    sensitivity.py     `cs-sensitivity`: scorecard binning settings varied one at a time
notebooks/             EDA, scorecard walkthrough, challenger + SHAP comparison
tests/                 synthetic-data fixtures: no download needed in CI
models/                versioned scorecard + metrics JSON, champion.json
```

## Development

- **Dependencies:** `uv` with a lockfile. Runtime dependencies (FastAPI, numpy,
  pydantic) are kept apart from the `train` extra (pandas, scikit-learn, optbinning,
  XGBoost), which keeps the image small.
- **Quality gates:** ruff lint and format run in pre-commit and again in CI, next to
  pytest and a Docker build with a container smoke test (`.github/workflows/ci.yml`).
- **Tests:** API contract and 422 cases, scorecard maths (scaling anchor, PD↔score
  inverse, bin edges, reason-code ordering), metrics, time split, and an AUC
  regression guard on the committed champion.

## Roadmap

- [x] EDA and model-comparison notebook (incl. SHAP for XGBoost)
- [x] Sensitivity check of the scorecard's binning settings (`make sensitivity`)
- [x] Scorecard v2: re-admit `purpose` and `emp_length` (IV cutoff 0.005, +0.003 AUC)
- [ ] Scorecard v3: finer top FICO bin (gap found by SHAP)
- [ ] PD recalibration (intercept / Platt) on the most recent matured months
- [ ] MLflow tracking
- [ ] Retraining loop: monthly ingest → validation → PSI/AUC check → retrain → promote on margin
- [ ] Push image to GHCR; deploy to AWS App Runner with OIDC
- [ ] `/metrics` endpoint with Prometheus

## Data license

CC BY 4.0, Sanz-Guerrero & Arroyo, *Lending Club loan dataset for granting models*,
Zenodo, 2024 ([record 11295916](https://zenodo.org/records/11295916)).

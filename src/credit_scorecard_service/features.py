"""Feature definitions shared by training and serving.

Every transformation from raw application fields to model inputs lives here,
so the training pipeline and the API cannot drift apart.

Deliberately excluded:
- addr_state, zip_code: geographic proxies for protected attributes.
- title, desc: free text.
- experience_c: constant in the dataset (25 non-default values in 1.3M rows).
- interest rate, grade, sub-grade: the lender's own risk view (not in the dataset).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

TARGET = "Default"
DATE_COLUMN = "issue_d"
ID_COLUMN = "id"

# Raw application fields the API accepts.
RAW_NUMERIC = ("revenue", "dti_n", "loan_amnt", "fico_n")
RAW_CATEGORICAL = ("emp_length", "purpose", "home_ownership_n")
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

# Model inputs: raw fields plus derived ones.
NUMERIC_FEATURES = (*RAW_NUMERIC, "loan_to_income")
CATEGORICAL_FEATURES = RAW_CATEGORICAL
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EMP_LENGTH_VALUES = (
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
    "NI",  # no information
)
PURPOSE_VALUES = (
    "car",
    "credit_card",
    "debt_consolidation",
    "educational",
    "home_improvement",
    "house",
    "major_purchase",
    "medical",
    "moving",
    "other",
    "renewable_energy",
    "small_business",
    "vacation",
    "wedding",
)
HOME_OWNERSHIP_VALUES = ("MORTGAGE", "RENT", "OWN", "OTHER")

CATEGORY_VALUES: dict[str, tuple[str, ...]] = {
    "emp_length": EMP_LENGTH_VALUES,
    "purpose": PURPOSE_VALUES,
    "home_ownership_n": HOME_OWNERSHIP_VALUES,
}

# Human-readable names used in reason codes.
FEATURE_LABELS = {
    "revenue": "Annual income",
    "dti_n": "Debt-to-income ratio",
    "loan_amnt": "Loan amount",
    "fico_n": "FICO score",
    "loan_to_income": "Loan amount relative to income",
    "emp_length": "Employment length",
    "purpose": "Loan purpose",
    "home_ownership_n": "Home ownership",
}


def build_features(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Turn raw columns into model features.

    `raw` maps column name to a sequence of values: a pandas DataFrame and a
    dict of lists both work, so batch training and single API requests go
    through the same code.
    """
    out: dict[str, np.ndarray] = {name: np.asarray(raw[name], dtype=float) for name in RAW_NUMERIC}
    for name in RAW_CATEGORICAL:
        out[name] = np.asarray(raw[name], dtype=object)

    revenue = out["revenue"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["loan_to_income"] = np.where(revenue > 0, out["loan_amnt"] / revenue, np.nan)
    return out


def records_to_columns(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Convert a list of row dicts to the column layout `build_features` expects."""
    return {name: [r[name] for r in records] for name in RAW_FEATURES}

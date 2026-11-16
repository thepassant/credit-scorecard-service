"""Request and response models for the prediction API.

Unknown categories are rejected with a 422 rather than silently mapped, so a
client sending a typo learns about it instead of getting a neutral score.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from credit_scorecard_service.features import (
    EMP_LENGTH_VALUES,
    HOME_OWNERSHIP_VALUES,
    PURPOSE_VALUES,
)

EmpLength = Literal[EMP_LENGTH_VALUES]  # type: ignore[valid-type]
Purpose = Literal[PURPOSE_VALUES]  # type: ignore[valid-type]
HomeOwnership = Literal[HOME_OWNERSHIP_VALUES]  # type: ignore[valid-type]


class Applicant(BaseModel):
    """Information known at application time."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "revenue": 65000,
                    "dti_n": 18.5,
                    "loan_amnt": 15000,
                    "fico_n": 702,
                    "emp_length": "5 years",
                    "purpose": "debt_consolidation",
                    "home_ownership_n": "RENT",
                }
            ]
        },
    )

    revenue: float = Field(gt=0, le=20_000_000, description="Self-declared annual income (USD)")
    dti_n: float = Field(ge=0, le=999, description="Debt-to-income ratio, % (999 = not available)")
    loan_amnt: float = Field(gt=0, le=100_000, description="Requested amount (USD)")
    fico_n: float = Field(ge=300, le=850, description="FICO score at application")
    emp_length: EmpLength
    purpose: Purpose
    home_ownership_n: HomeOwnership


class ReasonCodeOut(BaseModel):
    feature: str
    description: str
    value: Any
    points: float
    points_below_max: float = Field(description="Points lost versus the best bin for this feature")


class Prediction(BaseModel):
    probability_of_default: float
    score: int
    risk_band: str
    decision: Literal["approve", "decline"]
    reason_codes: list[ReasonCodeOut]
    model_version: str


class PredictionResponse(Prediction):
    request_id: str


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1)


class BatchResponse(BaseModel):
    request_id: str
    model_version: str
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str


class ModelInfo(BaseModel):
    version: str
    created_at: str | None
    git_commit: str | None
    features: list[str]
    score_range: list[float]
    cutoff_score: float
    risk_bands: list[dict[str, Any]]
    training_data: dict[str, Any]
    metrics: dict[str, Any]

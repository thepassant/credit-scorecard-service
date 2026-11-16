"""FastAPI prediction service.

The scorecard is loaded once at startup and never modified here: retraining
writes a new artifact and a redeploy picks it up.

    uv run uvicorn credit_scorecard_service.api:app --port 8000
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from credit_scorecard_service.config import N_REASON_CODES, ServiceSettings
from credit_scorecard_service.features import records_to_columns
from credit_scorecard_service.prediction_log import PredictionLog
from credit_scorecard_service.schemas import (
    Applicant,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionResponse,
)
from credit_scorecard_service.scorecard import Scorecard

log = logging.getLogger("credit_scorecard_service.api")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    log.handlers = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False


def create_app(settings: ServiceSettings | None = None) -> FastAPI:
    settings = settings or ServiceSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _configure_logging()
        app.state.scorecard = Scorecard.load(settings.model_path)
        app.state.prediction_log = PredictionLog(settings.prediction_log_path)
        log.info(
            "model loaded",
            extra={"extra_fields": {"model_version": app.state.scorecard.version}},
        )
        yield

    app = FastAPI(
        title="Credit Scorecard Service",
        description="Probability of default, credit score, risk band and reason codes.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            },
        )
        return response

    def score(request: Request, applicants: Sequence[Applicant]) -> list[Prediction]:
        if len(applicants) > settings.max_batch_size:
            raise HTTPException(413, f"Batch size exceeds limit of {settings.max_batch_size}")
        scorecard: Scorecard = request.app.state.scorecard
        records = [a.model_dump() for a in applicants]
        results = scorecard.score_records(records_to_columns(records), n_reasons=N_REASON_CODES)
        request.app.state.prediction_log.write(
            request.state.request_id, scorecard.version, records, results
        )
        return [Prediction(**asdict(r), model_version=scorecard.version) for r in results]

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", model_version=request.app.state.scorecard.version)

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request) -> ModelInfo:
        sc: Scorecard = request.app.state.scorecard
        meta = sc.metadata
        return ModelInfo(
            version=sc.version,
            created_at=meta.get("created_at"),
            git_commit=meta.get("git_commit"),
            features=[c.name for c in sc.features],
            score_range=[round(s, 1) for s in sc.score_range],
            cutoff_score=sc.cutoff_score,
            risk_bands=[asdict(b) for b in sc.risk_bands],
            training_data=meta.get("data", {}),
            metrics=meta.get("metrics", {}),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: Request, applicant: Applicant) -> PredictionResponse:
        (prediction,) = score(request, [applicant])
        return PredictionResponse(**prediction.model_dump(), request_id=request.state.request_id)

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: Request, body: BatchRequest) -> BatchResponse:
        return BatchResponse(
            request_id=request.state.request_id,
            model_version=request.app.state.scorecard.version,
            predictions=score(request, body.applicants),
        )

    @app.post("/predict/batch/csv", response_model=BatchResponse)
    def predict_batch_csv(
        request: Request, file: Annotated[UploadFile, File(description="CSV with a header row")]
    ) -> BatchResponse:
        text = file.file.read().decode("utf-8-sig")
        applicants = _parse_csv(text)
        if not applicants:
            raise HTTPException(422, "CSV contains no rows")
        return BatchResponse(
            request_id=request.state.request_id,
            model_version=request.app.state.scorecard.version,
            predictions=score(request, applicants),
        )

    return app


def _parse_csv(text: str) -> list[Applicant]:
    applicants, errors = [], []
    fields = set(Applicant.model_fields)
    for line_no, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        try:
            applicants.append(Applicant(**{k: v for k, v in row.items() if k in fields}))
        except ValidationError as exc:
            errors.extend(_located(exc.errors(), ("csv", f"line {line_no}")))
    if errors:
        raise RequestValidationError(errors)
    return applicants


def _located(errors: list[Any], prefix: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{**e, "loc": (*prefix, *e["loc"])} for e in errors]


app = create_app()

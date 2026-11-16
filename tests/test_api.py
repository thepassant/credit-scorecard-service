import sqlite3

import pytest
from fastapi.testclient import TestClient

from credit_scorecard_service.api import create_app
from credit_scorecard_service.config import ServiceSettings

APPLICANT = {
    "revenue": 65000,
    "dti_n": 18.5,
    "loan_amnt": 15000,
    "fico_n": 702,
    "emp_length": "5 years",
    "purpose": "debt_consolidation",
    "home_ownership_n": "RENT",
}


@pytest.fixture
def settings(artifact_path, tmp_path):
    return ServiceSettings(
        model_path=artifact_path,
        prediction_log_path=tmp_path / "predictions.db",
        max_batch_size=10,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test"}


def test_model_info(client, scorecard):
    body = client.get("/model-info").json()
    assert body["version"] == "test"
    assert body["features"] == [c.name for c in scorecard.features]


def test_predict(client, scorecard):
    r = client.post("/predict", json=APPLICANT, headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["probability_of_default"] < 1
    lo, hi = scorecard.score_range
    assert lo - 1 <= body["score"] <= hi + 1
    assert body["risk_band"] in {"A", "B", "C", "D", "E"}
    assert body["decision"] in {"approve", "decline"}
    assert len(body["reason_codes"]) <= 3
    assert body["model_version"] == "test"
    assert body["request_id"] == "abc123"
    assert r.headers["X-Request-ID"] == "abc123"


@pytest.mark.parametrize(
    "change",
    [
        {"fico_n": None},  # missing / null
        {"fico_n": "high"},  # wrong type
        {"fico_n": 900},  # out of range
        {"revenue": 0},  # must be positive
        {"purpose": "yacht"},  # unknown category
        {"zip_code": "100xx"},  # unexpected field (proxy for a protected attribute)
    ],
)
def test_predict_rejects_bad_input(client, change):
    payload = {**APPLICANT, **change}
    payload = {k: v for k, v in payload.items() if v is not None}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_batch(client):
    r = client.post(
        "/predict/batch", json={"applicants": [APPLICANT, {**APPLICANT, "fico_n": 800}]}
    )
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert preds[1]["score"] >= preds[0]["score"]


def test_batch_limits(client):
    assert client.post("/predict/batch", json={"applicants": []}).status_code == 422
    too_many = {"applicants": [APPLICANT] * 11}
    assert client.post("/predict/batch", json=too_many).status_code == 413


def test_batch_csv(client):
    header = ",".join(APPLICANT) + ",zip_code"
    row = ",".join(str(v) for v in APPLICANT.values()) + ",100xx"
    csv_text = f"{header}\n{row}\n{row}\n"
    r = client.post("/predict/batch/csv", files={"file": ("loans.csv", csv_text, "text/csv")})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_batch_csv_reports_bad_line(client):
    header = ",".join(APPLICANT)
    bad = ",".join(str(v) for v in {**APPLICANT, "fico_n": "abc"}.values())
    r = client.post(
        "/predict/batch/csv", files={"file": ("x.csv", f"{header}\n{bad}\n", "text/csv")}
    )
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][:2] == ["csv", "line 2"]


def test_predictions_are_logged(client, settings):
    client.post("/predict", json=APPLICANT, headers={"X-Request-ID": "log-me"})
    client.post("/predict/batch", json={"applicants": [APPLICANT] * 3})
    with sqlite3.connect(settings.prediction_log_path) as conn:
        rows = conn.execute("SELECT request_id, row_index FROM predictions").fetchall()
    assert ("log-me", 0) in rows
    assert len(rows) == 4

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="train the model first: make train")

from api.main import app  # noqa: E402

client = TestClient(app)
GOOD = dict(LIMIT_BAL=200000, SEX=2, EDUCATION=1, MARRIAGE=2, AGE=34,
            PAY_0=0, PAY_2=0, PAY_3=0, PAY_4=0, PAY_5=0, PAY_6=0,
            BILL_AMT1=20000, BILL_AMT2=19000, BILL_AMT3=18000, BILL_AMT4=17000, BILL_AMT5=16000, BILL_AMT6=15000,
            PAY_AMT1=20000, PAY_AMT2=19000, PAY_AMT3=18000, PAY_AMT4=17000, PAY_AMT5=16000, PAY_AMT6=15000)
RISKY = {**GOOD, "LIMIT_BAL": 20000, "PAY_0": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2,
         **{f"BILL_AMT{i}": 19500 for i in range(1, 7)}, **{f"PAY_AMT{i}": 0 for i in range(1, 7)}}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict_shape_and_reasons():
    r = client.post("/predict", json=GOOD).json()
    assert 0 <= r["default_probability"] <= 1 and len(r["top_reasons"]) == 3


def test_risky_scores_higher_than_good():
    g = client.post("/predict", json=GOOD).json()["default_probability"]
    b = client.post("/predict", json=RISKY).json()
    assert b["default_probability"] > g and b["decision"] == "review"


def test_sex_does_not_change_score():
    a = client.post("/predict", json={**GOOD, "SEX": 1}).json()["default_probability"]
    b = client.post("/predict", json={**GOOD, "SEX": 2}).json()["default_probability"]
    assert a == b


def test_validation_rejects_bad_input():
    assert client.post("/predict", json={**GOOD, "AGE": 10}).status_code == 422


def test_batch():
    r = client.post("/predict/batch", json={"applicants": [GOOD, RISKY]})
    assert r.status_code == 200 and len(r.json()) == 2

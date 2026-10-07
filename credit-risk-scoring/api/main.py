"""FastAPI service. Run: uvicorn api.main:app --reload  ->  http://localhost:8000/docs"""
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, conlist

from credit_risk import __version__
from credit_risk.predict import score

app = FastAPI(title="Credit Risk Scoring API", version=__version__,
              description="Predicts the probability a credit card customer defaults next month, with per-decision reasons.")


class Applicant(BaseModel):
    LIMIT_BAL: float = Field(..., gt=0, description="Credit limit (NT$)", examples=[50000])
    SEX: Literal[1, 2] = Field(1, description="Collected for audit only; NOT used by the model")
    EDUCATION: int = Field(..., ge=0, le=6, examples=[2])
    MARRIAGE: int = Field(..., ge=0, le=3, examples=[1])
    AGE: int = Field(..., ge=18, le=100, examples=[35])
    PAY_0: int = Field(..., ge=-2, le=9, description="Repayment status last month (months delayed)")
    PAY_2: int = Field(..., ge=-2, le=9)
    PAY_3: int = Field(..., ge=-2, le=9)
    PAY_4: int = Field(..., ge=-2, le=9)
    PAY_5: int = Field(..., ge=-2, le=9)
    PAY_6: int = Field(..., ge=-2, le=9)
    BILL_AMT1: float; BILL_AMT2: float; BILL_AMT3: float
    BILL_AMT4: float; BILL_AMT5: float; BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0); PAY_AMT2: float = Field(..., ge=0); PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0); PAY_AMT5: float = Field(..., ge=0); PAY_AMT6: float = Field(..., ge=0)


class Reason(BaseModel):
    feature: str; value: float; impact: float; direction: str


class Prediction(BaseModel):
    default_probability: float
    risk_band: Literal["low", "medium", "high"]
    decision: Literal["approve", "review"]
    model_version: str
    top_reasons: list[Reason] | None = None


class BatchRequest(BaseModel):
    applicants: conlist(Applicant, min_length=1, max_length=1000)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    try:
        from credit_risk.predict import _artifacts
        _artifacts()
        return {"status": "ok", "model_version": __version__}
    except Exception as e:  # pragma: no cover
        raise HTTPException(503, f"model not loaded: {e}")


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant, explain: bool = True):
    return score([applicant.model_dump()], explain=explain)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest, explain: bool = False):
    t = time.time()
    res = score([a.model_dump() for a in req.applicants], explain=explain)
    app.state.last_batch_ms = round((time.time() - t) * 1000, 1)
    return res

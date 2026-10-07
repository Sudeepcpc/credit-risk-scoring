"""Inference: load artefacts once, score raw applicant records, explain each decision."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from .config import MODEL_DIR
from .data import clean
from .features import add_features


@lru_cache(maxsize=1)
def _artifacts():
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    booster = joblib.load(MODEL_DIR / "explainer_booster.joblib")
    return bundle, shap.TreeExplainer(booster)


def risk_band(p: float) -> str:
    return "low" if p < 0.15 else "medium" if p < 0.35 else "high"


def score(records: list[dict], explain: bool = True, top_k: int = 3) -> list[dict]:
    bundle, explainer = _artifacts()
    raw = pd.DataFrame(records)
    raw = clean(raw)
    X = add_features(raw)[bundle["features"]]
    probs = bundle["model"].predict_proba(X)[:, 1]
    contribs = None
    if explain:
        sv = explainer.shap_values(X)
        contribs = sv[1] if isinstance(sv, list) else sv
    out = []
    for i, p in enumerate(probs):
        item = {
            "default_probability": round(float(p), 4),
            "risk_band": risk_band(float(p)),
            "decision": "review" if p >= bundle["threshold"] else "approve",
            "model_version": bundle["version"],
        }
        if explain:
            idx = np.argsort(-np.abs(contribs[i]))[:top_k]
            item["top_reasons"] = [
                {"feature": bundle["features"][j], "value": round(float(X.iloc[i, j]), 4),
                 "impact": round(float(contribs[i][j]), 4),
                 "direction": "raises risk" if contribs[i][j] > 0 else "lowers risk"}
                for j in idx
            ]
        out.append(item)
    return out

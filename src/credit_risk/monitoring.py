"""Population Stability Index (PSI) for detecting data drift in production."""
import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI < 0.1 stable, 0.1-0.25 watch, > 0.25 retrain."""
    rows = []
    for col in reference.columns:
        v = psi(reference[col], current[col])
        status = "stable" if v < 0.1 else "watch" if v < 0.25 else "retrain"
        rows.append({"feature": col, "psi": round(v, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)

import numpy as np
import pandas as pd

from credit_risk.data import clean
from credit_risk.features import add_features, ENGINEERED
from credit_risk.monitoring import psi
from credit_risk.evaluate import ks_statistic, best_threshold


def make_row(**kw):
    base = dict(LIMIT_BAL=50000, SEX=2, EDUCATION=2, MARRIAGE=1, AGE=30,
                **{c: 0 for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
                **{f"BILL_AMT{i}": 10000 for i in range(1, 7)},
                **{f"PAY_AMT{i}": 2000 for i in range(1, 7)})
    base.update(kw)
    return pd.DataFrame([base])


def test_clean_maps_undocumented_codes():
    df = clean(make_row(EDUCATION=6, MARRIAGE=0, PAY_0=-2))
    assert df.loc[0, "EDUCATION"] == 4 and df.loc[0, "MARRIAGE"] == 3 and df.loc[0, "PAY_0"] == 0


def test_engineered_features_present_and_finite():
    f = add_features(clean(make_row()))
    assert set(ENGINEERED) <= set(f.columns)
    assert np.isfinite(f[ENGINEERED].values).all()
    assert abs(f.loc[0, "utilization_last"] - 0.2) < 1e-9


def test_delays_counted():
    f = add_features(clean(make_row(PAY_0=2, PAY_2=1)))
    assert f.loc[0, "months_delayed"] == 2 and f.loc[0, "max_delay"] == 2


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_ks_and_threshold():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert ks_statistic(y, p) == 1.0
    assert 0.3 <= best_threshold(y, p) <= 0.8

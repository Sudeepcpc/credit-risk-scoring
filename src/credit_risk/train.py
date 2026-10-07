"""Train, compare, calibrate and persist the credit risk model.

Run: python -m credit_risk.train
"""
import json
import time

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .config import FIG_DIR, MODEL_DIR, RANDOM_STATE, REPORT_DIR, TEST_SIZE, TARGET
from .data import load
from .evaluate import best_threshold, report
from .features import split_xy
from .monitoring import drift_report


def fairness_audit(df_test, y, p, threshold) -> dict:
    """Approval-rate and recall by SEX (not a model input). 1 = male, 2 = female."""
    out = {}
    pred = p >= threshold
    for code, name in [(1, "male"), (2, "female")]:
        m = (df_test["SEX"] == code).values
        out[name] = {
            "n": int(m.sum()),
            "flag_rate": round(float(pred[m].mean()), 4),
            "default_rate": round(float(y[m].mean()), 4),
            "recall": round(float(pred[m & (y == 1)].mean()), 4),
        }
    out["flag_rate_ratio_female_to_male"] = round(out["female"]["flag_rate"] / out["male"]["flag_rate"], 3)
    return out


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE)
    X_tr, y_tr = split_xy(train_df)
    X_te, y_te = split_xy(test_df)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Baseline: scaled logistic regression (the traditional scorecard approach)
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
    lr_cv = cross_val_score(logreg, X_tr, y_tr, cv=cv, scoring="roc_auc")

    # 2. Gradient boosting with randomized hyper-parameter search
    search = RandomizedSearchCV(
        lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
        {
            "n_estimators": [200, 400, 600, 800],
            "learning_rate": [0.01, 0.02, 0.03, 0.05],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [20, 50, 100],
            "subsample": [0.7, 0.8, 1.0], "subsample_freq": [1],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_lambda": [0, 1, 5],
        },
        n_iter=20, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=1,
    )
    search.fit(X_tr, y_tr)
    best_lgb = search.best_estimator_

    # 3. Calibrate probabilities so a score of 0.3 really means ~30% default risk
    calibrated = CalibratedClassifierCV(lgb.LGBMClassifier(**best_lgb.get_params()), method="isotonic", cv=cv)
    calibrated.fit(X_tr, y_tr)

    # Threshold chosen on out-of-fold train predictions (never on test) to minimise business cost
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(lgb.LGBMClassifier(**best_lgb.get_params()), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    threshold = best_threshold(y_tr, oof)

    logreg.fit(X_tr, y_tr)
    p_lr = logreg.predict_proba(X_te)[:, 1]
    p_lgb = calibrated.predict_proba(X_te)[:, 1]
    oof_lr = cross_val_predict(logreg, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    th_lr = best_threshold(y_tr, oof_lr)

    metrics = {
        "model_version": __version__,
        "dataset": {"rows": len(df), "train": len(train_df), "test": len(test_df), "default_rate": round(float(df[TARGET].mean()), 4)},
        "logistic_regression": {"cv_roc_auc_mean": round(lr_cv.mean(), 4), "test": report(y_te, p_lr, th_lr)},
        "lightgbm_calibrated": {"cv_roc_auc_best": round(search.best_score_, 4), "best_params": search.best_params_, "test": report(y_te, p_lgb, threshold)},
    }
    metrics["fairness_audit"] = fairness_audit(test_df, y_te.values, p_lgb, threshold)
    # Baseline cost: approve everyone
    metrics["business_impact"] = {
        "cost_approve_everyone": round(float(y_te.mean() * 5.0), 4),
        "cost_with_model": metrics["lightgbm_calibrated"]["test"]["expected_cost_per_customer"],
    }
    bi = metrics["business_impact"]
    bi["cost_reduction_pct"] = round(100 * (1 - bi["cost_with_model"] / bi["cost_approve_everyone"]), 1)

    # Explainability: SHAP on the uncalibrated booster (same params, full train)
    booster = lgb.LGBMClassifier(**best_lgb.get_params()).fit(X_tr, y_tr)
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X_te.sample(2000, random_state=RANDOM_STATE))
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=X_te.columns).sort_values(ascending=False)
    metrics["top_features_shap"] = {k: round(float(v), 4) for k, v in imp.head(10).items()}

    # Drift check: train vs test should be stable
    drift = drift_report(X_tr, X_te)
    drift.to_csv(REPORT_DIR / "drift_train_vs_test.csv", index=False)
    metrics["max_psi_train_vs_test"] = float(drift["psi"].max())

    # Figures
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, p in [("Logistic regression", p_lr), ("LightGBM (calibrated)", p_lgb)]:
        fpr, tpr, _ = roc_curve(y_te, p); ax[0].plot(fpr, tpr, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_title("ROC curve (test)"); ax[0].legend(); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    frac, mean_p = calibration_curve(y_te, p_lgb, n_bins=10)
    ax[1].plot(mean_p, frac, "o-"); ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].set_title("Calibration (LightGBM)"); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Observed")
    imp.head(10)[::-1].plot.barh(ax=ax[2], color="#2a6f97"); ax[2].set_title("Top drivers (mean |SHAP|)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "model_report.png", dpi=130); plt.close()

    # Persist artefacts
    joblib.dump({"model": calibrated, "threshold": threshold, "features": X_tr.columns.tolist(), "version": __version__}, MODEL_DIR / "model.joblib")
    joblib.dump(booster, MODEL_DIR / "explainer_booster.joblib")
    X_tr.sample(3000, random_state=RANDOM_STATE).to_csv(MODEL_DIR / "reference_sample.csv", index=False)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

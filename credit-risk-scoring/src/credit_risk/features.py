"""Domain features a credit analyst would compute. Pure function: safe to reuse at serving time."""
import numpy as np
import pandas as pd

from .data import PAY_STATUS, BILL, PAID
from .config import PROTECTED, TARGET

ENGINEERED = [
    "utilization_last", "utilization_avg", "pay_ratio_avg", "pay_ratio_last",
    "months_delayed", "max_delay", "delay_trend", "bill_growth", "avg_payment", "zero_payment_months",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    limit = out["LIMIT_BAL"].clip(lower=1)
    bills = out[BILL].clip(lower=0)
    paid = out[PAID]

    out["utilization_last"] = bills["BILL_AMT1"] / limit
    out["utilization_avg"] = bills.mean(axis=1) / limit
    # payment made in month t settles the bill of month t+1
    ratios = np.column_stack([
        paid[f"PAY_AMT{i}"] / (bills[f"BILL_AMT{i+1}"] + 1) for i in range(1, 6)
    ]).clip(0, 5)
    out["pay_ratio_avg"] = ratios.mean(axis=1)
    out["pay_ratio_last"] = ratios[:, 0]
    out["months_delayed"] = (out[PAY_STATUS] > 0).sum(axis=1)
    out["max_delay"] = out[PAY_STATUS].max(axis=1)
    out["delay_trend"] = out[["PAY_0", "PAY_2", "PAY_3"]].mean(axis=1) - out[["PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
    out["bill_growth"] = (bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit
    out["avg_payment"] = paid.mean(axis=1)
    out["zero_payment_months"] = (paid == 0).sum(axis=1)
    return out


def split_xy(df: pd.DataFrame):
    X = add_features(df.drop(columns=[TARGET]))
    X = X.drop(columns=[c for c in PROTECTED if c in X.columns])
    return X, df[TARGET].astype(int)


def feature_names(df: pd.DataFrame) -> list[str]:
    return split_xy(df)[0].columns.tolist()

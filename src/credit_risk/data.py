"""Load and clean the UCI 'Default of Credit Card Clients' dataset (30,000 Taiwanese card holders)."""
import io
import zipfile
import urllib.request

import pandas as pd

from .config import RAW_PATH, DATA_URL, TARGET

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAID = [f"PAY_AMT{i}" for i in range(1, 7)]


def download(dest=RAW_PATH) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = urllib.request.urlopen(DATA_URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.endswith(".xls")][0]
        dest.write_bytes(z.read(name))


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    download(path)
    df = pd.read_excel(path, header=1)
    return df.rename(columns={"default payment next month": TARGET}).drop(columns=["ID"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix undocumented category codes found during EDA."""
    df = df.copy()
    # EDUCATION 0, 5, 6 are undocumented -> 4 ("other")
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3, 4]), 4)
    # MARRIAGE 0 is undocumented -> 3 ("other")
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].isin([1, 2, 3]), 3)
    # PAY_* -2 (no consumption) and -1 (paid in full) both mean "not delayed" -> 0
    for c in PAY_STATUS:
        df[c] = df[c].clip(lower=0)
    return df


def load() -> pd.DataFrame:
    return clean(load_raw())

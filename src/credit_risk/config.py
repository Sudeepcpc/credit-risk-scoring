from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "default of credit card clients.xls"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

TARGET = "default"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Business costs (in NT$ units, illustrative): approving a customer who defaults
# costs far more than declining a good customer (lost margin).
COST_FALSE_NEGATIVE = 5.0   # missed defaulter
COST_FALSE_POSITIVE = 1.0   # wrongly flagged good customer

# Protected attribute excluded from training on purpose (fair-lending practice).
# Kept only for the fairness audit.
PROTECTED = ["SEX"]

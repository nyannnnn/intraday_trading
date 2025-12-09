# config.py

from pathlib import Path

# =====================
# Universe & bar config
# =====================

UNIVERSE = [
    "SOFI",
    "CNCK",
    "QBTS",
    "RR",
    "QTTB",
    "FIG",
    "SYM",
    "AHMA",
]

BAR_INTERVAL_MIN = 5             # 5-minute bars
FUTURE_HORIZON_BARS = 6          # 30 minutes ahead (6 * 5min)
LABEL_UP_THRESHOLD = 0.006      # +0.6% threshold for "up" label

# =====================
# Trading / risk config
# =====================

P_UP_ENTRY_THRESHOLD = 0.00      # ML probability threshold to enter long
MAX_CONCURRENT_POSITIONS = 3    # max open trades at once
MAX_BARS_IN_TRADE = 15           
DAILY_LOSS_STOP_FRACTION = 0.03 # 3% daily loss stop
COOLDOWN_BARS_AFTER_STOP = 6
ATR_WINDOW = 20           # Lookback for volatility
ATR_STOP_MULT = 2.0       # Stop Loss = 2x ATR
ATR_TP_MULT = 3.0         # Take Profit = 3x ATR (1.5 R-Multiple)
RISK_PER_TRADE_FRACTION = 0.005

# Flat per-order fee (e.g. 1.50 per transaction)
FEE_PER_ORDER = 1.50

# =====================
# Paths
# =====================

PROJECT_ROOT = Path(__file__).resolve().parent  # folder with config.py
DATA_DIR = PROJECT_ROOT / "quant" / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# =====================
# ML feature config
# =====================

LABEL_COLUMN = "label_up"

# Features produced in quant_model.build_features_for_symbol(...)
# plus symbol_id (added inside ml_model.add_symbol_id_feature)
FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "vol_60",
    "vwap",
    "dev_vwap",
    "dev_vwap_z",
    "vol_z",
    "time_minutes",
    "time_of_day_norm",
]

# Time-based split between train / validation for ML
TEST_SPLIT_DATE = "2024-07-01"
TRAIN_VAL_SPLIT_DATE = "2025-10-10"

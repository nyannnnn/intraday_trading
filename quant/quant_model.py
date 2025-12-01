# quant_model.py

import numpy as np
import pandas as pd
from datetime import time

from config import FUTURE_HORIZON_BARS, LABEL_UP_THRESHOLD


def _intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute intraday VWAP, resetting each day.

    VWAP_t = sum_{i<=t} (price_i * volume_i) / sum_{i<=t} volume_i
    """
    idx_date = df.index.normalize()  # group by date (same day => same midnight)

    px_vol = (df["close"] * df["volume"]).groupby(idx_date).cumsum()
    cum_vol = df["volume"].groupby(idx_date).cumsum().replace(0, np.nan)

    vwap = px_vol / cum_vol
    vwap.name = "vwap"
    return vwap


def build_features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for ONE symbol's intraday OHLCV.
    Assumes df index is DatetimeIndex and columns: ['open','high','low','close','volume'].
    """

    df = df.sort_index().copy()

    # --- NEW: force OHLCV columns to be numeric in case CSV gave strings ---
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1-bar, 3-bar, 6-bar returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)

    # Rolling 60-bar volume (approx 5h on 5-min bars)
    df["vol_60"] = df["volume"].rolling(window=60, min_periods=10).mean()

    # Intraday VWAP and deviation vs VWAP
    df["vwap"] = _intraday_vwap(df)
    df["dev_vwap"] = df["close"] - df["vwap"]

    # Z-score of deviation vs VWAP over 60 bars
    df["dev_vwap_z"] = (
        df["dev_vwap"]
        .rolling(window=60, min_periods=10)
        .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False)
    )

    # Z-score of volume vs its 60-bar history
    df["vol_z"] = (
        df["vol_60"]
        .rolling(window=60, min_periods=10)
        .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False)
    )

    # Time-of-day features
    dt_index = df.index

    # minutes since midnight (easier to compute than "since market open")
    minutes = dt_index.hour * 60 + dt_index.minute
    df["time_minutes"] = minutes

    # Normalize into [0, 1] assuming a 6.5h trading day (390 minutes)
    df["time_of_day_norm"] = df["time_minutes"] / 390.0

    return df


def add_future_return_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a feature DataFrame (with 'close'), append:

      future_ret : forward return over FUTURE_HORIZON_BARS
      label_up   : 1 if future_ret >= LABEL_UP_THRESHOLD else 0
    """
    out = df.copy()

    # Forward return over N bars
    future_price = out["close"].shift(-FUTURE_HORIZON_BARS)
    out["future_ret"] = (future_price / out["close"]) - 1.0

    # Binary label for classification
    out["label_up"] = (out["future_ret"] >= LABEL_UP_THRESHOLD).astype(int)

    # It's fine to keep the tail rows with NaN future_ret;
    # the ML pipeline will drop NaNs before training.
    return out


def build_panel_features_and_labels(panel_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    panel_ohlcv: MultiIndex DataFrame with index (symbol, datetime)
                 and columns ['open','high','low','close','volume'].

    Returns a MultiIndex DataFrame with the same index structure plus
    all features and the label columns.
    """
    if not isinstance(panel_ohlcv.index, pd.MultiIndex):
        raise ValueError("panel_ohlcv must have a MultiIndex (symbol, datetime).")

    out_list = []

    # group by symbol (level 0)
    for sym, df_sym in panel_ohlcv.groupby(level=0):
        # remove the symbol level; index becomes datetime only
        df_sym = df_sym.droplevel(0)

        # build features and labels for this symbol
        df_feat = build_features_for_symbol(df_sym)
        df_lab = add_future_return_and_label(df_feat)
        df_lab["symbol"] = sym

        out_list.append(df_lab)

    panel = pd.concat(out_list, axis=0)

    # move 'symbol' into the index so we get (symbol, datetime)
    panel.set_index("symbol", append=True, inplace=True)
    # current index levels are (datetime, symbol); reorder:
    panel = panel.reorder_levels(["symbol", panel.index.names[0]])
    panel = panel.sort_index()

    return panel

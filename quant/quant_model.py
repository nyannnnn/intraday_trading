import numpy as np
import pandas as pd
from datetime import time

from config import FUTURE_HORIZON_BARS, LABEL_UP_THRESHOLD


def _intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute intraday VWAP per day based on close * volume."""
    idx_date = df.index.normalize()

    px_vol = (df["close"] * df["volume"]).groupby(idx_date).cumsum()
    cum_vol = df["volume"].groupby(idx_date).cumsum().replace(0, np.nan)

    vwap = px_vol / cum_vol
    vwap.name = "vwap"
    return vwap


def build_features_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Generate intraday return, volume, VWAP, and time-of-day features for a single symbol."""
    df = df.sort_index().copy()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ret_1"] = df["close"].pct_change(1, fill_method=None)
    df["ret_3"] = df["close"].pct_change(3, fill_method=None)
    df["ret_6"] = df["close"].pct_change(6, fill_method=None)

    df["vol_60"] = df["volume"].rolling(window=60, min_periods=10).mean()

    df["vwap"] = _intraday_vwap(df)
    df["dev_vwap"] = df["close"] - df["vwap"]

    df["dev_vwap_z"] = (
        df["dev_vwap"]
        .rolling(window=60, min_periods=10)
        .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False)
    )

    df["vol_z"] = (
        df["vol_60"]
        .rolling(window=60, min_periods=10)
        .apply(lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False)
    )

    dt_index = df.index
    minutes = dt_index.hour * 60 + dt_index.minute
    df["time_minutes"] = minutes
    df["time_of_day_norm"] = df["time_minutes"] / 390.0

    return df


def add_future_return_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Append forward returns and binary up/down labels for a given feature DataFrame."""
    out = df.copy()

    future_price = out["close"].shift(-FUTURE_HORIZON_BARS)
    out["future_ret"] = (future_price / out["close"]) - 1.0

    out["label_up"] = (out["future_ret"] >= LABEL_UP_THRESHOLD).astype(int)
    return out


def build_panel_features_and_labels(panel_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Construct feature/label panel for a MultiIndex (symbol, datetime) OHLCV input."""
    if not isinstance(panel_ohlcv.index, pd.MultiIndex):
        raise ValueError("panel_ohlcv must have a MultiIndex (symbol, datetime).")

    out_list = []

    for sym, df_sym in panel_ohlcv.groupby(level=0):
        df_sym = df_sym.droplevel(0)

        df_feat = build_features_for_symbol(df_sym)
        df_lab = add_future_return_and_label(df_feat)
        df_lab["symbol"] = sym

        out_list.append(df_lab)

    panel = pd.concat(out_list, axis=0)

    panel.set_index("symbol", append=True, inplace=True)
    panel = panel.reorder_levels(["symbol", panel.index.names[0]])
    panel = panel.sort_index()

    return panel

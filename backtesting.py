# backtesting.py

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from config import (
    UNIVERSE,
    DATA_DIR,
    MODEL_DIR,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    P_UP_ENTRY_THRESHOLD,
    MAX_CONCURRENT_POSITIONS,
    RISK_PER_TRADE_FRACTION,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_BARS_IN_TRADE,
    FEE_PER_ORDER,
    BAR_INTERVAL_MIN,
)
from quant.quant_model import build_panel_features_and_labels


# =========================
# Backtest configuration
# =========================

@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0

    p_up_entry_threshold: float = P_UP_ENTRY_THRESHOLD
    max_concurrent_positions: int = MAX_CONCURRENT_POSITIONS
    risk_per_trade_fraction: float = RISK_PER_TRADE_FRACTION
    stop_loss_pct: float = STOP_LOSS_PCT
    take_profit_pct: float = TAKE_PROFIT_PCT
    max_bars_in_trade: int = MAX_BARS_IN_TRADE
    fee_per_order: float = FEE_PER_ORDER

    feature_cols: Tuple[str, ...] = tuple(FEATURE_COLUMNS)
    label_col: str = LABEL_COLUMN


# =========================
# Data loading
# =========================

def load_panel_ohlcv(data_root: Path, universe) -> pd.DataFrame:
    """
    Load per-symbol OHLCV CSVs from quant/data into a panel DataFrame.

    Expected CSV columns: datetime, open, high, low, close, volume
    """
    dfs = []
    for symbol in universe:
        csv_path = data_root / f"{symbol}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing CSV for {symbol}: {csv_path}")
            continue

        df = pd.read_csv(
            csv_path,
            parse_dates=["datetime"],
        )

        # Normalize columns
        col_map = {c.lower(): c for c in df.columns}
        df = df.rename(
            columns={
                col_map.get("open", "open"): "open",
                col_map.get("high", "high"): "high",
                col_map.get("low", "low"): "low",
                col_map.get("close", "close"): "close",
                col_map.get("volume", "volume"): "volume",
                col_map.get("datetime", "datetime"): "datetime",
            }
        )

        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df["symbol"] = symbol
        df = df.set_index(["symbol", "datetime"]).sort_index()
        dfs.append(df)

    if not dfs:
        raise ValueError("No data frames loaded. Check DATA_DIR and UNIVERSE.")

    panel = pd.concat(dfs).sort_index()
    return panel


# =========================
# Model loading
# =========================

def load_classifier(model_dir: Path) -> object:
    """
    Load the trained classifier.
    We assume train_ml.py saved ONLY the classifier object as 'intraday_gbm.joblib'.
    """
    model_path = model_dir / "intraday_gbm.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    clf = joblib.load(model_path)
    return clf


# =========================
# Backtest core
# =========================

def run_backtest(panel_with_features: pd.DataFrame,
                 clf,
                 cfg: BacktestConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Simple multi-symbol long-only backtest driven by ML probability of 'up'.

    We iterate over (symbol, datetime) rows in time order, keep track of
    open positions per symbol, and update equity over time.
    """

    # Ensure sorted by time then symbol
    panel = panel_with_features.sort_index(level=["datetime", "symbol"])

    # Dictionaries to track positions and last prices
    positions: Dict[str, Dict[str, float]] = {}
    last_price: Dict[str, float] = {}

    cash = cfg.initial_capital

    equity_list = []
    dt_list = []

    # For classification diagnostics (optional)
    y_true = []
    y_score = []

    # Iterate over each bar (per-symbol)
    for (symbol, dt), row in panel.iterrows():
        price = float(row["close"])
        last_price[symbol] = price

        # ---------- 1) Manage existing position in this symbol ----------
        if symbol in positions:
            pos = positions[symbol]
            entry_price = pos["entry_price"]
            bars_in_trade = pos["bars_in_trade"]

            pnl_pct = (price - entry_price) / entry_price

            exit_reason = None
            if pnl_pct <= -cfg.stop_loss_pct:
                exit_reason = "stop"
            elif pnl_pct >= cfg.take_profit_pct:
                exit_reason = "take_profit"
            elif bars_in_trade >= cfg.max_bars_in_trade:
                exit_reason = "time"

            if exit_reason is not None:
                # Close the position
                shares = pos["shares"]
                cash += shares * price
                cash -= cfg.fee_per_order
                del positions[symbol]
            else:
                pos["bars_in_trade"] = bars_in_trade + 1

        # ---------- 2) Entry logic for this symbol ----------
        flat_for_symbol = symbol not in positions
        num_positions = len(positions)

        proba_up = np.nan

        # Only try to open new position if we are flat in this symbol and under max positions
        if flat_for_symbol and num_positions < cfg.max_concurrent_positions:
            feat = row[list(cfg.feature_cols)]

            # Skip if any feature is NaN
            if not feat.isna().any():
                # Use a DataFrame so sklearn sees feature names (avoids warnings)
                X_row = row[FEATURE_COLUMNS].to_frame().T   # 1-row DataFrame
                proba_up = clf.predict_proba(X_row)[0, 1]


                if proba_up >= cfg.p_up_entry_threshold:
                    # Position sizing
                    risk_cap = cash * cfg.risk_per_trade_fraction
                    shares = int(risk_cap / price)
                    if shares >= 1:
                        cash -= shares * price
                        cash -= cfg.fee_per_order
                        positions[symbol] = {
                            "shares": shares,
                            "entry_price": price,
                            "bars_in_trade": 0,
                        }

        # ---------- 3) Collect classification diagnostics ----------
        if cfg.label_col in row and not np.isnan(row[cfg.label_col]) and not np.isnan(proba_up):
            y_true.append(int(row[cfg.label_col]))
            y_score.append(proba_up)

        # ---------- 4) Compute current equity ----------
        equity = cash
        for sym, pos in positions.items():
            px = last_price.get(sym, pos["entry_price"])
            equity += pos["shares"] * px

        equity_list.append(equity)
        dt_list.append(dt)

    # ---------- Liquidate remaining positions at last known price ----------
    for sym, pos in positions.items():
        px = last_price.get(sym, pos["entry_price"])
        cash += pos["shares"] * px
        cash -= cfg.fee_per_order

    if equity_list:
        equity_list[-1] = cash  # final equity after liquidation

    equity_series = pd.Series(equity_list, index=pd.to_datetime(dt_list))
    equity_df = pd.DataFrame({"equity": equity_series})

    # ---------- Compute performance metrics ----------
    metrics = compute_performance_metrics(equity_series, cfg.initial_capital)

    # Optional classification metrics
    if len(y_true) > 0 and len(set(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score

            metrics["val_auc_backtest"] = roc_auc_score(y_true, y_score)
            y_pred = [1 if p >= cfg.p_up_entry_threshold else 0 for p in y_score]
            metrics["val_accuracy_backtest"] = accuracy_score(y_true, y_pred)
        except Exception:
            pass

    return equity_df, metrics


def compute_performance_metrics(equity: pd.Series,
                                initial_capital: float) -> Dict[str, float]:
    # 1. Clean equity series
    equity = equity.astype(float)

    # Drop any NaNs at the start/end (e.g., preallocated values that were never filled)
    equity = equity.dropna()

    if equity.empty:
        raise ValueError("Equity curve is empty after dropping NaNs; check backtest logic.")

    start_equity = float(equity.iloc[0])
    final_equity = float(equity.iloc[-1])

    # 2. Returns (explicit fill_method to avoid FutureWarning)
    returns = equity.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 3. Compute metrics
    total_return = final_equity / start_equity - 1.0

    # If your index is datetime-like, you can annualize:
    if hasattr(equity.index, "to_series"):
        # Assume intraday bars but at least multiple days;
        # use calendar days between first and last point
        n_days = (equity.index[-1] - equity.index[0]).days
        if n_days > 0:
            years = n_days / 252.0  # or 365.0, depending on your convention
            CAGR = (final_equity / start_equity) ** (1.0 / years) - 1.0
        else:
            CAGR = np.nan
    else:
        CAGR = np.nan
    minutes_per_year = 252 * 390
    bars_per_year = minutes_per_year / BAR_INTERVAL_MIN
    if returns.std() > 0:
        sharpe = math.sqrt(bars_per_year) * returns.mean() / returns.std()
    else:
        sharpe = float("nan")

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = float(drawdown.min())

    metrics = {
        "total_return": total_return,
        "CAGR": CAGR,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(equity.iloc[-1]),
        "start_equity": float(equity.iloc[0]),
    }
    return metrics


# =========================
# Script entry point
# =========================

def main():
    print("=== Step 1: Load raw OHLCV data ===")
    panel_ohlcv = load_panel_ohlcv(DATA_DIR, UNIVERSE)
    print("Panel OHLCV shape:", panel_ohlcv.shape)
    print("Panel OHLCV index example:", panel_ohlcv.index[:5])

    print("\n=== Step 2: Build features & labels ===")
    panel_with_features = build_panel_features_and_labels(panel_ohlcv)
    print("Panel with features+labels shape:", panel_with_features.shape)
    print("Panel columns (first 20):", list(panel_with_features.columns)[:20])

    print("\n=== Step 3: Load trained classifier ===")
    clf = load_classifier(MODEL_DIR)
    print("Loaded classifier:", type(clf))

    print("\n=== Step 4: Run backtest ===")
    bt_cfg = BacktestConfig()
    equity_df, metrics = run_backtest(panel_with_features, clf, bt_cfg)

    print("\nBacktest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Optionally save equity curve
    out_path = Path("equity_curve.csv")
    equity_df.to_csv(out_path)
    print(f"\nSaved equity curve to {out_path.resolve()}")


if __name__ == "__main__":
    main()

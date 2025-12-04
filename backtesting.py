from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score

from config import (
    UNIVERSE,
    DATA_DIR,
    MODEL_DIR,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    P_UP_ENTRY_THRESHOLD,
    FUTURE_HORIZON_BARS,
    BAR_INTERVAL_MIN,
    RISK_PER_TRADE_FRACTION,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
)

from quant.quant_model import build_panel_features_and_labels


def load_panel_ohlcv(data_dir: Path, universe: List[str]) -> pd.DataFrame:
    """Load per-symbol intraday OHLCV CSVs into a (symbol, datetime) panel."""
    print("=== Step 1: Load raw OHLCV data ===")

    frames = []
    for sym in universe:
        csv_path = data_dir / f"{sym}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing CSV for {sym}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]

        if "datetime" not in df.columns:
            df.rename(columns={df.columns[0]: "datetime"}, inplace=True)

        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        cols = []
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                cols.append(c)
            else:
                raise ValueError(f"Column '{c}' missing in {csv_path}")

        df = df[["datetime"] + cols]

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close"])

        df["symbol"] = sym
        df = df.set_index(["symbol", "datetime"])
        frames.append(df)

    if not frames:
        raise ValueError("No data frames loaded. Check DATA_DIR and UNIVERSE.")

    panel = pd.concat(frames).sort_index()

    print(f"Panel OHLCV shape: {panel.shape}")
    print("Panel OHLCV index example:", panel.index[:5])
    return panel


def load_latest_classifier(model_dir: Path):
    """Load the trained sklearn classifier saved by train_ml.py."""
    model_dir = Path(model_dir)
    model_path = model_dir / "intraday_gbm.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    clf = joblib.load(model_path)
    return clf


def build_buy_and_hold_equity(
    panel_ohlcv: pd.DataFrame, start_equity: float
) -> pd.Series:
    """Construct an equal-weight buy-and-hold benchmark across the universe."""
    close_df = (
        panel_ohlcv["close"]
        .unstack("symbol")
        .sort_index()
        .ffill()
        .dropna(how="all")
    )

    close_df = close_df.apply(pd.to_numeric, errors="coerce")
    close_df = close_df.dropna(how="all")

    first_prices = close_df.iloc[0]
    first_prices = pd.to_numeric(first_prices, errors="coerce")

    valid_syms = first_prices.dropna().index
    if len(valid_syms) == 0:
        raise ValueError("No valid symbols for buy & hold benchmark.")

    alloc_per_symbol = start_equity / len(valid_syms)
    shares = alloc_per_symbol / first_prices[valid_syms]

    close_df = close_df[valid_syms]

    equity_bh = (close_df * shares).sum(axis=1)
    equity_bh.name = "buy_hold_equity"
    return equity_bh


@dataclass
class BacktestResult:
    """Container for strategy equity, benchmark equity, trades, and summary metrics."""
    equity: pd.Series
    buy_hold_equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def run_backtest(
    panel_with_features: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    clf,
    start_equity: float = 100_000.0,
) -> BacktestResult:
    """Run an offline horizon-based backtest of the ML signal with simple risk caps."""
    print("=== Step 4: Run backtest ===")

    df = panel_with_features.copy()

    needed = FEATURE_COLUMNS + ["future_ret", LABEL_COLUMN]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in panel_with_features.")

    df = df.dropna(subset=FEATURE_COLUMNS + ["future_ret"])
    df = df.sort_index()

    X = df[FEATURE_COLUMNS]
    proba_up = clf.predict_proba(X)[:, 1]
    df["proba_up"] = proba_up

    y_true = df[LABEL_COLUMN].astype(int)
    try:
        val_auc = roc_auc_score(y_true, proba_up)
    except ValueError:
        val_auc = np.nan
    y_pred = (proba_up >= P_UP_ENTRY_THRESHOLD).astype(int)
    val_acc = accuracy_score(y_true, y_pred)

    trades = df[df["proba_up"] >= P_UP_ENTRY_THRESHOLD].copy()
    if trades.empty:
        print("No trades triggered by the ML model with current threshold.")
        all_dts = (
            panel_ohlcv.index.get_level_values("datetime")
            .sort_values()
            .unique()
        )
        equity = pd.Series(start_equity, index=all_dts, name="equity")
        bh_equity = build_buy_and_hold_equity(panel_ohlcv, start_equity)

        metrics = {
            "total_return": 0.0,
            "CAGR": 0.0,
            "sharpe": np.nan,
            "max_drawdown": 0.0,
            "final_equity": start_equity,
            "start_equity": start_equity,
            "n_trades": 0,
            "win_rate": np.nan,
            "avg_gain": np.nan,
            "avg_loss": np.nan,
            "val_auc_backtest": val_auc,
            "val_accuracy_backtest": val_acc,
        }

        return BacktestResult(
            equity=equity,
            buy_hold_equity=bh_equity,
            trades=trades,
            metrics=metrics,
        )

    trades = trades.reset_index()
    trades = trades.rename(columns={"datetime": "entry_dt"})

    horiz_minutes = FUTURE_HORIZON_BARS * BAR_INTERVAL_MIN
    trades["exit_dt"] = trades["entry_dt"] + pd.to_timedelta(horiz_minutes, unit="m")

    trades["future_ret"] = trades["future_ret"].astype(float)
    trades["trade_ret_raw"] = trades["future_ret"]
    trades["trade_ret"] = trades["future_ret"].clip(
        lower=-STOP_LOSS_PCT, upper=TAKE_PROFIT_PCT
    )

    notional_per_trade = start_equity * RISK_PER_TRADE_FRACTION
    trades["pnl_dollars"] = notional_per_trade * trades["trade_ret"]

    trades["entry_price"] = trades["close"]
    trades["exit_price_est"] = trades["entry_price"] * (1.0 + trades["trade_ret"])

    pnl_by_exit = trades.groupby("exit_dt")["pnl_dollars"].sum().sort_index()
    equity = pnl_by_exit.cumsum() + start_equity
    equity.name = "strategy_equity"

    all_dts = (
        panel_ohlcv.index.get_level_values("datetime")
        .sort_values()
        .unique()
    )
    equity = equity.reindex(all_dts, method="ffill").fillna(start_equity)

    buy_hold_equity = build_buy_and_hold_equity(panel_ohlcv, start_equity)

    returns = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    if len(equity.index) > 1:
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25
        if n_years > 0:
            cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0
        else:
            cagr = np.nan
    else:
        cagr = np.nan

    if returns.std() > 0:
        sharpe = np.sqrt(252.0) * returns.mean() / returns.std()
    else:
        sharpe = np.nan

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()

    n_trades = len(trades)
    win_mask = trades["trade_ret"] > 0
    lose_mask = trades["trade_ret"] < 0

    win_rate = win_mask.mean() if n_trades > 0 else np.nan
    avg_gain = trades.loc[win_mask, "trade_ret"].mean() if win_mask.any() else np.nan
    avg_loss = trades.loc[lose_mask, "trade_ret"].mean() if lose_mask.any() else np.nan

    metrics = {
        "total_return": float(total_return),
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "max_drawdown": float(max_dd),
        "final_equity": float(equity.iloc[-1]),
        "start_equity": float(equity.iloc[0]),
        "n_trades": int(n_trades),
        "win_rate": float(win_rate) if pd.notna(win_rate) else np.nan,
        "avg_gain": float(avg_gain) if pd.notna(avg_gain) else np.nan,
        "avg_loss": float(avg_loss) if pd.notna(avg_loss) else np.nan,
        "val_auc_backtest": float(val_auc) if pd.notna(val_auc) else np.nan,
        "val_accuracy_backtest": float(val_acc) if pd.notna(val_acc) else np.nan,
    }

    return BacktestResult(
        equity=equity,
        buy_hold_equity=buy_hold_equity,
        trades=trades,
        metrics=metrics,
    )


def analyze_proba_buckets(panel_with_features: pd.DataFrame, clf) -> None:
    """
    Slice future returns by proba_up buckets to see where the real edge is.
    Buckets: [0–0.5, 0.5–0.6, 0.6–0.7, 0.7–0.8, 0.8–0.9, 0.9–1.0].
    """
    df = panel_with_features.copy()

    needed = FEATURE_COLUMNS + ["future_ret", LABEL_COLUMN]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in panel_with_features.")

    df = df.dropna(subset=FEATURE_COLUMNS + ["future_ret"])
    df = df.sort_index()

    X = df[FEATURE_COLUMNS]
    proba_up = clf.predict_proba(X)[:, 1]
    df["proba_up"] = proba_up

    # raw and clipped returns, same clipping as backtest
    df["future_ret"] = df["future_ret"].astype(float)
    df["trade_ret"] = df["future_ret"].clip(
        lower=-STOP_LOSS_PCT, upper=TAKE_PROFIT_PCT
    )

    # define buckets
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001]
    labels = [
        "0.0–0.5",
        "0.5–0.6",
        "0.6–0.7",
        "0.7–0.8",
        "0.8–0.9",
        "0.9–1.0",
    ]
    df["proba_bucket"] = pd.cut(df["proba_up"], bins=bins, labels=labels, right=False)

    grouped = df.groupby("proba_bucket")

    stats = grouped.agg(
        n=("future_ret", "size"),
        proba_mean=("proba_up", "mean"),
        future_ret_mean=("future_ret", "mean"),
        trade_ret_mean=("trade_ret", "mean"),
        win_rate=("future_ret", lambda x: (x > 0).mean() if len(x) > 0 else np.nan),
    ).reset_index()

    print("\n=== Proba_up bucket analysis ===")
    print(
        "Each row shows how the next-horizon return behaves for different "
        "ranges of model probability."
    )
    with pd.option_context("display.float_format", "{:0.5f}".format):
        print(stats.to_string(index=False))


def main():
    """Run offline backtest, print diagnostics, and persist curves/logs to disk."""
    project_root = Path(__file__).resolve().parent

    panel_ohlcv = load_panel_ohlcv(DATA_DIR, UNIVERSE)

    print("\n=== Step 2: Build features & labels ===")
    panel_with_features = build_panel_features_and_labels(panel_ohlcv)
    print(f"Panel with features+labels shape: {panel_with_features.shape}")
    print(
        "Panel columns (first 20):",
        list(panel_with_features.columns[:20]),
    )

    print("\n=== Step 3: Load trained classifier ===")
    clf = load_latest_classifier(MODEL_DIR)

    # New: bucket analysis for proba_up -> future_ret
    analyze_proba_buckets(panel_with_features, clf)

    result = run_backtest(panel_with_features, panel_ohlcv, clf, start_equity=100_000.0)

    equity = result.equity
    buy_hold_equity = result.buy_hold_equity
    trades = result.trades
    metrics = result.metrics

    print("\nBacktest metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if not trades.empty:
        print("\nTrade summary (first 10 trades):")
        cols = [
            "symbol",
            "entry_dt",
            "exit_dt",
            "entry_price",
            "exit_price_est",
            "proba_up",
            "trade_ret",
            "pnl_dollars",
        ]
        cols = [c for c in cols if c in trades.columns]
        print(trades[cols].head(10).to_string(index=False))

        trades_path = project_root / "trades.csv"
        trades.to_csv(trades_path, index=False)
        print(f"\nSaved full trade log to {trades_path}")

    equity_df = pd.DataFrame(
        {
            "strategy_equity": equity,
            "buy_hold_equity": buy_hold_equity.reindex(equity.index, method="ffill"),
        }
    )
    eq_path = project_root / "equity_curve.csv"
    equity_df.to_csv(eq_path)
    print(f"Saved equity curves to {eq_path}")

    plt.figure(figsize=(11, 5))
    plt.plot(equity.index, equity.values, label="ML strategy")
    bh_aligned = buy_hold_equity.reindex(equity.index, method="ffill")
    plt.plot(
        bh_aligned.index,
        bh_aligned.values,
        label="Buy & hold (equal-weight)",
        linestyle="--",
    )
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.title("ML Strategy vs Buy & Hold")
    plt.legend()
    plt.tight_layout()

    plot_path = project_root / "equity_vs_buyhold.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved equity comparison plot to {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()

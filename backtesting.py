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
    RISK_PER_TRADE_FRACTION,
    BAR_INTERVAL_MIN,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_BARS_IN_TRADE,
    MAX_CONCURRENT_POSITIONS,
    COOLDOWN_BARS_AFTER_STOP,
    DAILY_LOSS_STOP_FRACTION,
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
    start_equity: float = 1000000.0,
    threshold_override: float | None = None,
) -> BacktestResult:
    """Backtest using live-style SL/TP/MAX_BARS exits and live-style risk controls."""
    entry_thr = P_UP_ENTRY_THRESHOLD if threshold_override is None else threshold_override
    print("=== Step 4: Run backtest (live-style exits + risk controls) ===")

    df = panel_with_features.copy()

    needed = FEATURE_COLUMNS + ["open", "high", "low", "close", LABEL_COLUMN]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in panel_with_features.")

    # ----- model probabilities & classification metrics -----
    feat_mask = df[FEATURE_COLUMNS].notna().all(axis=1)
    proba = np.full(len(df), np.nan)
    if feat_mask.any():
        X = df.loc[feat_mask, FEATURE_COLUMNS].values
        proba[feat_mask.to_numpy()] = clf.predict_proba(X)[:, 1]
    df["proba_up"] = proba

    valid_mask = (~df["proba_up"].isna()) & df[LABEL_COLUMN].notna()
    if valid_mask.any():
        y_true = df.loc[valid_mask, LABEL_COLUMN].astype(int)
        y_score = df.loc[valid_mask, "proba_up"].astype(float)
        try:
            val_auc = roc_auc_score(y_true, y_score)
        except ValueError:
            val_auc = np.nan
        y_pred = (y_score >= entry_thr).astype(int)
        val_acc = accuracy_score(y_true, y_pred)
    else:
        val_auc = np.nan
        val_acc = np.nan

    # If nothing ever crosses the threshold, return flat equity
    trigger_mask = df["proba_up"] >= entry_thr
    if not trigger_mask.any():
        print("No trades triggered by the ML model with current threshold.")
        all_dts = (
            panel_ohlcv.index.get_level_values("datetime")
            .sort_values()
            .unique()
        )
        equity = pd.Series(start_equity, index=all_dts, name="strategy_equity")
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

        trades_empty = pd.DataFrame(
            columns=[
                "symbol",
                "entry_dt",
                "exit_dt",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "r_multiple",
                "trade_ret",
                "reason",
                "p_up",
            ]
        )

        return BacktestResult(
            equity=equity,
            buy_hold_equity=bh_equity,
            trades=trades_empty,
            metrics=metrics,
        )

    # ----- sort by time & symbol for bar-by-bar simulation -----
    df_reset = df.reset_index().rename(columns={"datetime": "dt"})
    df_reset["dt"] = pd.to_datetime(df_reset["dt"])
    df_reset = df_reset.sort_values(["dt", "symbol"]).reset_index(drop=True)

    equity_idx = []
    equity_vals = []

    # Realized PnL tracking
    realized_pnl_cum = 0.0          # total realized over whole backtest
    realized_pnl_today = 0.0        # realized within current trading date

    positions: dict[str, dict] = {}
    last_stop_bar: dict[str, pd.Timestamp] = {}

    trades_out = []

    current_trading_date: Optional[dt.date] = None
    daily_loss_stop_active = False   # block new entries once hit for the day

    def unrealized_pnl() -> float:
        """Unrealized PnL over all open positions."""
        total = 0.0
        for pos in positions.values():
            total += (pos["last_price"] - pos["entry_price"]) * pos["size"]
        return total

    def equity_for_risk() -> float:
        """Equity notion used for risk sizing / daily-loss-stop (matches live style)."""
        return start_equity + realized_pnl_today + unrealized_pnl()

    def equity_for_curve() -> float:
        """Equity notion used for equity curve / final metrics."""
        return start_equity + realized_pnl_cum + unrealized_pnl()

    bar_dt_unit = pd.to_timedelta(BAR_INTERVAL_MIN, unit="m")

    # ----- main time loop: exits, risk checks, entries, equity mark -----
    for dt_val, frame in df_reset.groupby("dt"):
        trade_date = dt_val.date()

        # roll to new trading date
        if current_trading_date is None or trade_date != current_trading_date:
            current_trading_date = trade_date
            realized_pnl_today = 0.0
            daily_loss_stop_active = False

        just_exited = set()

        # 1) exits and mark latest prices
        for _, row in frame.iterrows():
            sym = row["symbol"]
            close_px = float(row["close"])
            high_px = float(row["high"])

            if sym in positions:
                pos = positions[sym]
                pos["bars_held"] += 1
                pos["last_price"] = close_px

                exit_price = None
                reason = None

                if close_px <= pos["stop_price"]:
                    exit_price = close_px
                    reason = "STOP"
                elif high_px >= pos["take_profit_price"]:
                    exit_price = pos["take_profit_price"]
                    reason = "TP"
                elif pos["bars_held"] >= MAX_BARS_IN_TRADE:
                    exit_price = close_px
                    reason = "MAX_BARS"

                if exit_price is not None:
                    pnl = (exit_price - pos["entry_price"]) * pos["size"]
                    realized_pnl_cum += pnl
                    realized_pnl_today += pnl

                    risk_per_share = pos["entry_price"] * STOP_LOSS_PCT
                    denom = risk_per_share * pos["size"] if risk_per_share > 0 else np.nan
                    r_mult = pnl / denom if denom and denom != 0 else np.nan

                    notional = pos["entry_price"] * pos["size"]
                    trade_ret = pnl / notional if notional != 0 else np.nan

                    trades_out.append(
                        {
                            "symbol": sym,
                            "entry_dt": pos["entry_dt"],
                            "exit_dt": dt_val,
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "size": pos["size"],
                            "pnl": pnl,
                            "r_multiple": r_mult,
                            "trade_ret": trade_ret,
                            "reason": reason,
                            "p_up": pos["p_up"],
                            "bars_held": pos["bars_held"],
                        }
                    )

                    if reason == "STOP":
                        last_stop_bar[sym] = dt_val

                    del positions[sym]
                    just_exited.add(sym)

        # 2) update daily loss stop status based on risk equity
        eq_risk = equity_for_risk()
        drawdown = (eq_risk - start_equity) / start_equity
        if drawdown <= -DAILY_LOSS_STOP_FRACTION:
            daily_loss_stop_active = True

        # 3) entries based on ML signal, risk sizing, concurrency, cooldown, daily stop
        for _, row in frame.iterrows():
            sym = row["symbol"]

            # no new entries if:
            # - already have position
            # - just exited this bar
            # - daily loss stop hit
            # - max concurrent positions reached
            if (
                sym in positions
                or sym in just_exited
                or daily_loss_stop_active
                or len(positions) >= MAX_CONCURRENT_POSITIONS
            ):
                continue

            p_up = row["proba_up"]
            if pd.isna(p_up) or p_up < entry_thr:
                continue

            # per-symbol cooldown after STOP
            last_stop = last_stop_bar.get(sym)
            if last_stop is not None:
                bars_since_stop = (dt_val - last_stop) / bar_dt_unit
                if bars_since_stop < COOLDOWN_BARS_AFTER_STOP:
                    continue

            eq_now = equity_for_risk()
            close_px = float(row["close"])

            stop_price = close_px * (1.0 - STOP_LOSS_PCT)
            risk_per_share = close_px - stop_price
            if risk_per_share <= 0:
                continue

            risk_capital = eq_now * RISK_PER_TRADE_FRACTION
            size = int(risk_capital / risk_per_share)
            if size <= 0:
                continue

            positions[sym] = {
                "entry_price": close_px,
                "stop_price": stop_price,
                "take_profit_price": close_px * (1.0 + TAKE_PROFIT_PCT),
                "size": size,
                "entry_dt": dt_val,
                "bars_held": 0,
                "last_price": close_px,
                "p_up": float(p_up),
            }

        # 4) mark equity at end of this bar time (curve equity)
        eq_curve = equity_for_curve()
        equity_idx.append(dt_val)
        equity_vals.append(eq_curve)

    # ----- force-close remaining positions at final mark -----
    if positions:
        final_dt = equity_idx[-1]
        for sym, pos in list(positions.items()):
            exit_price = pos["last_price"]
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
            realized_pnl_cum += pnl
            realized_pnl_today += pnl

            risk_per_share = pos["entry_price"] * STOP_LOSS_PCT
            denom = risk_per_share * pos["size"] if risk_per_share > 0 else np.nan
            r_mult = pnl / denom if denom and denom != 0 else np.nan

            notional = pos["entry_price"] * pos["size"]
            trade_ret = pnl / notional if notional != 0 else np.nan

            trades_out.append(
                {
                    "symbol": sym,
                    "entry_dt": pos["entry_dt"],
                    "exit_dt": final_dt,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "size": pos["size"],
                    "pnl": pnl,
                    "r_multiple": r_mult,
                    "trade_ret": trade_ret,
                    "reason": "EOD_FORCED",
                    "p_up": pos["p_up"],
                    "bars_held": pos["bars_held"],
                }
            )

            del positions[sym]

        # last equity point after final realization
        equity_vals[-1] = equity_for_curve()

    equity = pd.Series(
        equity_vals,
        index=pd.to_datetime(equity_idx),
        name="strategy_equity",
    )

    # ----- benchmark & metrics -----
    buy_hold_equity = build_buy_and_hold_equity(panel_ohlcv, start_equity)
    buy_hold_equity = (
        buy_hold_equity.reindex(equity.index)
        .ffill()
        .bfill()
    )

    trades_df = pd.DataFrame(trades_out)

    returns = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    if len(equity.index) > 1:
        n_years = (equity.index[-1] - equity.index[0]).days / 365.25
        if n_years > 0:
            cagr = (1.0 + total_return) ** (1.0 / n_years) - 1.0
        else:
            cagr = np.nan
    else:
        cagr = np.nan

    if returns.std() > 0:
        sharpe = float(np.sqrt(252.0) * returns.mean() / returns.std())
    else:
        sharpe = np.nan

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = float(drawdown.min())

    n_trades = len(trades_df)
    if n_trades > 0:
        win_mask = trades_df["trade_ret"] > 0
        lose_mask = trades_df["trade_ret"] < 0

        win_rate = float(win_mask.mean())
        avg_gain = (
            float(trades_df.loc[win_mask, "trade_ret"].mean()) if win_mask.any() else np.nan
        )
        avg_loss = (
            float(trades_df.loc[lose_mask, "trade_ret"].mean()) if lose_mask.any() else np.nan
        )
    else:
        win_rate = np.nan
        avg_gain = np.nan
        avg_loss = np.nan

    metrics = {
        "total_return": total_return,
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(equity.iloc[-1]),
        "start_equity": float(equity.iloc[0]),
        "n_trades": int(n_trades),
        "win_rate": win_rate,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "val_auc_backtest": float(val_auc) if pd.notna(val_auc) else np.nan,
        "val_accuracy_backtest": float(val_acc) if pd.notna(val_acc) else np.nan,
    }

    return BacktestResult(
        equity=equity,
        buy_hold_equity=buy_hold_equity,
        trades=trades_df,
        metrics=metrics,
    )


def sweep_entry_thresholds(
    panel_with_features: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    clf,
    thresholds: list[float],
    start_equity: float = 100_000.0,
) -> pd.DataFrame:
    """Sweep p_up entry thresholds and summarize key backtest metrics."""
    rows = []

    for thr in thresholds:
        print("\n" + "=" * 60)
        print(f"Sweep run for entry threshold = {thr:.3f}")
        print("=" * 60)

        result = run_backtest(
            panel_with_features,
            panel_ohlcv,
            clf,
            start_equity=start_equity,
            threshold_override=thr,
        )
        m = result.metrics

        rows.append(
            {
                "threshold": thr,
                "n_trades": m["n_trades"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
                "total_return": m["total_return"],
            }
        )

    df_res = pd.DataFrame(rows).set_index("threshold")
    print("\n=== Threshold sweep summary ===")
    print(df_res.to_string(float_format=lambda x: f"{x: .4f}"))
    return df_res


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

def print_backtest_summary(result: BacktestResult) -> None:
    """Print key backtest metrics and a short trade sample."""
    m = result.metrics

    print("\nBacktest metrics:")
    print(f"  total_return: {m['total_return']:.4f}")
    print(f"  CAGR: {m['CAGR']:.4f}")
    print(f"  sharpe: {m['sharpe']:.4f}")
    print(f"  max_drawdown: {m['max_drawdown']:.4f}")
    print(f"  final_equity: {m['final_equity']:.4f}")
    print(f"  start_equity: {m['start_equity']:.4f}")
    print(f"  n_trades: {m['n_trades']}")
    print(f"  win_rate: {m['win_rate']:.4f}" if m["n_trades"] > 0 else "  win_rate: nan")
    print(f"  avg_gain: {m['avg_gain']:.4f}" if m["n_trades"] > 0 else "  avg_gain: nan")
    print(f"  avg_loss: {m['avg_loss']:.4f}" if m["n_trades"] > 0 else "  avg_loss: nan")
    print(f"  val_auc_backtest: {m['val_auc_backtest']:.4f}")
    print(f"  val_accuracy_backtest: {m['val_accuracy_backtest']:.4f}")

    trades = result.trades
    if trades is not None and not trades.empty:
        cols = ["symbol", "entry_dt", "exit_dt", "entry_price", "trade_ret"]
        print("\nTrade summary (first 10 trades):")
        print(trades[cols].head(10).to_string(index=False))
    else:
        print("\nNo trades to display.")

def summarize_trades_by_symbol(result: BacktestResult) -> None:
    """Summarize performance by symbol (n trades, PnL, returns, win rate)."""
    trades = result.trades
    if trades is None or trades.empty:
        print("\n[Symbol summary] No trades to summarize.")
        return

    df = trades.copy()
    # Ensure trade_ret exists; if not, derive from pnl and entry_price*size
    if "trade_ret" not in df.columns or df["trade_ret"].isna().all():
        notional = df["entry_price"] * df["size"].abs()
        df["trade_ret"] = df["pnl"] / notional.replace(0, np.nan)

    # Aggregate by symbol
    grouped = df.groupby("symbol")
    summary = grouped.agg(
        n_trades=("symbol", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        avg_ret=("trade_ret", "mean"),
        win_rate=("trade_ret", lambda x: (x > 0).mean()),
    )

    # Sort by total_pnl descending so you see biggest contributors first
    summary = summary.sort_values("total_pnl", ascending=False)

    print("\n=== Symbol-level performance summary ===")
    print(summary.to_string(float_format=lambda x: f"{x: .4f}"))

def summarize_trades_by_holding_period(result: BacktestResult) -> None:
    """Summarize performance by holding duration in bars."""
    trades = result.trades
    if trades is None or trades.empty or "bars_held" not in trades.columns:
        print("\n[Holding-period summary] No trades or bars_held not tracked.")
        return

    df = trades.copy()

    # Define some rough buckets: 0–3, 4–6, 7–12, 13–24, 25+ bars
    bins = [0, 3, 6, 12, 24, np.inf]
    labels = ["0–3", "4–6", "7–12", "13–24", "25+"]
    df["bars_bucket"] = pd.cut(df["bars_held"], bins=bins, labels=labels, right=True)

    grouped = df.groupby("bars_bucket")
    summary = grouped.agg(
        n_trades=("bars_held", "count"),
        avg_ret=("trade_ret", "mean"),
        win_rate=("trade_ret", lambda x: (x > 0).mean()),
    )

    print("\n=== Holding-period performance summary (bars_held) ===")
    print(summary.to_string(float_format=lambda x: f"{x: .44f}"))


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

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    sweep_df = sweep_entry_thresholds(
        panel_with_features,
        panel_ohlcv,
        clf,
        thresholds=thresholds,
        start_equity=100_000.0,
    )

    # 2) optionally run once at your current config threshold for full detail
    #    (using the threshold from config.py)
    print("\n\n=== Full backtest at config threshold ===")
    result = run_backtest(
        panel_with_features,
        panel_ohlcv,
        clf,
        start_equity=100_000.0,
        threshold_override=None,  # use P_UP_ENTRY_THRESHOLD from config
    )
    print_backtest_summary(result)
    summarize_trades_by_symbol(result)
    summarize_trades_by_holding_period(result)

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

    equity_df = pd.DataFrame(
        {
            "strategy_equity": equity,
            "buy_hold_equity": buy_hold_equity.reindex(equity.index, method="ffill"),
        }
    )

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

    plt.show()


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Paper trading strategy (IBKR, 5-min bars):

- Data: polls 5-min TRADES bars for UNIVERSE via reqHistoricalData.
- Entry: ML classifier; open long if p_up >= P_UP_ENTRY_THRESHOLD and risk checks pass.
- Exit: stop-loss on close, intrabar take-profit on high, early TP at +2R, or MAX_BARS_IN_TRADE.
- Risk: size = RISK_PER_TRADE_FRACTION * equity / price, per-symbol cooldown after STOP,
  and DAILY_LOSS_STOP_FRACTION to halt new entries.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from types import SimpleNamespace

import datetime as dt

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

import logging
import os
from logging.handlers import RotatingFileHandler

# basic rotating file + console logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "paper_trade.log")

logger = logging.getLogger("paper_trade")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


from config import (
    UNIVERSE,
    MODEL_DIR,
    FEATURE_COLUMNS,
    BAR_INTERVAL_MIN,
    P_UP_ENTRY_THRESHOLD,
    RISK_PER_TRADE_FRACTION,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_CONCURRENT_POSITIONS,
    DAILY_LOSS_STOP_FRACTION,
    MAX_BARS_IN_TRADE,
    COOLDOWN_BARS_AFTER_STOP,
)

from quant.quant_model import build_features_for_symbol
from backtesting import load_latest_classifier  # same model as backtest

STARTING_EQUITY = 100_000.0
MAX_BUFFER_LENGTH = 500
ORDER_WAIT_TIMEOUT_SEC = 15  # max time to wait for order fills


@dataclass
class Position:
    """Single open long position with entry, size, risk levels, and model p_up."""
    symbol: str
    size: int
    entry_price: float
    entry_dt: pd.Timestamp
    stop_price: float
    take_profit_price: float
    p_up: float  # model probability at entry


class PaperTrader:
    """Polling-based paper trader that applies ML signals on 5-min bars."""

    def __init__(self) -> None:
        """Initialize IB connection, model handle, and trading state."""
        self.ib: Optional[IB] = None
        self.clf = None
        self.contracts: Dict[str, Stock] = {}
        self.ohlcv_buffers: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Position] = {}

        self.start_of_day_equity: float = STARTING_EQUITY
        self.realized_pnl_today: float = 0.0
        self.last_stop_bar: Dict[str, pd.Timestamp] = {}

        self.trades_today: list[dict] = []
        self.current_trading_date: Optional[dt.date] = None

    # ------------ logging helper ------------

    @staticmethod
    def _log(msg: str) -> None:
        """Write a line to both file and console logs."""
        logger.info(msg)

    # ------------ setup / connection ------------

    def connect_and_load(self) -> None:
        """Connect to IBKR, load model, qualify contracts, and init buffers."""
        self._log("Connecting to IBKR TWS/Gateway...")
        self.ib = IB()
        self.ib.connect("127.0.0.1", 7497, clientId=1)
        self._log("Connected to IBKR.")

        self._log("Loading latest classifier...")
        self.clf = load_latest_classifier(MODEL_DIR)
        self._log("Classifier loaded.")

        self._log(f"Creating contracts for universe: {UNIVERSE}")
        for sym in UNIVERSE:
            self.contracts[sym] = Stock(sym, "SMART", "USD")

        self._log("Qualifying contracts with IBKR...")
        self.ib.qualifyContracts(*self.contracts.values())
        self._log("Contracts qualified.")

        self._log("Initializing OHLCV buffers...")
        for sym in UNIVERSE:
            self.ohlcv_buffers[sym] = self._empty_buffer()
        self._log("Buffers initialized.")

    @staticmethod
    def _empty_buffer() -> pd.DataFrame:
        """Return an empty OHLCV buffer for a symbol."""
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols)

    # ------------ sync existing IB positions on startup ------------

    def _load_existing_ib_positions(self) -> None:
        """Mirror any existing IBKR long positions into local state."""
        self._log("Loading existing IBKR positions into local state...")
        ib_positions = self.ib.positions()

        for p in ib_positions:
            sym = p.contract.symbol
            if sym not in UNIVERSE:
                continue

            size = int(p.position)
            if size <= 0:
                continue

            entry_price = float(p.avgCost)
            stop_price = entry_price * (1.0 - STOP_LOSS_PCT)
            take_profit_price = entry_price * (1.0 + TAKE_PROFIT_PCT)

            # We don't know original p_up; store a sentinel (0.0) just so the field exists
            self.positions[sym] = Position(
                symbol=sym,
                size=size,
                entry_price=entry_price,
                entry_dt=pd.Timestamp.utcnow(),  # unknown true open time
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                p_up=0.0,
            )

            self._log(
                f"[SYNC] Loaded existing position from IB: {sym}, "
                f"size={size}, entry_price={entry_price:.4f}, "
                f"stop={stop_price:.4f}, tp={take_profit_price:.4f}"
            )

    # ------------ risk / equity helpers ------------

    def _current_equity(self) -> float:
        """Estimate equity as starting capital plus realized PnL."""
        return self.start_of_day_equity + self.realized_pnl_today

    def _daily_loss_exceeded(self) -> bool:
        """Check if daily loss exceeds configured drawdown limit."""
        eq = self._current_equity()
        if eq <= 0:
            self._log("Equity is non-positive; daily loss exceeded.")
            return True

        drawdown_fraction = -self.realized_pnl_today / eq
        exceeded = drawdown_fraction >= DAILY_LOSS_STOP_FRACTION
        if exceeded:
            self._log(
                f"Daily loss limit exceeded: drawdown={drawdown_fraction:.4f}, "
                f"limit={DAILY_LOSS_STOP_FRACTION:.4f}"
            )
        return exceeded

    def _can_open_position(self, symbol: str, bar_time: pd.Timestamp) -> bool:
        """Gate entries by concurrency, daily loss, and post-STOP cooldown."""
        if symbol in self.positions:
            self._log(f"Skip entry for {symbol}: already in an open position.")
            return False

        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            self._log(
                f"Skip entry for {symbol}: reached MAX_CONCURRENT_POSITIONS="
                f"{MAX_CONCURRENT_POSITIONS}."
            )
            return False

        if self._daily_loss_exceeded():
            self._log(f"Skip entry for {symbol}: daily loss limit hit.")
            return False

        # Symbol-specific cooldown after a stop-loss exit
        last_stop = self.last_stop_bar.get(symbol)
        if last_stop is not None:
            bars_since_stop = (bar_time - last_stop) / dt.timedelta(
                minutes=BAR_INTERVAL_MIN
            )
            if bars_since_stop < COOLDOWN_BARS_AFTER_STOP:
                self._log(
                    f"Skip entry for {symbol}: in cooldown after STOP "
                    f"({bars_since_stop:.1f} bars ago, cooldown="
                    f"{COOLDOWN_BARS_AFTER_STOP} bars)."
                )
                return False

        return True

    def _calc_position_size(self, last_price: float) -> int:
        """Size trade as a fixed fraction of equity divided by price."""
        eq = self._current_equity()
        notional = RISK_PER_TRADE_FRACTION * eq
        if last_price <= 0:
            self._log(
                f"Invalid last_price={last_price:.4f}; position size set to 0."
            )
            return 0
        size = int(notional / last_price)
        self._log(
            f"Calculated position size: equity={eq:.2f}, risk_frac={RISK_PER_TRADE_FRACTION}, "
            f"notional={notional:.2f}, last_price={last_price:.4f}, size={size}"
        )
        return max(size, 0)

    # ------------ open positions logging ------------

    def _log_open_positions(self) -> None:
        """Log snapshot of all open positions and their current R multiples."""
        if not self.positions:
            self._log("POSITIONS: none open.")
            return

        self._log("POSITIONS SNAPSHOT:")
        for sym, pos in self.positions.items():
            buf = self.ohlcv_buffers.get(sym)
            if buf is not None and not buf.empty:
                current_price = float(buf["close"].iloc[-1])
            else:
                current_price = pos.entry_price

            pnl_per_share = current_price - pos.entry_price
            pnl_dollar = pnl_per_share * pos.size
            pnl_pct = (current_price / pos.entry_price - 1.0) * 100.0

            risk_per_share = pos.entry_price - pos.stop_price
            r_multiple = (
                pnl_per_share / risk_per_share if risk_per_share > 0 else float("nan")
            )

            self._log(
                "  {sym}: size={size}, entry={entry:.4f}, "
                "last={last:.4f}, pnl={pnl:.2f} ({pnl_pct:.2f}%), "
                "stop={sl:.4f}, target_exit={tp:.4f}, R={r:.2f}, "
                "held_since={entry_dt}, p_up={p_up:.3f}".format(
                    sym=pos.symbol,
                    size=pos.size,
                    entry=pos.entry_price,
                    last=current_price,
                    pnl=pnl_dollar,
                    pnl_pct=pnl_pct,
                    sl=pos.stop_price,
                    tp=pos.take_profit_price,
                    r=r_multiple,
                    entry_dt=pos.entry_dt,
                    p_up=pos.p_up,
                )
            )

    # ------------ model / signal ------------

    def _compute_p_up(self, symbol: str) -> Optional[float]:
        """Compute model p_up for latest bar of a symbol."""
        buf = self.ohlcv_buffers[symbol]

        if buf.empty or len(buf) < 50:
            self._log(
                f"Not enough data to compute p_up for {symbol}: "
                f"len(buf)={len(buf)} (need >= 50)."
            )
            return None

        df = buf.copy()
        df.index = pd.to_datetime(df.index)

        feat_df = build_features_for_symbol(df)
        latest = feat_df.iloc[-1]

        if latest[FEATURE_COLUMNS].isna().any():
            self._log(f"NaNs in feature columns for {symbol}; skipping signal.")
            return None

        X = latest[FEATURE_COLUMNS].to_frame().T
        p_up = self.clf.predict_proba(X)[:, 1][0]
        p_up = float(p_up)
        self._log(f"Computed p_up for {symbol}: {p_up:.4f}")
        return p_up

    # ------------ order handling with fill tracking ------------

    def _wait_for_trade_fill(self, trade) -> Optional[float]:
        """Block until order is filled/cancelled or timeout, and return fill price."""
        deadline = dt.datetime.now() + dt.timedelta(seconds=ORDER_WAIT_TIMEOUT_SEC)

        while True:
            status = trade.orderStatus.status
            filled = trade.orderStatus.filled
            remaining = trade.orderStatus.remaining

            self._log(
                f"[ORDER] status={status}, filled={filled}, remaining={remaining}"
            )

            if status in ("Filled", "ApiCancelled", "Cancelled", "Inactive"):
                break

            if dt.datetime.now() >= deadline:
                self._log(
                    f"[ORDER] Timeout waiting for fill; last status={status}, "
                    f"filled={filled}, remaining={remaining}"
                )
                break

            self.ib.sleep(1)

        status = trade.orderStatus.status
        if status == "Filled" and trade.orderStatus.filled > 0:
            fill_price = (
                trade.orderStatus.avgFillPrice or trade.orderStatus.lastFillPrice
            )
            self._log(
                f"[ORDER] Filled: orderId={trade.order.orderId}, "
                f"avgFillPrice={fill_price:.4f}"
            )
            return float(fill_price)

        self._log(
            f"[ORDER] Not filled (final status={status}); treating as no trade."
        )
        return None

    def _send_market_order(
        self, symbol: str, action: str, size: int, price_hint: float
    ) -> Optional[float]:
        """Submit market order and return actual fill price if filled."""
        self._log(
            f"Placing market order: symbol={symbol}, action={action}, size={size}, "
            f"price_hint={price_hint:.4f}"
        )
        contract = self.contracts[symbol]
        order = MarketOrder(action, size)
        order.tif = "DAY"

        trade = self.ib.placeOrder(contract, order)
        fill_price = self._wait_for_trade_fill(trade)

        if fill_price is None:
            self._log(
                f"[ORDER] {action} {symbol} size={size} was not filled; "
                "skipping position update."
            )
            return None

        return fill_price

    def _handle_exit(
        self,
        symbol: str,
        last_price: float,
        high_price: float,
        bar_time: pd.Timestamp,
    ) -> None:
        """Exit on stop-loss, intrabar TP, +2R target, or max holding bars."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        bars_held = (bar_time - pos.entry_dt) / dt.timedelta(
            minutes=BAR_INTERVAL_MIN
        )

        risk_per_share = pos.entry_price - pos.stop_price
        pnl_per_share = last_price - pos.entry_price
        r_multiple = (
            pnl_per_share / risk_per_share if risk_per_share > 0 else float("nan")
        )

        hit_stop = last_price <= pos.stop_price
        hit_tp = high_price >= pos.take_profit_price       # intrabar touch
        hit_r2 = r_multiple >= 2.0                         # early take profit at +2R
        hit_max_bars = bars_held >= MAX_BARS_IN_TRADE

        if not (hit_stop or hit_tp or hit_r2 or hit_max_bars):
            return

        if hit_stop:
            reason = "STOP"
        elif hit_tp or hit_r2:
            reason = "TAKE_PROFIT"
        else:
            reason = "MAX_BARS"

        self._log(
            f"Exit condition met for {symbol}: last_price={last_price:.4f}, "
            f"stop={pos.stop_price:.4f}, tp={pos.take_profit_price:.4f}, "
            f"R={r_multiple:.2f}, bars_held={bars_held:.1f}, reason={reason}"
        )

        exit_price = self._send_market_order(symbol, "SELL", pos.size, last_price)
        if exit_price is None:
            self._log(
                f"[EXIT] Order to close {symbol} was not filled; leaving position open."
            )
            return

        pnl = (exit_price - pos.entry_price) * pos.size
        self.realized_pnl_today += pnl

        trade_record = {
            "symbol": symbol,
            "direction": "LONG",
            "entry_dt": pos.entry_dt,
            "exit_dt": bar_time,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size": pos.size,
            "pnl": pnl,
            "reason": reason,
            "hold_bars": float(bars_held),
            "p_up": pos.p_up,
        }
        self.trades_today.append(trade_record)

        self._log(
            f"[EXIT] {symbol}: price={exit_price:.2f}, pnl={pnl:.2f}, "
            f"realized_pnl_today={self.realized_pnl_today:.2f}"
        )

        # Only STOP exits trigger cooldown
        if hit_stop:
            self.last_stop_bar[symbol] = bar_time

        del self.positions[symbol]

    def _handle_entry(
        self, symbol: str, last_price: float, bar_time: pd.Timestamp
    ) -> None:
        """Run ML signal for a symbol and open long if conditions are met."""
        self._log(
            f"Evaluating entry for {symbol} at {bar_time}, last_price={last_price:.4f}"
        )

        if not self._can_open_position(symbol, bar_time):
            return

        p_up = self._compute_p_up(symbol)
        if p_up is None:
            self._log(f"No valid p_up for {symbol}; skipping entry.")
            return

        if p_up < P_UP_ENTRY_THRESHOLD:
            self._log(
                f"p_up below threshold for {symbol}: p_up={p_up:.4f}, "
                f"threshold={P_UP_ENTRY_THRESHOLD:.4f}. No trade."
            )
            return

        size = self._calc_position_size(last_price)
        if size <= 0:
            self._log(f"Calculated size <= 0 for {symbol}; skipping entry.")
            return

        stop_price = last_price * (1.0 - STOP_LOSS_PCT)
        take_profit_price = last_price * (1.0 + TAKE_PROFIT_PCT)

        entry_price = self._send_market_order(symbol, "BUY", size, last_price)
        if entry_price is None:
            self._log(
                f"[ENTRY] Order to open {symbol} was not filled; no position created."
            )
            return

        self.positions[symbol] = Position(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            entry_dt=bar_time,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            p_up=p_up,
        )

        self._log(
            f"[ENTRY] {symbol}: size={size}, entry={entry_price:.2f}, "
            f"p_up={p_up:.3f}, stop={stop_price:.2f}, tp={take_profit_price:.2f}"
        )

    # ------------ daily summary helpers ------------

    def _log_daily_summary(self, summary_date: dt.date) -> None:
        """Log daily stats, trade list, and end-of-day open positions."""
        self._log("")
        self._log("=" * 70)
        self._log(f"DAILY SUMMARY for {summary_date.isoformat()}")
        self._log("=" * 70)

        n_trades = len(self.trades_today)
        total_realized = self.realized_pnl_today

        wins = [t for t in self.trades_today if t["pnl"] > 0]
        losses = [t for t in self.trades_today if t["pnl"] < 0]
        flats = [t for t in self.trades_today if t["pnl"] == 0]

        max_win = max((t["pnl"] for t in wins), default=0.0)
        max_loss = min((t["pnl"] for t in losses), default=0.0)

        avg_pnl = total_realized / n_trades if n_trades > 0 else 0.0
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0

        self._log(
            f"Trades: {n_trades}, wins={len(wins)}, losses={len(losses)}, flats={len(flats)}"
        )
        self._log(
            f"Total realized PnL: {total_realized:.2f}, "
            f"avg PnL/trade: {avg_pnl:.2f}, "
            f"win_rate: {win_rate:.1%}"
        )
        self._log(f"Max win: {max_win:.2f}, max loss: {max_loss:.2f}")

        if n_trades > 0:
            self._log("-" * 70)
            self._log("Per-trade details:")
            for t in self.trades_today:
                self._log(
                    f"  {t['symbol']} | dir={t['direction']} | size={t['size']} | "
                    f"entry={t['entry_price']:.4f} @ {t['entry_dt']} | "
                    f"exit={t['exit_price']:.4f} @ {t['exit_dt']} | "
                    f"pnl={t['pnl']:.2f} | reason={t['reason']} | "
                    f"hold_bars={t['hold_bars']:.1f} | p_up={t['p_up']:.3f}"
                )
        else:
            self._log("No closed trades today.")

        if self.positions:
            self._log("-" * 70)
            self._log("Open positions at end of day:")
            for sym, pos in self.positions.items():
                last_price = pos.entry_price
                buf = self.ohlcv_buffers.get(sym)
                if buf is not None and not buf.empty:
                    last_price = float(buf["close"].iloc[-1])
                unrealized = (last_price - pos.entry_price) * pos.size
                self._log(
                    f"  {sym} | size={pos.size} | entry={pos.entry_price:.4f} | "
                    f"last={last_price:.4f} | unrealized_pnl={unrealized:.2f} | "
                    f"p_up={pos.p_up:.3f}"
                )
        else:
            self._log("No open positions at end of day.")

        self._log("=" * 70)
        self._log("")

                # --- NEW: persist trades_today to CSV for analysis ---
        if self.trades_today:
            import csv

            out_dir = "live_trades"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"trades_{summary_date.isoformat()}.csv")

            fieldnames = [
                "symbol",
                "direction",
                "entry_dt",
                "exit_dt",
                "entry_price",
                "exit_price",
                "size",
                "pnl",
                "reason",
                "hold_bars",
                "p_up",
            ]
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for t in self.trades_today:
                    writer.writerow(t)

            self._log(f"Saved daily trades to {out_path}")

    def _maybe_roll_trading_day(self, bar_time: pd.Timestamp) -> None:
        """Detect day changes, log summary, and reset daily counters."""
        trade_date = bar_time.date()

        if self.current_trading_date is None:
            self.current_trading_date = trade_date
            self.start_of_day_equity = self._current_equity()
            self._log(
                f"Starting new trading day: {trade_date}, "
                f"starting_equity={self.start_of_day_equity:.2f}"
            )
            return

        if trade_date == self.current_trading_date:
            return

        self._log_daily_summary(self.current_trading_date)

        self.start_of_day_equity = self._current_equity()
        self.realized_pnl_today = 0.0
        self.trades_today.clear()
        self.last_stop_bar.clear()

        self.current_trading_date = trade_date
        self._log(
            f"Rolled to new trading day: {trade_date}, "
            f"starting_equity={self.start_of_day_equity:.2f}"
        )

    # ------------ bar handling ------------

    def _on_bar(self, symbol: str, bar) -> None:
        """Update buffer with new bar, then handle exits and entries."""
        ts = pd.to_datetime(getattr(bar, "date", dt.datetime.utcnow()))
        self._maybe_roll_trading_day(ts)

        last_price = float(bar.close)
        high_price = float(bar.high)
        volume = float(getattr(bar, "volume", 0.0))

        self._log(
            f"[BAR] {symbol}: ts={ts}, o={bar.open:.4f}, h={bar.high:.4f}, "
            f"l={bar.low:.4f}, c={last_price:.4f}, v={volume}"
        )

        buf = self.ohlcv_buffers[symbol]
        new_row = pd.DataFrame(
            {
                "open": [float(bar.open)],
                "high": [float(bar.high)],
                "low": [float(bar.low)],
                "close": [last_price],
                "volume": [volume],
            },
            index=[ts],
        )

        buf = pd.concat([buf, new_row])
        if len(buf) > MAX_BUFFER_LENGTH:
            buf = buf.iloc[-MAX_BUFFER_LENGTH:]

        self.ohlcv_buffers[symbol] = buf

        self._handle_exit(symbol, last_price, high_price, ts)
        self._handle_entry(symbol, last_price, ts)

    # ------------ polling instead of streaming ------------

    def _poll_bars_once(self) -> None:
        """Snapshot latest 5-min bars from IBKR and process only new ones."""
        for sym, contract in self.contracts.items():
            self._log(f"[POLL] Requesting latest bars for {sym}...")
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting=f"{BAR_INTERVAL_MIN} mins",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )

            if not bars:
                self._log(f"[POLL] No data returned for {sym}.")
                continue

            records = []
            for bar in bars:
                ts = pd.to_datetime(getattr(bar, "date", dt.datetime.utcnow()))
                records.append(
                    (
                        ts,
                        float(bar.open),
                        float(bar.high),
                        float(bar.low),
                        float(bar.close),
                        float(getattr(bar, "volume", 0.0)),
                    )
                )

            df = pd.DataFrame(
                records, columns=["date", "open", "high", "low", "close", "volume"]
            ).set_index("date")

            buf = self.ohlcv_buffers[sym]

            if buf.empty:
                self._log(
                    f"[POLL] {sym}: seeding buffer with {len(df)} bars (no trading yet)."
                )
                self.ohlcv_buffers[sym] = df
                continue

            last_ts = buf.index[-1]
            new_df = df[df.index > last_ts]

            if new_df.empty:
                self._log(f"[POLL] {sym}: no new bars since {last_ts}.")
                continue

            self._log(f"[POLL] {sym}: processing {len(new_df)} new bars.")

            for ts, row in new_df.iterrows():
                bar_obj = SimpleNamespace(
                    date=ts,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
                self._on_bar(sym, bar_obj)

    # ------------ main loop ------------

    def run(self) -> None:
        """Main loop: poll bars, trade, then sleep for BAR_INTERVAL_MIN minutes."""
        self._log("Initializing paper trader...")
        self.connect_and_load()
        self._load_existing_ib_positions()

        self._log(
            f"Starting polling bar loop (interval={BAR_INTERVAL_MIN} minutes)..."
        )
        while True:
            self._poll_bars_once()
            self._log_open_positions()

            self._log(
                f"[POLL] Sleeping for {BAR_INTERVAL_MIN} minutes before next poll..."
            )
            self.ib.sleep(BAR_INTERVAL_MIN * 60)


def main() -> None:
    """Instantiate a PaperTrader and run the live loop."""
    trader = PaperTrader()
    trader.run()


if __name__ == "__main__":
    main()

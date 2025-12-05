from __future__ import annotations

"""
Paper trading strategy (IBKR, 5-min bars):

- Data: polls 5-min TRADES bars for UNIVERSE via reqHistoricalData.
- Entry: ML classifier; open long if p_up >= P_UP_ENTRY_THRESHOLD and risk checks pass.
- Exit: stop-loss on close, intrabar take-profit on high, or MAX_BARS_IN_TRADE.
- Risk: size = RISK_PER_TRADE_FRACTION * equity / price, per-symbol cooldown after STOP,
  and DAILY_LOSS_STOP_FRACTION to halt new entries.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from types import SimpleNamespace

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

import logging
import os
import csv
import time
from logging.handlers import RotatingFileHandler

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
from backtesting import load_latest_classifier


# === Logging setup ===

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "paper_trade.log")
TRADES_CSV_PATH = os.path.join(LOG_DIR, "paper_trades.csv")

logger = logging.getLogger("paper_trade")
logger.setLevel(logging.INFO)

# File handler (rotating)
file_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=5)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logger.addHandler(console_handler)

# Separate heartbeat logger (used outside RTH)
heartbeat_logger = logging.getLogger("paper_trade_heartbeat")
heartbeat_logger.setLevel(logging.INFO)
heartbeat_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "paper_trade_heartbeat.log"),
    maxBytes=1_000_000,
    backupCount=3,
)
heartbeat_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
heartbeat_logger.addHandler(heartbeat_handler)

STARTING_EQUITY = 100_000.0

# Reduced warmup so you can trade earlier in the session
MIN_BARS_FOR_FEATURES = 12  # was 30; ~1 hour of 5-min bars instead of 2.5h


# ================
# Regular trading hours helper
# ================

RTH_START = dt.time(9, 00)
RTH_END = dt.time(16, 30)

try:
    RTH_TZ = ZoneInfo("America/New_York")
except Exception:
    # Fallback: use local system time if zoneinfo is not available
    RTH_TZ = None

# Heartbeat interval when outside RTH (in minutes)
HEARTBEAT_INTERVAL_MIN_OUTSIDE_RTH = 60


def is_rth_now() -> bool:
    """Return True if current time is within US equity RTH."""
    if RTH_TZ is not None:
        now = dt.datetime.now(RTH_TZ)
    else:
        now = dt.datetime.now()
    current_time = now.time()
    return RTH_START <= current_time <= RTH_END


def minutes_until_next_rth_open() -> float:
    """Return minutes until next RTH open (today or tomorrow)."""
    if RTH_TZ is not None:
        now = dt.datetime.now(RTH_TZ)
    else:
        now = dt.datetime.now()

    current_time = now.time()

    # If before today's open, next open is today; otherwise tomorrow.
    if current_time < RTH_START:
        target_date = now.date()
    else:
        target_date = now.date() + dt.timedelta(days=1)

    if RTH_TZ is not None:
        target_dt = dt.datetime.combine(target_date, RTH_START, tzinfo=RTH_TZ)
    else:
        target_dt = dt.datetime.combine(target_date, RTH_START)

    delta = target_dt - now
    return max(delta.total_seconds() / 60.0, 0.0)


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
    """Paper trader that mirrors backtest logic using IBKR 5-min bars."""

    def __init__(self) -> None:
        # IB connection + model
        self.ib: Optional[IB] = None
        self.clf = None

        # Universe + data buffers + positions
        self.contracts: Dict[str, Stock] = {}
        self.ohlcv_buffers: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Position] = {}

        # Equity tracking
        self.start_of_day_equity: float = STARTING_EQUITY
        self.realized_pnl_today: float = 0.0
        self.last_stop_bar: Dict[str, pd.Timestamp] = {}

        # Per-session tracking
        self.trades_today: list[dict] = []
        self.current_trading_date: Optional[dt.date] = None

    # ------------ logging helpers ------------

    @staticmethod
    def _log(msg: str) -> None:
        """Write a line to both file and console logs."""
        logger.info(msg)

    def _append_trade_to_csv(self, trade_record: dict) -> None:
        """Append a single closed trade to a CSV file for later analysis."""
        os.makedirs(LOG_DIR, exist_ok=True)
        file_exists = os.path.exists(TRADES_CSV_PATH)

        fieldnames = [
            "symbol",
            "entry_dt",
            "exit_dt",
            "entry_price",
            "exit_price",
            "size",
            "pnl",
            "r_multiple",
            "reason",
        ]

        rec = dict(trade_record)
        entry_dt = rec.get("entry_dt")
        exit_dt = rec.get("exit_dt")

        if isinstance(entry_dt, (pd.Timestamp, dt.datetime)):
            rec["entry_dt"] = entry_dt.isoformat()
        if isinstance(exit_dt, (pd.Timestamp, dt.datetime)):
            rec["exit_dt"] = exit_dt.isoformat()

        try:
            with open(TRADES_CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(rec)
        except Exception as e:
            self._log(f"[TRADE-LOG-ERROR] Failed to append trade to CSV: {e}")

    # ------------ time helpers ------------

    @staticmethod
    def _normalize_ts(ts) -> pd.Timestamp:
        """Ensure timestamps are tz-aware UTC."""
        if isinstance(ts, pd.Timestamp):
            out = ts
        else:
            out = pd.Timestamp(ts)

        if out.tzinfo is None:
            out = out.tz_localize("UTC")
        else:
            out = out.tz_convert("UTC")
        return out

    def _ib_now(self) -> pd.Timestamp:
        """Return 'now' from IBKR's perspective as tz-aware UTC."""
        if self.ib is None:
            return self._normalize_ts(pd.Timestamp.utcnow())
        # Using local system clock as IB reference is fine for paper trading.
        return self._normalize_ts(pd.Timestamp.utcnow())

    # ------------ equity / risk helpers ------------

    def _ib_net_liquidation(self) -> Optional[float]:
        """Fetch NetLiquidation from IB account summary."""
        if self.ib is None:
            return None
        try:
            accs = self.ib.accountSummary()
            for row in accs:
                if row.tag == "NetLiquidation":
                    return float(row.value)
        except Exception as e:
            self._log(f"[IB] Failed to fetch NetLiquidation: {e}")
        return None

    def _current_equity(self) -> float:
        """Return current equity estimate (start-of-day + realized PnL)."""
        return self.start_of_day_equity + self.realized_pnl_today

    def _max_daily_loss_reached(self) -> bool:
        """Return True if daily loss exceeds configured fraction."""
        equity_now = self._current_equity()
        drop = (equity_now - self.start_of_day_equity) / self.start_of_day_equity
        if drop <= -DAILY_LOSS_STOP_FRACTION:
            return True
        return False

    def _calc_position_size(self, price: float) -> int:
        """Compute size based on RISK_PER_TRADE_FRACTION and STOP_LOSS_PCT."""
        if price <= 0:
            return 0
        equity = self._current_equity()
        risk_capital = RISK_PER_TRADE_FRACTION * equity
        per_share_risk = STOP_LOSS_PCT * price
        if per_share_risk <= 0:
            return 0
        size = int(risk_capital / per_share_risk)
        return max(size, 0)

    def _position_risk_ok(self, symbol: str, bar_time: pd.Timestamp) -> bool:
        """Check symbol not open, max positions, and cooldown after STOP."""
        if symbol in self.positions:
            return False

        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return False

        last_stop = self.last_stop_bar.get(symbol)
        if last_stop is not None:
            bars_since_stop = (bar_time - last_stop) / dt.timedelta(
                minutes=BAR_INTERVAL_MIN
            )
            if bars_since_stop < COOLDOWN_BARS_AFTER_STOP:
                self._log(
                    f"[COOLDOWN] Skipping {symbol}: {bars_since_stop:.1f} bars since STOP < "
                    f"{COOLDOWN_BARS_AFTER_STOP}."
                )
                return False

        if self._max_daily_loss_reached():
            self._log(
                "[DAILY-STOP] Daily loss limit reached; no new entries will be opened."
            )
            return False

        return True

    # ------------ buffers & features ------------

    @staticmethod
    def _empty_buffer() -> pd.DataFrame:
        """Create an empty OHLCV buffer for a symbol."""
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"], dtype=float
        )

    def _update_buffer_from_bars(
        self, symbol: str, bars: list
    ) -> Optional[pd.Series]:
        """Update buffer with last bar for symbol and return latest row."""
        if not bars:
            return None

        last_bar = bars[-1]
        ts = self._normalize_ts(last_bar.date)
        row = {
            "open": last_bar.open,
            "high": last_bar.high,
            "low": last_bar.low,
            "close": last_bar.close,
            "volume": last_bar.volume,
        }

        buf = self.ohlcv_buffers.get(symbol)
        if buf is None or buf.empty:
            buf = self._empty_buffer()

        buf.loc[ts] = row
        # Keep buffer to a reasonable length (mainly memory control)
        max_len = 500
        if len(buf) > max_len:
            buf = buf.iloc[-max_len:]

        self.ohlcv_buffers[symbol] = buf
        return buf.iloc[-1]

    # ------------ IB connection & setup ------------

    def _connect_ib_and_setup(self) -> None:
        """Connect to IBKR, load model, create contracts, and init buffers."""
        self._log("Connecting to IBKR TWS/Gateway...")
        self.ib = IB()

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                self.ib.connect("127.0.0.1", 7497, clientId=1)
                if self.ib.isConnected():
                    self._log(f"Connected to IBKR on attempt {attempt}.")
                    break
            except ConnectionRefusedError as e:
                self._log(
                    f"IBKR connection refused (attempt {attempt}/{max_retries}). "
                    f"Is TWS/IB Gateway running with API enabled? Error: {e}"
                )
            except Exception as e:
                self._log(
                    f"Unexpected error while connecting to IBKR "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )

            if attempt < max_retries:
                self._log("Sleeping 5 seconds before next connection attempt...")
                time.sleep(5)

        else:
            raise RuntimeError("Could not connect to IBKR after multiple attempts.")

        self._log("Fetching IBKR NetLiquidation for starting equity...")
        net_liq = self._ib_net_liquidation()
        if net_liq is not None:
            self.start_of_day_equity = net_liq
            self._log(
                f"Start-of-day equity from IBKR NetLiquidation: {net_liq:.2f}"
            )
        else:
            self._log(
                f"Could not fetch NetLiquidation; using default "
                f"STARTING_EQUITY={STARTING_EQUITY:.2f}"
            )

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

    def _ensure_ib_connected(self) -> bool:
        """Ensure IBKR connection is alive; attempt quick reconnect if needed."""
        if self.ib is None:
            self._log("[IB] No IB instance; cannot trade.")
            return False

        if self.ib.isConnected():
            return True

        self._log("[IB] Connection appears lost; attempting to reconnect...")
        try:
            try:
                self.ib.disconnect()
            except Exception:
                pass

            self.ib.connect("127.0.0.1", 7497, clientId=1)
            if self.ib.isConnected():
                self._log("[IB] Successfully reconnected to IBKR.")
                return True
        except Exception as e:
            self._log(f"[IB] Reconnect failed: {e}")

        return False

    def connect_and_load(self) -> None:
        """Public wrapper to connect and set up."""
        self._connect_ib_and_setup()

    # ------------ positions & PnL helpers ------------

    def _load_existing_ib_positions(self) -> None:
        """Seed positions from IBKR portfolio for symbols in UNIVERSE."""
        if self.ib is None or not self.ib.isConnected():
            return
        try:
            port = self.ib.positions()
            for p in port:
                sym = p.contract.symbol
                if sym not in UNIVERSE:
                    continue
                pos_size = int(p.position)
                if pos_size == 0:
                    continue
                # We don't know original stop/TP; treat as unmanaged placeholder.
                self._log(
                    f"[EXISTING] Found existing IB position in {sym} (size={pos_size}); "
                    f"tracking without SL/TP."
                )
                # Use last price as entry placeholder; user should flatten manually.
                self.positions[sym] = Position(
                    symbol=sym,
                    size=pos_size,
                    entry_price=float(p.avgCost or 0.0),
                    entry_dt=self._ib_now(),
                    stop_price=0.0,
                    take_profit_price=0.0,
                    p_up=0.0,
                )
        except Exception as e:
            self._log(f"[IB] Failed to load existing positions: {e}")

    # ------------ order helpers ------------

    def _submit_market_order(
        self, symbol: str, size: int
    ) -> Optional[SimpleNamespace]:
        """Submit a market order via ib_insync and wait briefly for fill."""
        contract = self.contracts[symbol]
        action = "BUY" if size > 0 else "SELL"

        self._log(f"[ORDER] Submitting {action} {abs(size)} {symbol} (MKT)...")
        order = MarketOrder(action, abs(size))
        trade = self.ib.placeOrder(contract, order)

        # Wait briefly for fills to arrive
        self.ib.sleep(1)
        elapsed = 0.0
        timeout = 15.0
        while not trade.isDone() and elapsed < timeout:
            self.ib.sleep(0.5)
            elapsed += 0.5

        fills = trade.fills
        if not fills:
            self._log("[ORDER] No fills received before timeout; treating as no fill.")
            return None

        avg_price = sum(f.fillPrice * f.execution.shares for f in fills) / sum(
            f.execution.shares for f in fills
        )

        self._log(
            f"[ORDER-FILLED] {action} {abs(size)} {symbol} avg_price={avg_price:.4f}"
        )

        return SimpleNamespace(
            avg_price=avg_price, size=size, action=action, raw_trade=trade
        )

    def _close_position(
        self,
        symbol: str,
        pos: Position,
        exit_price: float,
        exit_ts: pd.Timestamp,
        reason: str,
    ) -> None:
        """Submit closing order, update PnL, log, and write CSV row."""
        size = pos.size
        fills = self._submit_market_order(symbol, -size)
        if fills is None:
            self._log(
                f"[EXIT-FAILED] {symbol}: attempted close but no fills; leaving position open."
            )
            return

        realized = (fills.avg_price - pos.entry_price) * size
        r_multiple = realized / (STOP_LOSS_PCT * pos.entry_price * size)

        self.realized_pnl_today += realized
        self.last_stop_bar[symbol] = exit_ts if reason == "STOP" else self.last_stop_bar.get(
            symbol
        )

        trade_record = {
            "symbol": symbol,
            "entry_dt": pos.entry_dt,
            "exit_dt": exit_ts,
            "entry_price": pos.entry_price,
            "exit_price": fills.avg_price,
            "size": size,
            "pnl": realized,
            "r_multiple": r_multiple,
            "reason": reason,
        }
        self.trades_today.append(trade_record)
        self._append_trade_to_csv(trade_record)

        self._log(
            f"[EXIT-{reason}] {symbol}: size={size}, entry={pos.entry_price:.4f}, "
            f"exit={fills.avg_price:.4f}, pnl={realized:.2f}, R={r_multiple:.2f}"
        )

        del self.positions[symbol]

    # ------------ main trading logic ------------

    def _maybe_roll_trading_date(self, now_ts: pd.Timestamp) -> None:
        """Reset daily stats when calendar day changes."""
        today = now_ts.date()
        if self.current_trading_date is None:
            self.current_trading_date = today
            return

        if today != self.current_trading_date:
            # Log summary for previous day
            self._log_daily_summary()
            # Reset for new day
            self.current_trading_date = today
            self.realized_pnl_today = 0.0
            self.trades_today = []
            eq = self._ib_net_liquidation()
            if eq is not None:
                self.start_of_day_equity = eq

    def _log_daily_summary(self) -> None:
        """Log summary of today's trades and performance."""
        if not self.trades_today:
            self._log("[SUMMARY] No trades today.")
            return

        total_pnl = sum(t["pnl"] for t in self.trades_today)
        n_trades = len(self.trades_today)
        wins = [t for t in self.trades_today if t["pnl"] > 0]
        losses = [t for t in self.trades_today if t["pnl"] < 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        max_win = max((t["pnl"] for t in self.trades_today), default=0.0)
        max_loss = min((t["pnl"] for t in self.trades_today), default=0.0)

        self._log(
            f"[SUMMARY] date={self.current_trading_date}, n_trades={n_trades}, "
            f"total_pnl={total_pnl:.2f}, win_rate={win_rate:.3f}, "
            f"max_win={max_win:.2f}, max_loss={max_loss:.2f}"
        )

    def _poll_bars_once(self) -> None:
        """Fetch latest bars for all symbols, manage exits, then entries."""
        now_ts = self._ib_now()
        self._maybe_roll_trading_date(now_ts)

        # === 1) Fetch latest bar for each symbol ===
        for sym in UNIVERSE:
            contract = self.contracts[sym]
            try:
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=f"{BAR_INTERVAL_MIN * 2} M",
                    barSizeSetting=f"{BAR_INTERVAL_MIN} mins",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
            except Exception as e:
                self._log(f"[BAR-ERROR] {sym}: reqHistoricalData failed: {e}")
                continue

            if not bars:
                self._log(f"[BAR] {sym}: no bars returned from IBKR.")
                continue

            last = self._update_buffer_from_bars(sym, bars)
            if last is None:
                self._log(f"[BAR] {sym}: buffer update returned None.")
                continue

            buf = self.ohlcv_buffers.get(sym)
            buf_len = len(buf) if buf is not None else 0
            self._log(
                f"[BAR] {sym}: last_close={float(last['close']):.4f}, "
                f"buffer_len={buf_len}"
            )

        # === 2) Exits first: use the latest bar close/high/low ===
        for sym, pos in list(self.positions.items()):
            buf = self.ohlcv_buffers.get(sym)
            if buf is None or buf.empty:
                continue
            last_ts = buf.index[-1]
            last_row = buf.iloc[-1]
            close = float(last_row["close"])
            high = float(last_row["high"])
            low = float(last_row["low"])

            # Stop-loss if low breaches stop
            if low <= pos.stop_price:
                self._log(
                    f"[EXIT-CHECK] {sym}: STOP hit (low={low:.4f} <= SL={pos.stop_price:.4f})"
                )
                self._close_position(sym, pos, pos.stop_price, last_ts, reason="STOP")
                continue

            # Take-profit if high breaches TP
            if high >= pos.take_profit_price:
                self._log(
                    f"[EXIT-CHECK] {sym}: TP hit (high={high:.4f} >= TP={pos.take_profit_price:.4f})"
                )
                self._close_position(sym, pos, pos.take_profit_price, last_ts, reason="TP")
                continue

            # Time stop: max bars in trade
            bars_held = (last_ts - pos.entry_dt) / dt.timedelta(minutes=BAR_INTERVAL_MIN)
            if bars_held >= MAX_BARS_IN_TRADE:
                self._log(
                    f"[EXIT-CHECK] {sym}: MAX_BARS reached ({bars_held:.1f} >= {MAX_BARS_IN_TRADE})"
                )
                self._close_position(sym, pos, close, last_ts, reason="MAX_BARS")
                continue

        # === 3) Entries (only if daily loss not hit) ===
        if self._max_daily_loss_reached():
            self._log("[DAILY-STOP] Max daily loss reached; skipping new entries.")
            return

        for sym in UNIVERSE:
            if sym in self.positions:
                continue

            buf = self.ohlcv_buffers.get(sym)
            if buf is None:
                self._log(f"[ENTRY-SKIP] {sym}: no buffer yet.")
                continue
            if len(buf) < MIN_BARS_FOR_FEATURES:
                self._log(
                    f"[ENTRY-SKIP] {sym}: buffer_len={len(buf)} < {MIN_BARS_FOR_FEATURES} (warmup)."
                )
                continue  # need enough history for features

            # Build features using same logic as training/backtest
            panel = build_features_for_symbol(buf)
            if panel.empty:
                self._log(f"[ENTRY-SKIP] {sym}: feature panel empty.")
                continue
            row = panel.iloc[-1]

            # Predict using 1-row DataFrame to preserve feature names
            x = row[FEATURE_COLUMNS].to_frame().T
            p_up = float(self.clf.predict_proba(x)[0, 1])

            self._log(
                f"[SIGNAL] {sym}: p_up={p_up:.3f}, "
                f"threshold={P_UP_ENTRY_THRESHOLD:.3f}"
            )

            if p_up < P_UP_ENTRY_THRESHOLD:
                self._log(f"[ENTRY-SKIP] {sym}: p_up below threshold.")
                continue

            bar_time = buf.index[-1]
            if not self._position_risk_ok(sym, bar_time):
                self._log(f"[ENTRY-SKIP] {sym}: position risk check failed.")
                continue

            price = float(row["close"])
            size = self._calc_position_size(price)
            if size <= 0:
                self._log(
                    f"[ENTRY-SKIP] {sym}: calc size <= 0 at price={price:.4f}."
                )
                continue

            fills = self._submit_market_order(sym, size)
            if fills is None:
                self._log(f"[ENTRY-SKIP] {sym}: market order returned no fills.")
                continue

            entry_price = fills.avg_price
            stop_price = entry_price * (1.0 - STOP_LOSS_PCT)
            tp_price = entry_price * (1.0 + TAKE_PROFIT_PCT)

            pos = Position(
                symbol=sym,
                size=size,
                entry_price=entry_price,
                entry_dt=bar_time,
                stop_price=stop_price,
                take_profit_price=tp_price,
                p_up=p_up,
            )
            self.positions[sym] = pos

            self._log(
                f"[ENTRY] {sym}: size={size}, entry={entry_price:.4f}, "
                f"SL={stop_price:.4f}, TP={tp_price:.4f}, p_up={p_up:.3f}"
            )

    def _log_open_positions(self) -> None:
        """Log current open positions for monitoring."""
        if not self.positions:
            self._log("[POSITIONS] No open positions.")
            return

        for sym, pos in self.positions.items():
            self._log(
                f"[POSITIONS] {sym}: size={pos.size}, entry={pos.entry_price:.4f}, "
                f"SL={pos.stop_price:.4f}, TP={pos.take_profit_price:.4f}, "
                f"p_up={pos.p_up:.3f}"
            )

    # ------------ main loop ------------

    def run(self) -> None:
        """Main loop: RTH-only trading; hourly heartbeat outside RTH with loop timing."""
        self._log("Initializing paper trader...")
        self.connect_and_load()
        self._load_existing_ib_positions()

        self._log(
            f"Starting polling bar loop (interval={BAR_INTERVAL_MIN} minutes, RTH-only)..."
        )

        in_rth_last = False

        while True:
            loop_start = dt.datetime.now()
            in_rth_now = is_rth_now()

            if not in_rth_now:
                # Outside regular trading hours:
                #  - NO market data pulls
                #  - Heartbeat only once per hour (or less if close to open)
                minutes_to_open = minutes_until_next_rth_open()

                if minutes_to_open > HEARTBEAT_INTERVAL_MIN_OUTSIDE_RTH:
                    sleep_minutes = HEARTBEAT_INTERVAL_MIN_OUTSIDE_RTH
                else:
                    # Within last hour: sleep straight into the open (min 1 minute)
                    sleep_minutes = max(1.0, minutes_to_open)

                heartbeat_logger.info(
                    f"[HEARTBEAT] Outside RTH; no trading. "
                    f"Next RTH open in ~{minutes_to_open:.1f} minutes. "
                    f"Sleeping for {sleep_minutes:.1f} minutes..."
                )
                in_rth_last = False

                if self.ib is not None and self.ib.isConnected():
                    sleep_start = dt.datetime.now()
                    self.ib.sleep(sleep_minutes * 60)
                    sleep_end = dt.datetime.now()
                    heartbeat_logger.info(
                        f"[HEARTBEAT] Slept {(sleep_end - sleep_start).total_seconds():.1f}s outside RTH."
                    )
                else:
                    time.sleep(sleep_minutes * 60)
                continue

            # === Inside regular trading hours (RTH) ===
            work_start = dt.datetime.now()

            # Ensure we have a live IB connection; no heartbeat logs here.
            if not self._ensure_ib_connected():
                self._log(
                    "[IB] Not connected during RTH; will retry after short sleep."
                )
                time.sleep(15)
                continue

            if not in_rth_last:
                # First iteration after entering RTH for the day.
                self._log("Entered regular trading hours; trading enabled.")
                in_rth_last = True

            try:
                # Market data + trades only happen inside RTH.
                self._poll_bars_once()
                self._log_open_positions()
            except Exception as e:
                self._log(f"[ERROR] Exception in polling loop: {e}")
                in_rth_last = False

            work_end = dt.datetime.now()
            work_sec = (work_end - work_start).total_seconds()
            self._log(f"[LOOP] Work (poll + positions) took {work_sec:.1f}s this cycle.")

            self._log(
                f"[POLL] Sleeping for {BAR_INTERVAL_MIN} minutes before next poll..."
            )
            sleep_start = dt.datetime.now()
            self.ib.sleep(BAR_INTERVAL_MIN * 60)
            sleep_end = dt.datetime.now()
            sleep_sec = (sleep_end - sleep_start).total_seconds()

            loop_end = sleep_end
            total_sec = (loop_end - loop_start).total_seconds()
            self._log(
                f"[LOOP] Slept {sleep_sec:.1f}s; total loop time {total_sec:.1f}s."
            )


def main() -> None:
    """Instantiate a PaperTrader and run the live loop."""
    trader = PaperTrader()
    trader.run()


if __name__ == "__main__":
    main()

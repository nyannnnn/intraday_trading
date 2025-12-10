"""
Paper trading strategy (IBKR, 5-min bars):

- Data: polls 5-min TRADES bars for UNIVERSE via reqHistoricalData.
- Entry: ML classifier; open long if p_up >= P_UP_ENTRY_THRESHOLD and risk checks pass.
- Exit: stop-loss on close, intrabar take-profit on high, or MAX_BARS_IN_TRADE.
- Risk: size = RISK_PER_TRADE_FRACTION * equity / price, per-symbol cooldown after STOP,
  and DAILY_LOSS_STOP_FRACTION to halt new entries.
"""
import csv
import datetime as dt
import logging
import os
import smtplib
import time
import traceback
from dataclasses import dataclass
from email.mime.text import MIMEText
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from ib_insync import IB, MarketOrder, Stock, LimitOrder, StopOrder

# =================
# Project config
# =================
try:
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
except Exception as e:
    raise RuntimeError(f"Failed to import config or model modules: {e}")

# ============
# Constants
# ============
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "paper_trade.log")
TRADES_CSV_PATH = os.path.join(LOG_DIR, "paper_trades.csv")

STARTING_EQUITY = 100_000.0
MAX_BUFFER_LENGTH = 500  # number of 5-min bars to keep in memory
ORDER_WAIT_TIMEOUT_SEC = 15  # how long to wait for order fills

# Email alert config (optional; no email if TRADER_ALERT_EMAIL_TO is unset)
ALERT_EMAIL_TO = os.environ.get("TRADER_ALERT_EMAIL_TO")
ALERT_EMAIL_FROM = os.environ.get("TRADER_ALERT_EMAIL_FROM")
SMTP_HOST = os.environ.get("TRADER_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("TRADER_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("TRADER_SMTP_USER")
SMTP_PASS = os.environ.get("TRADER_SMTP_PASS")

# Trading hours (Regular Trading Hours window for polling / entries)
RTH_START = dt.time(9, 0)
RTH_END = dt.time(16, 30)
US_EQUITY_OPEN = dt.time(9, 30)
US_EQUITY_OPEN_WARMUP_END = dt.time(9, 45)

try:
    RTH_TZ = ZoneInfo("America/New_York")
except Exception:
    RTH_TZ = None

HEARTBEAT_INTERVAL_MIN_OUTSIDE_RTH = 60

# ============
# Logging setup
# ============
logger = logging.getLogger("paper_trade")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_console)

_file = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3)
_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_file)

heartbeat_logger = logging.getLogger("heartbeat")
heartbeat_logger.setLevel(logging.INFO)
heartbeat_logger.handlers.clear()
hb_file = RotatingFileHandler(os.path.join(LOG_DIR, "heartbeat.log"), maxBytes=500_000, backupCount=2)
hb_file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
heartbeat_logger.addHandler(hb_file)


# ============
# Helpers
# ============
def is_rth_now() -> bool:
    if RTH_TZ is not None:
        now = dt.datetime.now(RTH_TZ)
    else:
        now = dt.datetime.now()
    return RTH_START <= now.time() <= RTH_END


def minutes_until_next_rth_open() -> float:
    if RTH_TZ is not None:
        now = dt.datetime.now(RTH_TZ)
    else:
        now = dt.datetime.now()

    today_open = dt.datetime.combine(now.date(), RTH_START)
    if RTH_TZ is not None:
        today_open = today_open.replace(tzinfo=RTH_TZ)

    if now <= today_open:
        delta = today_open - now
        return max(delta.total_seconds() / 60.0, 0.0)

    tomorrow = now.date() + dt.timedelta(days=1)
    next_open = dt.datetime.combine(tomorrow, RTH_START)
    if RTH_TZ is not None:
        next_open = next_open.replace(tzinfo=RTH_TZ)
    delta = next_open - now
    return max(delta.total_seconds() / 60.0, 0.0)


@dataclass
class Position:
    """Single open long position with entry, size, and risk levels."""
    symbol: str
    size: int
    entry_price: float
    entry_dt: pd.Timestamp


class PaperTrader:
    def __init__(self) -> None:
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
        # NEW: prevent double-closing / double-logging the same trade
        self.closed_trade_keys: set[tuple] = set()

    @staticmethod
    def _log(msg: str) -> None:
        logger.info(msg)

    def _send_email_alert(self, subject: str, body: str) -> None:
        """Send a simple text email if TRADER_ALERT_EMAIL_TO is configured."""
        if not ALERT_EMAIL_TO:
            return
        try:
            msg = MIMEText(body)
            from_addr = ALERT_EMAIL_FROM or ALERT_EMAIL_TO
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = ALERT_EMAIL_TO

            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                if SMTP_USER and SMTP_PASS:
                    server.starttls()
                    server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
            self._log(f"[ALERT] Sent email to {ALERT_EMAIL_TO}.")
        except Exception as e:
            self._log(f"[EMAIL-ERROR] Failed to send email: {e}")

    def _append_trade_to_csv(self, trade_record: dict) -> None:
        """Append a single trade to trades CSV, skipping duplicates by key."""
        os.makedirs(LOG_DIR, exist_ok=True)
        file_exists = os.path.exists(TRADES_CSV_PATH)
        fieldnames = [
            "symbol", "entry_dt", "exit_dt", "entry_price", "exit_price",
            "size", "pnl", "r_multiple", "reason",
        ]

        rec = dict(trade_record)
        entry_dt = rec.get("entry_dt")
        exit_dt = rec.get("exit_dt")

        # Normalize timestamps to ISO strings for stable comparison
        if isinstance(entry_dt, (pd.Timestamp, dt.datetime)):
            rec["entry_dt"] = entry_dt.isoformat()
        if isinstance(exit_dt, (pd.Timestamp, dt.datetime)):
            rec["exit_dt"] = exit_dt.isoformat()

        # Build a simple fingerprint for duplicate detection
        key_fields = ("symbol", "entry_dt", "entry_price", "exit_price", "size", "reason")
        new_key = tuple(str(rec.get(k, "")) for k in key_fields)

        # 1) If file exists, scan to avoid duplicate rows
        if file_exists:
            try:
                with open(TRADES_CSV_PATH, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_key = tuple(str(row.get(k, "")) for k in key_fields)
                        if existing_key == new_key:
                            self._log(
                                f"[TRADE-LOG-SKIP] Duplicate trade detected for {rec['symbol']}, not appending to CSV."
                            )
                            return
            except Exception as e:
                self._log(f"[TRADE-LOG-ERROR] Failed to check CSV for duplicates: {e}")

        # 2) Append new row
        try:
            with open(TRADES_CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(rec)
        except Exception as e:
            self._log(f"[TRADE-LOG-ERROR] Failed to append trade to CSV: {e}")

    @staticmethod
    def _normalize_ts(ts) -> pd.Timestamp:
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
        return self._normalize_ts(pd.Timestamp.utcnow())

    def _is_within_open_warmup(self, now_ts: pd.Timestamp) -> bool:
        if RTH_TZ is None:
            return False
        local = now_ts.tz_convert(RTH_TZ)
        open_dt = dt.datetime.combine(local.date(), US_EQUITY_OPEN).replace(tzinfo=RTH_TZ)
        warmup_end_dt = dt.datetime.combine(local.date(), US_EQUITY_OPEN_WARMUP_END).replace(tzinfo=RTH_TZ)
        return open_dt <= local <= warmup_end_dt

    def _current_equity(self) -> float:
        try:
            if self.ib is None or not self.ib.isConnected():
                return self.start_of_day_equity
            acc_vals = self.ib.accountValues()
            for v in acc_vals:
                if v.tag == "NetLiquidation":
                    return float(v.value)
        except Exception as e:
            self._log(f"[EQUITY-ERROR] {e}")
        return self.start_of_day_equity

    def _daily_loss_limit_hit(self) -> bool:
        eq = self._current_equity()
        drop = (eq - self.start_of_day_equity) / max(self.start_of_day_equity, 1.0)
        return drop <= -DAILY_LOSS_STOP_FRACTION

    def _calc_position_size(self, price: float) -> int:
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
        # 1. No duplicate positions per symbol
        if symbol in self.positions:
            return False

        # 2. Daily loss stop
        if self._daily_loss_limit_hit():
            self._log("[RISK] Daily loss limit hit, not opening new positions.")
            return False

        # 3. Avoid entering too close to market close
        if RTH_TZ is not None:
            local = bar_time.tz_convert(RTH_TZ)
            close_dt = dt.datetime.combine(local.date(), RTH_END).replace(tzinfo=RTH_TZ)
            if (close_dt - local) <= dt.timedelta(minutes=10):
                self._log(f"[RISK] Skipping entries near close for {symbol}.")
                return False

        # 4. Max concurrent positions
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            self._log(f"[RISK] Max concurrent positions reached ({MAX_CONCURRENT_POSITIONS})")
            return False

        # 5. Cooldown after STOP (simple 3-bar cooldown)
        last_stop_ts = self.last_stop_bar.get(symbol)
        if last_stop_ts is not None:
            if (bar_time - last_stop_ts) < dt.timedelta(minutes=BAR_INTERVAL_MIN * 3):
                self._log(f"[RISK] {symbol} in cooldown after STOP.")
                return False

        return True

    def _empty_buffer(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def connect_and_load(self) -> None:
        self.ib = IB()
        for attempt in range(1, 6):
            try:
                self._log(f"Connecting to IBKR TWS/Gateway... attempt {attempt}")
                self.ib.connect("127.0.0.1", 7497, clientId=1)
                self._log(f"Connected to IBKR on attempt {attempt}.")
                break
            except Exception as e:
                self._log(f"IBKR connection attempt {attempt} failed: {e}")
                time.sleep(5)
        else:
            raise RuntimeError("Could not connect to IBKR after multiple attempts.")

        # Get start-of-day equity
        self._log("Fetching IBKR NetLiquidation for starting equity...")
        try:
            acc_vals = self.ib.accountValues()
            for v in acc_vals:
                if v.tag == "NetLiquidation":
                    self.start_of_day_equity = float(v.value)
                    break
            self._log(f"Start-of-day equity from IBKR NetLiquidation: {self.start_of_day_equity:.2f}")
        except Exception as e:
            self._log(f"[WARN] Could not fetch NetLiquidation, using default {STARTING_EQUITY:.2f}: {e}")

        # Load ML model
        self._log("Loading latest classifier...")
        self.clf = load_latest_classifier(MODEL_DIR)
        self._log("Classifier loaded.")

        # Build IB contracts
        self._log(f"Creating contracts for universe: {UNIVERSE}")
        for sym in UNIVERSE:
            self.contracts[sym] = Stock(sym, "SMART", "USD")
        self._log("Qualifying contracts...")
        self.ib.qualifyContracts(*self.contracts.values())

        # Initialize buffers & backfill
        self._log("Initializing buffers...")
        for sym in UNIVERSE:
            self.ohlcv_buffers[sym] = self._empty_buffer()
        self._backfill_buffers_on_startup()

    def _ensure_ib_connected(self) -> bool:
        if self.ib is None:
            return False
        if self.ib.isConnected():
            return True

        try:
            self._log("[IB] Disconnected, trying to reconnect...")
            self.ib.connect("127.0.0.1", 7497, clientId=1)
            if self.ib.isConnected():
                self._log("[IB] Reconnected successfully.")
                return True
        except Exception as e:
            self._log(f"[IB-ERROR] Reconnect failed: {e}")
        return False

    def _backfill_symbol(self, symbol: str, end_dt: dt.datetime, bars: int = MAX_BUFFER_LENGTH) -> None:
        contract = self.contracts[symbol]
        duration = f"{bars * BAR_INTERVAL_MIN} M"
        bar_size = f"{BAR_INTERVAL_MIN} mins"

        self._log(f"[BACKFILL] {symbol}: requesting {duration} of {bar_size} ending {end_dt}.")
        bars_data = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars_data:
            self._log(f"[BACKFILL] {symbol}: no data returned.")
            return

        rows = []
        for b in bars_data:
            rows.append(
                {
                    "datetime": self._normalize_ts(pd.Timestamp(b.date)),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
            )
        df = pd.DataFrame(rows).set_index("datetime")
        df = df[~df.index.duplicated(keep="last")].sortIndex()
        self.ohlcv_buffers[symbol] = df.tail(MAX_BUFFER_LENGTH)
        self._log(f"[BACKFILL] {symbol}: buffer now {len(df)} rows.")

    def _backfill_buffers_on_startup(self) -> None:
        now = dt.datetime.utcnow()
        for sym in UNIVERSE:
            try:
                self._backfill_symbol(sym, now)
            except Exception as e:
                self._log(f"[BACKFILL-ERROR] {sym}: {e}")

    def _load_existing_ib_positions(self) -> None:
        if self.ib is None:
            return
        try:
            open_positions = self.ib.positions()
        except Exception as e:
            self._log(f"[IB-POSITIONS-ERROR] {e}")
            return

        if not open_positions:
            self._log("[RESCUE] No existing IBKR positions at startup.")
            return

        for p in open_positions:
            sym = p.contract.symbol
            if sym not in UNIVERSE:
                continue
            avg_price = float(p.avgCost)
            size = int(p.position)
            if size == 0:
                continue
            pos = Position(
                symbol=sym,
                size=size,
                entry_price=avg_price,
                entry_dt=self._ib_now(),
            )
            self.positions[sym] = pos
            self._log(f"[RESCUE] {sym}: restored position size={size} entry_price={avg_price:.4f}")

    def _poll_bars_once(self) -> None:
        if not self._ensure_ib_connected():
            raise RuntimeError("IB connection lost and could not reconnect.")
        now_ts = self._ib_now()
        self._maybe_roll_trading_date(now_ts)

        for sym in UNIVERSE:
            try:
                self._poll_symbol(sym)
            except Exception as e:
                self._log(f"[POLL-ERROR] {sym}: {e}")

    def _poll_symbol(self, symbol: str) -> None:
        contract = self.contracts[symbol]
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{BAR_INTERVAL_MIN} M",
            barSizeSetting=f"{BAR_INTERVAL_MIN} mins",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return

        b = bars[-1]
        bar_ts = self._normalize_ts(pd.Timestamp(b.date))
        buf = self.ohlcv_buffers[symbol]
        if bar_ts in buf.index:
            return

        new_row = pd.DataFrame(
            {
                "open": [b.open],
                "high": [b.high],
                "low": [b.low],
                "close": [b.close],
                "volume": [b.volume],
            },
            index=[bar_ts],
        )
        buf = pd.concat([buf, new_row])
        buf = buf[~buf.index.duplicated(keep="last")].sort_index()
        self.ohlcv_buffers[symbol] = buf.tail(MAX_BUFFER_LENGTH)

        if bar_ts.tzinfo is None:
            self._log(f"[WARN] Bar {symbol} has no tzinfo: {bar_ts}")
        self._maybe_enter_or_manage(symbol, bar_ts)

    def _submit_market_order(self, symbol: str, size: int) -> Optional[SimpleNamespace]:
        contract = self.contracts[symbol]
        action = "BUY" if size > 0 else "SELL"
        order = MarketOrder(action, abs(size))
        trade = self.ib.placeOrder(contract, order)
        self._log(f"[ORDER] {action} {abs(size)} {symbol} as market order submitted.")

        elapsed = 0.0
        while not trade.isDone() and elapsed < ORDER_WAIT_TIMEOUT_SEC:
            self.ib.waitOnUpdate(timeout=1)
            elapsed += 0.5
        fills = trade.fills
        if not fills:
            self._log("[ORDER] No fills received.")
            return None
        total_shares = sum(abs(f.execution.shares) for f in fills)
        if total_shares == 0:
            return None
        avg_price = sum(f.execution.price * abs(f.execution.shares) for f in fills) / total_shares
        self._log(f"[ORDER-FILLED] {action} {abs(size)} {symbol} avg={avg_price:.4f}")
        return SimpleNamespace(avg_price=avg_price, size=size, action=action, raw_trade=trade)

    def _place_bracket_order(self, symbol: str, size: int, p_up: float, current_price: float, bar_time: pd.Timestamp):
        contract = self.contracts[symbol]
        stop_price = round(current_price * (1.0 - STOP_LOSS_PCT), 2)
        tp_price = round(current_price * (1.0 + TAKE_PROFIT_PCT), 2)

        parent = LimitOrder("BUY", size, current_price, transmit=False)
        parent.orderType = "MKT"
        parent.tif = "DAY"

        take_profit = LimitOrder("SELL", size, tp_price, parentId=parent.orderId, transmit=False)
        stop_loss = StopOrder("SELL", size, stop_price, parentId=parent.orderId, transmit=True)

        self.ib.qualifyContracts(contract)
        trade_parent = self.ib.placeOrder(contract, parent)
        trade_tp = self.ib.placeOrder(contract, take_profit)
        trade_sl = self.ib.placeOrder(contract, stop_loss)

        self._log(
            f"[ENTRY-BRACKET] {symbol} sent. Size={size} EstPrice={current_price:.2f} "
            f"SL={stop_price} TP={tp_price} p_up={p_up:.3f}"
        )

        return SimpleNamespace(
            parent=trade_parent,
            take_profit=trade_tp,
            stop_loss=trade_sl,
            stop_price=stop_price,
            take_profit_price=tp_price,
            p_up=p_up,
        )

    def _maybe_enter_or_manage(self, symbol: str, bar_time: pd.Timestamp) -> None:
        buf = self.ohlcv_buffers[symbol]
        if len(buf) < 20:
            return

        # Gate by RTH and daily loss stop
        if not is_rth_now():
            return
        if self._daily_loss_limit_hit():
            return

        try:
            panel = build_features_for_symbol(buf)
        except Exception as e:
            self._log(f"[FEATURE-ERROR] {symbol}: {e}")
            return

        if panel.empty:
            return

        row = panel.iloc[-1]
        feat_vals = row[FEATURE_COLUMNS]
        if not np.isfinite(feat_vals.values.astype(float)).all():
            return

        x = feat_vals.to_frame().T
        p_up = float(self.clf.predict_proba(x)[0, 1])

        if p_up > 0.5:
            self._log(f"[SIGNAL] {symbol}: p_up={p_up:.3f}")

        # ========== Entry logic ==========
        if symbol not in self.positions:
            if p_up < P_UP_ENTRY_THRESHOLD:
                return
            if not self._position_risk_ok(symbol, bar_time):
                return

            current_price = float(buf["close"].iloc[-1])
            size = self._calc_position_size(current_price)
            if size <= 0:
                return

            bracket = self._place_bracket_order(symbol, size, p_up, current_price, bar_time)
            pos = Position(
                symbol=symbol,
                size=size,
                entry_price=current_price,
                entry_dt=bar_time,
            )
            self.positions[symbol] = pos
            self._log(
                f"[POSITION-OPEN] {symbol}: size={size}, entry={current_price:.4f}, "
                f"SL={bracket.stop_price}, TP={bracket.take_profit_price}"
            )
            return

        # ========== Exit logic (fallback, in case bracket doesn't fire) ==========
        pos = self.positions[symbol]
        stop_price = pos.entry_price * (1.0 - STOP_LOSS_PCT)
        tp_price = pos.entry_price * (1.0 + TAKE_PROFIT_PCT)

        current_price = float(buf["close"].iloc[-1])
        if current_price <= stop_price:
            self._log(f"[STOP-HIT] {symbol} at price {current_price:.4f}")
            self._close_position(symbol, pos, current_price, bar_time, reason="STOP")
        elif current_price >= tp_price:
            self._log(f"[TAKE-PROFIT] {symbol} at price {current_price:.4f}")
            self._close_position(symbol, pos, current_price, bar_time, reason="TAKE_PROFIT")

    def _close_position(self, symbol: str, pos: Position, exit_price: float, exit_ts: pd.Timestamp, reason: str) -> None:
        """
        Close a position once (idempotent), cancel any pending orders, and log the result.
        """
        # --- 0) Idempotency guard: skip if we've already closed this economic trade ---
        trade_key = (
            symbol,
            float(pos.entry_price),
            pos.entry_dt.isoformat() if hasattr(pos.entry_dt, "isoformat") else str(pos.entry_dt),
            int(pos.size),
        )
        if trade_key in self.closed_trade_keys:
            self._log(f"[EXIT-SKIP] {symbol}: trade already closed, skipping duplicate.")
            return
        self.closed_trade_keys.add(trade_key)

        # 1. Cancel any open orders for this symbol (cleanup bracket children)
        for t in self.ib.openTrades():
            if t.contract.symbol == symbol:
                self.ib.cancelOrder(t.order)

        # 2. If the exit reason is NOT 'BRACKET_HIT', we need to send a market sell.
        if reason != "BRACKET_HIT":
            size = pos.size
            fills = self._submit_market_order(symbol, -size)
            if fills is None:
                self._log(f"[EXIT-FAILED] {symbol}: no fills on close.")
                return
            realized_price = fills.avg_price
        else:
            realized_price = exit_price

        # 3. Calculate PnL (R-multiple based on fixed % stop distance)
        realized = (realized_price - pos.entry_price) * pos.size
        r_multiple = realized / (STOP_LOSS_PCT * pos.entry_price * pos.size)
        self.realized_pnl_today += realized

        if reason == "STOP":
            self.last_stop_bar[symbol] = exit_ts

        # 4. Log & record trade
        trade_record = {
            "symbol": symbol,
            "entry_dt": pos.entry_dt,
            "exit_dt": exit_ts,
            "entry_price": pos.entry_price,
            "exit_price": realized_price,
            "size": pos.size,
            "pnl": realized,
            "r_multiple": r_multiple,
            "reason": reason,
        }
        self.trades_today.append(trade_record)
        self._append_trade_to_csv(trade_record)
        self._log(f"[EXIT-{reason}] {symbol}: pnl={realized:.2f}, R={r_multiple:.2f}")

        # 5. Remove from local tracking
        if symbol in self.positions:
            del self.positions[symbol]

    def _maybe_roll_trading_date(self, now_ts: pd.Timestamp) -> None:
        today = now_ts.date()
        if self.current_trading_date is None:
            self.current_trading_date = today
            return
        if today != self.current_trading_date:
            self._log_daily_summary()
            self.start_of_day_equity = self._current_equity()
            self._log(f"[ROLL] New day {today}; SOD Equity={self.start_of_day_equity:.2f}")
            self.current_trading_date = today
            self.realized_pnl_today = 0.0
            self.trades_today = []

    def _log_daily_summary(self) -> None:
        if not self.trades_today:
            self._log("[SUMMARY] No trades today.")
            return
        total_pnl = sum(t["pnl"] for t in self.trades_today)
        avg_pnl = total_pnl / max(len(self.trades_today), 1)
        self._log(f"[SUMMARY] Trades={len(self.trades_today)} TotalPnL={total_pnl:.2f} AvgPnL={avg_pnl:.2f}")

        body_lines = [
            f"Date: {self.current_trading_date}",
            f"Start-of-day equity: {self.start_of_day_equity:.2f}",
            f"Realized PnL today: {self.realized_pnl_today:.2f}",
            f"Number of trades: {len(self.trades_today)}",
            "",
            "Trades:",
        ]
        for t in self.trades_today:
            body_lines.append(
                f"{t['symbol']} {t['entry_dt']} -> {t['exit_dt']} "
                f"pnl={t['pnl']:.2f} R={t['r_multiple']:.2f} reason={t['reason']}"
            )
        body = "\n".join(body_lines)
        self._send_email_alert(
            subject=f"Paper Trade Summary {self.current_trading_date}",
            body=body,
        )

    def _log_open_positions(self) -> None:
        """Log open positions using last bar close as mark price for open PnL."""
        if not self.positions:
            self._log("[POSITIONS] None")
            return

        for sym, pos in self.positions.items():
            buf = self.ohlcv_buffers.get(sym)
            if buf is not None and not buf.empty:
                last_price = float(buf["close"].iloc[-1])
            else:
                # Fallback: no recent bar, assume flat PnL
                last_price = float(pos.entry_price)

            open_pnl = (last_price - pos.entry_price) * pos.size
            self._log(
                f"[POSITIONS] {sym}: size={pos.size}, last={last_price:.2f}, open_pnl={open_pnl:.2f}"
            )

    def run(self) -> None:
        self._log("Initializing paper trader...")
        self.connect_and_load()
        self._load_existing_ib_positions()
        self._log("Starting polling loop...")

        while True:
            if not is_rth_now():
                min_to_open = minutes_until_next_rth_open()
                sleep_min = min(min_to_open, HEARTBEAT_INTERVAL_MIN_OUTSIDE_RTH)
                sleep_min = max(1.0, sleep_min)
                heartbeat_logger.info(f"[HEARTBEAT] Outside RTH. Sleeping {sleep_min:.1f} min.")
                if self.ib and self.ib.isConnected():
                    self.ib.sleep(sleep_min * 60)
                else:
                    time.sleep(sleep_min * 60)
                continue

            if not self._ensure_ib_connected():
                heartbeat_logger.info("[HEARTBEAT] IB disconnected, retrying in 60s.")
                time.sleep(60)
                continue

            start = dt.datetime.now()
            try:
                self._poll_bars_once()
                self._log_open_positions()
            except Exception as e:
                self._log(f"[ERROR] Polling loop: {e}")

            elapsed = (dt.datetime.now() - start).total_seconds()
            sleep_sec = max(1.0, (BAR_INTERVAL_MIN * 60) - elapsed)
            self._log(f"[LOOP] Done in {elapsed:.1f}s. Sleeping {sleep_sec:.1f}s.")
            self.ib.sleep(sleep_sec)


def main() -> None:
    trader = PaperTrader()
    try:
        trader.run()
    except KeyboardInterrupt:
        trader._log("Shutting down due to KeyboardInterrupt.")
    except Exception as e:
        trader._log(f"[FATAL] {e}")
        trader._log(traceback.format_exc())


if __name__ == "__main__":
    main()

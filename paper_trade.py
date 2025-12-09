from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, MarketOrder, LimitOrder, StopOrder
import logging
import os
import csv
import random
import time
import json
import smtplib
import traceback
from logging.handlers import RotatingFileHandler
from email.mime.text import MIMEText

# === Config Imports ===
from config import (
    UNIVERSE, MODEL_DIR, FEATURE_COLUMNS, BAR_INTERVAL_MIN,
    P_UP_ENTRY_THRESHOLD, RISK_PER_TRADE_FRACTION,
    ATR_WINDOW, ATR_STOP_MULT, ATR_TP_MULT,
    MAX_CONCURRENT_POSITIONS, DAILY_LOSS_STOP_FRACTION,
    MAX_BARS_IN_TRADE, COOLDOWN_BARS_AFTER_STOP
)
from quant.quant_model import build_features_for_symbol
from backtesting import load_latest_classifier

# === Safety Constraints ===
MAX_NOTIONAL_PER_TRADE = 25000.0
MAX_SHARES_PER_TRADE = 2000

# === Persistence ===
STATE_FILE = "trade_state.json"

@dataclass
class Position:
    """Tracks state of a live trade."""
    symbol: str
    entry_dt: dt.datetime
    size: int
    entry_price: float
    stop_price: float
    take_profit_price: float
    p_up: float
    bars_held: int

    def to_dict(self):
        d = asdict(self)
        d['entry_dt'] = self.entry_dt.isoformat()
        return d

    @staticmethod
    def from_dict(d):
        d['entry_dt'] = dt.datetime.fromisoformat(d['entry_dt'])
        return Position(**d)

def save_state_to_disk(positions: Dict[str, Position]):
    try:
        data = {sym: pos.to_dict() for sym, pos in positions.items()}
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"[STATE] Failed to save state: {e}")

def load_state_from_disk() -> Dict[str, Position]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        return {sym: Position.from_dict(d) for sym, d in data.items()}
    except Exception as e:
        logging.error(f"[STATE] Failed to load state: {e}")
        return {}

# === Logging Setup ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "paper_trade.log")
TRADES_CSV_PATH = os.path.join(LOG_DIR, "paper_trades.csv")

logger = logging.getLogger("paper_trade")
logger.setLevel(logging.INFO)
logger.addHandler(RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=5))
logger.addHandler(logging.StreamHandler())

heartbeat_logger = logging.getLogger("paper_trade_heartbeat")
heartbeat_logger.setLevel(logging.INFO)
heartbeat_logger.addHandler(RotatingFileHandler(os.path.join(LOG_DIR, "paper_trade_heartbeat.log"), maxBytes=1_000_000, backupCount=3))

# === Constants & Env Vars ===
STARTING_EQUITY = 100000.0
MIN_BARS_FOR_FEATURES = 12
MAX_BUFFER_LENGTH = 500
BACKFILL_DURATION_STR = "5 D"

ALERT_EMAIL_TO = os.environ.get("TRADER_ALERT_EMAIL_TO")
ALERT_EMAIL_FROM = os.environ.get("TRADER_ALERT_EMAIL_FROM")
SMTP_HOST = os.environ.get("TRADER_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("TRADER_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("TRADER_SMTP_USER")
SMTP_PASS = os.environ.get("TRADER_SMTP_PASS")

RTH_START = dt.time(9, 0)
RTH_END = dt.time(16, 30)
US_EQUITY_OPEN = dt.time(9, 30)
US_EQUITY_OPEN_WARMUP_END = dt.time(9, 45)

try:
    RTH_TZ = ZoneInfo("America/New_York")
except Exception:
    RTH_TZ = None

HEARTBEAT_INTERVAL_MIN_OUTSIDE_RTH = 60

# === Helpers ===

def is_rth_now() -> bool:
    """Check if current time is within trading hours (approximate)."""
    now = dt.datetime.now(RTH_TZ) if RTH_TZ else dt.datetime.now()
    return RTH_START <= now.time() <= RTH_END

def minutes_until_next_rth_open() -> float:
    """Calculate minutes until next RTH open to optimize sleep cycles."""
    now = dt.datetime.now(RTH_TZ) if RTH_TZ else dt.datetime.now()
    if now.time() < RTH_START:
        target_date = now.date()
    else:
        target_date = now.date() + dt.timedelta(days=1)
    target_dt = dt.datetime.combine(target_date, RTH_START, tzinfo=now.tzinfo)
    return max((target_dt - now).total_seconds() / 60.0, 0.0)

def send_fatal_error_email(subject: str, body: str) -> None:
    if not ALERT_EMAIL_TO: return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = ALERT_EMAIL_FROM or ALERT_EMAIL_TO
    msg["To"] = ALERT_EMAIL_TO
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"[ALERT-ERROR] Failed to send email: {e}")

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute Average True Range (ATR) for volatility sizing."""
    if len(df) < period + 1: return 0.0
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return float(atr.iloc[-1])

# === Core Trader ===

class PaperTrader:
    def __init__(self) -> None:
        self.ib: Optional[IB] = None
        self.clf = None
        self.contracts: Dict[str, Stock] = {}
        self.ohlcv_buffers: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Position] = load_state_from_disk()

        self.start_of_day_equity: float = STARTING_EQUITY
        self.realized_pnl_today: float = 0.0
        self.last_stop_bar: Dict[str, pd.Timestamp] = {}
        self.trades_today: list[dict] = []
        self.current_trading_date: Optional[dt.date] = None

    @staticmethod
    def _log(msg: str) -> None:
        logger.info(msg)

    def _append_trade_to_csv(self, trade_record: dict) -> None:
        """Log closed trades to CSV."""
        os.makedirs(LOG_DIR, exist_ok=True)
        file_exists = os.path.exists(TRADES_CSV_PATH)
        fieldnames = ["symbol", "entry_dt", "exit_dt", "entry_price", "exit_price", "size", "pnl", "r_multiple", "reason"]
        rec = dict(trade_record)
        for k in ["entry_dt", "exit_dt"]:
            if isinstance(rec.get(k), (pd.Timestamp, dt.datetime)):
                rec[k] = rec[k].isoformat()
        try:
            with open(TRADES_CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists: writer.writeheader()
                writer.writerow(rec)
        except Exception as e:
            self._log(f"[TRADE-LOG-ERROR] {e}")

    def _ib_now(self) -> pd.Timestamp:
        return pd.Timestamp.utcnow()

    # --- Risk Management ---

    def _current_equity(self) -> float:
        return self.start_of_day_equity + self.realized_pnl_today

    def _max_daily_loss_reached(self) -> bool:
        """Return True if daily loss exceeds configured fraction."""
        equity_now = self._current_equity()
        drop = (equity_now - self.start_of_day_equity) / self.start_of_day_equity
        return drop <= -DAILY_LOSS_STOP_FRACTION

    def _calc_position_size(self, price: float, atr: float) -> int:
        """Calculate position size using Volatility Sizing & Hard Caps."""
        if price <= 0 or atr <= 0: return 0
        equity = self._current_equity()
        risk_capital = RISK_PER_TRADE_FRACTION * equity
        per_share_risk = max(atr * ATR_STOP_MULT, 0.01)
        vol_size = int(risk_capital / per_share_risk)
        notional_size = int(MAX_NOTIONAL_PER_TRADE / price)
        return max(min(vol_size, notional_size, MAX_SHARES_PER_TRADE), 0)

    def _position_risk_ok(self, symbol: str, bar_time: pd.Timestamp) -> bool:
        if symbol in self.positions: return False
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS: 
            self._log(f"[ENTRY-SKIP] {symbol}: Max concurrent positions reached.")
            return False
        
        last_stop = self.last_stop_bar.get(symbol)
        if last_stop:
            bars_since = (bar_time - last_stop) / dt.timedelta(minutes=BAR_INTERVAL_MIN)
            if bars_since < COOLDOWN_BARS_AFTER_STOP:
                self._log(f"[ENTRY-SKIP] {symbol}: In cooldown ({bars_since:.1f} < {COOLDOWN_BARS_AFTER_STOP}).")
                return False
                
        if self._max_daily_loss_reached():
            self._log(f"[ENTRY-SKIP] {symbol}: Daily loss limit hit.")
            return False
        return True

    # --- Data & Connection ---

    def _update_buffer_from_bars(self, symbol: str, bars: list) -> Optional[pd.Series]:
        """Append latest bar to buffer."""
        if not bars: return None
        last_bar = bars[-1]
        ts = last_bar.date.tz_localize("UTC") if last_bar.date.tzinfo is None else last_bar.date.tz_convert("UTC")
        row = {"open": last_bar.open, "high": last_bar.high, "low": last_bar.low, 
               "close": last_bar.close, "volume": last_bar.volume}
        
        buf = self.ohlcv_buffers.get(symbol, pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float))
        buf.loc[ts] = row
        if len(buf) > MAX_BUFFER_LENGTH: buf = buf.iloc[-MAX_BUFFER_LENGTH:]
        self.ohlcv_buffers[symbol] = buf
        return buf.iloc[-1]

    def _backfill_buffers_on_startup(self) -> None:
        """
        Backfill OHLCV buffers with a fallback strategy.
        Attempt 1: 2 Days, Outside RTH (Preferred).
        Attempt 2: 1 Day, Inside RTH (Lighter request if servers are slow).
        """
        if self.ib is None or not self.ib.isConnected(): return
        
        self._log(f"[BACKFILL] Starting backfill sequence...")
        
        for sym in UNIVERSE:
            contract = self.contracts[sym]
            bars = None
            
            # --- Attempt 1: Ideal Data (2 Days, Ext Hours) ---
            try:
                self.ib.sleep(2.0) # Throttle
                bars = self.ib.reqHistoricalData(
                    contract, "", "2 D", f"{BAR_INTERVAL_MIN} mins", "TRADES", 
                    useRTH=False, formatDate=1, keepUpToDate=False, timeout=45
                )
            except Exception:
                pass # Fail silently and try fallback

            # --- Attempt 2: Fallback (1 Day, RTH Only) ---
            if not bars:
                self._log(f"[BACKFILL-RETRY] {sym}: Timed out. Retrying lighter request (1 D, RTH)...")
                try:
                    self.ib.sleep(2.0)
                    bars = self.ib.reqHistoricalData(
                        contract, "", "1 D", f"{BAR_INTERVAL_MIN} mins", "TRADES", 
                        useRTH=True, formatDate=1, keepUpToDate=False, timeout=45
                    )
                except Exception as e:
                    self._log(f"[BACKFILL-FAIL] {sym}: Could not load history. Starting empty. ({e})")
                    continue

            # --- Process Data ---
            if bars:
                data = [{"datetime": pd.Timestamp(b.date).tz_localize("UTC"), "open": b.open, "high": b.high, 
                         "low": b.low, "close": b.close, "volume": b.volume} for b in bars]
                df = pd.DataFrame(data).set_index("datetime").sort_index()
                
                # Deduplicate and sort
                df = df[~df.index.duplicated(keep='last')].sort_index()
                
                if len(df) > MAX_BUFFER_LENGTH: df = df.iloc[-MAX_BUFFER_LENGTH:]
                self.ohlcv_buffers[sym] = df
                self._log(f"[BACKFILL] {sym}: Loaded {len(df)} bars.")
        
        self._sync_positions_with_ib()

    def _connect_ib_and_setup(self) -> None:
        """Initialize IBKR connection and load data."""
        self._log("Connecting to IBKR...")
        self.ib = IB()
        try:
            rand_id = random.randint(1, 999)
            self.ib.connect("127.0.0.1", 7497, clientId=rand_id)
            self._log(f"Connected with Client ID: {rand_id}")        
        except Exception as e:
            raise RuntimeError(f"IBKR Connection failed: {e}")

        self.clf = load_latest_classifier(MODEL_DIR)
        
        for sym in UNIVERSE:
            self.contracts[sym] = Stock(sym, "SMART", "USD")
        self.ib.qualifyContracts(*self.contracts.values())

        # Perform Backfill
        self._backfill_buffers_on_startup()

    def _sync_positions_with_ib(self):
        """Reconcile local state with IBKR portfolio."""
        ib_positions = {p.contract.symbol: p.position for p in self.ib.positions()}
        for sym in list(self.positions.keys()):
            if sym not in ib_positions or ib_positions[sym] == 0:
                self._log(f"[SYNC] Removing {sym} (Closed at IBKR).")
                del self.positions[sym]
        for sym, size in ib_positions.items():
            if size != 0 and sym in UNIVERSE and sym not in self.positions:
                self._log(f"[SYNC] Zombie found: {sym}. Tracking loosely.")
                self.positions[sym] = Position(
                    symbol=sym, entry_dt=self._ib_now(), size=int(size),
                    entry_price=0.0, stop_price=0.0, take_profit_price=0.0,
                    p_up=0.0, bars_held=0
                )
        save_state_to_disk(self.positions)

    def _ensure_ib_connected(self) -> bool:
        if self.ib and self.ib.isConnected(): return True
        try:
            self.ib.disconnect()
            self.ib.connect("127.0.0.1", 7497, clientId=1)
            return True
        except: return False

    def connect_and_load(self) -> None:
        self._connect_ib_and_setup()

    # --- Execution Logic ---

    def _submit_bracket_order(self, symbol: str, quantity: int, current_price: float, p_up: float, atr: float) -> None:
        """Submit Entry + Stop Loss + Take Profit orders."""
        stop_dist = round(atr * ATR_STOP_MULT, 2)
        tp_dist = round(atr * ATR_TP_MULT, 2)
        stop_price = round(current_price - stop_dist, 2)
        take_profit_price = round(current_price + tp_dist, 2)
        
        parent = MarketOrder('BUY', quantity)
        parent.transmit = False 
        tp_order = LimitOrder('SELL', quantity, take_profit_price)
        tp_order.transmit = False
        sl_order = StopOrder('SELL', quantity, stop_price)
        sl_order.transmit = True 
        
        for o in [parent, tp_order, sl_order]:
            self.ib.placeOrder(o)
        
        self._log(f"[ENTRY] Bracket {symbol}: Size={quantity}, Price={current_price:.2f}, SL={stop_price}, TP={take_profit_price}, p_up={p_up:.3f}")

        self.positions[symbol] = Position(
            symbol=symbol, entry_dt=dt.datetime.now(dt.timezone.utc),
            size=quantity, entry_price=current_price,
            stop_price=stop_price, take_profit_price=take_profit_price, 
            p_up=p_up, bars_held=0
        )
        save_state_to_disk(self.positions)

    def _check_exit_signals(self, symbol: str, current_price: float) -> None:
        """Check for Time-based exits (Price exits handled by IBKR)."""
        pos = self.positions.get(symbol)
        if not pos: return

        ib_positions = {p.contract.symbol: p.position for p in self.ib.positions()}
        if symbol not in ib_positions or ib_positions[symbol] == 0:
            self._log(f"[EXIT-SYNC] {symbol} closed at IBKR. Removing state.")
            self._log_trade_result(symbol, "BROKER_EXIT", current_price, pos)
            del self.positions[symbol]
            save_state_to_disk(self.positions)
            return

        pos.bars_held += 1
        save_state_to_disk(self.positions)

        if pos.bars_held >= MAX_BARS_IN_TRADE:
            self._log(f"[EXIT-TIME] {symbol} hit MAX_BARS. Force closing.")
            self._cancel_bracket_and_close(symbol)

    def _cancel_bracket_and_close(self, symbol: str):
        """Force close a position."""
        for order in self.ib.openOrders():
            if order.contract.symbol == symbol:
                self.ib.cancelOrder(order)
        self.ib.sleep(0.5) 
        
        positions = [p for p in self.ib.positions() if p.contract.symbol == symbol]
        if positions and positions[0].position > 0:
            self.ib.placeOrder(MarketOrder('SELL', positions[0].position))
                
        if symbol in self.positions:
            self._log_trade_result(symbol, "MAX_BARS", 0.0, self.positions[symbol])
            del self.positions[symbol]
            save_state_to_disk(self.positions)

    def _log_trade_result(self, symbol: str, reason: str, exit_price: float, pos: Position):
        realized = (exit_price - pos.entry_price) * pos.size
        risk_dist = pos.entry_price - pos.stop_price
        r_mult = realized / (risk_dist * pos.size) if risk_dist > 0 and exit_price > 0 else 0.0
        
        self.realized_pnl_today += realized
        rec = {
            "symbol": symbol, "entry_dt": pos.entry_dt, "exit_dt": self._ib_now(),
            "entry_price": pos.entry_price, "exit_price": exit_price,
            "size": pos.size, "pnl": realized, "r_multiple": r_mult, "reason": reason
        }
        self.trades_today.append(rec)
        self._append_trade_to_csv(rec)

    def _log_open_positions(self) -> None:
        if not self.positions:
            self._log("[POSITIONS] No open positions.")
            return
        for sym, pos in self.positions.items():
            self._log(
                f"[POSITIONS] {sym}: size={pos.size}, entry={pos.entry_price:.2f}, "
                f"SL={pos.stop_price:.2f}, TP={pos.take_profit_price:.2f}, "
                f"p_up={pos.p_up:.3f}, bars={pos.bars_held}"
            )

    # --- Main Loop ---

    def _poll_bars_once(self) -> None:
        """Poll data -> Check Exits -> Generate Features -> Check Entries."""
        now_ts = self._ib_now()
        
        if self.current_trading_date != now_ts.date():
            self.start_of_day_equity = self._current_equity()
            self.current_trading_date = now_ts.date()
            self.realized_pnl_today = 0.0
            self.trades_today = []

        # 1. Data Ingestion
        for sym in UNIVERSE:
            try:
                # useRTH=False allows testing data flow anytime
                # Added timeout=30 to prevent long hangs during maintenance
                bars = self.ib.reqHistoricalData(self.contracts[sym], "", f"{BAR_INTERVAL_MIN * 2} M", 
                                                 f"{BAR_INTERVAL_MIN} mins", "TRADES", 
                                                 useRTH=False, formatDate=1, keepUpToDate=False, timeout=30)
                last = self._update_buffer_from_bars(sym, bars)
                if last is not None:
                    buf = self.ohlcv_buffers[sym]
                    self._log(f"[BAR] {sym}: last_close={last['close']:.4f}, buffer_len={len(buf)}")
            except: continue

        # 2. Exits
        for sym in list(self.positions.keys()):
            buf = self.ohlcv_buffers.get(sym)
            price = buf.iloc[-1]['close'] if buf is not None and not buf.empty else 0.0
            self._check_exit_signals(sym, price)

        # 3. Entries
        if self._max_daily_loss_reached(): return
        
        local_time = now_ts.tz_convert(RTH_TZ).time() if RTH_TZ else now_ts.time()
        if US_EQUITY_OPEN <= local_time < US_EQUITY_OPEN_WARMUP_END: 
             self._log("[ENTRY-SKIP] Open warmup window.")
             return

        for sym in UNIVERSE:
            if sym in self.positions: continue

            buf = self.ohlcv_buffers.get(sym)
            if buf is None or len(buf) < MIN_BARS_FOR_FEATURES:
                self._log(f"[ENTRY-SKIP] {sym}: Insufficient history ({len(buf) if buf else 0}).")
                continue

            panel = build_features_for_symbol(buf)
            if panel.empty: continue
            row = panel.iloc[-1]
            feat_vals = row[FEATURE_COLUMNS].fillna(0).replace([np.inf, -np.inf], 0)

            current_atr = calculate_atr(buf, period=ATR_WINDOW)
            if current_atr <= 0:
                self._log(f"[ENTRY-SKIP] {sym}: ATR invalid.")
                continue

            x = feat_vals.to_frame().T
            p_up = float(self.clf.predict_proba(x)[0, 1])
            self._log(f"[SIGNAL] {sym}: p_up={p_up:.3f}, thresh={P_UP_ENTRY_THRESHOLD:.3f}, ATR={current_atr:.2f}")

            if p_up >= P_UP_ENTRY_THRESHOLD:
                if not self._position_risk_ok(sym, buf.index[-1]): continue
                
                price = float(row["close"])
                size = self._calc_position_size(price, current_atr)
                if size > 0:
                    self._submit_bracket_order(sym, size, price, p_up, current_atr)
                else:
                    self._log(f"[ENTRY-SKIP] {sym}: Calc size is 0.")
            else:
                 self._log(f"[ENTRY-SKIP] {sym}: p_up below threshold.")

        self._log_open_positions()

    def run(self) -> None:
        self._log("Initializing paper trader...")
        self.connect_and_load()
        self._log("Running loop...")

        while True:
            loop_start = dt.datetime.now()
            
            if not is_rth_now():
                sleep_min = min(minutes_until_next_rth_open(), HEARTBEAT_INTERVAL_MIN_OUTSIDE_RTH)
                heartbeat_logger.info(f"[HEARTBEAT] Sleeping {sleep_min:.1f}m until open.")
                time.sleep(sleep_min * 60)
                continue

            if not self._ensure_ib_connected():
                time.sleep(10)
                continue

            work_start = dt.datetime.now()
            try:
                self._poll_bars_once()
            except Exception as e:
                self._log(f"[ERROR] Polling loop error: {e}")
            work_end = dt.datetime.now()
            
            work_sec = (work_end - work_start).total_seconds()
            sleep_sec = max(1.0, (BAR_INTERVAL_MIN * 60) - work_sec)
            
            self._log(f"[LOOP] Work took {work_sec:.1f}s. Sleeping {sleep_sec:.1f}s.")
            self.ib.sleep(sleep_sec)
            
            total_sec = (dt.datetime.now() - loop_start).total_seconds()
            self._log(f"[LOOP] Cycle finished. Total time: {total_sec:.1f}s")

def main() -> None:
    while True:
        try:
            trader = PaperTrader()
            trader.run()
        except KeyboardInterrupt:
            logger.info("Manual Stop.")
            break
        except Exception as e:
            logger.error(f"[CRASH] {e}")
            send_fatal_error_email("Trader Crashed", traceback.format_exc())
            time.sleep(30)

if __name__ == "__main__":
    main()
"""
Paper trading strategy (IBKR, 5-min bars).
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

# === Config Imports ===
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
except ImportError:
    # --- SAFETY DEFAULTS (For standalone testing) ---
    print("WARNING: Config not found, using defaults.")
    UNIVERSE = ['SOFI', 'AAPL']
    MODEL_DIR = "models"
    FEATURE_COLUMNS = ['close', 'volume']
    BAR_INTERVAL_MIN = 5
    P_UP_ENTRY_THRESHOLD = 0.55
    RISK_PER_TRADE_FRACTION = 0.01
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.04
    MAX_CONCURRENT_POSITIONS = 3
    DAILY_LOSS_STOP_FRACTION = 0.02
    MAX_BARS_IN_TRADE = 24
    COOLDOWN_BARS_AFTER_STOP = 3

    def build_features_for_symbol(df):
        df['f1'] = df['close'].pct_change()
        return df

    class MockClf:
        def predict_proba(self, X): return np.array([[0.4, 0.6]])
    def load_latest_classifier(d): return MockClf()


# === Logging setup ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "paper_trade.log")
TRADES_CSV_PATH = os.path.join(LOG_DIR, "paper_trades.csv")

logger = logging.getLogger("paper_trade")
logger.setLevel(logging.INFO)

# Only add handlers if they don't exist yet to prevent double logging
if not logger.handlers:
    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=5)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(console_handler)

heartbeat_logger = logging.getLogger("paper_trade_heartbeat")
heartbeat_logger.setLevel(logging.INFO)
if not heartbeat_logger.handlers:
    heartbeat_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "paper_trade_heartbeat.log"),
        maxBytes=1_000_000,
        backupCount=3,
    )
    heartbeat_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    heartbeat_logger.addHandler(heartbeat_handler)

STARTING_EQUITY = 100000.0
MIN_BARS_FOR_FEATURES = 12
MAX_BUFFER_LENGTH = 500
BACKFILL_DURATION_STR = "5 D"

# === Email alert config ===
ALERT_EMAIL_TO = os.environ.get("TRADER_ALERT_EMAIL_TO")
ALERT_EMAIL_FROM = os.environ.get("TRADER_ALERT_EMAIL_FROM")
SMTP_HOST = os.environ.get("TRADER_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("TRADER_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("TRADER_SMTP_USER")
SMTP_PASS = os.environ.get("TRADER_SMTP_PASS")

# === Trading Hours ===
RTH_START = dt.time(9, 0)
RTH_END = dt.time(16, 30)
US_EQUITY_OPEN = dt.time(9, 30)
US_EQUITY_OPEN_WARMUP_END = dt.time(9, 45)

try:
    RTH_TZ = ZoneInfo("America/New_York")
except Exception:
    RTH_TZ = None

HEARTBEAT_INTERVAL_MIN_OUTSIDE_RTH = 60


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

    current_time = now.time()
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


def send_fatal_error_email(subject: str, body: str) -> None:
    if not ALERT_EMAIL_TO:
        return
    from_addr = ALERT_EMAIL_FROM or ALERT_EMAIL_TO
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ALERT_EMAIL_TO
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            if SMTP_USER and SMTP_PASS:
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logger.info(f"[ALERT] Sent fatal error email to {ALERT_EMAIL_TO}.")
    except Exception as e:
        logger.error(f"[ALERT-ERROR] Failed to send fatal error email: {e}")


@dataclass
class Position:
    symbol: str
    size: int
    entry_price: float
    entry_dt: pd.Timestamp
    stop_price: float
    take_profit_price: float
    p_up: float


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

    @staticmethod
    def _log(msg: str) -> None:
        logger.info(msg)

    def _append_trade_to_csv(self, trade_record: dict) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_exists = os.path.exists(TRADES_CSV_PATH)
        fieldnames = [
            "symbol", "entry_dt", "exit_dt", "entry_price", "exit_price",
            "size", "pnl", "r_multiple", "reason"
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
        t = local.time()
        return US_EQUITY_OPEN <= t < US_EQUITY_OPEN_WARMUP_END

    def _ib_net_liquidation(self) -> Optional[float]:
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
        return self.start_of_day_equity + self.realized_pnl_today

    def _max_daily_loss_reached(self) -> bool:
        equity_now = self._current_equity()
        drop = (equity_now - self.start_of_day_equity) / self.start_of_day_equity
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
        if symbol in self.positions:
            return False
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return False
        last_stop = self.last_stop_bar.get(symbol)
        if last_stop is not None:
            bars_since_stop = (bar_time - last_stop) / dt.timedelta(minutes=BAR_INTERVAL_MIN)
            if bars_since_stop < COOLDOWN_BARS_AFTER_STOP:
                self._log(f"[COOLDOWN] Skipping {symbol}: {bars_since_stop:.1f} bars since STOP.")
                return False
        if self._max_daily_loss_reached():
            self._log("[DAILY-STOP] Daily loss limit reached; no new entries.")
            return False
        return True

    @staticmethod
    def _empty_buffer() -> pd.DataFrame:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)

    def _update_buffer_from_bars(self, symbol: str, bars: list) -> Optional[pd.Series]:
        if not bars:
            return None
        last_bar = bars[-1]
        ts = self._normalize_ts(last_bar.date)
        row = {
            "open": last_bar.open, "high": last_bar.high, "low": last_bar.low,
            "close": last_bar.close, "volume": last_bar.volume,
        }
        buf = self.ohlcv_buffers.get(symbol)
        if buf is None or buf.empty:
            buf = self._empty_buffer()
        buf.loc[ts] = row
        if len(buf) > MAX_BUFFER_LENGTH:
            buf = buf.iloc[-MAX_BUFFER_LENGTH:]
        self.ohlcv_buffers[symbol] = buf
        return buf.iloc[-1]

    def _backfill_buffers_on_startup(self) -> None:
        if self.ib is None or not self.ib.isConnected():
            return
        self._log(f"[BACKFILL] Requesting {BACKFILL_DURATION_STR} of {BAR_INTERVAL_MIN}-min RTH bars...")
        for sym in UNIVERSE:
            contract = self.contracts[sym]
            try:
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr=BACKFILL_DURATION_STR,
                    barSizeSetting=f"{BAR_INTERVAL_MIN} mins", whatToShow="TRADES",
                    useRTH=True, formatDate=1,
                )
            except Exception as e:
                self._log(f"[BACKFILL-ERROR] {sym}: {e}")
                continue
            if not bars:
                continue
            data = []
            for b in bars:
                ts = self._normalize_ts(b.date)
                data.append({
                    "datetime": ts, "open": b.open, "high": b.high,
                    "low": b.low, "close": b.close, "volume": b.volume,
                })
            df = pd.DataFrame(data).set_index("datetime").sort_index()
            if len(df) > MAX_BUFFER_LENGTH:
                df = df.iloc[-MAX_BUFFER_LENGTH:]
            self.ohlcv_buffers[sym] = df
            self._log(f"[BACKFILL] {sym}: loaded {len(df)} bars.")

    def _connect_ib_and_setup(self) -> None:
        self._log("Connecting to IBKR TWS/Gateway...")
        self.ib = IB()
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                self.ib.connect("127.0.0.1", 7497, clientId=1)
                if self.ib.isConnected():
                    self._log(f"Connected to IBKR on attempt {attempt}.")
                    break
            except Exception as e:
                self._log(f"Connection error (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(5)
        else:
            raise RuntimeError("Could not connect to IBKR.")

        self._log(f"Using internal STARTING_EQUITY={self.start_of_day_equity:.2f}")
        net_liq = self._ib_net_liquidation()
        if net_liq:
            self._log(f"[INFO] IBKR NetLiquidation={net_liq:.2f} (monitoring only)")

        self._log("Loading latest classifier...")
        self.clf = load_latest_classifier(MODEL_DIR)
        self._log(f"Creating contracts for universe: {UNIVERSE}")
        for sym in UNIVERSE:
            self.contracts[sym] = Stock(sym, "SMART", "USD")
        self._log("Qualifying contracts...")
        self.ib.qualifyContracts(*self.contracts.values())
        
        self._log("Initializing buffers...")
        for sym in UNIVERSE:
            self.ohlcv_buffers[sym] = self._empty_buffer()
        self._backfill_buffers_on_startup()

    def _ensure_ib_connected(self) -> bool:
        if self.ib is None: return False
        if self.ib.isConnected(): return True
        self._log("[IB] Connection lost; attempting reconnect...")
        try:
            try: self.ib.disconnect()
            except Exception: pass
            self.ib.connect("127.0.0.1", 7497, clientId=1)
            if self.ib.isConnected():
                self._log("[IB] Reconnected.")
                return True
        except Exception as e:
            self._log(f"[IB] Reconnect failed: {e}")
        return False

    def connect_and_load(self) -> None:
        self._connect_ib_and_setup()

    def _load_existing_ib_positions(self) -> None:
        """
        Seed positions from IBKR portfolio.
        If a position has no existing brackets (is naked), automatically attach 'Rescue' orders.
        """
        if self.ib is None or not self.ib.isConnected():
            return

        try:
            # 1. Get actual held positions
            ib_positions = self.ib.positions()
            
            # 2. Get Open Trades to find attached SL/TP
            self.ib.reqAllOpenOrders()
            open_trades = self.ib.openTrades()
            
            for p in ib_positions:
                sym = p.contract.symbol
                if sym not in UNIVERSE:
                    continue
                
                pos_size = int(p.position)
                if pos_size == 0:
                    continue

                avg_cost = float(p.avgCost or 0.0)
                stop_price = 0.0
                tp_price = 0.0
                
                # --- Scan for existing orders ---
                relevant_trades = [
                    t for t in open_trades 
                    if t.contract.symbol == sym and t.order.action == 'SELL'
                ]

                for t in relevant_trades:
                    order = t.order
                    if order.orderType in ['STP', 'TRAIL']:
                        stop_price = order.auxPrice
                    elif order.orderType == 'LMT':
                        # Assume Limit Sell above cost is TP
                        if order.lmtPrice > avg_cost:
                            tp_price = order.lmtPrice

                # --- RESCUE LOGIC: If Naked, Protect It ---
                if stop_price == 0 and tp_price == 0:
                    self._log(f"[RESCUE] {sym} is NAKED. Attaching new Hard Bracket...")
                    
                    # Calculate new levels based on Average Cost
                    new_sl = round(avg_cost * (1.0 - STOP_LOSS_PCT), 2)
                    new_tp = round(avg_cost * (1.0 + TAKE_PROFIT_PCT), 2)
                    
                    # Define Orders
                    # We use OCA (One Cancels All) to link them without a parent order
                    oca_group_name = f"RESCUE_{sym}_{int(time.time())}"
                    
                    sl_order = StopOrder('SELL', pos_size, new_sl)
                    sl_order.ocaGroup = oca_group_name
                    sl_order.ocaType = 1 # 1 = Cancel all remaining orders with block
                    
                    tp_order = LimitOrder('SELL', pos_size, new_tp)
                    tp_order.ocaGroup = oca_group_name
                    tp_order.ocaType = 1

                    # Place them
                    self.ib.placeOrder(p.contract, sl_order)
                    self.ib.placeOrder(p.contract, tp_order)
                    
                    stop_price = new_sl
                    tp_price = new_tp
                    self._log(f"[RESCUE-SENT] {sym}: SL={new_sl}, TP={new_tp}")

                # --- Log Status ---
                log_msg = f"[EXISTING] {sym} size={pos_size} @ {avg_cost:.2f}."
                if stop_price > 0:
                    log_msg += f" SL={stop_price:.2f}"
                if tp_price > 0:
                    log_msg += f" TP={tp_price:.2f}"
                self._log(log_msg)

                # Reconstruct Position object
                self.positions[sym] = Position(
                    symbol=sym,
                    size=pos_size,
                    entry_price=avg_cost,
                    entry_dt=self._ib_now(), 
                    stop_price=stop_price,
                    take_profit_price=tp_price,
                    p_up=0.0 
                )

        except Exception as e:
            self._log(f"[IB] Failed to load/rescue positions: {e}")
            logger.error(traceback.format_exc())

    def _submit_market_order(self, symbol: str, size: int) -> Optional[SimpleNamespace]:
        # Used for closing only
        contract = self.contracts[symbol]
        action = "BUY" if size > 0 else "SELL"
        self._log(f"[ORDER] Submitting {action} {abs(size)} {symbol} (MKT)...")
        order = MarketOrder(action, abs(size))
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        elapsed = 0.0
        while not trade.isDone() and elapsed < 15.0:
            self.ib.sleep(0.5)
            elapsed += 0.5
        fills = trade.fills
        if not fills:
            self._log("[ORDER] No fills received.")
            return None
        total_shares = sum(abs(f.execution.shares) for f in fills)
        if total_shares == 0: return None
        avg_price = sum(f.execution.price * abs(f.execution.shares) for f in fills) / total_shares
        self._log(f"[ORDER-FILLED] {action} {abs(size)} {symbol} avg={avg_price:.4f}")
        return SimpleNamespace(avg_price=avg_price, size=size, action=action, raw_trade=trade)

    def _place_bracket_order(self, symbol: str, size: int, p_up: float, current_price: float, bar_time: pd.Timestamp):
        """
        Submit a real Bracket Order (Parent Market + Child Limit + Child Stop).
        """
        contract = self.contracts[symbol]
        
        # Calculate prices
        stop_price = round(current_price * (1.0 - STOP_LOSS_PCT), 2)
        tp_price = round(current_price * (1.0 + TAKE_PROFIT_PCT), 2)

        # 1. Parent Order (Market Buy)
        parent = MarketOrder('BUY', size)
        parent.transmit = False # Important: Don't send yet
        
        # 2. Take Profit (Limit Sell)
        takeProfit = LimitOrder('SELL', size, tp_price)
        takeProfit.transmit = False
        
        # 3. Stop Loss (Stop Sell)
        stopLoss = StopOrder('SELL', size, stop_price)
        stopLoss.transmit = True # Sending this triggers the group

        # Place orders (ib_insync handles the grouping if we link them or use list)
        # Note: In newer ib_insync, we can use the bracket list approach,
        # but manual linking is safest for correct OCO behavior.
        
        # We place parent first to get an ID? No, ib_insync assigns IDs.
        # We just need to link them via parentId
        
        orders = [parent, takeProfit, stopLoss]
        trades = []
        for o in orders:
            # We must qualify to get IDs? No, placeOrder does it.
            # But children need parentId.
            # Standard approach:
            pass

        # Cleaner approach with ib_insync built-in linkage
        parent.orderId = self.ib.client.getReqId()
        takeProfit.parentId = parent.orderId
        stopLoss.parentId = parent.orderId
        
        # Place them
        parent_trade = self.ib.placeOrder(contract, parent)
        tp_trade = self.ib.placeOrder(contract, takeProfit)
        sl_trade = self.ib.placeOrder(contract, stopLoss)
        
        # Wait a moment for local confirmation (optional)
        self.ib.sleep(0.5)

        # Assuming fill is immediate for Market, but we record the intent
        entry_price = current_price # Approximation until fill

        self.positions[symbol] = Position(
            symbol=symbol, size=size, entry_price=entry_price, entry_dt=bar_time,
            stop_price=stop_price, take_profit_price=tp_price, p_up=p_up
        )
        self._log(f"[ENTRY-BRACKET] {symbol} sent. Size={size} EstPrice={current_price:.2f} SL={stop_price} TP={tp_price}")

    def _close_position(self, symbol: str, pos: Position, exit_price: float, exit_ts: pd.Timestamp, reason: str) -> None:
        # Cancel any open orders for this symbol first (Clean up the bracket children)
        open_orders = self.ib.openOrders()
        for o in open_orders:
            if o.contract.symbol == symbol:
                self.ib.cancelOrder(o)
        
        # Now close the position
        size = pos.size
        fills = self._submit_market_order(symbol, -size)
        
        if fills is None:
            self._log(f"[EXIT-FAILED] {symbol}: no fills on close.")
            return
            
        realized = (fills.avg_price - pos.entry_price) * size
        r_multiple = realized / (STOP_LOSS_PCT * pos.entry_price * size)
        self.realized_pnl_today += realized
        if reason == "STOP":
            self.last_stop_bar[symbol] = exit_ts
            
        trade_record = {
            "symbol": symbol, "entry_dt": pos.entry_dt, "exit_dt": exit_ts,
            "entry_price": pos.entry_price, "exit_price": fills.avg_price,
            "size": size, "pnl": realized, "r_multiple": r_multiple, "reason": reason,
        }
        self.trades_today.append(trade_record)
        self._append_trade_to_csv(trade_record)
        self._log(f"[EXIT-{reason}] {symbol}: pnl={realized:.2f}, R={r_multiple:.2f}")
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
        self._log(f"[SUMMARY] PnL={total_pnl:.2f}, Trades={len(self.trades_today)}")

    def _poll_bars_once(self) -> None:
        now_ts = self._ib_now()
        self._maybe_roll_trading_date(now_ts)

        # 1. Fetch Bars
        for sym in UNIVERSE:
            contract = self.contracts[sym]
            try:
                bars = self.ib.reqHistoricalData(
                    contract, endDateTime="", durationStr=f"{BAR_INTERVAL_MIN * 2} M",
                    barSizeSetting=f"{BAR_INTERVAL_MIN} mins", whatToShow="TRADES",
                    useRTH=True, formatDate=1,
                )
            except Exception:
                continue
            if not bars: continue
            last = self._update_buffer_from_bars(sym, bars)
            if last is None: continue
            # self._log(f"[BAR] {sym}: close={float(last['close']):.4f}") # Too noisy

        # 2. Check Exits (Wait, if we use Brackets, IBKR exits for us.
        # BUT we still monitor here to log the exit if it happens externally)
        for sym, pos in list(self.positions.items()):
            # Check if position is gone in IBKR (hit bracket)
            ib_positions = {p.contract.symbol: p.position for p in self.ib.positions()}
            if sym not in ib_positions or ib_positions[sym] == 0:
                self._log(f"[EXIT-DETECTED] {sym} no longer in portfolio (Bracket hit?).")
                # We don't know exact exit price here easily without execution details
                # Just assuming close price for logging approximate PnL
                buf = self.ohlcv_buffers.get(sym)
                close = buf.iloc[-1]['close'] if buf is not None else pos.entry_price
                self._close_position(sym, pos, close, now_ts, "BRACKET_HIT")
                continue
            
            # Time Stop (We still need to enforce this manually)
            buf = self.ohlcv_buffers.get(sym)
            if buf is None or buf.empty: continue
            last_ts = buf.index[-1]
            bars_held = (last_ts - pos.entry_dt) / dt.timedelta(minutes=BAR_INTERVAL_MIN)
            if bars_held >= MAX_BARS_IN_TRADE:
                self._log(f"[EXIT-CHECK] {sym}: MAX_BARS hit.")
                self._close_position(sym, pos, buf.iloc[-1]['close'], last_ts, "MAX_BARS")

        # 3. Check Entries
        if self._max_daily_loss_reached(): return
        if self._is_within_open_warmup(now_ts): return

        for sym in UNIVERSE:
            if sym in self.positions: continue
            buf = self.ohlcv_buffers.get(sym)
            if buf is None or len(buf) < MIN_BARS_FOR_FEATURES: continue
            
            panel = build_features_for_symbol(buf)
            if panel.empty: continue
            
            row = panel.iloc[-1]
            feat_vals = row[FEATURE_COLUMNS]
            if ~np.isfinite(feat_vals.values.astype(float)).all():
                continue

            x = feat_vals.to_frame().T
            p_up = float(self.clf.predict_proba(x)[0, 1])

            # Filter noise from log
            if p_up > 0.5:
                self._log(f"[SIGNAL] {sym}: p_up={p_up:.3f}")

            if p_up < P_UP_ENTRY_THRESHOLD: continue
            if not self._position_risk_ok(sym, buf.index[-1]): continue

            price = float(row["close"])
            size = self._calc_position_size(price)
            if size <= 0: continue

            # USE BRACKET ENTRY
            self._place_bracket_order(sym, size, p_up, price, buf.index[-1])

    def _log_open_positions(self) -> None:
        if not self.positions:
            self._log("[POSITIONS] None")
        for sym, pos in self.positions.items():
            self._log(f"[POSITIONS] {sym}: size={pos.size}, pnl={(pos.entry_price*pos.size):.2f}")

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
                if self.ib and self.ib.isConnected(): self.ib.sleep(sleep_min * 60)
                else: time.sleep(sleep_min * 60)
                continue

            if not self._ensure_ib_connected():
                time.sleep(15)
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
        logger.info("Manual Stop.")
    except Exception as e:
        logger.error(f"[FATAL] {e}")
        send_fatal_error_email("Paper Trader Crashed", traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
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
from zoneinfo import ZoneInfo

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

import logging
import os
import time
from logging.handlers import RotatingFileHandler
import csv

# basic rotating file + console logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "paper_trade.log")
TRADES_CSV_PATH = os.path.join(LOG_DIR, "paper_trades.csv")

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
logger.propagate = False

# console-only logger for heartbeats (no file handler)
heartbeat_logger = logging.getLogger("paper_trade_heartbeat")
heartbeat_logger.setLevel(logging.INFO)
hb_console_handler = logging.StreamHandler()
hb_console_handler.setFormatter(formatter)
heartbeat_logger.addHandler(hb_console_handler)
heartbeat_logger.propagate = False


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

STARTING_EQUITY = 1000_000.0
MAX_BUFFER_LENGTH = 500 # max bars to keep in OHLCV buffer per symbol
ORDER_WAIT_TIMEOUT_SEC = 15  # max time to wait for order fills


# ================
# Regular trading hours helper
# ================

RTH_START = dt.time(9, 30)
RTH_END = dt.time(16, 0)

try:
    RTH_TZ = ZoneInfo("America/New_York")
except Exception:
    # Fallback: use local system time if zoneinfo is not available
    RTH_TZ = None


def is_rth_now() -> bool:
    """Return True if the current time is within regular trading hours (US equities).

    This lets the paper trader run 24/7 while only trading/logging during RTH.
    """
    if RTH_TZ is not None:
        now = dt.datetime.now(RTH_TZ)
    else:
        now = dt.datetime.now()
    current_time = now.time()
    return RTH_START <= current_time <= RTH_END


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

        # equity tracking (start-of-day comes from IB NetLiquidation when possible)
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

    def _append_trade_to_csv(self, trade_record: dict) -> None:
        """Append a single closed trade to a CSV file for later analysis."""
        # Make sure log directory exists (should already from global setup)
        os.makedirs(LOG_DIR, exist_ok=True)

        file_exists = os.path.exists(TRADES_CSV_PATH)

        # Fixed column order for easier downstream analysis
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

        # Convert timestamps to ISO strings so CSV is clean & portable
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
                    writer.writeheader()  # write header once on first file creation
                writer.writerow(rec)
        except Exception as e:
            # Don't crash trading loop if logging fails; just log the error.
            self._log(f"[TRADE-LOG-ERROR] Failed to append trade to CSV: {e}")


    # ------------ time normalization helpers -----------

    @staticmethod
    def _normalize_ts(ts) -> pd.Timestamp:
        """Ensure we always work with tz-aware UTC timestamps."""
        if isinstance(ts, pd.Timestamp):
            out = ts
        else:
            out = pd.Timestamp(ts)

        if out.tzinfo is None:
            out = out.tz_localize("UTC")
        else:
            out = out.tz_convert("UTC")
        return out

    @staticmethod
    def _ts_to_trading_date(ts: pd.Timestamp) -> dt.date:
        """Return the 'trading date' (UTC-normalized) from a timestamp."""
        ts = PaperTrader._normalize_ts(ts)
        return ts.date()

    # ------------ IB connection & setup ------------

    def _ib_net_liquidation(self) -> Optional[float]:
        """Fetch NetLiquidation from IBKR account values (returns None if unavailable)."""

        """ self.ib.reqAccountUpdates(True)
        self.ib.sleep(1)

        net_liq = None
        for v in self.ib.accountValues():
            if v.tag == "NetLiquidation":
                try:
                    net_liq = float(v.value)
                except ValueError:
                    net_liq = None
                break

        self.ib.reqAccountUpdates(False) """

        return None # temporarily disabled for testing without IBKR connection


    def _connect_ib_and_setup(self) -> None:
        """Connect to IBKR, load model, create contracts, and init buffers."""
        self._log("Connecting to IBKR TWS/Gateway...")
        self.ib = IB()

        max_retries = 10
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
            # All attempts failed
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
                f"Could not fetch NetLiquidation; using default STARTING_EQUITY={STARTING_EQUITY:.2f}"
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
        """Ensure IBKR connection is alive; try a quick reconnect if needed."""
        if self.ib is None:
            self._log("[IB] No IB instance; cannot trade.")
            return False

        if self.ib.isConnected():
            return True

        self._log("[IB] Connection appears lost; attempting to reconnect...")
        try:
            # Best-effort cleanup of any stale connection
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


    @staticmethod
    def _empty_buffer() -> pd.DataFrame:
        """Return an empty OHLCV buffer for a symbol."""
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols)

    def connect_and_load(self) -> None:
        """Public wrapper to connect IB and set up internal state."""
        self._connect_ib_and_setup()

    # ------------ loading existing positions ------------

    def _load_existing_ib_positions(self) -> None:
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

            entry_dt = self._normalize_ts(pd.Timestamp.utcnow())

            # We don't know original p_up; store a sentinel (0.0) just so the field exists
            self.positions[sym] = Position(
                symbol=sym,
                size=size,
                entry_price=entry_price,
                entry_dt=entry_dt,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                p_up=0.0,
            )

            self._log(
                f"[SYNC] Loaded existing position from IB: {sym}, "
                f"size={size}, entry_price={entry_price:.4f}, "
                f"stop={stop_price:.4f}, tp={take_profit_price:.4f}, "
                f"entry_dt={entry_dt}"
            )

    # ------------ equity & trade-tracking helpers ------------

    def _current_equity_estimate(self) -> float:
        """Estimate current equity as start_of_day + realized + unrealized."""
        unrealized = 0.0

        # Rough unrealized estimate from current close vs. entry for each open position
        for sym, pos in self.positions.items():
            buf = self.ohlcv_buffers.get(sym)
            if buf is None or buf.empty:
                continue

            last_price = float(buf["close"].iloc[-1])
            unrealized += (last_price - pos.entry_price) * pos.size

        equity = self.start_of_day_equity + self.realized_pnl_today + unrealized
        return equity

    def _maybe_roll_trading_date(self, ts: pd.Timestamp) -> None:
        """Check if trading date has changed and reset daily state if so."""
        trading_date = self._ts_to_trading_date(ts)
        if self.current_trading_date is None:
            self.current_trading_date = trading_date
            self._log(f"Trading date initialized to {trading_date}")
            return

        if trading_date != self.current_trading_date:
            # new day
            self._log_daily_summary()
            self.current_trading_date = trading_date
            self.realized_pnl_today = 0.0
            self.trades_today = []
            self._log(f"New trading date: {trading_date}")

    def _log_daily_summary(self) -> None:
        """Log a daily summary of realized PnL and trades."""
        if not self.trades_today:
            self._log("No trades today; skipping daily summary.")
            return

        summary_date = self.current_trading_date or dt.date.today()

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

        self._log(f"Trades today: {n_trades}")
        self._log(f"Realized PnL: {total_realized:.2f}")
        self._log(f"Avg PnL/trade: {avg_pnl:.2f}")
        self._log(f"Win rate: {win_rate:.2%}")
        self._log(f"Max win: {max_win:.2f}")
        self._log(f"Max loss: {max_loss:.2f}")
        self._log(f"Winners: {len(wins)}, Losers: {len(losses)}, Flats: {len(flats)}")
        self._log("=" * 70)

    # ------------ order helpers ------------

    def _submit_market_order(self, symbol: str, size: int) -> Optional[SimpleNamespace]:
        """Submit a market order via ib_insync and wait briefly for fill."""
        contract = self.contracts[symbol]
        action = "BUY" if size > 0 else "SELL"

        self._log(f"[ORDER] Submitting {action} {abs(size)} {symbol} (MKT)...")
        order = MarketOrder(action, abs(size))
        trade = self.ib.placeOrder(contract, order)

        # Wait for a fill or for the timeout
        self.ib.sleep(1)
        elapsed = 0.0
        while not trade.isDone() and elapsed < ORDER_WAIT_TIMEOUT_SEC:
            self.ib.sleep(1)
            elapsed += 1.0

        if not trade.isDone():
            self._log(
                f"[ORDER] {action} {symbol} not filled within timeout; "
                f"still in state={trade.orderStatus.status}."
            )

        fills = trade.fills
        if not fills:
            self._log(f"[ORDER] No fills for {symbol}.")
            return None

        avg_fill_price = sum(f.fillPrice * f.execution.shares for f in fills) / sum(
            f.execution.shares for f in fills
        )

        self._log(
            f"[ORDER] Fills for {symbol}: {len(fills)} fills, avg price={avg_fill_price:.4f}"
        )
        return SimpleNamespace(trade=trade, avg_price=avg_fill_price)

    # ------------ position & risk logic ------------

    def _calc_position_size(self, symbol: str, price: float) -> int:
        """Compute position size based on risk fraction and stop distance."""
        equity = self._current_equity_estimate()
        risk_capital = equity * RISK_PER_TRADE_FRACTION

        stop_price = price * (1.0 - STOP_LOSS_PCT)
        risk_per_share = price - stop_price
        if risk_per_share <= 0:
            return 0

        size = int(risk_capital / risk_per_share)
        return max(size, 0)

    def _max_daily_loss_reached(self) -> bool:
        """Return True if today's realized PnL breaches DAILY_LOSS_STOP_FRACTION."""
        equity = self._current_equity_estimate()
        drawdown = (equity - self.start_of_day_equity) / self.start_of_day_equity
        if drawdown <= -DAILY_LOSS_STOP_FRACTION:
            self._log(
                f"[RISK] Daily loss limit reached: drawdown={drawdown:.2%}, "
                f"limit={DAILY_LOSS_STOP_FRACTION:.2%}. No new entries today."
            )
            return True
        return False

    def _position_risk_ok(self, symbol: str, bar_time: pd.Timestamp) -> bool:
        """Check symbol-level risk constraints (cooldown, concurrency, etc.)."""
        # Per-symbol: skip if already have a position
        if symbol in self.positions:
            self._log(f"Skip entry for {symbol}: already have open position.")
            return False

        # Max concurrent positions
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            self._log(
                f"Skip entry for {symbol}: max concurrent positions "
                f"({MAX_CONCURRENT_POSITIONS}) reached."
            )
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

    # ------------ entry & exit logic ------------

    def _handle_exit(
        self, symbol: str, last_price: float, high_price: float, ts: pd.Timestamp
    ) -> None:
        """Check exit conditions (stop-loss, TP, early TP at +2R, max bars)."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        holding_bars = 0
        buf = self.ohlcv_buffers.get(symbol)
        if buf is not None and not buf.empty:
            # Count how many bars since entry
            mask = buf.index >= pos.entry_dt
            holding_bars = mask.sum()

        # Stop-loss on close
        if last_price <= pos.stop_price:
            self._log(
                f"[EXIT-STOP] {symbol}: close={last_price:.4f} <= stop={pos.stop_price:.4f}"
            )
            self._close_position(symbol, last_price, ts, reason="STOP")
            self.last_stop_bar[symbol] = ts
            return

        # Take-profit on intrabar high
        if high_price >= pos.take_profit_price:
            self._log(
                f"[EXIT-TP] {symbol}: high={high_price:.4f} >= tp={pos.take_profit_price:.4f}"
            )
            self._close_position(symbol, pos.take_profit_price, ts, reason="TP")
            return

        # Optional: early partial TP at +2R could go here (not implemented yet)

        # Max bars in trade
        if holding_bars >= MAX_BARS_IN_TRADE:
            self._log(
                f"[EXIT-MAXBARS] {symbol}: holding_bars={holding_bars} "
                f">= MAX_BARS_IN_TRADE={MAX_BARS_IN_TRADE}"
            )
            self._close_position(symbol, last_price, ts, reason="MAX_BARS")
            return

    def _close_position(
        self, symbol: str, exit_price: float, ts: pd.Timestamp, reason: str
    ) -> None:
        """Close an existing position at exit_price using a market order."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        size = pos.size
        action_size = -size  # sell to close

        fills = self._submit_market_order(symbol, action_size)
        if fills is None:
            self._log(f"[EXIT] No fills for {symbol}; leaving position open.")
            return

        realized = (fills.avg_price - pos.entry_price) * size
        self.realized_pnl_today += realized

        r_multiple = realized / (pos.entry_price * STOP_LOSS_PCT * size)

        trade_record = {
            "symbol": symbol,
            "entry_dt": pos.entry_dt,
            "exit_dt": self._normalize_ts(ts),
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

    def _handle_entry(self, symbol: str, last_price: float, ts: pd.Timestamp) -> None:
        """Check ML signal and risk, and open a new long position if conditions met."""
        if self._max_daily_loss_reached():
            return

        buf = self.ohlcv_buffers.get(symbol)
        if buf is None or buf.empty:
            return

        # Build single-row features from buffer
        df_feat = build_features_for_symbol(buf)
        if df_feat.empty:
            return

        row = df_feat.iloc[-1]
        x = row[FEATURE_COLUMNS].values.reshape(1, -1)

        p_up = float(self.clf.predict_proba(x)[0, 1])

        if p_up < P_UP_ENTRY_THRESHOLD:
            return

        bar_time = self._normalize_ts(buf.index[-1])
        if not self._position_risk_ok(symbol, bar_time):
            return

        size = self._calc_position_size(symbol, last_price)
        if size <= 0:
            self._log(f"[ENTRY-SKIP] {symbol}: computed size <= 0, skipping.")
            return

        fills = self._submit_market_order(symbol, size)
        if fills is None:
            self._log(f"[ENTRY-FAIL] {symbol}: order not filled; no position opened.")
            return

        entry_price = fills.avg_price
        stop_price = entry_price * (1.0 - STOP_LOSS_PCT)
        take_profit_price = entry_price * (1.0 + TAKE_PROFIT_PCT)

        pos = Position(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            entry_dt=self._normalize_ts(ts),
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            p_up=p_up,
        )
        self.positions[symbol] = pos

        self._log(
            f"[ENTRY] {symbol}: size={size}, entry={entry_price:.4f}, "
            f"stop={stop_price:.4f}, tp={take_profit_price:.4f}, p_up={p_up:.3f}"
        )

    # ------------ bar ingestion & polling ------------

    def _on_new_bar(self, symbol: str, bar) -> None:
        """Update buffers with a new bar and process entries/exits."""
        ts = self._normalize_ts(bar.date)
        self._maybe_roll_trading_date(ts)

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
        """Poll historical data (2D lookback) and feed the latest bar into the strategy."""
        end_str = ""
        duration_str = "2 D"
        bar_size = f"{BAR_INTERVAL_MIN} mins"

        for sym, contract in self.contracts.items():
            self._log(f"[POLL] Requesting latest bars for {sym}...")
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                self._log(f"[POLL] No bars returned for {sym}.")
                continue

            bar = bars[-1]
            self._on_new_bar(sym, bar)

    def _log_open_positions(self) -> None:
        """Log current open positions and equity estimate."""
        equity = self._current_equity_estimate()
        if not self.positions:
            self._log(f"[STATUS] No open positions. Est. equity={equity:.2f}")
            return

        self._log(f"[STATUS] Open positions (est. equity={equity:.2f}):")
        for sym, pos in self.positions.items():
            self._log(
                f"  - {sym}: size={pos.size}, entry={pos.entry_price:.4f}, "
                f"stop={pos.stop_price:.4f}, tp={pos.take_profit_price:.4f}, "
                f"entry_dt={pos.entry_dt}, p_up={pos.p_up:.3f}"
            )

    # ------------ main loop (RTH-only) ------------

    def run(self) -> None:
        """Main loop: RTH-only trading; heartbeat outside RTH goes to console only."""
        self._log("Initializing paper trader...")
        self.connect_and_load()
        self._load_existing_ib_positions()

        self._log(
            f"Starting polling bar loop (interval={BAR_INTERVAL_MIN} minutes, RTH-only)..."
        )

        in_rth_last = False

        while True:
            in_rth_now = is_rth_now()

            if not in_rth_now:
                # Outside regular trading hours: heartbeat only to console, not to log file.
                heartbeat_logger.info(
                    f"[HEARTBEAT] Outside RTH; no trading. "
                    f"Sleeping for {BAR_INTERVAL_MIN} minutes..."
                )
                in_rth_last = False

                # If IB is up, use ib.sleep so event loop stays active; otherwise time.sleep.
                if self.ib is not None and self.ib.isConnected():
                    self.ib.sleep(BAR_INTERVAL_MIN * 60)
                else:
                    time.sleep(BAR_INTERVAL_MIN * 60)
                continue

            # Inside RTH: make sure we actually have a live IB connection.
            if not self._ensure_ib_connected():
                heartbeat_logger.error(
                    "[IB] Not connected during RTH; will retry after short sleep."
                )
                time.sleep(15)
                continue

            if not in_rth_last:
                # First iteration after entering RTH for the day.
                self._log("Entered regular trading hours; trading enabled.")
                in_rth_last = True

            try:
                self._poll_bars_once()
                self._log_open_positions()
            except Exception as e:
                self._log(f"[ERROR] Exception in polling loop: {e}")
                # On error, drop back to next iteration and re-check connectivity
                in_rth_last = False

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

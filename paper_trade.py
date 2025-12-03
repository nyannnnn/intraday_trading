from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from types import SimpleNamespace

import datetime as dt

import pandas as pd
from ib_insync import IB, Stock, MarketOrder

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


STARTING_EQUITY = 100_000.0
MAX_BUFFER_LENGTH = 500

# How long to wait for an order to leave "Pending/Submitted" state (seconds)
ORDER_WAIT_TIMEOUT_SEC = 15


@dataclass
class Position:
    """Single open long position with entry, size, and risk levels."""
    symbol: str
    size: int
    entry_price: float
    entry_dt: pd.Timestamp
    stop_price: float
    take_profit_price: float


class PaperTrader:
    """Polling-based paper trader that pulls IBKR bars and applies the ML signal."""

    def __init__(self) -> None:
        """Initialize empty IB connection, model handle, and trading state."""
        self.ib: Optional[IB] = None
        self.clf = None
        self.contracts: Dict[str, Stock] = {}
        self.ohlcv_buffers: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Position] = {}
        self.start_of_day_equity: float = STARTING_EQUITY
        self.realized_pnl_today: float = 0.0
        self.last_stop_bar: Dict[str, pd.Timestamp] = {}
        
    # ------------ logging helper ------------

    @staticmethod
    def _log(msg: str) -> None:
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}")

    # ------------ setup / connection ------------

    def connect_and_load(self) -> None:
        """Connect to IBKR, load model, qualify contracts, and initialize buffers."""
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
        """Create an empty OHLCV buffer for a symbol."""
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols)

    # ------------ sync existing IB positions on startup ------------

    def _load_existing_ib_positions(self) -> None:
        """
        Load existing IBKR positions into self.positions so restarts keep track.
        Only handles long stock positions in our UNIVERSE.
        """
        self._log("Loading existing IBKR positions into local state...")
        ib_positions = self.ib.positions()  # list of ib_insync.Position

        for p in ib_positions:
            sym = p.contract.symbol

            # Only track symbols in our universe
            if sym not in UNIVERSE:
                continue

            size = int(p.position)
            if size <= 0:
                # Ignore flat/short for now
                continue

            entry_price = float(p.avgCost)
            stop_price = entry_price * (1.0 - STOP_LOSS_PCT)
            take_profit_price = entry_price * (1.0 + TAKE_PROFIT_PCT)

            self.positions[sym] = Position(
                symbol=sym,
                size=size,
                entry_price=entry_price,
                entry_dt=pd.Timestamp.utcnow(),  # unknown true open time
                stop_price=stop_price,
                take_profit_price=take_profit_price,
            )

            self._log(
                f"[SYNC] Loaded existing position from IB: {sym}, "
                f"size={size}, entry_price={entry_price:.4f}, "
                f"stop={stop_price:.4f}, tp={take_profit_price:.4f}"
            )

    # ------------ risk / equity helpers ------------

    def _current_equity(self) -> float:
        """Approximate intraday equity as starting capital plus realized PnL."""
        return self.start_of_day_equity + self.realized_pnl_today

    def _daily_loss_exceeded(self) -> bool:
        """Check whether intraday drawdown breached the configured loss limit."""
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
        """Enforce position, concurrency, daily loss, and cooldown constraints."""
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

        # --- Cooldown after STOP for this symbol ---
        last_stop = self.last_stop_bar.get(symbol)
        if last_stop is not None:
            bars_since_stop = (bar_time - last_stop) / dt.timedelta(minutes=BAR_INTERVAL_MIN)
            if bars_since_stop < COOLDOWN_BARS_AFTER_STOP:
                self._log(
                    f"Skip entry for {symbol}: in cooldown after STOP "
                    f"({bars_since_stop:.1f} bars ago, cooldown="
                    f"{COOLDOWN_BARS_AFTER_STOP} bars)."
                )
                return False

        return True


    def _calc_position_size(self, last_price: float) -> int:
        """Compute trade size as a fixed fraction of equity divided by price."""
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

    # ------------ model / signal ------------

    def _compute_p_up(self, symbol: str) -> Optional[float]:
        """Generate p_up from the model for the latest bar of a symbol."""
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
        """
        Wait for a trade to be filled or cancelled, up to ORDER_WAIT_TIMEOUT_SEC.
        Returns the avg fill price if filled, otherwise None.
        """
        deadline = dt.datetime.now() + dt.timedelta(seconds=ORDER_WAIT_TIMEOUT_SEC)

        while True:
            status = trade.orderStatus.status
            filled = trade.orderStatus.filled
            remaining = trade.orderStatus.remaining

            # Log once in a while if still pending
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

            # Let IB process events
            self.ib.sleep(1)

        status = trade.orderStatus.status
        if status == "Filled" and trade.orderStatus.filled > 0:
            fill_price = trade.orderStatus.avgFillPrice or trade.orderStatus.lastFillPrice
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
        """
        Send a market order to IBKR and return the actual fill price if filled.
        If the order is cancelled/rejected or times out, returns None.
        """
        self._log(
            f"Placing market order: symbol={symbol}, action={action}, size={size}, "
            f"price_hint={price_hint:.4f}"
        )
        contract = self.contracts[symbol]
        order = MarketOrder(action, size)
        # Make TIF explicit to align with presets and reduce warnings
        order.tif = "DAY"

        trade = self.ib.placeOrder(contract, order)
        fill_price = self._wait_for_trade_fill(trade)

        if fill_price is None:
            # No fill — do not treat this as a valid position change
            self._log(
                f"[ORDER] {action} {symbol} size={size} was not filled; "
                "skipping position update."
            )
            return None

        return fill_price

    def _handle_exit(self, symbol: str, last_price: float, bar_time: pd.Timestamp) -> None:
        """
        Close positions that have hit stop-loss, take-profit, or max-hold time.
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return

        # How many bars have we held this position?
        bars_held = (bar_time - pos.entry_dt) / dt.timedelta(minutes=BAR_INTERVAL_MIN)

        hit_stop = last_price <= pos.stop_price
        hit_tp = last_price >= pos.take_profit_price
        hit_max_bars = bars_held >= MAX_BARS_IN_TRADE

        if not (hit_stop or hit_tp or hit_max_bars):
            return

        if hit_stop:
            reason = "STOP"
        elif hit_tp:
            reason = "TAKE_PROFIT"
        else:
            reason = "MAX_BARS"

        self._log(
            f"Exit condition met for {symbol}: last_price={last_price:.4f}, "
            f"stop={pos.stop_price:.4f}, tp={pos.take_profit_price:.4f}, "
            f"bars_held={bars_held:.1f}, reason={reason}"
        )

        exit_price = self._send_market_order(symbol, "SELL", pos.size, last_price)
        if exit_price is None:
            # Exit did not execute; keep position open
            self._log(
                f"[EXIT] Order to close {symbol} was not filled; leaving position open."
            )
            return

        pnl = (exit_price - pos.entry_price) * pos.size
        self.realized_pnl_today += pnl

        self._log(
            f"[EXIT] {symbol}: price={exit_price:.2f}, pnl={pnl:.2f}, "
            f"realized_pnl_today={self.realized_pnl_today:.2f}"
        )

        # Record STOP for cooldown logic (but not for TP or time exit)
        if hit_stop:
            self.last_stop_bar[symbol] = bar_time

        del self.positions[symbol]


    def _handle_entry(
        self, symbol: str, last_price: float, bar_time: pd.Timestamp
    ) -> None:
        """Evaluate signal and, if permitted by risk, open a new long position."""
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
            # Entry order did not fill -> do not open position locally
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
        )

        self._log(
            f"[ENTRY] {symbol}: size={size}, entry={entry_price:.2f}, "
            f"p_up={p_up:.3f}, stop={stop_price:.2f}, tp={take_profit_price:.2f}"
        )

    # ------------ bar handling ------------

    def _on_bar(self, symbol: str, bar) -> None:
        """Update symbol buffer with a new bar and run exit/entry logic."""
        ts = pd.to_datetime(getattr(bar, "date", dt.datetime.utcnow()))
        last_price = float(bar.close)
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

        # First manage exits on existing positions, then consider new entries
        self._handle_exit(symbol, last_price, ts)
        self._handle_entry(symbol, last_price, ts)

    # ------------ polling instead of streaming ------------

    def _poll_bars_once(self) -> None:
        """
        Poll latest historical bars for each symbol and process only new bars.
        This replaces streaming with keepUpToDate, which can be flaky.
        """
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
                keepUpToDate=False,  # snapshot each poll
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

            # First time: seed buffer but DO NOT trade on historical bars
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

            # For each truly new bar, call the same logic as live streaming would
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
        """Start the paper-trading loop by polling bars every BAR_INTERVAL_MIN minutes."""
        self._log("Initializing paper trader...")
        self.connect_and_load()

        # Sync any existing IB paper positions into our local state
        self._load_existing_ib_positions()

        self._log(
            f"Starting polling bar loop (interval={BAR_INTERVAL_MIN} minutes)..."
        )
        while True:
            self._poll_bars_once()
            self._log(
                f"[POLL] Sleeping for {BAR_INTERVAL_MIN} minutes before next poll..."
            )
            # Use ib.sleep so IB's internal event loop continues to process messages
            self.ib.sleep(BAR_INTERVAL_MIN * 60)


def main() -> None:
    """Instantiate a PaperTrader and execute the live paper-trading loop."""
    trader = PaperTrader()
    trader.run()


if __name__ == "__main__":
    main()

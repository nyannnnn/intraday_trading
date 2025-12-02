from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import datetime as dt
from pathlib import Path

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
)

from quant.quant_model import build_features_for_symbol
from backtesting import load_latest_classifier


STARTING_EQUITY = 100_000.0
MAX_BUFFER_LENGTH = 500


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
    """Event-driven paper trader that streams IBKR bars and applies the ML signal."""

    def __init__(self) -> None:
        """Initialize empty IB connection, model handle, and trading state."""
        self.ib: Optional[IB] = None
        self.clf = None
        self.contracts: Dict[str, Stock] = {}
        self.ohlcv_buffers: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Position] = {}
        self.start_of_day_equity: float = STARTING_EQUITY
        self.realized_pnl_today: float = 0.0
        self.bar_subscriptions = {}

    def connect_and_load(self) -> None:
        """Connect to IBKR, load model, qualify contracts, and initialize buffers."""
        self.ib = IB()
        self.ib.connect("127.0.0.1", 7497, clientId=1)

        self.clf = load_latest_classifier(MODEL_DIR)

        for sym in UNIVERSE:
            self.contracts[sym] = Stock(sym, "SMART", "USD")

        self.ib.qualifyContracts(*self.contracts.values())

        for sym in UNIVERSE:
            self.ohlcv_buffers[sym] = self._empty_buffer()

    @staticmethod
    def _empty_buffer() -> pd.DataFrame:
        """Create an empty OHLCV buffer for a symbol."""
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols)

    def _current_equity(self) -> float:
        """Approximate intraday equity as starting capital plus realized PnL."""
        return self.start_of_day_equity + self.realized_pnl_today

    def _daily_loss_exceeded(self) -> bool:
        """Check whether intraday drawdown breached the configured loss limit."""
        eq = self._current_equity()
        if eq <= 0:
            return True

        drawdown_fraction = -self.realized_pnl_today / eq
        return drawdown_fraction >= DAILY_LOSS_STOP_FRACTION

    def _can_open_position(self, symbol: str) -> bool:
        """Enforce position, concurrency, and daily loss constraints before entry."""
        if symbol in self.positions:
            return False
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return False
        if self._daily_loss_exceeded():
            return False
        return True

    def _calc_position_size(self, last_price: float) -> int:
        """Compute trade size as a fixed fraction of equity divided by price."""
        eq = self._current_equity()
        notional = RISK_PER_TRADE_FRACTION * eq
        if last_price <= 0:
            return 0
        size = int(notional / last_price)
        return max(size, 0)

    def _compute_p_up(self, symbol: str) -> Optional[float]:
        """Generate p_up from the model for the latest bar of a symbol."""
        buf = self.ohlcv_buffers[symbol]

        if buf.empty or len(buf) < 50:
            return None

        df = buf.copy()
        df.index = pd.to_datetime(df.index)

        feat_df = build_features_for_symbol(df)
        latest = feat_df.iloc[-1]

        if latest[FEATURE_COLUMNS].isna().any():
            return None

        X = latest[FEATURE_COLUMNS].to_frame().T
        p_up = self.clf.predict_proba(X)[:, 1][0]
        return float(p_up)

    def _send_market_order(self, symbol: str, action: str, size: int, price_hint: float) -> float:
        """Send a market order to IBKR and return an approximate fill price."""
        contract = self.contracts[symbol]
        order = MarketOrder(action, size)
        trade = self.ib.placeOrder(contract, order)

        return float(price_hint)

    def _handle_exit(self, symbol: str, last_price: float) -> None:
        """Close positions that have hit stop-loss or take-profit thresholds."""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        hit_stop = last_price <= pos.stop_price
        hit_tp = last_price >= pos.take_profit_price

        if not (hit_stop or hit_tp):
            return

        exit_price = self._send_market_order(symbol, "SELL", pos.size, last_price)
        pnl = (exit_price - pos.entry_price) * pos.size
        self.realized_pnl_today += pnl

        print(f"[EXIT] {symbol}: price={exit_price:.2f}, pnl={pnl:.2f}")
        del self.positions[symbol]

    def _handle_entry(self, symbol: str, last_price: float, bar_time: pd.Timestamp) -> None:
        """Evaluate signal and, if permitted by risk, open a new long position."""
        if not self._can_open_position(symbol):
            return

        p_up = self._compute_p_up(symbol)
        if p_up is None:
            return
        if p_up < P_UP_ENTRY_THRESHOLD:
            return

        size = self._calc_position_size(last_price)
        if size <= 0:
            return

        stop_price = last_price * (1.0 - STOP_LOSS_PCT)
        take_profit_price = last_price * (1.0 + TAKE_PROFIT_PCT)

        entry_price = self._send_market_order(symbol, "BUY", size, last_price)

        self.positions[symbol] = Position(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            entry_dt=bar_time,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
        )

        print(
            f"[ENTRY] {symbol}: size={size}, entry={entry_price:.2f}, "
            f"p_up={p_up:.3f}, stop={stop_price:.2f}, tp={take_profit_price:.2f}"
        )

    def _on_bar(self, symbol: str, bar) -> None:
        """Update symbol buffer with a new bar and run exit/entry logic."""
        ts = pd.to_datetime(getattr(bar, "date", dt.datetime.utcnow()))
        last_price = float(bar.close)
        volume = float(getattr(bar, "volume", 0.0))

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

        self._handle_exit(symbol, last_price)
        self._handle_entry(symbol, last_price, ts)

    def _make_bar_handler(self, symbol: str):
        """Create an ib_insync updateEvent handler bound to a given symbol."""
        def handler(bars, hasNewBar: bool) -> None:
            if not hasNewBar:
                return
            bar = bars[-1]
            self._on_bar(symbol, bar)

        return handler

    def subscribe_bars(self) -> None:
        """Subscribe to streaming intraday bars for the configured universe."""
        for sym, contract in self.contracts.items():
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting=f"{BAR_INTERVAL_MIN} mins",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=True,
            )
            bars.updateEvent += self._make_bar_handler(sym)
            self.bar_subscriptions[sym] = bars

    def run(self) -> None:
        """Start the paper-trading event loop with live IBKR data."""
        self.connect_and_load()
        self.subscribe_bars()
        print("Starting IBKR paper trading loop...")
        self.ib.run()


def main() -> None:
    """Instantiate a PaperTrader and execute the live paper-trading loop."""
    trader = PaperTrader()
    trader.run()


if __name__ == "__main__":
    main()

# paper_trade.py
"""
Live paper-trading loop for the intraday ML model using IBKR.

Folder layout assumed (your screenshot):
---------------------------------------
INTRADAY_TRADING/
    ML/
        __init__.py
        ml_model.py
    quant/
        __init__.py
        quant_model.py
        data/...
    models/
        intraday_gbm.joblib
    backtesting.py
    config.py
    download_data.py
    train_ml.py
    paper_trade.py   <-- this file

What this script does:
----------------------
1. Connects to IBKR paper TWS / Gateway via ib_insync.
2. Subscribes to streaming 5-min bars (BAR_INTERVAL_MIN from config) for UNIVERSE.
3. Maintains an OHLCV buffer per symbol in memory.
4. On each new completed bar for a symbol:
   - Append the bar to that symbol's buffer.
   - Recompute features with quant.quant_model.build_features_for_symbol.
   - Compute p_up with the trained GBM classifier.
   - If p_up >= P_UP_ENTRY_THRESHOLD and risk checks pass, send BUY order.
5. For open positions:
   - Checks stop-loss / take-profit on each new bar.
   - If hit, sends SELL order and updates realized PnL.
6. Tracks simple intraday realized PnL and applies DAILY_LOSS_STOP_FRACTION.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import datetime as dt
from pathlib import Path

import pandas as pd
from ib_insync import IB, Stock, MarketOrder  # pip install ib_insync

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

# NOTE: quant_model is inside the 'quant' package folder
from quant.quant_model import build_features_for_symbol

from backtesting import load_latest_classifier


# ----------------------------------------------
# High-level trading parameters for this script
# ----------------------------------------------

# Used only for internal risk sizing / daily PnL tracking.
# IB itself will still track real paper PnL.
STARTING_EQUITY = 100_000.0

# Max number of bars we keep per symbol in memory
MAX_BUFFER_LENGTH = 500


# ----------------------------------------------
# Position state
# ----------------------------------------------

@dataclass
class Position:
    """
    Represents one open long position in a symbol.
    """
    symbol: str
    size: int
    entry_price: float
    entry_dt: pd.Timestamp
    stop_price: float
    take_profit_price: float


# ----------------------------------------------
# Main trader class
# ----------------------------------------------

class PaperTrader:
    def __init__(self) -> None:
        # IBKR connection and classifier
        self.ib: Optional[IB] = None
        self.clf = None

        # symbol -> IB contract
        self.contracts: Dict[str, Stock] = {}

        # symbol -> OHLCV DataFrame (DatetimeIndex, cols [open, high, low, close, volume])
        self.ohlcv_buffers: Dict[str, pd.DataFrame] = {}

        # symbol -> Position
        self.positions: Dict[str, Position] = {}

        # PnL and risk
        self.start_of_day_equity: float = STARTING_EQUITY
        self.realized_pnl_today: float = 0.0

        # symbol -> BarDataList subscription
        self.bar_subscriptions = {}

    # =========================
    # Setup / initialization
    # =========================

    def connect_and_load(self) -> None:
        """
        1. Connect to IBKR.
        2. Load the trained classifier.
        3. Build and qualify contracts.
        4. Initialize OHLCV buffers.
        """
        # --- 1. IBKR connection (paper account) ---
        self.ib = IB()
        # Assumes TWS/Gateway is running in PAPER mode on port 7497
        # Change port if your setup is different.
        self.ib.connect("127.0.0.1", 7497, clientId=1)

        # --- 2. Load classifier from MODEL_DIR/intraday_gbm.joblib ---
        self.clf = load_latest_classifier(MODEL_DIR)

        # --- 3. Build Stock contracts for each symbol in UNIVERSE ---
        for sym in UNIVERSE:
            # If these are TSX symbols, change currency/exchange accordingly
            self.contracts[sym] = Stock(sym, "SMART", "USD")

        # Qualify contracts so IBKR fills in ids, exchanges, etc.
        self.ib.qualifyContracts(*self.contracts.values())

        # --- 4. Empty OHLCV buffers ---
        for sym in UNIVERSE:
            self.ohlcv_buffers[sym] = self._empty_buffer()

    @staticmethod
    def _empty_buffer() -> pd.DataFrame:
        cols = ["open", "high", "low", "close", "volume"]
        return pd.DataFrame(columns=cols)

    # =========================
    # Risk helpers
    # =========================

    def _current_equity(self) -> float:
        """
        Approximate intraday equity used for sizing:
        starting_equity + realized_pnl_today.
        (Does not include unrealized PnL.)
        """
        return self.start_of_day_equity + self.realized_pnl_today

    def _daily_loss_exceeded(self) -> bool:
        """
        Returns True if we hit DAILY_LOSS_STOP_FRACTION.
        """
        eq = self._current_equity()
        if eq <= 0:
            return True

        drawdown_fraction = -self.realized_pnl_today / eq
        return drawdown_fraction >= DAILY_LOSS_STOP_FRACTION

    def _can_open_position(self, symbol: str) -> bool:
        """
        Check:
          - not already long that symbol
          - max concurrent positions not exceeded
          - daily loss stop not exceeded
        """
        if symbol in self.positions:
            return False
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return False
        if self._daily_loss_exceeded():
            return False
        return True

    def _calc_position_size(self, last_price: float) -> int:
        """
        Fixed-fraction sizing:
        notional_per_trade = RISK_PER_TRADE_FRACTION * equity
        """
        eq = self._current_equity()
        notional = RISK_PER_TRADE_FRACTION * eq
        if last_price <= 0:
            return 0
        size = int(notional / last_price)
        return max(size, 0)

    # =========================
    # ML signal computation
    # =========================

    def _compute_p_up(self, symbol: str) -> Optional[float]:
        """
        Turn the OHLCV buffer for one symbol into features,
        then ask the GBM model for p_up.
        """
        buf = self.ohlcv_buffers[symbol]

        # Need enough history for rolling windows (VWAP, 60-bar vol, z-scores, etc.)
        if buf.empty or len(buf) < 50:
            return None

        # build_features_for_symbol expects DatetimeIndex + OHLCV columns
        df = buf.copy()
        df.index = pd.to_datetime(df.index)

        feat_df = build_features_for_symbol(df)

        # Take the most recent row as "current bar features"
        latest = feat_df.iloc[-1]

        # Ensure all required feature columns exist and are non-NaN
        if latest[FEATURE_COLUMNS].isna().any():
            return None

        X = latest[FEATURE_COLUMNS].to_frame().T  # shape (1, n_features)
        p_up = self.clf.predict_proba(X)[:, 1][0]
        return float(p_up)

    # =========================
    # Order helpers
    # =========================

    def _send_market_order(self, symbol: str, action: str, size: int, price_hint: float) -> float:
        """
        Send a MarketOrder to IBKR.

        For internal PnL, we approximate fill at price_hint
        (e.g., close of the bar). In real usage you can
        extend this to wait for trade.fills and use the
        actual fill prices.
        """
        contract = self.contracts[symbol]
        order = MarketOrder(action, size)
        trade = self.ib.placeOrder(contract, order)

        # If you want real fill prices, you can:
        # while not trade.isDone():
        #     self.ib.waitOnUpdate()
        # if trade.fills:
        #     price_hint = trade.fills[-1].price

        return float(price_hint)

    # =========================
    # Entry / exit logic
    # =========================

    def _handle_exit(self, symbol: str, last_price: float) -> None:
        """
        If we have an open position and the last_price
        hits stop-loss or take-profit, close the position.
        """
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
        """
        Check ML signal and risk constraints; if passed, open a long.
        """
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

    # =========================
    # Bar handling
    # =========================

    def _on_bar(self, symbol: str, bar) -> None:
        """
        Core per-bar handler:
          - Append new bar to buffer.
          - First evaluate exits, then potential entry.
        """
        # bar.date is typically a datetime or a string like '20251128 10:35:00'
        ts = pd.to_datetime(getattr(bar, "date", dt.datetime.utcnow()))
        last_price = float(bar.close)
        volume = float(getattr(bar, "volume", 0.0))

        # Append to buffer
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

        # 1) Exit logic first (respect stops / targets)
        self._handle_exit(symbol, last_price)

        # 2) Then entry logic (if flat)
        self._handle_entry(symbol, last_price, ts)

    def _make_bar_handler(self, symbol: str):
        """
        Closure that adapts ib_insync's bars.updateEvent callback
        to call _on_bar(symbol, latest_bar).
        """

        def handler(bars, hasNewBar: bool) -> None:
            if not hasNewBar:
                return
            bar = bars[-1]
            self._on_bar(symbol, bar)

        return handler

    # =========================
    # Subscription & main loop
    # =========================

    def subscribe_bars(self) -> None:
        """
        Subscribe to streaming 5-min bars for each symbol in UNIVERSE.
        """
        for sym, contract in self.contracts.items():
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="2 D",  # enough history to seed features
                barSizeSetting=f"{BAR_INTERVAL_MIN} mins",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=True,  # <-- streaming updates
            )
            bars.updateEvent += self._make_bar_handler(sym)
            self.bar_subscriptions[sym] = bars

    def run(self) -> None:
        """
        Entry point: connect, subscribe, and start ib_insync event loop.
        """
        self.connect_and_load()
        self.subscribe_bars()
        print("Starting IBKR paper trading loop...")
        self.ib.run()


# ----------------------------------------------
# Script entry point
# ----------------------------------------------

def main() -> None:
    trader = PaperTrader()
    trader.run()


if __name__ == "__main__":
    main()

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import logging
import os
import csv
import time
import json
import smtplib
import traceback
import random
from logging.handlers import RotatingFileHandler
from email.mime.text import MIMEText

# === IB Imports ===
from ib_insync import (
    IB, Stock, MarketOrder, LimitOrder, StopOrder, 
    Trade, BarDataList, Contract
)

# === Config Imports ===
# Assuming these exist in your config.py. 
# If not, the script uses defaults defined below in SAFETY DEFAULTS.
try:
    from config import (
        UNIVERSE, MODEL_DIR, FEATURE_COLUMNS, BAR_INTERVAL_MIN,
        P_UP_ENTRY_THRESHOLD, RISK_PER_TRADE_FRACTION,
        ATR_WINDOW, ATR_STOP_MULT, ATR_TP_MULT,
        MAX_CONCURRENT_POSITIONS, DAILY_LOSS_STOP_FRACTION,
        MAX_BARS_IN_TRADE, COOLDOWN_BARS_AFTER_STOP
    )
    from quant.quant_model import build_features_for_symbol
    from backtesting import load_latest_classifier
except ImportError:
    # --- SAFETY DEFAULTS FOR TESTING (If config.py is missing) ---
    print("WARNING: Config not found, using defaults.")
    UNIVERSE = ['SOFI', 'AAPL', 'AMD']
    MODEL_DIR = "models"
    FEATURE_COLUMNS = ['close', 'volume'] # Placeholder
    BAR_INTERVAL_MIN = 5
    P_UP_ENTRY_THRESHOLD = 0.55
    RISK_PER_TRADE_FRACTION = 0.01
    ATR_WINDOW = 14
    ATR_STOP_MULT = 2.0
    ATR_TP_MULT = 3.0
    MAX_CONCURRENT_POSITIONS = 3
    DAILY_LOSS_STOP_FRACTION = 0.02
    MAX_BARS_IN_TRADE = 24
    COOLDOWN_BARS_AFTER_STOP = 3
    
    # Mocking external functions for standalone run capability
    def build_features_for_symbol(df): 
        # Simple Mock Feature
        df['f1'] = df['close'].pct_change()
        return df
        
    class MockClf:
        def predict_proba(self, X): return np.array([[0.4, 0.6]]) # Always returns 0.6 prob
    def load_latest_classifier(d): return MockClf()


# === System Constants ===
MAX_NOTIONAL_PER_TRADE = 25000.0
MAX_SHARES_PER_TRADE = 2000
STATE_FILE = "trade_state.json"
STARTING_EQUITY = 100000.0
MAX_BUFFER_LENGTH = 200 # Keep short for speed, only need enough for features
MIN_BARS_REQUIRED = 30  # Minimum bars to start calculating features

# === Email/Alert Config ===
ALERT_EMAIL_TO = os.environ.get("TRADER_ALERT_EMAIL_TO")
ALERT_EMAIL_FROM = os.environ.get("TRADER_ALERT_EMAIL_FROM")
SMTP_HOST = os.environ.get("TRADER_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("TRADER_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("TRADER_SMTP_USER")
SMTP_PASS = os.environ.get("TRADER_SMTP_PASS")

# === Logging Setup ===
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "paper_trade.log")
TRADES_CSV_PATH = os.path.join(LOG_DIR, "paper_trades.csv")

logger = logging.getLogger("paper_trade")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

if not logger.handlers:
    fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

# === Helper Classes ===

@dataclass
class Position:
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

# === Helper Functions ===

def save_state_to_disk(positions: Dict[str, Position]):
    try:
        data = {sym: pos.to_dict() for sym, pos in positions.items()}
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logger.error(f"[STATE] Save failed: {e}")

def load_state_from_disk() -> Dict[str, Position]:
    if not os.path.exists(STATE_FILE): return {}
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        return {sym: Position.from_dict(d) for sym, d in data.items()}
    except Exception as e:
        logger.error(f"[STATE] Load failed: {e}")
        return {}

def send_email(subject: str, body: str):
    if not ALERT_EMAIL_TO or not SMTP_USER: 
        logger.warning(f"[EMAIL-SKIP] No creds. Subject: {subject}")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = ALERT_EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
    except Exception as e:
        logger.error(f"[EMAIL-FAIL] {e}")

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1: return 0.0
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return float(tr.rolling(window=period).mean().iloc[-1])

# === Main Trader Class ===

class PaperTrader:
    def __init__(self):
        self.ib = IB()
        self.clf = None
        self.contracts: Dict[str, Stock] = {}
        
        # New: Live subscriptions container
        self.live_bars: Dict[str, BarDataList] = {} 
        
        self.positions: Dict[str, Position] = load_state_from_disk()
        self.last_stop_bar: Dict[str, pd.Timestamp] = {}
        
        self.start_equity = STARTING_EQUITY
        self.realized_pnl = 0.0
        self.trading_date = dt.date.today()

    def _log(self, msg: str):
        logger.info(msg)

    # --- Connection & Setup ---
    
    def connect(self):
        """Connects to IB, sets up data subscriptions, and waits for history."""
        self._log("Connecting to IB Gateway...")
        try:
            # Use a random client ID to avoid conflicts
            cid = random.randint(100, 9999)
            self.ib.connect('127.0.0.1', 7497, clientId=cid, timeout=15)
            self._log(f"Connected (ClientID: {cid})")
        except Exception as e:
            raise ConnectionError(f"Could not connect to IB: {e}")

        # Load Model
        try:
            self.clf = load_latest_classifier(MODEL_DIR)
            self._log("Model loaded successfully.")
        except Exception as e:
            self._log(f"[WARN] Model load failed: {e}. Ensure models exist.")

        # --- SUBSCRIPTION PHASE ---
        self._log(f"Subscribing to {len(UNIVERSE)} symbols...")
        for sym in UNIVERSE:
            contract = Stock(sym, 'SMART', 'USD')
            self.contracts[sym] = contract
            
            # This request fetches 2 days of history AND keeps appending new bars
            bars = self.ib.reqHistoricalData(
                contract, endDateTime='', durationStr='2 D',
                barSizeSetting=f'{BAR_INTERVAL_MIN} mins',
                whatToShow='TRADES', useRTH=True,
                formatDate=1, keepUpToDate=True
            )
            self.live_bars[sym] = bars
        
        # --- WARMUP WAIT PHASE ---
        self._log("Waiting for historical data to download...")
        start_wait = time.time()
        
        while True:
            # Check how many symbols have enough bars
            loaded_count = 0
            for sym, bars in self.live_bars.items():
                if len(bars) >= MIN_BARS_REQUIRED:
                    loaded_count += 1
            
            # If all are ready, break
            if loaded_count == len(UNIVERSE):
                self._log(f"History loaded for all {len(UNIVERSE)} symbols.")
                break
            
            # Timeout check (e.g., 45 seconds)
            if time.time() - start_wait > 45:
                self._log(f"[WARN] Timeout waiting for history. Proceeding with {loaded_count}/{len(UNIVERSE)} ready.")
                break
                
            self.ib.sleep(1) # Allow background thread to process messages

        self._log("Data subscriptions active.")
        self._sync_positions()

    def _sync_positions(self):
        """Syncs local state with actual IB portfolio."""
        ib_pos = {p.contract.symbol: p.position for p in self.ib.positions()}
        
        # Remove ghosts
        for sym in list(self.positions.keys()):
            if sym not in ib_pos or ib_pos[sym] == 0:
                self._log(f"[SYNC] {sym} not in IB portfolio. Removing local state.")
                del self.positions[sym]
        
        # Save clean state
        save_state_to_disk(self.positions)

    # --- Trading Logic ---

    def _process_data_to_df(self, bars: BarDataList) -> pd.DataFrame:
        """Converts ib_insync bars to clean DataFrame."""
        if not bars: return pd.DataFrame()
        
        data = [{'date': b.date, 'open': b.open, 'high': b.high, 
                 'low': b.low, 'close': b.close, 'volume': b.volume} for b in bars]
        df = pd.DataFrame(data)
        
        # Force UTC to avoid tz_localize errors
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        return df

    def _place_bracket_order(self, symbol: str, quantity: int, price: float, atr: float, p_up: float):
        contract = self.contracts[symbol]
        
        stop_dist = max(round(atr * ATR_STOP_MULT, 2), 0.05)
        tp_dist = max(round(atr * ATR_TP_MULT, 2), 0.05)
        
        # Create Order Objects
        parent = MarketOrder('BUY', quantity)
        parent.transmit = False # Do not send yet
        
        take_profit = LimitOrder('SELL', quantity, round(price + tp_dist, 2))
        take_profit.transmit = False
        
        stop_loss = StopOrder('SELL', quantity, round(price - stop_dist, 2))
        stop_loss.transmit = True # Sending this triggers the whole group
        
        # Place Parent first to establish the group (IB insync handles wrapping)
        try:
            parent_trade = self.ib.placeOrder(contract, parent)
            
            take_profit.parentId = parent.orderId
            stop_loss.parentId = parent.orderId
            
            tp_trade = self.ib.placeOrder(contract, take_profit)
            sl_trade = self.ib.placeOrder(contract, stop_loss)
            
            self._log(f"[ORDER] {symbol} Bracket Sent. Entry est: {price}, TP: {take_profit.lmtPrice}, SL: {stop_loss.auxPrice}")
            
            # Record State
            self.positions[symbol] = Position(
                symbol=symbol, entry_dt=dt.datetime.utcnow(), size=quantity,
                entry_price=price, stop_price=stop_loss.auxPrice,
                take_profit_price=take_profit.lmtPrice, p_up=p_up, bars_held=0
            )
            save_state_to_disk(self.positions)
            
        except Exception as e:
            self._log(f"[ORDER-FAIL] {symbol}: {e}")

    def _check_exit_conditions(self, symbol: str):
        # Time-based exit only. TP/SL are handled by IB server (Bracket Orders).
        if symbol not in self.positions: return
        
        pos = self.positions[symbol]
        pos.bars_held += 1
        
        # Check Max Bars
        if pos.bars_held >= MAX_BARS_IN_TRADE:
            self._log(f"[EXIT] {symbol} Max bars reached ({MAX_BARS_IN_TRADE}). Closing.")
            self._close_position(symbol, "MAX_BARS")
        
        save_state_to_disk(self.positions)

    def _close_position(self, symbol: str, reason: str):
        # Cancel any open orders for this symbol first
        open_orders = self.ib.openOrders()
        for o in open_orders:
            if o.contract.symbol == symbol:
                self.ib.cancelOrder(o)
        
        # Close position if it exists
        positions = self.ib.positions()
        for p in positions:
            if p.contract.symbol == symbol and p.position != 0:
                direction = 'SELL' if p.position > 0 else 'BUY'
                order = MarketOrder(direction, abs(p.position))
                self.ib.placeOrder(self.contracts[symbol], order)
                self._log(f"[CLOSE] {symbol} closed via Market. Reason: {reason}")
        
        if symbol in self.positions:
            del self.positions[symbol]
            save_state_to_disk(self.positions)

    # --- The Loop ---

    def _run_cycle(self):
        """Single iteration of the logic."""
        
        # 1. Update Portfolio Stats (Robust Method)
        try:
            current_eq = STARTING_EQUITY
            summaries = self.ib.accountSummary()
            if summaries:
                # Find the tag manually in the list
                net_liq_obj = next((s for s in summaries if s.tag == 'NetLiquidation'), None)
                if net_liq_obj:
                    current_eq = float(net_liq_obj.value)
        except Exception as e:
            self._log(f"[ACCT-WARN] Could not read equity: {e}")
            current_eq = STARTING_EQUITY
        
        # 2. Iterate Universe
        for symbol in UNIVERSE:
            bars = self.live_bars.get(symbol)
            if not bars or len(bars) < MIN_BARS_REQUIRED:
                continue
            
            # Convert to DF (Only need last N bars for features)
            df = self._process_data_to_df(bars[-MAX_BUFFER_LENGTH:])
            if df.empty: continue
            
            # Check if Data is Fresh (within last 10 mins)
            last_ts = df.index[-1]
            now_utc = pd.Timestamp.utcnow()
            if (now_utc - last_ts).seconds > (BAR_INTERVAL_MIN * 60 * 2):
                # Data is stale (market closed or lag), skip
                continue

            # --- Manage Exits ---
            if symbol in self.positions:
                self._check_exit_conditions(symbol)
                continue # Skip entry logic if in position

            # --- Manage Entries ---
            
            # Feature Engineering
            try:
                # Assuming build_features_for_symbol handles standard pandas logic
                # Pass a copy to ensure thread safety
                features_df = build_features_for_symbol(df.copy())
                current_row = features_df.iloc[-1]
                
                # Check for NaN in critical columns
                if current_row[FEATURE_COLUMNS].isna().any():
                    continue

                # Predict
                X = current_row[FEATURE_COLUMNS].values.reshape(1, -1)
                p_up = self.clf.predict_proba(X)[0][1]
                atr = calculate_atr(df, ATR_WINDOW)
                
                # Default safety for threshold
                thresh = P_UP_ENTRY_THRESHOLD if P_UP_ENTRY_THRESHOLD > 0 else 0.55

                # Log signal periodically (e.g. if high enough to be interesting)
                if p_up > 0.5:
                     self._log(f"[SIGNAL] {symbol}: p={p_up:.3f} (Req: {thresh}) ATR={atr:.2f}")

                if p_up >= thresh and atr > 0:
                    current_price = df['close'].iloc[-1]
                    
                    # Size Logic
                    risk_amt = STARTING_EQUITY * RISK_PER_TRADE_FRACTION
                    risk_per_share = atr * ATR_STOP_MULT
                    qty = int(risk_amt / risk_per_share)
                    
                    # Cap constraints
                    qty = min(qty, int(MAX_NOTIONAL_PER_TRADE / current_price))
                    qty = min(qty, MAX_SHARES_PER_TRADE)
                    
                    if qty > 0 and len(self.positions) < MAX_CONCURRENT_POSITIONS:
                         self._place_bracket_order(symbol, qty, current_price, atr, p_up)

            except Exception as e:
                logger.error(f"[Loop Error] {symbol}: {e}")
                # Don't crash the whole loop for one symbol error

    def run(self):
        self.connect()
        self._log("Starting Event Loop...")
        
        while True:
            try:
                # 1. Allow IB thread to process network messages
                # This updates the live_bars in the background
                self.ib.sleep(2.0)
                
                # 2. Run Strategy Logic
                # Since data is in memory, this is fast
                self._run_cycle()
                
                # 3. Heartbeat log every so often
                if int(time.time()) % 60 == 0:
                    self._log(f"[HEARTBEAT] Connected. Open Pos: {len(self.positions)}")

            except KeyboardInterrupt:
                self._log("Stopping...")
                self.ib.disconnect()
                break
            except Exception as e:
                self._log(f"[CRASH RECOVERY] {e}")
                send_email("Trader Crash", traceback.format_exc())
                self.ib.sleep(10)
                # Attempt reconnect
                if not self.ib.isConnected():
                    try: 
                        self.ib.connect('127.0.0.1', 7497, clientId=random.randint(100,999))
                    except: pass

if __name__ == "__main__":
    trader = PaperTrader()
    trader.run()
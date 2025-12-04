# Intraday ML Trading System

Intraday, ML-powered equity trading system with:

- End-to-end **feature engineering → model training → backtesting → live paper trading**
- Integration with **Interactive Brokers (IBKR)** via `ib_insync`
- **Risk-managed execution** (stops, targets, daily loss limit, cooldown logic)
- **Cloud deployment** on AWS Lightsail with IB Gateway and `tmux`
- **Detailed logging** and **daily trade summaries**

---

## Overview

This project implements an **intraday long-only equity strategy** driven by a machine learning classifier.

The high-level pipeline:

1. **Download / prepare intraday OHLCV data**
2. **Build engineered features** per symbol (`quant_model.py`)
3. **Train a classifier** to predict the probability of an upward move (`train_ml.py`, `ml_model.py`)
4. **Backtest the strategy** with realistic risk management (`backtesting.py`)
5. **Run live on IBKR paper** using a polling-based engine (`paper_trade.py`)

The live engine connects to **IB Gateway/TWS**, polls recent bars via `reqHistoricalData`, builds features, generates signals, and sends market orders via `ib_insync` while enforcing strict risk and position management constraints.

---

## Features

### Strategy & ML

- Uses a **classification model** (e.g. scikit-learn) to estimate `p_up` = P(price up).
- Strategy enters a long position when:
  - enough bars are available for features,
  - no cooldown/daily loss constraint is violated,
  - and `p_up >= P_UP_ENTRY_THRESHOLD`.
- Exit logic:
  - **Stop-loss**
  - **Take-profit**
  - **Maximum hold duration** (in bars)

### Risk Management

- Fixed-fraction position sizing: allocate `RISK_PER_TRADE_FRACTION * equity` per trade.
- **Daily loss stop**: stop opening new trades once an intraday drawdown threshold is hit.
- **Max concurrent positions**: cap how many symbols can be long at the same time.
- **Symbol-level cooldown** after a STOP exit: avoid overtrading a symbol that just hit a stop.

### Architecture

- `ib_insync`-based IBKR API client
- Polling using `reqHistoricalData` instead of streaming (more robust for long-running processes)
- ML model loaded from a serialized `.joblib` file
- Designed to run:
  - locally on your machine, or
  - remotely on an AWS Lightsail Ubuntu instance with IB Gateway/TWS

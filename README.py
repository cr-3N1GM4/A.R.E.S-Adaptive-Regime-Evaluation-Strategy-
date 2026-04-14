# =============================================================================
# README.py
# =============================================================================
#
#   1. DAILY stock selection (cross-sectional factor model)
#   2. INTRADAY reinforcement learning execution (PPO)
#   3. PORTFOLIO-level backtesting
#
# The system is designed for:
#   - Research-grade backtesting
#   - Competitions
#   - Serious quantitative experimentation
#
# =============================================================================
# TABLE OF CONTENTS
# =============================================================================
#
# 1. High-Level Architecture
# 2. Project Folder Structure
# 3. Data Requirements (CSV format)
# 4. Intraday RL Engine (RL.py)
# 5. Daily Stock Selection (StockSelection.py)
# 6. Portfolio Testing
# 7. Commands & Usage
# 8. Configuration
# 9. Output Files & Metrics
# 10. Lookahead-Free Guarantees
# 11. Troubleshooting
# 12. Recommended Workflow
#
# =============================================================================
# 1. HIGH-LEVEL ARCHITECTURE
# =============================================================================
#
# ┌────────────────────────────┐
# │  DAILY DATA (yfinance)     │
# └────────────┬───────────────┘
#              │
#              ▼
# ┌────────────────────────────┐
# │ StockSelection.py          │
# │ - Momentum                 │
# │ - Volatility               │
# │ - Liquidity                │
# │ - Beta                     │
# └────────────┬───────────────┘
#              │
#              ▼
# ┌────────────────────────────┐
# │ selection_map.json         │
# │ (Date → Selected Stocks)  │
# └────────────┬───────────────┘
#              │
#              ▼
# ┌────────────────────────────┐
# │ RL.py (Intraday PPO)       │
# │ - HMM + XGBoost regimes    │
# │ - PPO execution            │
# │ - Close-to-close trading   │
# └────────────┬───────────────┘
#              │
#              ▼
# ┌────────────────────────────┐
# │ Portfolio Backtest         │
# │ - Capital allocation       │
# │ - Aggregated PnL           │
# └────────────────────────────┘
#
# =============================================================================
# 2. PROJECT FOLDER STRUCTURE
# =============================================================================
#
# project_folder/
# ├── RL.py                         # Main RL engine
# ├── StockSelection.py             # Daily stock selection
# ├── README.py                     # THIS FILE
# ├── selection_map.json            # Generated daily selections
# │
# ├── data/                         # RAW DATA
# │   ├── SWIGGY_minute.csv
# │   ├── RELIANCE_minute.csv
# │   ├── ABB_NS.csv
# │   ├── HDFCBANK_NS.csv
# │   └── MARKET_NSEI.csv
# │
# ├── SWIGGY_2min/                  # Auto-generated intraday days
# │   ├── day0.csv
# │   ├── day1.csv
# │   └── ...
# │
# ├── Models_SWIGGY/                # Saved models
# │   ├── ppo_trading_model_SWIGGY.zip
# │   ├── ppo_trading_model_SWIGGY_vecnormalize.pkl
# │   ├── regime_detector_model.pkl
# │   └── hmm_model.pkl
# │
# ├── signals_SWIGGY/               # BUY / SELL / EXIT signals
# ├── test_trade_plots/             # Intraday charts
# ├── test_results/                 # Performance reports
# └── portfolio_test_results/       # Portfolio equity & metrics
#
# =============================================================================
# 3. DATA REQUIREMENTS (CSV FORMAT)
# =============================================================================
#
# INTRADAY DATA (for RL.py):
#
# Filename:
#   TICKER_minute.csv
# Example:
#   SWIGGY_minute.csv
#
# Required columns (case-insensitive):
#   Time / Date / Datetime
#   Open
#   High
#   Low
#   Close
#   Volume   (MANDATORY)
#
# Example row:
#   2024-01-15 09:15:00,550.25,551.00,549.50,550.75,125000
#
# DAILY DATA (for StockSelection.py):
#
# Automatically downloaded via yfinance and cached in data/
#
# =============================================================================
# 4. INTRADAY RL ENGINE (RL.py)
# =============================================================================
#
# Core features:
#   - PPO (Stable-Baselines3)
#   - Close-to-close execution
#   - Regime detection using:
#       * Hidden Markov Models (HMM)
#       * XGBoost classifier
#
# EXECUTION MODEL:
#
# Step t:
#   - Agent sees indicators computed from Close[t-1] and earlier
#   - Agent chooses action
#   - Trade executes at Close[t]
#
# Step t+1:
#   - Stop-loss / trailing-stop evaluated at Close[t+1]
#
# No intrabar assumptions. No future access.
#
# =============================================================================
# 5. DAILY STOCK SELECTION (StockSelection.py)
# =============================================================================
#
# Purpose:
#   - Select top-K stocks EACH DAY from a large universe
#   - Provide inputs for portfolio-level RL trading
#
# Factors used:
#   - Long momentum (60-day)
#   - Medium momentum (20-day)
#   - Volatility
#   - Liquidity
#   - NATR
#   - Beta vs NSE Index
#
# Method:
#   - Cross-sectional z-scoring PER DAY
#   - Weighted composite score
#   - Top-K ranking (default K = 10)
#
# Output:
#   selection_map.json
#
# Format:
# {
#   "2024-01-11": ["HDFC", "ICICIBANK", "RELIANCE"],
#   "2024-01-12": ["TCS", "INFY", "LT"]
# }
#
# IMPORTANT:
#   Selection for day D+1 uses ONLY information from day D.
#
# =============================================================================
# 6. PORTFOLIO TESTING
# =============================================================================
#
# Command:
#   python RL.py portfolio_test
#
# What happens:
#   - Reads selection_map.json
#   - For each day:
#       * Loads selected stocks
#       * Runs each stock’s RL agent independently
#       * Allocates capital using inverse-volatility weighting
#   - Aggregates daily portfolio PnL
#
# This simulates a REALISTIC multi-asset trading desk.
#
# =============================================================================
# 7. COMMANDS & USAGE
# =============================================================================
#
# Train ONE stock:
#   python RL.py train SWIGGY
#
# Train ALL stocks:
#   python RL.py train_all
#
# Test ONE stock:
#   python RL.py test SWIGGY
#
# Verify NO lookahead:
#   python RL.py verify SWIGGY
#
# Run DAILY stock selection:
#   python StockSelection.py
#
# Run PORTFOLIO backtest:
#   python RL.py portfolio_test
#
# =============================================================================
# 8. CONFIGURATION (RL.py → PARAMS)
# =============================================================================
#
# PARAMS = {
#   'WARMUP_MINUTES': 30,
#   'TRANSACTION_COST': 0.0005,
#   'WINDOW_SIZE': 10,
#   'STOP_LOSS_TR': -0.0005,
#   'TRAIL_PCT_TR': 0.0005,
#   'STOP_LOSS_TE': -0.0004,
#   'TRAIL_PCT_TE': 0.0004,
#   'NUM_PROCS': 8,
#   'EPISODES': 1_000_000,
#   'CANDLE_FREQUENCY': '2min',
#   'SEED': 2
# }
#
# =============================================================================
# 9. OUTPUT METRICS
# =============================================================================
#
# Example:
#
# FINAL SUMMARY - SWIGGY
# ---------------------
# Total Days: 150
# Initial Capital: $100,000
# Final Equity: $125,432
# Total PnL: $25,432
# CAGR: 42.3%
# Sharpe: 1.85
# Max Drawdown: 8.2%
# Calmar: 5.15
#
# =============================================================================
# 10. LOOKAHEAD-FREE GUARANTEES
# =============================================================================
#
# This system is MATHEMATICALLY LOOKAHEAD-FREE:
#
# 1. All indicators are computed then shifted by 1 bar
# 2. Decisions at t only see data ≤ t-1
# 3. Close-to-close execution
# 4. Chronological train/test split
# 5. VecNormalize stats frozen during testing
# 6. Regime models trained strictly on past data
# 7. Explicit verification via:
#
#       python RL.py verify TICKER
#
# =============================================================================
# 11. TROUBLESHOOTING
# =============================================================================
#
# No data found:
#   - Check TICKER_minute.csv exists
#
# Volume missing:
#   - Volume column is mandatory
#
# Slow training:
#   - Reduce EPISODES
#   - Check CUDA availability
#
# =============================================================================
# 12. RECOMMENDED WORKFLOW
# =============================================================================
#
# 1. Place intraday CSVs in project folder
# 2. Run daily stock selection:
#       python StockSelection.py
# 3. Train all RL models:
#       python RL.py train_all
# 4. Verify lookahead safety:
#       python RL.py verify TICKER
# 5. Run portfolio backtest:
#       python RL.py portfolio_test
#
# =============================================================================
# END OF README
# =============================================================================

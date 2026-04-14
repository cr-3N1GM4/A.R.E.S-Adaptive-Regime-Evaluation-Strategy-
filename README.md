# A.R.E.S-Adaptive-Regime-Evaluation-Strategy-
This project builds a regime-aware trading system using quantitative factors, ML, and reinforcement learning. It selects stocks, detects market conditions, and uses a PPO agent to trade adaptively, aiming for consistent returns with strong risk management and low drawdowns.

## 🚀 Key Features

* **Strict Lookahead-Free Execution:** A mathematically sound "close-to-close" execution model. All features are rigorously pre-shifted. The agent observes data up to $t-1$ to make a decision executed at the close of $t$.
* **Dual-Model Regime Detection:** * **HMM:** An online, forward-algorithm Gaussian HMM calculates real-time probabilities of high-volatility/risky market states.
    * **XGBoost:** Predicts localized concurrent volatility regimes (categorized into quantiles) without future data leakage.
* **Deep Feature Engineering:** Computes over 50 technical and microstructural features, including Johnny Ribbon regimes, multi-iteration Heikin Ashi, KAMA, VWAP deviation, and Hurst exponents.
* **Dynamic Portfolio Selection:** `StockSelection.py` ranks a universe of NSE stocks daily based on momentum, volatility, liquidity, normalized ATR, and Beta, generating a regime-aware trading universe (`selection_map.json`).
* **Advanced Risk Management:** Built-in dynamic trailing stops, strict stop-losses, and time-based EOD forced closures integrated directly into the reward function logic.

---

## 🏗️ System Architecture

1. **Universe Selection (`StockSelection.py`):** Downloads historical data via `yfinance`, calculates z-scored fundamental and technical metrics, and builds a JSON map of the top $K$ assets to trade for every forward business day.
2. **Data Preparation & Indicator Computation (`RL.py`):** Resamples 1-minute data to customizable intraday frequencies (e.g., 2-minute candles) and processes the complex feature space.
3. **Regime Training:** Isolates the first 10% of historical data chronologically to train the HMM and XGBoost models, completely separated from the RL training phase.
4. **Agent Training:** The next 50% of the timeline is used to train the Stable-Baselines3 PPO agent across parallelized SubprocVecEnv/DummyVecEnv instances. 
5. **Out-of-Sample Testing & Portfolio Simulation:** The final 40% of data is used to test individual asset performance or simulate the entire dynamic portfolio sequentially.

---

## ⚙️ Installation

**Prerequisites:** Python 3.9+ 

Clone the repository and install the required scientific and quantitative libraries:

```bash
pip install numpy pandas yfinance torch stable-baselines3 gym gymnasium xgboost hmmlearn scipy tqdm matplotlib "kagglehub[pandas-datasets]"
```
💾 Data Acquisition: Historical 1-Minute Data

While StockSelection.py utilizes yfinance for broader daily metrics, historical 1-minute intraday data is restricted on standard free APIs. You can easily source high-quality 1-minute Nifty 50 data directly from Kaggle into your /data directory using kagglehub.
Python

import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the specific file path you'd like to load, or leave blank to download the dataset directory
file_path = ""

# Load the latest version of the Nifty 50 1-min dataset into a pandas DataFrame
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "debashis74017/stock-market-data-nifty-50-stocks-1-min-data",
  file_path,
)

print("First 5 records:\n", df.head())

📖 RL Trading System - Usage Guide
📋 Quick Start Example

    Place your data in the project folder:
    Plaintext

    project_folder/SWIGGY_minute.csv

    Open terminal in project folder and train:
    Bash

    python RL.py train SWIGGY

    Test:
    Bash

    python RL.py test SWIGGY

    Verify no lookahead:
    Bash

    python RL.py verify SWIGGY

🚀 Commands
1. Train a Model
Bash

python RL.py train SWIGGY

What it does:

    Looks for SWIGGY_minute.csv or SWIGGY.csv in the current folder.

    Resamples to 2-minute candles → saves in SWIGGY_2min/.

    Splits data: 10% → Regime detector, 50% → RL training, 40% → Testing.

    Trains HMM + XGBoost regime detector.

    Trains PPO agent (1,000,000 timesteps by default).

    Saves models in Models_SWIGGY/.
    Expected runtime: 30-60 minutes (depends on GPU and data size).

2. Test a Trained Model
Bash

python RL.py test SWIGGY

What it does:

    Loads the trained model from Models_SWIGGY/.

    Tests on the last 40% of data (held-out test set).

    Generates trade charts in test_trade_plots/.

    Saves signals in signals_SWIGGY/.

    Outputs metrics (CAGR, Sharpe, Max Drawdown, etc.).

3. Verify No Lookahead
Bash

python RL.py verify SWIGGY

What it does:

    Corrupts future data and checks if the observation changes.

    If observation is unchanged → ✅ No lookahead.

    Mathematically proves the backtesting is valid.

4. Train All Stocks
Bash

python RL.py train_all

What it does:

    Scans the current folder for all *_minute.csv files.

    Trains a model for each ticker found.

⚙️ Configuration

Edit PARAMS at the top of RL.py to adjust system behavior:
Python

PARAMS = {
    'WARMUP_MINUTES': 30,           # Skip first 30 min each day
    'TRANSACTION_COST': 0.0005,     # 0.05% per side (0.10% round-trip)
    'WINDOW_SIZE': 10,              # Lookback window for features
    'STOP_LOSS_TR': -0.0005,        # Training stop-loss (-0.05%)
    'TRAIL_PCT_TR': 0.0005,         # Training trailing stop
    'TAKE_PROFIT_TR': 1.0,          # Training take-profit (100% = disabled)
    'STOP_LOSS_TE': -0.0004,        # Testing stop-loss
    'TRAIL_PCT_TE': 0.0004,         # Testing trailing stop
    'TAKE_PROFIT_TE': 1.0,          # Testing take-profit
    'NUM_PROCS': 8,                 # Parallel environments
    'EPISODES': 1_000_000,          # Training timesteps
    'CANDLE_FREQUENCY': '2min',     # Candle resampling frequency
    'SEED': 2,                      # Random seed for reproducibility
}

📁 File Structure
Plaintext

project_folder/
├── RL.py                          # Main script
├── StockSelection.py              # Screener script
├── SWIGGY_minute.csv              # Source data file (YOUR DATA)
├── RELIANCE_minute.csv            # Another stock (optional)
│
├── SWIGGY_2min/                   # Auto-created after prepare_data
│   ├── day0.csv
│   ├── day1.csv
│   └── ...
│
├── Models_SWIGGY/                 # Auto-created after training
│   ├── ppo_trading_model_SWIGGY.zip
│   ├── ppo_trading_model_SWIGGY_vecnormalize.pkl
│   ├── regime_detector_model.pkl
│   └── hmm_model.pkl
│
├── test_results/                  # Auto-created after testing
│   └── test_results_SWIGGY.txt
│
├── test_trade_plots/              # Auto-created - trade charts
│   └── SWIGGY_day_1_day0.png
│
└── signals_SWIGGY/                # Auto-created - signal files
    └── day0.csv

📊 Data Format

Your source CSV file (e.g., SWIGGY_minute.csv) should have these columns:
Column	Description	Example
Time or Date or Datetime	Timestamp	2024-01-15 09:15:00
Open	Open price	550.25
High	High price	551.00
Low	Low price	549.50
Close	Close price	550.75
Volume	Volume	125000

Note: Column names are case-insensitive (open, Open, OPEN all work).
📈 Output Metrics

After running python RL.py test SWIGGY, you will see an output block summarizing the performance:
Plaintext

================================================================
FINAL SUMMARY - SWIGGY
================================================================
Total Days: 150
Initial Capital: $100,000.00
Final Equity: $125,432.50
Total PnL: $25,432.50
Winning Days: 85
Losing Days: 55
Total Trades: 312

Risk Metrics:
CAGR (Compounded): 42.35%
Sharpe Ratio: 1.85
Max Drawdown: 8.23%
Calmar Ratio: 5.15
================================================================

📉 Signal File Format

After testing, the file located at signals_SWIGGY/day0.csv will log exact agent actions:
Time	Price	BUY	SELL	EXIT
09:32:00	550.25	0	0	0
09:34:00	551.00	1	0	0
09:36:00	550.50	0	0	0
09:38:00	549.00	0	0	1

    BUY=1: Open long position

    SELL=1: Open short position

    EXIT=1: Close position

🔒 Lookahead Prevention Guarantee

This code is strictly designed to eliminate lookahead bias via the following mechanisms:

    Feature Shifting: All indicators are shifted by 1 bar before the agent sees them.

    Decision Segregation: The decision at time t uses data exclusively from t−1. The agent never sees the current bar's indicators before acting.

    Close-to-Close Execution: A consistent, realistic execution model.

    Chronological Splits: Train and test datasets are strictly time-ordered.

    Verification Suite: Run python RL.py verify TICKER to mathematically prove the integrity of the environment.

🔧 Troubleshooting

    "No data found for TICKER": Ensure TICKER_minute.csv or TICKER.csv exists in the folder and contains the correct OHLCV columns.

    "Volume column missing": Your CSV must have a Volume column. If missing, you can add a dummy column with a constant value (e.g., 100000).

    Training is slow: Reduce PARAMS['EPISODES'] to 500,000 for faster training. Ensure CUDA is available via torch.cuda.is_available().

    Model not improving: Experiment with different stop-loss/trailing-stop values, increase training episodes, or verify your data quality (ensure no gaps and proper OHLCV structures).

Author: Charvit Rajani

Disclaimer: This software is for educational and research purposes only. It is not financial advice. The models and strategies provided do not guarantee profits and may result in the loss of capital. Use at your own risk.

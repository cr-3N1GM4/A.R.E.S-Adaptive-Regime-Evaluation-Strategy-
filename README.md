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
pip install numpy pandas yfinance torch stable-baselines3 gym gymnasium xgboost hmmlearn scipy tqdm matplotlib
```
💻 Usage Interface

The system is managed entirely through the CLI provided in RL.py.
Step 1: Generate the Selection Map

Run the screener to download NSE data and generate the daily stock selection mapping.
Bash

python StockSelection.py

Step 2: Validate Environment Integrity

Before training, run the built-in corruption test to mathematically prove the environment is free of lookahead bias.
Bash

python RL.py verify <TICKER>
# Example: python RL.py verify RELIANCE

Step 3: Train the RL Agent

You can train a specialized model for a single ticker or iterate over your entire data folder.
Command	Description
python RL.py train <TICKER>	Prepares data, trains regime models, and trains the PPO agent for a specific stock.
python RL.py train_all	Scans the /data directory and sequentially trains models for all available CSV files.
Step 4: Test & Evaluate

Generate tear sheets, equity curves, drawdown charts, and raw trade logs.
Command	Description
python RL.py test <TICKER>	Runs the deterministic out-of-sample test for a single trained ticker model. Saves plots to /test_trade_plots and logs to /test_results.
python RL.py portfolio_test	Simulates the full dynamic portfolio by reading selection_map.json, loading the respective specialized models, and trading the selected universe chronologically.
📁 Project Structure

    RL.py: The core reinforcement learning environment, regime detector training, and PPO implementation.

    StockSelection.py: Cross-sectional momentum and volatility screener for the NSE universe.

    data/: Directory containing raw downloaded ticker CSVs (e.g., MARKET_NSEI.csv, RELIANCE.csv).

    Models_<TICKER>/: Auto-generated directories containing the saved .zip PPO models, serialized HMM/XGBoost .pkl files, and VecNormalize statistics.

    test_results/: Text-based trade logs and summary metrics (Sharpe, Calmar, Max DD).

    test_trade_plots/: High-resolution visual overlays of the agent's entry/exit points on the price curve.

Author: Charvit Rajani

Disclaimer: This software is for educational and research purposes only. It is not financial advice. The models and strategies provided do not guarantee profits and may result in the loss of capital. Use at your own risk.

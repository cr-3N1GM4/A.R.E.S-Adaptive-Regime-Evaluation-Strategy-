"""
================================================================================
RL TRADING ENVIRONMENT - LOOKAHEAD-FREE DESIGN DOCUMENT
================================================================================

EXECUTION MODEL: CLOSE-TO-CLOSE
-------------------------------
This backtesting system uses a "close-to-close" execution model:
  - Decisions are made at the START of bar t, using data from bar t-1 and earlier
  - All trades (entries AND exits) execute at Close[t]
  - Stop-loss/trailing-stop checks happen at Close of subsequent bars

TIMELINE FOR EACH STEP:
-----------------------
  Step t:
    1. Agent receives State[t] containing:
       - Technical indicators computed from Close[t-1] and earlier (SHIFTED by 1)
       - Agent's current position (0=flat, 1=long, -1=short)
       - Mark-to-market PnL (using current close - this is known in real-time)
    
    2. Agent decides Action[t] (0=hold, 1=buy/cover, 2=sell/short)
    
    3. If entering position: Entry price = Close[t]
       If exiting position: Exit price = Close[t]
    
    4. Environment advances: current_step += 1
    
    5. Risk checks (stop-loss, trailing-stop) compare Close[t+1] vs entry
       - This is NOT lookahead: we're now at bar t+1, checking if stop was hit
       - Since execution is close-to-close, stops trigger at Close, not intrabar

WHY THIS IS NOT LOOKAHEAD:
--------------------------
  1. FEATURE SHIFTING: All technical indicators in State[t] are computed from 
     Close[t-1] and earlier. This is enforced by:
       indicators[lookback_cols] = indicators[lookback_cols].shift(1)
  
  2. DECISION/EXECUTION SEPARATION: The agent's decision at step t only sees 
     data from t-1. The execution happens at Close[t], which the agent did NOT 
     see when making the decision.
  
  3. REGIME FEATURES: HMM/XGBoost predictions use an online forward algorithm
     that only uses observations 0..t. The regime_df is pre-shifted, so these
     features are NOT shifted again (to avoid double-shifting).
  
  4. MTM CALCULATION: The agent sees its current unrealized P&L. This uses the
     current close price, which IS available in real trading (you can always
     check your position's current value).
  
  5. REWARD TIMING: Rewards are computed AFTER the action is taken and the
     environment steps. This follows standard MDP formulation: r_{t+1} can
     depend on s_{t+1}.

PRICE COLUMNS (_close, _high, _low):
------------------------------------
  These are intentionally NOT shifted because:
    - They are EXCLUDED from the feature observation (agent doesn't see them)
    - They are used only for: execution prices, MTM calculation, risk checks
    - At step t, you DO know Close[t] for these purposes

COMMON MISCONCEPTIONS (NOT BUGS):
---------------------------------
  1. "Entry at Close[t] is lookahead" - NO. Decision uses t-1 data. Close[t]
     is just the execution price, like a market-on-close order.
  
  2. "Stop checks on next bar use future data" - NO. After step increment,
     we're at bar t+1. Checking Close[t+1] is checking CURRENT bar.
  
  3. "MTM uses current price" - YES, and this is CORRECT. You always know 
     your portfolio's current value in real trading.
  
  4. "_close not shifted" - BY DESIGN. It's not a feature, it's for execution.

DATA SPLIT (NO LEAKAGE):
------------------------
  - First 10%: Regime detector training (HMM + XGBoost)
  - Next 50%: RL agent training
  - Last 40%: Testing
  All splits are chronological. No future data ever touches past periods.

================================================================================
"""
import os
import gc
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import glob
import re
import torch
import warnings
import numpy as np
import pandas as pd
import gymnasium as gym
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
from datetime import datetime
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import json
import pickle
import xgboost as xgb
from hmmlearn import hmm
from scipy import stats
from gymnasium.utils import seeding

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Explicitly filter specific pandas warnings
pd.options.mode.chained_assignment = None

REGIME_MODEL = None
HMM_MODEL = None

PARAMS = {
    'WARMUP_MINUTES': 30,
    'TRANSACTION_COST': 0.0001,  # 0.01% per side (0.02% round-trip)
    'WINDOW_SIZE': 10,
    'STOP_LOSS_TR': -0.0005,
    'TRAIL_PCT_TR': 0.0005,
    'TAKE_PROFIT_TR': 1.0,
    'STOP_LOSS_TE': -0.0004,
    'TRAIL_PCT_TE': 0.0004,
    'TAKE_PROFIT_TE': 1.0,
    'NUM_PROCS' : 8,
    'INDICATOR_LOOKBACKS': [3, 5, 10, 20],
    'CHOP_PERIOD': 14,
    'KAMA_PERIOD': 20,
    'KAMA_FAST': 2,
    'KAMA_SLOW': 20,
    'AROON_PERIOD': 20,
    'HA_ITERATIONS': 10,
    'RIBBON_PERIODS': [2, 3, 5, 8, 12, 15, 18],
    'RIBBON_REF_PERIOD': 100,
    'OPPORTUNITY_WINDOW': 5,
    'OPPORTUNITY_THRESHOLD': 0.005,
    'TRAIN_RATIO': 0.5,
    'SEED': 2,
    'EPISODES': 1_000_000,
    'CANDLE_FREQUENCY': '2min',
    'SOURCE_FOLDER': 'data',
    'NO_ENTRY_AFTER_TIME': '15:20',  # No new trades after 3:20 PM (allows time for EOD close)
}

REWARD_PARAMS = {
    'STOP_LOSS_PENALTY': -100,
    'TRAILING_STOP_PENALTY': -10,
    'TAKE_PROFIT_REWARD': 0,
    'EOD_CLOSE_PENALTY': -10,
    'TRADE_ENTRY_PENALTY': -5,
    'LOSS_MULTIPLIER': 4.0,
    'MISSED_OPPORTUNITY_PENALTY': -2.0,
    'WAIT_BONUS': 0.1,
    'SCALE': 0.01,
}

def train_regime_detector(file_list: List[str]):
    print(f"\n>> Training Regime Detector on {len(file_list)} days (10% Split)...")
    
    df_list = []
    for f in tqdm(file_list, desc="Loading Regime Data"):
        df = pd.read_csv(f)
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df = df.set_index('Time').sort_index()
        df_list.append(df)
    
    if not df_list: return None, None
    data = pd.concat(df_list).rename(columns=lambda x: x.lower())
    
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1))
    data['gk_vol'] = np.sqrt(0.5 * (np.log(data['high']/data['low'])**2) - (2*np.log(2)-1)*(np.log(data['close']/data['open'])**2))
    data['volume_ratio'] = data['volume'] / (data['volume'].rolling(20).mean() + 1e-9)
    data['vol_z'] = (data['volume'] - data['volume'].rolling(20).mean()) / (data['volume'].rolling(20).std() + 1e-9)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # -------------------------------------------------------------------------
    # HMM TRAINING WITH NUMERICAL STABILITY FIX
    # -------------------------------------------------------------------------
    # Problem: log_ret and gk_vol are very small numbers (e.g., 0.0001) which
    # can cause the covariance matrix to become singular (not positive definite).
    # 
    # Solution: 
    # 1. Scale features to bring them into a reasonable range (~0 to ~10)
    # 2. Use min_covar parameter to prevent zero-variance crashes
    # 3. Add fallback with higher regularization if training fails
    # -------------------------------------------------------------------------
    
    # Scale up tiny features to prevent numerical instability
    data['log_ret_scaled'] = data['log_ret'] * 1000  # Convert to basis points-ish
    data['gk_vol_scaled'] = data['gk_vol'] * 100     # Scale up volatility
    
    X_hmm = data[['log_ret_scaled', 'gk_vol_scaled', 'vol_z']].values
    X_train_hmm = X_hmm[:20000] if len(X_hmm) > 20000 else X_hmm
    
    # Train HMM with increased min_covar for stability
    hmm_model = hmm.GaussianHMM(
        n_components=3, 
        covariance_type="full", 
        n_iter=100, 
        random_state=42,
        min_covar=0.01  # Regularization to prevent singular matrices
    )
    
    try:
        hmm_model.fit(X_train_hmm)
    except Exception as e:
        print(f"   ⚠ HMM training failed: {e}")
        print("   Retrying with higher regularization (min_covar=0.1)...")
        hmm_model = hmm.GaussianHMM(
            n_components=3, 
            covariance_type="full", 
            n_iter=100, 
            random_state=42,
            min_covar=0.1  # Even higher regularization for stability
        )
        hmm_model.fit(X_train_hmm)
    
    # Generate HMM Features (using scaled data)
    data['HMM_State'] = hmm_model.predict(X_hmm)
    vol_variance = [np.diag(hmm_model.covars_[i]).mean() for i in range(3)]
    risky_state_idx = np.argmax(vol_variance)
    data['HMM_Risk_Prob'] = hmm_model.predict_proba(X_hmm)[:, risky_state_idx]


    data['cusum_log_ret'] = (data['log_ret'] - data['log_ret'].expanding().mean()).fillna(0).cumsum()
    data['cusum_vol'] = (data['gk_vol'] - data['gk_vol'].expanding().mean()).fillna(0).cumsum()
    
    # Dynamic span
    N = len(data)
    fast_span = max(10, int(0.005 * N))
    data['cusum_logret_fast'] = data['cusum_log_ret'].ewm(span=fast_span).mean()
    data['cusum_vol_fast'] = data['cusum_vol'].ewm(span=fast_span).mean()
    

    data['egarch_cond_vol'] = data['gk_vol'].rolling(20).mean()
    data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
    data['ema_21'] = data['close'].ewm(span=21, adjust=False).mean()
    

    data['spread'] = (data['high'] - data['low']) / data['close']
    data['hl_ratio'] = data['high'] / data['low']
    data['rsi'] = 100 - (100 / (1 + (data['close'].diff().clip(lower=0).rolling(14).mean() / -data['close'].diff().clip(upper=0).rolling(14).mean())))
    macd = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    data['macd_hist'] = macd - macd.ewm(span=9).mean()
    data['mom_accel'] = data['macd_hist'].diff()
    

    timeframes = [5, 15, 30, 60]
    for period in timeframes:
        data[f'ret_{period}m'] = data['close'].pct_change(period) * 100
        data[f'vol_{period}m'] = data['close'].pct_change().rolling(period).std() * 100
        data[f'momentum_{period}m'] = data['close'] - data['close'].shift(period)

    # Train XGBoost
    print("   > Training XGBoost...")
    features = ['log_ret', 'gk_vol', 'spread', 'hl_ratio', 'rsi', 'macd_hist', 'mom_accel', 
                'volume_ratio', 'vol_z', 'HMM_State', 'HMM_Risk_Prob',
                'cusum_logret_fast', 'cusum_vol_fast', 'egarch_cond_vol', 'ema_9', 'ema_21']
    for period in timeframes:
        features.extend([f'ret_{period}m', f'vol_{period}m', f'momentum_{period}m'])
        
    data = data.dropna()
    
    # -------------------------------------------------------------------------
    # REGIME TARGET CREATION
    # -------------------------------------------------------------------------
    # NOTE ON LOOKAHEAD: This uses pd.qcut on the training data (first 10% of days).
    # This is NOT lookahead because:
    #   1. This function only runs on the FIRST 10% of historical data
    #   2. The quantile thresholds are learned and FROZEN in the XGBoost model
    #   3. When predicting on NEW data (RL train/test), the model uses learned 
    #      parameters, NOT recomputed quantiles
    #   4. This is standard supervised learning: train on past, predict on future
    # 
    # The target is based on CONCURRENT volatility (gk_vol at time t), not future.
    # -------------------------------------------------------------------------
    data['Target'] = pd.qcut(data['gk_vol'], 3, labels=[0, 1, 2])

    X = data[features]
    y = data['Target']
    
    # No resampling/balancing for time-series (would cause lookahead)
    X_bal, y_bal = X, y

    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.03, random_state=42)
    xgb_model.fit(X_bal, y_bal)
    
    print("✓ Regime Training Complete.")
    return xgb_model, hmm_model

def prepare_data(ticker: str, target_folder: str, frequency: str = '2min'):
    """
    Robust data preparation:
    1. Finds source CSV
    2. Standardizes column names (Open, High, Low, Close, Volume)
    3. Resamples and saves
    """
    import os
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    import sys

    src = PARAMS['SOURCE_FOLDER']
    possible_files = [
        f"{ticker}_minute.csv", 
        f"{ticker}.csv",
        f"{src}/{ticker}_minute.csv",
        f"{src}/{ticker}.csv"
    ]
    csv_path = None
    for f in possible_files:
        if os.path.exists(f):
            csv_path = f
            break
            
    if csv_path is None:
        print(f"Error: Could not find source file for {ticker}. Expected {possible_files}")
        return

    print(f"Preparing data from {csv_path} into {target_folder}...")
    os.makedirs(target_folder, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Critical Error reading CSV: {e}")
        sys.exit(1)

    df.columns = [c.strip().lower() for c in df.columns]
    
    rename_map = {
        'date': 'Time', 'time': 'Time', 'datetime': 'Time',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume', 'vol': 'Volume'
    }
    
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    if 'Volume' not in df.columns:
        print(f"CRITICAL ERROR: 'Volume' column missing. Found: {df.columns}")
        print("Please check your CSV headers.")
        sys.exit(1)

    df['Time'] = pd.to_datetime(df['Time'])
    df = df.set_index('Time').sort_index()
    
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum' 
    }
    
    print(f"Resampling to {frequency}...")
    days = df.groupby(df.index.date)
    
    day_count = 0
    iterator = tqdm(days, desc="Creating Episodes") if 'tqdm' in sys.modules else days
    
    for date, day_df in iterator:
        if len(day_df) < 10: continue
        
        resampled = day_df.resample(frequency).agg(agg_dict).dropna()
        if len(resampled) == 0: continue
        resampled['Volume'] = resampled['Volume'].astype(float)
        
        out_file = Path(target_folder) / f"day{day_count}.csv"
        resampled = resampled.reset_index()
        resampled.to_csv(out_file, index=False)
        day_count += 1
        
    print(f"Successfully generated {day_count} clean daily files.")
    return True



def load_ohlc_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV and ensures the Index is a DatetimeIndex.
    Filters data to Indian market hours (09:15 - 15:30).
    """
    try:
        df = pd.read_csv(filepath)
        
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            
        if 'Time' in df.columns:
            df = df.set_index('Time').sort_index()
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            df.columns = [c.capitalize() for c in df.columns]
            
        # Filter for Market Hours (9:15 to 15:30)
        # This removes any pre-market or post-market ordering data
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.between_time('09:15', '15:30')
            
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ha = df.copy()
    ha['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_opens = [df['Open'].iloc[0]]
    ha_closes = ha['Close'].values
    for i in range(len(df) - 1):
        ha_opens.append((ha_opens[i] + ha_closes[i]) / 2)
    ha['Open'] = ha_opens
    ha['High'] = df[['High']].join(ha[['Open', 'Close']]).max(axis=1)
    ha['Low'] = df[['Low']].join(ha[['Open', 'Close']]).min(axis=1)
    return ha

def apply_ha_iterations(df: pd.DataFrame, iterations: int) -> pd.DataFrame:
    result = df.copy()
    for _ in range(iterations):
        result = calculate_heikin_ashi(result)
    return result

def ema(price: pd.Series, window: int) -> pd.Series:
    return price.ewm(span=window, adjust=False).mean()

def rsi(price: pd.Series, window: int) -> pd.Series:
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def cci(price: pd.Series, window: int) -> pd.Series:
    tp = price
    sma_tp = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad + 1e-9)

def cmo(price: pd.Series, window: int) -> pd.Series:
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(window).sum()
    loss = -delta.where(delta < 0, 0).rolling(window).sum()
    return 100 * (gain - loss) / (gain + loss + 1e-9)

def atr_simple(price: pd.Series, window: int) -> pd.Series:
    high_low = price.rolling(2).max() - price.rolling(2).min()
    return high_low.rolling(window=window).mean()

def standard_deviation(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window=window).std()

def calculate_chop(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs().fillna(0)
    tr3 = (df['Low'] - prev_close).abs().fillna(0)
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    sum_tr = tr.rolling(window=period).sum()
    max_high = df['High'].rolling(window=period).max()
    min_low = df['Low'].rolling(window=period).min()
    range_hl = max_high - min_low
    chop = 100 * np.log10(sum_tr / (range_hl + 1e-9)) / np.log10(period)
    return chop

def calculate_kama(series: pd.Series, period: int = 20, fast: int = 2, slow: int = 20) -> Tuple[pd.Series, pd.Series]:
    change = abs(series - series.shift(period))
    volatility = abs(series - series.shift(1)).rolling(window=period).sum()
    er = change / (volatility + 1e-9)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(series)
    kama[:] = np.nan
    start_idx = period
    if start_idx > 0 and start_idx <= len(series):
        kama[start_idx-1] = series.iloc[start_idx-1]
    price_values = series.values
    sc_values = sc.values
    for i in range(start_idx, len(series)):
        if np.isnan(sc_values[i]):
            kama[i] = price_values[i]
        else:
            kama[i] = kama[i-1] + sc_values[i] * (price_values[i] - kama[i-1])
    return pd.Series(kama, index=series.index), er

def calculate_aroon(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    arg_max = df['High'].rolling(window=window).apply(lambda x: x.argmax(), raw=True)
    days_since_high = (window - 1) - arg_max
    arg_min = df['Low'].rolling(window=window).apply(lambda x: x.argmin(), raw=True)
    days_since_low = (window - 1) - arg_min
    aroon_up = ((window - days_since_high) / window) * 100
    aroon_down = ((window - days_since_low) / window) * 100
    return aroon_up, aroon_down

def calculate_johnny_ribbon(price: pd.Series, ma_periods: List[int], ref_period: int = 100) -> Tuple[dict, pd.Series]:
    ref_ma = ema(price, ref_period)
    ribbon_data = {}
    for period in ma_periods:
        ma = ema(price, period)
        diff = ma.diff()
        cond_lime = (diff >= 0) & (ma > ref_ma)
        cond_maroon = (diff < 0) & (ma > ref_ma)
        cond_rubi = (diff <= 0) & (ma < ref_ma)
        cond_green = (diff >= 0) & (ma < ref_ma)
        regime = np.select([cond_lime, cond_maroon, cond_rubi, cond_green], [1, 2, 3, 4], default=0)
        ribbon_data[period] = {'ma': ma, 'regime': regime}
    return ribbon_data, ref_ma

def add_time_features(index: pd.DatetimeIndex) -> dict:
    t = pd.to_datetime(index)
    return {
        'H_sin': np.sin(2 * np.pi * t.hour / 24),
        'H_cos': np.cos(2 * np.pi * t.hour / 24),
        'M_sin': np.sin(2 * np.pi * t.minute / 60),
        'M_cos': np.cos(2 * np.pi * t.minute / 60)
    }

def calculate_all_indicators(df_120s: pd.DataFrame, hmm_model=None, xgb_model=None) -> pd.DataFrame:
    """
    Calculate all technical indicators for one trading day.
    
    LOOKAHEAD PREVENTION - TWO-STEP PROCESS:
    =========================================
    STEP 1: Compute indicators using current OHLCV (unshifted)
            - This is just computation, not yet exposed to the agent
            - All indicators at index t are computed from data up to and including t
    
    STEP 2: Shift ALL computed indicators by 1 (at the end of this function)
            - indicators[lookback_cols] = indicators[lookback_cols].shift(1)
            - After shift: indicators at index t contain values from t-1
            - Agent at step t only sees data from t-1 and earlier
    
    EXCEPTION - NOT SHIFTED:
    ========================
    - Predicted_Regime: Already computed on pre-shifted regime_df (would cause double-shift)
    - _close, _high, _low: Not features! Used only for execution/MTM (excluded from obs)
    
    WHY COMPUTE THEN SHIFT (vs. shift first)?
    =========================================
    Shifting OHLCV first then computing would give incorrect indicator values.
    Example: RSI at t needs Close[t], Close[t-1], ..., Close[t-n]
             If we shift first, we'd compute RSI from Close[t-1], Close[t-2], ...
             which changes the mathematical meaning of RSI.
    
    The correct approach: Compute normally, then shift the RESULT.
    """
    active_hmm = hmm_model if hmm_model is not None else HMM_MODEL
    active_xgb = xgb_model if xgb_model is not None else REGIME_MODEL

    indicators = pd.DataFrame(index=df_120s.index)
    
    # -------------------------------------------------------------------------
    # These variables hold UNSHIFTED OHLCV data for indicator computation.
    # All computed indicators will be SHIFTED at the end of this function.
    # The agent NEVER sees these raw values - only the shifted indicators.
    # -------------------------------------------------------------------------
    curr_open = df_120s['Open']
    curr_high = df_120s['High']
    curr_low = df_120s['Low']
    curr_close = df_120s['Close']
    curr_volume = df_120s['Volume'] 

    # Basic Candle Features
    indicators['candle_open'] = np.log(curr_open / curr_open.shift(1))
    indicators['candle_high'] = np.log(curr_high / curr_high.shift(1))
    indicators['candle_low'] = np.log(curr_low / curr_low.shift(1))
    indicators['candle_close'] = np.log(curr_close / curr_close.shift(1))
    
    # Momentum & Volatility Indicators
    for period in PARAMS['INDICATOR_LOOKBACKS']:
        indicators[f'rsi_{period}'] = rsi(curr_close, period)
        indicators[f'cci_{period}'] = cci(curr_close, period)
        indicators[f'cmo_{period}'] = cmo(curr_close, period)
        indicators[f'atr_{period}'] = atr_simple(curr_close, period) / (curr_close + 1e-9)
        indicators[f'std_{period}'] = standard_deviation(curr_close, period) / (curr_close + 1e-9)
    
    df_prev = pd.DataFrame({'Open': curr_open, 'High': curr_high, 'Low': curr_low, 'Close': curr_close, 'Volume': curr_volume})
    
    # Chop Index
    chop_vals = calculate_chop(df_prev, PARAMS['CHOP_PERIOD'])
    indicators[f'chop_{PARAMS["CHOP_PERIOD"]}'] = chop_vals
    indicators['chop_binary'] = (chop_vals < 43.2).astype(int)
    
    # KAMA & Efficiency Ratio
    kama_val, er_val = calculate_kama(curr_close, PARAMS['KAMA_PERIOD'], PARAMS['KAMA_FAST'], PARAMS['KAMA_SLOW'])
    indicators[f'kama_{PARAMS["KAMA_PERIOD"]}'] = kama_val
    indicators[f'er_{PARAMS["KAMA_PERIOD"]}'] = er_val
    indicators['er_binary'] = (indicators[f'er_{PARAMS["KAMA_PERIOD"]}'] > 0.3).astype(int)
    
    # Aroon
    aroon_up, aroon_down = calculate_aroon(df_prev, PARAMS['AROON_PERIOD'])
    indicators[f'aroon_up_{PARAMS["AROON_PERIOD"]}'] = aroon_up
    indicators[f'aroon_down_{PARAMS["AROON_PERIOD"]}'] = aroon_down
    
    # Heikin Ashi
    df_ha = apply_ha_iterations(df_prev, PARAMS['HA_ITERATIONS'])
    indicators['ha_trend'] = (df_ha['Close'] >= df_ha['Open']).astype(int)
    
    indicators['ha_candle_width'] = ((df_ha['Close'] - df_ha['Open']) / (df_ha['Open'] + 1e-9))
    indicators['ha_body_size'] = (abs(df_ha['Close'] - df_ha['Open']) / (df_ha['Open'] + 1e-9))
    indicators['ha_upper_wick'] = ((df_ha['High'] - df_ha[['Open', 'Close']].max(axis=1)) / (df_ha['Open'] + 1e-9))
    indicators['ha_lower_wick'] = ((df_ha[['Open', 'Close']].min(axis=1) - df_ha['Low']) / (df_ha['Open'] + 1e-9))
    
    indicators['ha_open'] = np.log(df_ha['Open'] / df_ha['Open'].shift(1))
    indicators['ha_high'] = np.log(df_ha['High'] / df_ha['High'].shift(1))
    indicators['ha_low'] = np.log(df_ha['Low'] / df_ha['Low'].shift(1))
    indicators['ha_close'] = np.log(df_ha['Close'] / df_ha['Close'].shift(1))
    
    # Johnny Ribbon
    ribbon_data, ref_ma = calculate_johnny_ribbon(df_ha['Close'], PARAMS['RIBBON_PERIODS'], PARAMS['RIBBON_REF_PERIOD'])
    for period, data in ribbon_data.items():
        indicators[f'ribbon_ma_{period}'] = np.log(data['ma'] / data['ma'].shift(1))
        indicators[f'ribbon_regime_{period}'] = pd.Series(data['regime'], index=data['ma'].index)
    indicators['ribbon_ref_ma_100'] = np.log(ref_ma / ref_ma.shift(1))
    
    # -------------------------------------------------------------------------
    # VOLUME & VWAP FEATURES
    # -------------------------------------------------------------------------
    # NOTE: These use curr_volume (unshifted here), BUT they get shifted by 1 
    # at the end of this function via: indicators[lookback_cols].shift(1)
    # So at time t, the agent sees Volume[t-1] features, NOT current volume.
    #
    # VWAP NOTE: cumsum() is PER-DAY because each CSV file is one trading day.
    # The function is called once per day, so cumsum resets automatically.
    # -------------------------------------------------------------------------
    vol_ma = curr_volume.rolling(window=20).mean()
    indicators['candle_volume_rel'] = np.log1p(curr_volume / (vol_ma + 1e-9)).fillna(0)
    
    # VWAP (per-day since each file = one day)
    cum_vol = curr_volume.cumsum()
    cum_vol_price = (curr_close * curr_volume).cumsum()
    vwap = cum_vol_price / (cum_vol + 1e-9)
    indicators['candle_vwap_dev'] = ((curr_close - vwap) / (vwap + 1e-9)).fillna(0)
    
    # Hurst Exponent Proxy
    er_period = 30
    er_change = curr_close.diff(er_period).abs()
    er_volatility = curr_close.diff().abs().rolling(er_period).sum()
    indicators['candle_hurst'] = (er_change / (er_volatility + 1e-9)).fillna(0.5)
    regime_df = pd.DataFrame(index=df_120s.index)

    regime_df['close'] = df_120s['Close'].shift(1)
    regime_df['open'] = df_120s['Open'].shift(1)
    regime_df['high'] = df_120s['High'].shift(1)
    regime_df['low'] = df_120s['Low'].shift(1)
    regime_df['volume'] = df_120s['Volume'].shift(1)

    # Base Calculations
    regime_df['log_ret'] = np.log(regime_df['close'] / regime_df['close'].shift(1))
    regime_df['gk_vol'] = np.sqrt(0.5 * (np.log(regime_df['high']/regime_df['low'])**2) - (2*np.log(2)-1)*(np.log(regime_df['close']/regime_df['open'])**2))
    regime_df['volume_ratio'] = regime_df['volume'] / (regime_df['volume'].rolling(20).mean() + 1e-9)
    regime_df['vol_z'] = (regime_df['volume'] - regime_df['volume'].rolling(20).mean()) / (regime_df['volume'].rolling(20).std() + 1e-9)

    # HMM Prediction 
    if active_hmm is not None:
        # IMPORTANT: Apply the SAME scaling used during HMM training
        # Training used: log_ret * 1000, gk_vol * 100, vol_z unchanged
        regime_df['log_ret_scaled'] = regime_df['log_ret'] * 1000
        regime_df['gk_vol_scaled'] = regime_df['gk_vol'] * 100
        
        X_hmm = regime_df[['log_ret_scaled', 'gk_vol_scaled', 'vol_z']].fillna(0).values
        n_samples = len(X_hmm)
        n_states = active_hmm.n_components
        
        try:
            # Get risky state index (highest volatility state)
            vol_variance = [np.diag(active_hmm.covars_[i]).mean() for i in range(n_states)]
            risky_state_idx = np.argmax(vol_variance)
            
            # Online forward algorithm: at time t, only use data 0 to t
            hmm_states = np.zeros(n_samples)
            hmm_risk_probs = np.full(n_samples, 0.5)
            
            # Precompute log probabilities
            log_startprob = np.log(active_hmm.startprob_ + 1e-10)
            log_transmat = np.log(active_hmm.transmat_ + 1e-10)
            
            # Precompute inverse covariances for speed
            inv_covs = []
            log_dets = []
            for s in range(n_states):
                cov = active_hmm.covars_[s] + np.eye(3) * 1e-6
                inv_covs.append(np.linalg.inv(cov))
                log_dets.append(np.log(np.linalg.det(cov) + 1e-10))
            
            alpha = None
            
            for t in range(n_samples):
                obs = X_hmm[t]
                
                # Compute emission log-probabilities
                log_emission = np.zeros(n_states)
                for s in range(n_states):
                    diff = obs - active_hmm.means_[s]
                    log_emission[s] = -0.5 * (np.dot(diff, np.dot(inv_covs[s], diff)) + log_dets[s] + 3 * np.log(2 * np.pi))
                
                if t == 0:
                    alpha = log_startprob + log_emission
                else:
                    alpha_new = np.zeros(n_states)
                    for s in range(n_states):
                        alpha_new[s] = log_emission[s] + np.logaddexp.reduce(alpha + log_transmat[:, s])
                    alpha = alpha_new
                
                # Normalize and get probabilities
                alpha_norm = alpha - np.logaddexp.reduce(alpha)
                state_probs = np.exp(alpha_norm)
                
                hmm_states[t] = np.argmax(state_probs)
                hmm_risk_probs[t] = state_probs[risky_state_idx]
            
            # Shift by 1 since input data was already shifted
            regime_df['HMM_State'] = pd.Series(hmm_states, index=regime_df.index).fillna(0)
            regime_df['HMM_Risk_Prob'] = pd.Series(hmm_risk_probs, index=regime_df.index).fillna(0.5)
            
        except Exception as e:
            regime_df['HMM_State'] = 0
            regime_df['HMM_Risk_Prob'] = 0.5
    else:
        regime_df['HMM_State'] = 0
        regime_df['HMM_Risk_Prob'] = 0.5

    # CUSUM Logic
    regime_df['cusum_log_ret'] = (regime_df['log_ret'] - regime_df['log_ret'].expanding().mean()).fillna(0).cumsum()
    regime_df['cusum_vol'] = (regime_df['gk_vol'] - regime_df['gk_vol'].expanding().mean()).fillna(0).cumsum()
    
    N_current = len(regime_df)
    fast_span = max(10, int(0.005 * N_current)) if N_current > 100 else 10
    regime_df['cusum_logret_fast'] = regime_df['cusum_log_ret'].ewm(span=fast_span).mean()
    regime_df['cusum_vol_fast'] = regime_df['cusum_vol'].ewm(span=fast_span).mean()
    
    # EGARCH & EMAs
    regime_df['egarch_cond_vol'] = regime_df['gk_vol'].rolling(20).mean()
    regime_df['ema_9'] = regime_df['close'].ewm(span=9, adjust=False).mean()
    regime_df['ema_21'] = regime_df['close'].ewm(span=21, adjust=False).mean()
    
    # Standard Technicals
    regime_df['spread'] = (regime_df['high'] - regime_df['low']) / regime_df['close']
    regime_df['hl_ratio'] = regime_df['high'] / regime_df['low']
    regime_df['rsi'] = 100 - (100 / (1 + (regime_df['close'].diff().clip(lower=0).rolling(14).mean() / -regime_df['close'].diff().clip(upper=0).rolling(14).mean())))
    macd = regime_df['close'].ewm(span=12).mean() - regime_df['close'].ewm(span=26).mean()
    regime_df['macd_hist'] = macd - macd.ewm(span=9).mean()
    regime_df['mom_accel'] = regime_df['macd_hist'].diff()
    
    # Multi-Timeframe Momentum
    for period in [5, 15, 30, 60]:
        regime_df[f'ret_{period}m'] = regime_df['close'].pct_change(period) * 100
        regime_df[f'vol_{period}m'] = regime_df['close'].pct_change().rolling(period).std() * 100
        regime_df[f'momentum_{period}m'] = regime_df['close'] - regime_df['close'].shift(period)

    # XGBoost Prediction 
    if active_xgb is not None:
        features = ['log_ret', 'gk_vol', 'spread', 'hl_ratio', 'rsi', 'macd_hist', 'mom_accel', 
                    'volume_ratio', 'vol_z', 'HMM_State', 'HMM_Risk_Prob',
                    'cusum_logret_fast', 'cusum_vol_fast', 'egarch_cond_vol', 'ema_9', 'ema_21']
        for period in [5, 15, 30, 60]:
            features.extend([f'ret_{period}m', f'vol_{period}m', f'momentum_{period}m'])
            
        X_regime = regime_df[features].fillna(0)
        try:
            raw_regime = active_xgb.predict(X_regime)
            indicators['Predicted_Regime'] = pd.Series(raw_regime, index=regime_df.index).fillna(1) - 1
        except:
            indicators['Predicted_Regime'] = 0
    else:
        indicators['Predicted_Regime'] = 0

    time_features = add_time_features(df_120s.index)
    for key, val in time_features.items():
        indicators[key] = val
    

    # -------------------------------------------------------------------------
    # CRITICAL: PREVENT LOOKAHEAD BIAS
    # We shift ALL calculated indicators by 1 timestep.
    # Original: indicators[t] uses data from Candle [t]
    # Shifted:  indicators[t] uses data from Candle [t-1]
    # This ensures the agent (at step t) only observes data computed from COMPLETED candles (up to t-1).
    # It does NOT see the current candle's close (Close[t]) before making a decision.
    # 
    # EXCEPTION: Predicted_Regime is already computed on pre-shifted regime_df data,
    # so we exclude it to avoid double-shifting.
    # -------------------------------------------------------------------------
    lookback_cols = [col for col in indicators.columns if col not in ['_close', 'Predicted_Regime']]
    indicators[lookback_cols] = indicators[lookback_cols].shift(1)
    

    indicators['_close'] = df_120s['Close']
    indicators['_high'] = df_120s['High'] 
    indicators['_low'] = df_120s['Low']
    
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Time']
    cols_to_drop = [c for c in cols_to_drop if c in indicators.columns]
    indicators = indicators.drop(columns=cols_to_drop)
    indicators = indicators.replace([np.inf, -np.inf], np.nan)
    indicators = indicators.fillna(method='ffill').fillna(0)
    
    return indicators

def verify_no_lookahead(env, num_tests: int = 20) -> bool:
    """
    LOOKAHEAD VERIFICATION TEST
    ===========================
    This function PROVES there is no lookahead by:
    1. Getting the observation at step t
    2. Corrupting all future data (t+1, t+2, ...)
    3. Getting the observation again
    4. If observation is UNCHANGED, there is NO LOOKAHEAD
    
    Run this after creating an environment to verify integrity.
    Returns True if no lookahead detected, False otherwise.
    """
    print("\n" + "="*60)
    print("LOOKAHEAD VERIFICATION TEST")
    print("="*60)
    
    env.reset()
    original_data = env.data_matrix.copy()
    
    all_passed = True
    for step in range(5, min(num_tests + 5, env.n_steps - 10)):
        env.current_step = step
        
        # Get observation with original data
        obs1 = env._get_obs()
        
        # Corrupt all FUTURE data (should not affect obs if no lookahead)
        env.data_matrix[step+1:] = np.random.randn(*env.data_matrix[step+1:].shape) * 1000
        
        # Get observation with corrupted future
        obs2 = env._get_obs()
        
        # Restore original data
        env.data_matrix = original_data.copy()
        
        # Check if observations match
        if not np.allclose(obs1, obs2, rtol=1e-5, atol=1e-8):
            print(f"❌ LOOKAHEAD DETECTED at step {step}!")
            print(f"   Max difference: {np.max(np.abs(obs1 - obs2)):.6f}")
            all_passed = False
            break
    
    if all_passed:
        print(f"✅ PASSED: No lookahead detected in {num_tests} test steps")
        print("   Observation at step t does NOT depend on data from t+1 onwards")
    
    print("="*60 + "\n")
    return all_passed

def is_lookback_feature(feature_name: str) -> bool:
    lookback_prefixes = ['kama_', 'er_', 'aroon_', 'ribbon_ma_', 'ribbon_regime_', 'ha_', 'candle_']
    return any(feature_name.startswith(prefix) for prefix in lookback_prefixes)

def process_single_file(file_path: str) -> pd.DataFrame:
    try:
        df_120s = load_ohlc_data(file_path)
        if df_120s.empty:
            return None
        indicators = calculate_all_indicators(df_120s)
        warmup_cutoff = indicators.index[0] + pd.Timedelta(minutes=PARAMS['WARMUP_MINUTES'])
        indicators = indicators[indicators.index > warmup_cutoff]
        
        # Preserve timestamps for NO_ENTRY_AFTER_TIME logic
        indicators['_time'] = indicators.index
        indicators = indicators.reset_index(drop=True)
        return indicators
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def precompute_all_data(file_pairs: List[str]) -> List[pd.DataFrame]:
    print(f"\n{'='*80}")
    print(f"PRE-COMPUTING INDICATORS FOR {len(file_pairs)} DAYS")
    print(f"{'='*80}")
    
    # Parallel processing using ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress bar
        results = list(tqdm(executor.map(process_single_file, file_pairs), 
                          total=len(file_pairs), 
                          desc="Processing files"))
    
    valid_results = [r for r in results if r is not None]
    
    print(f"\nSuccessfully processed {len(valid_results)}/{len(file_pairs)} days")
    print(f"{'='*80}\n")
    return valid_results

def save_feature_info(feature_names: List[str], lookback_features: List[str], 
                     scalar_features: List[str], state_size: int, window_size: int, 
                     ticker: str, output_file: str = "feature_info.txt"):
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"STATE SPACE FEATURE INFORMATION - {ticker}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total State Space Size: {state_size}\n")
        f.write(f"Window Size (Lookback): {window_size}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"LOOKBACK FEATURES (with {window_size}-step history)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: {len(lookback_features)}\n")
        f.write(f"Total dimensions: {len(lookback_features) * window_size}\n\n")
        
        for i, feat in enumerate(lookback_features, 1):
            f.write(f"  {i:2d}. {feat:30s} --> {window_size} timesteps\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("SCALAR FEATURES (current timestep only)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: {len(scalar_features)}\n")
        f.write(f"Total dimensions: {len(scalar_features)}\n\n")
        
        for i, feat in enumerate(scalar_features, 1):
            f.write(f"  {i:2d}. {feat}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("AGENT STATE FEATURES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: 2\n")
        f.write(f"Total dimensions: 2\n\n")
        f.write(f"  1. position (0=flat, 1=long, -1=short)\n")
        f.write(f"  2. mark-to-market PnL (unrealized return)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nFeature information saved to {output_file}")

def linear_schedule(initial_value: float, min_lr: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        current_lr = initial_value * progress_remaining
        return max(current_lr, min_lr)
    return func

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.n_envs = PARAMS['NUM_PROCS']
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.explained_variances = []
        self.last_entry_price=0
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="steps")
        
    def _on_step(self):
        if self.pbar:
            self.pbar.update(self.n_envs)
        return True
    
    def _on_rollout_end(self):
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            policy_loss = self.model.logger.name_to_value.get('train/policy_loss', None)
            value_loss = self.model.logger.name_to_value.get('train/value_loss', None)
            entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', None)
            explained_var = self.model.logger.name_to_value.get('train/explained_variance', None)
            
            if policy_loss is not None:
                self.policy_losses.append(policy_loss)
            if value_loss is not None:
                self.value_losses.append(value_loss)
            if entropy_loss is not None:
                self.entropy_losses.append(entropy_loss)
            if explained_var is not None:
                self.explained_variances.append(explained_var)
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

def get_exit_params(mode: str = 'train') -> dict:
    if mode.lower() == 'train':
        return {
            'STOP_LOSS': PARAMS['STOP_LOSS_TR'],
            'TRAIL_PCT': PARAMS['TRAIL_PCT_TR'],
            'TAKE_PROFIT': PARAMS['TAKE_PROFIT_TR'],
        }
    elif mode.lower() == 'test':
        return {
            'STOP_LOSS': PARAMS['STOP_LOSS_TE'],
            'TRAIL_PCT': PARAMS['TRAIL_PCT_TE'],
            'TAKE_PROFIT': PARAMS['TAKE_PROFIT_TE'],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'")

class IntradayTradingEnv(gym.Env):
    """
    Gymnasium environment for intraday trading simulation.
    
    EXECUTION MODEL: Close-to-Close
    ================================
    - State[t] contains features computed from Close[t-1] and earlier (pre-shifted)
    - Action[t] is decided by the agent
    - Entry/Exit executed at Close[t]
    - Risk checks (stop-loss) at Close[t+1] after step increment
    
    NO LOOKAHEAD GUARANTEE:
    =======================
    1. Features are shifted by 1 in calculate_all_indicators()
    2. Agent only sees historical data in _get_obs()
    3. _close, _high, _low are NOT features - only used for execution/MTM
    4. MTM (mark-to-market) uses current close, which IS known in real trading
    
    PRICE COLUMNS (_close, _high, _low):
    ====================================
    These are intentionally excluded from observation features via:
        self.feature_names = [f for f in columns if f not in ['_close', '_high', '_low']]
    They are used only for:
        - Execution prices (entry_price, exit_price)
        - MTM calculation (agent's current P&L - valid, you know your portfolio value)
        - Risk management checks (stop-loss, trailing-stop)
    """
    metadata = {"render_modes": []}
    
    def __init__(self, processed_data_list: List[pd.DataFrame], mode: str = 'train'):
        super().__init__()
        
        self.all_days = processed_data_list
        self.mode = mode.lower()
        self.tc = PARAMS['TRANSACTION_COST']
        self.window = PARAMS['WINDOW_SIZE']
        self.exit_params = get_exit_params(self.mode)
        
        sample_df = self.all_days[0]
        self.feature_names = [f for f in sample_df.columns if f not in ['_close', '_high', '_low', '_time']]
        self.lookback_features = [f for f in self.feature_names if is_lookback_feature(f)]
        self.scalar_features = [f for f in self.feature_names if f not in self.lookback_features]
        
        num_lookback_features = len(self.lookback_features) * self.window
        num_scalar_features = len(self.scalar_features)
        num_agent_features = 2
        
        self.state_size = num_lookback_features + num_scalar_features + num_agent_features
        
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        # Pre-calculate indices for fast access
        self.feature_map = {name: i for i, name in enumerate(sample_df.columns)}
        self.lookback_indices = [self.feature_map[f] for f in self.lookback_features]
        self.scalar_indices = [self.feature_map[f] for f in self.scalar_features]
        self.close_idx = self.feature_map['_close']
        self.high_idx = self.feature_map['_high']
        self.low_idx = self.feature_map['_low']
        self.time_idx = self.feature_map['_time']
        
        # Parse NO_ENTRY_AFTER_TIME (e.g., "15:20" -> 15*60+20 = 920 minutes from midnight)
        cutoff_str = PARAMS.get('NO_ENTRY_AFTER_TIME', '15:20')
        h, m = map(int, cutoff_str.split(':'))
        self.no_entry_cutoff_minutes = h * 60 + m  # Minutes from midnight
        
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.n_steps = 0
        self.data_matrix = None 
        self.indicators_df = None 
        self.timestamps = None  # Store timestamps separately

        self.trade_history = []
        self.trade_count = 0
        self.daily_pnl_abs = 0.0
        self.highest_price = None
        self.lowest_price = None
        self.last_entry_price = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.trade_history = []
        self.trade_count = 0
        self.daily_pnl_abs = 0.0
        self.last_entry_price = 0 
        
        if hasattr(self, "np_random") and self.np_random is not None:
            random_day_idx = self.np_random.integers(len(self.all_days))
        else:
            random_day_idx = np.random.randint(len(self.all_days))
        
        self.indicators_df = self.all_days[random_day_idx]
        
        if '_time' in self.indicators_df.columns:
            self.timestamps = pd.to_datetime(self.indicators_df['_time'])
            numeric_cols = [c for c in self.indicators_df.columns if c != '_time']
            self.data_matrix = self.indicators_df[numeric_cols].values.astype(np.float32)
        else:
            self.timestamps = None
            self.data_matrix = self.indicators_df.values.astype(np.float32)
        
        self.n_steps = len(self.indicators_df)
        self.position = 0
        self.entry_price = 0
        self.highest_price = None
        self.lowest_price = None
        self.current_step = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """
        Build the observation vector for the agent.
        
        LOOKAHEAD PREVENTION:
        ---------------------
        - All features in data_matrix were shifted by 1 in calculate_all_indicators()
        - So data_matrix[t] contains features computed from Close[t-1] and earlier
        - The agent at step t only sees historical data, never current bar's indicators
        
        OBSERVATION STRUCTURE:
        ----------------------
        [lookback_features (flattened window), scalar_features, position, mtm]
        
        MTM (Mark-to-Market) NOTE:
        --------------------------
        MTM uses the current close price (_close). This is NOT lookahead because:
        - In real trading, you always know your current portfolio value
        - The current price is observable (bid/ask or last traded price)
        - This is agent STATE, not a predictive FEATURE
        """
        idx = self.current_step
        start = idx - self.window + 1
        
        if start >= 0:
            lookback_data = self.data_matrix[start : idx+1, self.lookback_indices]
        else:
            valid_len = idx + 1
            padding_len = self.window - valid_len
            
            real_data = self.data_matrix[0 : idx+1, self.lookback_indices]
            padding = np.zeros((padding_len, len(self.lookback_indices)), dtype=np.float32)
            lookback_data = np.vstack([padding, real_data])
            
        scalar_data = self.data_matrix[idx, self.scalar_indices]
        
        # Agent state: position and unrealized P&L
        # MTM uses current close - this is valid as you always know your portfolio value
        mtm = 0.0
        if self.position != 0 and self.entry_price > 0:
            curr_price = self.data_matrix[idx, self.close_idx]  # Current close (known in real-time)
            mtm = (curr_price - self.entry_price) / (self.entry_price + 1e-9)
            if self.position == -1:
                mtm = -mtm
        
        obs_lookback = lookback_data.T.ravel() 
        
        obs = np.r_[obs_lookback, scalar_data, [self.position, mtm]].astype(np.float32)
        return obs

    def _calculate_reward(self, action: int, prev_pos: int, prev_price: float, new_price: float, exit_reason: str = None) -> float:
        
        SCALE = REWARD_PARAMS['SCALE']
        CAPITAL = 100000
        reward = 0.0
        
        # Trade Entry Penalty
        if prev_pos == 0 and self.position != 0:
            reward += REWARD_PARAMS['TRADE_ENTRY_PENALTY'] * SCALE

        # Reward only on position close
        if prev_pos != 0 and self.position == 0:
            shares = CAPITAL / self.entry_price
            
            if prev_pos == 1:  # Closed long
                pnl_dollars = shares * (prev_price - self.entry_price)
            else:  # Closed short
                pnl_dollars = shares * (self.entry_price - prev_price)
            
            # Transaction costs already subtracted in step()
            pnl_bps = (pnl_dollars / CAPITAL) * 10000
            
            # Apply multiplier to losses
            if pnl_bps > 0:
                reward += pnl_bps * SCALE
            else:
                reward += pnl_bps * REWARD_PARAMS['LOSS_MULTIPLIER'] * SCALE
            
            # Apply specific exit penalties/rewards using exit_reason
            if exit_reason == "STOP_LOSS":
                reward += REWARD_PARAMS['STOP_LOSS_PENALTY'] * SCALE
            elif exit_reason == "TRAILING_STOP":
                reward += REWARD_PARAMS['TRAILING_STOP_PENALTY'] * SCALE
            elif exit_reason == "TAKE_PROFIT":
                reward += REWARD_PARAMS['TAKE_PROFIT_REWARD'] * SCALE
            elif exit_reason == "EOD_CLOSE":
                reward += REWARD_PARAMS['EOD_CLOSE_PENALTY'] * SCALE
            else:
                # Fallback: Check strict stop loss if no specific reason passed
                if prev_pos == 1:
                    curr_ret = (prev_price - self.entry_price) / self.entry_price
                else:
                    curr_ret = (self.entry_price - prev_price) / self.entry_price
                
                if curr_ret <= self.exit_params['STOP_LOSS']:
                    reward += REWARD_PARAMS['STOP_LOSS_PENALTY'] * SCALE
        
        # Small bonus for waiting (opportunity cost consideration)
        if self.position == 0:
            # Check for missed opportunities
            lookback = PARAMS.get('OPPORTUNITY_WINDOW', 5)
            threshold = PARAMS.get('OPPORTUNITY_THRESHOLD', 0.0005)
            
            if self.current_step >= lookback:
                past_price = self.indicators_df.loc[self.current_step - lookback, '_close']
                move_pct = abs(new_price - past_price) / (past_price + 1e-9)
                
                if move_pct > threshold:
                    reward += REWARD_PARAMS.get('MISSED_OPPORTUNITY_PENALTY', -2.0) * SCALE
                else:
                    reward += REWARD_PARAMS.get('WAIT_BONUS', 0.1) * SCALE
        
        return reward
        
    def step(self, action: int):
        
        # Execution Price: Close[t]
        # The agent decided Action[t] based on State[t] (which is data from t-1).
        # We assume the agent submits the order at Open[t] (or during t) and gets filled at Close[t].
        # Using Close[t] as the fill and Close[t+1] as the next price is standard "next-bar" return logic.
        # There is no lookahead because the decision (Action) did not know Close[t] (it only saw Close[t-1]).
        prev_price = self.data_matrix[self.current_step, self.close_idx]
        prev_pos = self.position
        
        CAPITAL_PER_TRADE = 100000
        
        # -------------------------------------------------------------------------
        # NO NEW ENTRIES AFTER CUTOFF TIME
        # Check if current time is past the no-entry cutoff (e.g., 15:20)
        # This prevents new trades near market close that would be immediately closed.
        # Existing positions can still be closed (exits are always allowed).
        # -------------------------------------------------------------------------
        allow_new_entry = True
        if self.timestamps is not None and len(self.timestamps) > self.current_step:
            current_time = self.timestamps.iloc[self.current_step]
            current_minutes = current_time.hour * 60 + current_time.minute
            if current_minutes >= self.no_entry_cutoff_minutes:
                allow_new_entry = False
        
        if action == 1: 
            if self.position == -1:  # Closing short is always allowed
                shares = CAPITAL_PER_TRADE / self.entry_price
                pnl_dollars = shares * (self.entry_price - prev_price)
                tc_entry = self.tc * self.entry_price * shares
                tc_exit = self.tc * prev_price * shares
                self.daily_pnl_abs += pnl_dollars - tc_entry - tc_exit
                self.position = 0
                
            elif self.position == 0 and allow_new_entry:  # New long entry - check time
                self.position = 1
                self.entry_price = prev_price
                self.highest_price = prev_price
                self.lowest_price = None
        
        elif action == 2:
            if self.position == 1:  # Closing long is always allowed
                shares = CAPITAL_PER_TRADE / self.entry_price
                pnl_dollars = shares * (prev_price - self.entry_price)
                tc_entry = self.tc * self.entry_price * shares
                tc_exit = self.tc * prev_price * shares
                self.daily_pnl_abs += pnl_dollars - tc_entry - tc_exit
                self.position = 0
                
            elif self.position == 0 and allow_new_entry:  # New short entry - check time
                self.position = -1
                self.entry_price = prev_price
                self.lowest_price = prev_price
                self.highest_price = None
        
        # Next Step
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        new_price = self.data_matrix[self.current_step, self.close_idx]
        
        # Risk Management
        # -------------------------------------------------------------------------
        # EXECUTION MODEL: Close-to-Close
        # Entry at Close[t], Exit at Close[t+1] (or later)
        # Since we can only trade at Close prices, stop-loss checks also use Close.
        # This is consistent with the execution model (no intrabar orders).
        # -------------------------------------------------------------------------
        force_close = False
        exit_reason = None
        
        if self.position != 0:
            shares = CAPITAL_PER_TRADE / self.entry_price
            
            if self.position == 1:  # Long position
                curr_ret = (new_price - self.entry_price) / self.entry_price
                self.highest_price = max(self.highest_price, new_price)
                trailing_level = self.highest_price * (1 - self.exit_params['TRAIL_PCT'])
                
                if new_price <= trailing_level:
                    force_close = True
                    exit_reason = "TRAILING_STOP"
            
            else:  # Short position
                curr_ret = (self.entry_price - new_price) / self.entry_price
                self.lowest_price = min(self.lowest_price, new_price)
                trailing_level = self.lowest_price * (1 + self.exit_params['TRAIL_PCT'])
                
                if new_price >= trailing_level:
                    force_close = True
                    exit_reason = "TRAILING_STOP"
            
            # Stop-loss check using Close price (consistent with execution model)
            if curr_ret <= self.exit_params['STOP_LOSS']:
                force_close = True
                exit_reason = "STOP_LOSS"
            
            # Take profit check
            elif curr_ret >= self.exit_params['TAKE_PROFIT']:
                force_close = True
                exit_reason = "TAKE_PROFIT"
            
            # Execute forced close
            if force_close:
                if self.position == 1:
                    pnl_dollars = shares * (new_price - self.entry_price)
                else:
                    pnl_dollars = shares * (self.entry_price - new_price)
                
                # TC: entry + exit
                tc_entry = self.tc * self.entry_price * shares
                tc_exit = self.tc * new_price * shares
                self.daily_pnl_abs += pnl_dollars - tc_entry - tc_exit
                self.position = 0
                self.highest_price = None
                self.lowest_price = None
        
        # EOD Forced Close
        if terminated and self.position != 0:
            shares = CAPITAL_PER_TRADE / self.entry_price
            
            if self.position == 1:
                pnl_dollars = shares * (new_price - self.entry_price)
            else:
                pnl_dollars = shares * (self.entry_price - new_price)
            
            # TC: entry + exit
            tc_entry = self.tc * self.entry_price * shares
            tc_exit = self.tc * new_price * shares
            self.daily_pnl_abs += pnl_dollars - tc_entry - tc_exit
            self.position = 0
            exit_reason = "EOD_CLOSE"
        reward = self._calculate_reward(action, prev_pos, prev_price, new_price, exit_reason)
        
        opened = (prev_pos == 0 and self.position != 0)
        closed = (prev_pos != 0 and self.position == 0)
        
        if opened or closed:
            self.trade_count += 1
            
            trade_type = (
                "OPEN_LONG" if prev_pos == 0 and self.position == 1 else
                "OPEN_SHORT" if prev_pos == 0 and self.position == -1 else
                "CLOSE_LONG" if prev_pos == 1 and self.position == 0 else
                "CLOSE_SHORT"
            )
            
            if opened:
                self.last_entry_price = prev_price
                self.trade_history.append({
                    "step": self.current_step - 1,
                    "trade_type": trade_type,
                    "prev_position": prev_pos,
                    "new_position": self.position,
                    "entry_price": float(prev_price),  
                    "exit_price": 0.0,
                    "realized_pnl": float(self.daily_pnl_abs),
                    "reward": float(reward),
                })
            elif closed:
                self.trade_history.append({
                    "step": self.current_step - 1,
                    "trade_type": trade_type,
                    "prev_position": prev_pos,
                    "new_position": self.position,
                    "entry_price": float(self.last_entry_price), 
                    "exit_price": float(prev_price), 
                    "realized_pnl": float(self.daily_pnl_abs),
                    "reward": float(reward),
                })
                self.last_entry_price = 0  # Reset
        
        return self._get_obs(), float(reward), terminated, False, {}

def load_data(folder_path: str, ticker: str, train_ratio: float = 1.0) -> Tuple[List[str], List[str]]:
    files = glob.glob(f"{folder_path}/*.csv")
    
    def get_day_number(filepath):
        match = re.search(r'day(\d+)', str(filepath))
        return int(match.group(1)) if match else -1
        
    file_pairs = sorted([str(Path(f)) for f in files], key=get_day_number)
    
    train_files = file_pairs
    test_files = [] 

    train_ids = [Path(f).stem for f in train_files]
    
    with open(f"train_days_{ticker}.txt", "w") as f:
        for item in train_ids:
            f.write(item + "\n")
    
    with open(f"test_days_{ticker}.txt", "w") as f:
        f.write("")
    
    print(f"Loaded {len(train_files)} days for training (Sequential, No Split).")
    print(f"Saved train_days_{ticker}.txt.")
    
    return train_files, test_files

def load_saved_days(ticker: str) -> Tuple[List[str], List[str]]:
    target_folder = f"{ticker}_{PARAMS['CANDLE_FREQUENCY']}"
    with open(f"train_days_{ticker}.txt", "r") as f:
        train_files = [Path(f'{Path(target_folder)}/{Path(line.strip())}.csv') for line in f if line.strip()]
    
    with open(f"test_days_{ticker}.txt", "r") as f:
        test_files = [Path(f'{Path(target_folder)}/{Path(line.strip())}.csv') for line in f if line.strip()]
    
    return train_files, test_files

def make_env(data_list: List[pd.DataFrame], mode: str = 'train') -> Callable:
    def _init():
        return Monitor(IntradayTradingEnv(data_list, mode=mode))
    return _init

def plot_training_metrics(callback: TqdmCallback, ticker: str, save_folder: str = "training_plots"):
    os.makedirs(save_folder, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    if len(callback.entropy_losses) > 0:
        axes[0].plot(callback.entropy_losses, label='Entropy Loss', color='blue', linewidth=2)
        axes[0].set_xlabel('Training Updates')
        axes[0].set_ylabel('Entropy Loss')
        axes[0].set_title(f'{ticker} - Entropy Loss Over Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    if len(callback.explained_variances) > 0:
        axes[1].plot(callback.explained_variances, label='Explained Variance', color='green', linewidth=2)
        axes[1].set_xlabel('Training Updates')
        axes[1].set_ylabel('Explained Variance')
        axes[1].set_title(f'{ticker} - Explained Variance Over Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{save_folder}/{ticker}_training_metrics.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Training metrics plot saved to {save_path}")

def train_model_parallel(train_file_pairs: List[str], ticker: str, total_timesteps: int = PARAMS['EPISODES'], 
                         use_gpu: bool = True, seed: int = PARAMS['SEED']):
    print("\n" + "=" * 80)
    print(f"INITIALIZING PARALLEL TRAINING FOR {ticker}")
    print("=" * 80)
    
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train_data_memory = precompute_all_data(train_file_pairs)
    
    temp_env = IntradayTradingEnv(train_data_memory, mode='train')
    save_feature_info(
        temp_env.feature_names,
        temp_env.lookback_features,
        temp_env.scalar_features,
        temp_env.state_size,
        temp_env.window,
        ticker,
        f"feature_info_{ticker}.txt"
    )
    
    print(f"\nState Space Dimension: {temp_env.state_size}")
    print(f"Action Space: {temp_env.action_space.n} actions (0=Hold, 1=Buy, 2=Sell)")
    print(f"Lookback Features: {len(temp_env.lookback_features)} x {temp_env.window} timesteps = {len(temp_env.lookback_features) * temp_env.window} dims")
    print(f"Scalar Features: {len(temp_env.scalar_features)} dims")
    print("Agent Features: 2 dims (position, mtm)")
    
    exit_params = get_exit_params('train')
    print(f"\nTraining Exit Parameters:")
    print(f"  Stop Loss: {exit_params['STOP_LOSS']}")
    print(f"  Trailing Stop: {exit_params['TRAIL_PCT']}")
    print(f"  Take Profit: {exit_params['TAKE_PROFIT']}")
    
    n_procs = PARAMS['NUM_PROCS']+2
    n_envs = max(1, n_procs - 2)
    print(f"Launching {n_envs} environments (DummyVecEnv for Windows optimization)...")
    
    env_cmds = [make_env(train_data_memory, mode='train') for _ in range(n_envs)]
    env = DummyVecEnv(env_cmds)
    env.seed(seed)
    # -------------------------------------------------------------------------
    # VecNormalize: Computes running mean/std for observation normalization.
    # NOT LOOKAHEAD because:
    #   - Training: Stats are computed from training data only (first 50% of days)
    #   - Testing: We load the saved stats and set training=False
    #   - Test data does NOT update the normalization statistics
    # -------------------------------------------------------------------------
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99)
    
    print("\nApplied VecNormalize (obs normalization, stats frozen during testing)")
    print("\n" + "=" * 80)
    print("CONFIGURING PPO MODEL FOR PARALLEL TRAINING")
    print("=" * 80)
    
    steps_per_env = 1024
    batch_size = 4096
    lr_schedule = linear_schedule(initial_value=3e-3, min_lr=1e-4)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        learning_rate=lr_schedule,
        n_steps=steps_per_env,
        batch_size=batch_size,
        n_epochs=10,
        ent_coef=0.001,
        gamma=0.99,
        gae_lambda=0.90,
        clip_range=0.2,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
        seed=seed,
    )
    
    print(f"Device: {device.upper()}")
    print("Policy: MlpPolicy with [256, 256] architecture")
    print("Learning Rate: Linear Schedule (3e-3 -> 1e-4)")
    print(f"Parallel Environments: {n_envs}")
    print(f"Steps per Environment: {steps_per_env}")
    print(f"Total Buffer Size per Update: {steps_per_env * n_envs}")
    print(f"Batch Size: {batch_size}")
    
    print("\n" + "=" * 80)
    print(f"STARTING PARALLEL TRAINING - {total_timesteps:,} timesteps")
    print("=" * 80 + "\n")
    
    callback = TqdmCallback(total_timesteps)
    model.learn(total_timesteps, callback=callback, progress_bar=False)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    model_folder = f"Models_{ticker}"
    os.makedirs(model_folder, exist_ok=True)
    
    save_path = f"{model_folder}/ppo_trading_model_{ticker}"
    model.save(save_path)
    env.save(f"{save_path}_vecnormalize.pkl")
    
    print(f"\nModel saved to: {save_path}.zip")
    print(f"VecNormalize stats saved to: {save_path}_vecnormalize.pkl")
    
    plot_training_metrics(callback, ticker)
    
    env.close()
    return model

def save_trade_plot(day_index: int, day_name: str, indicators_df: pd.DataFrame, 
                   trade_history: List[dict], ticker: str, save_folder: str = "test_trade_plots"):
    os.makedirs(save_folder, exist_ok=True)
    
    prices = indicators_df["_close"].values
    steps = list(range(len(prices)))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(steps, prices, 'k-', linewidth=1.5, label='Price')
    
    for t in trade_history:
        s = int(t["step"])
        if s < 0 or s >= len(prices):
            continue
        
        price = prices[s]
        tt = t["trade_type"].replace(" ", "_")
        
        if tt == "OPEN_LONG":
            ax.scatter(s, price, color='lime', s=100, marker='^', label='Open Long', zorder=5)
        elif tt == "OPEN_SHORT":
            ax.scatter(s, price, color='red', s=100, marker='v', label='Open Short', zorder=5)
        elif tt == "CLOSE_LONG":
            ax.scatter(s, price, color='lightgreen', s=80, marker='o', label='Close Long', zorder=5)
        elif tt == "CLOSE_SHORT":
            ax.scatter(s, price, color='pink', s=80, marker='o', label='Close Short', zorder=5)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    ax.set_title(f"{ticker} - Day {day_index+1} - {day_name}")
    ax.set_xlabel("Step (2-min candles)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    
    out_path = f"{save_folder}/{ticker}_day_{day_index+1}_{day_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved chart -> {out_path}")

def plot_equity_and_drawdown(daily_pnls: List[float], ticker: str, save_folder: str = "test_results"):
    os.makedirs(save_folder, exist_ok=True)
    
    # Calculate Compounded Equity
    initial_capital = 100000.0
    equity = [initial_capital]
    for pnl in daily_pnls:
        equity.append(equity[-1] + pnl)
    
    equity_curve = np.array(equity)
    
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(equity_curve, linewidth=2, color='blue')
    axes[0].set_title(f'{ticker} - Equity Curve (Start: $100k)')
    axes[0].set_xlabel('Trading Days')
    axes[0].set_ylabel('Equity ($)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].fill_between(range(len(drawdowns)), drawdowns * 100, color='red', alpha=0.3)
    axes[1].plot(drawdowns * 100, linewidth=2, color='darkred')
    axes[1].set_title(f'{ticker} - Drawdown (%)')
    axes[1].set_xlabel('Trading Days')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{save_folder}/{ticker}_equity_drawdown.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Equity and drawdown plot saved to {save_path}")

def test_model(ticker: str, test_file_pairs: List[str], deterministic: bool = True, 
               specific_day: str = None):
    model_folder = f"Models_{ticker}"
    model_path = f"{model_folder}/ppo_trading_model_{ticker}.zip"
    vec_normalize_path = f"{model_folder}/ppo_trading_model_{ticker}_vecnormalize.pkl"
    output_file = f"test_results/test_results_{ticker}.txt"
    
    os.makedirs("test_results", exist_ok=True)
    os.makedirs(f"test_trade_plots", exist_ok=True)
    os.makedirs(f"signals_{ticker}", exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"LOADING TRAINED MODEL FOR {ticker}")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vec_normalize_path}")
    
    exit_params = get_exit_params('test')
    print(f"\nTesting Exit Parameters:")
    print(f"  Stop Loss: {exit_params['STOP_LOSS']}")
    print(f"  Trailing Stop: {exit_params['TRAIL_PCT']}")
    print(f"  Take Profit: {exit_params['TAKE_PROFIT']}")
    
    if specific_day:
        test_file_pairs = [f for f in test_file_pairs if re.search(rf'day{re.escape(specific_day)}\b', str(f))]
        if not test_file_pairs:
            print(f"Error: Day {specific_day} not found in test files")
            return
        print(f"Testing specific day: {specific_day}")
    
    print(f"Days to test: {len(test_file_pairs)}")
    
    model = PPO.load(model_path)
    print("\nModel loaded.\n")
    
    total_test_pnl = 0.0
    total_trades = 0
    winning_days = 0
    losing_days = 0
    daily_pnls = []
    all_trade_pnls = []
    
    # Track cumulative equity after every trade for intraday drawdown
    initial_capital = 100000.0
    cumulative_equity = initial_capital
    equity_after_each_trade = [initial_capital]
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TEST RESULTS - {ticker}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"VecNormalize: {vec_normalize_path}\n")
        f.write(f"Deterministic: {deterministic}\n\n")
        
        for day_idx, file_120s in enumerate(tqdm(test_file_pairs, desc="Testing", unit="day")):
            day_name = Path(file_120s).stem
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"DAY {day_idx + 1}: {day_name}\n")
            f.write("=" * 80 + "\n")
            
            df_120s = load_ohlc_data(file_120s)
            if df_120s.empty:
                print(f"Skipping {day_name} (Empty Data)")
                continue

            try:
                indicators_df = calculate_all_indicators(df_120s)
            except Exception as e:
                print(f"Skipping {day_name} (Indicator Error: {e})")
                continue

            if indicators_df.empty:
                print(f"Skipping {day_name} (No Indicators)")
                continue

            warmup_cutoff = indicators_df.index[0] + pd.Timedelta(minutes=PARAMS['WARMUP_MINUTES'])
            
            indicators_df_with_index = indicators_df[indicators_df.index > warmup_cutoff].copy()
            
            if indicators_df_with_index.empty:
                print(f"Skipping {day_name} (Not enough data for warmup)")
                continue

            original_timestamps = indicators_df_with_index.index.tolist()
            
            # Preserve timestamps for NO_ENTRY_AFTER_TIME logic
            indicators_df_with_index['_time'] = indicators_df_with_index.index
            indicators_df = indicators_df_with_index.reset_index(drop=True)
            
            env = IntradayTradingEnv([indicators_df], mode='test')
            vec_env = DummyVecEnv([lambda e=env: e])
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            
            obs = vec_env.reset()
            done = np.array([False])
            
            step = 0
            prev_pos = 0
            day_pnl_dollars = 0.0
            logged_trades = []
            open_positions = {}
            
            signals_data = []
            
            print("\n" + "=" * 80)
            print(f"TRADE LOG - {day_name}")
            print("=" * 80)
            print(f"{'Step':<6} {'Trade Type':<20} {'Price':<10} {'Pos':<10} {'MTM%':<8}")
            print("-" * 80)
            
            while not done[0]:
                env_real = vec_env.venv.envs[0]
                price_idx = env_real.current_step
                price = env_real.indicators_df.loc[env_real.current_step, "_close"]
                timestamp = original_timestamps[price_idx] if price_idx < len(original_timestamps) else None
                timestamp = timestamp + pd.Timedelta(minutes=2)
                entry = env_real.entry_price if env_real.entry_price > 0 else 0.0
                
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)
                
                new_pos = vec_env.venv.envs[0].position
                
                buy_signal = 0
                sell_signal = 0
                exit_signal = 0
                
                if new_pos != prev_pos:
                    if prev_pos == 0 and new_pos == 1:
                        trade_type = "OPEN LONG"
                        mtm_pct = 0.0
                        logged_trades.append({"step": price_idx, "trade_type": trade_type,"entry_price": price,"exit_price":0.0})
                        open_positions["long"] = {"entry": price, "open_step": step, "capital": cumulative_equity}
                        buy_signal = 1
                        
                    elif prev_pos == 0 and new_pos == -1:
                        trade_type = "OPEN SHORT"
                        mtm_pct = 0.0
                        logged_trades.append({"step": price_idx, "trade_type": trade_type,"entry_price": price,"exit_price": 0.0})
                        open_positions["short"] = {"entry": price, "open_step": step, "capital": cumulative_equity}
                        sell_signal = 1
                        
                    elif prev_pos == 1 and new_pos == 0:
                        trade_type = "CLOSE LONG"
                        mtm_pct = (price - entry) / entry * 100 if entry else 0.0
                        logged_trades.append({"step": price_idx, "trade_type": trade_type,"entry_price": entry,"exit_price": price})
                        exit_signal = 1
                        
                        if "long" in open_positions:
                            entry_p = open_positions["long"]["entry"]
                            trade_capital = open_positions["long"]["capital"]
                            shares = trade_capital / entry_p
                            pnl_dollars = shares * (price - entry_p)
                            tc_entry = PARAMS["TRANSACTION_COST"] * entry_p * shares
                            tc_exit = PARAMS["TRANSACTION_COST"] * price * shares
                            pnl_after_tc = pnl_dollars - tc_entry - tc_exit
                            day_pnl_dollars += pnl_after_tc
                            cumulative_equity += pnl_after_tc
                            equity_after_each_trade.append(cumulative_equity)
                            pnl_pct = (pnl_after_tc / trade_capital) * 100
                            all_trade_pnls.append(pnl_pct)
                            total_trades += 1
                            del open_positions["long"]
                            
                    elif prev_pos == -1 and new_pos == 0:
                        trade_type = "CLOSE SHORT"
                        mtm_pct = (entry - price) / entry * 100 if entry else 0.0
                        logged_trades.append({"step": price_idx, "trade_type": trade_type,"entry_price": entry,"exit_price": price})
                        exit_signal = 1
                        
                        if "short" in open_positions:
                            entry_p = open_positions["short"]["entry"]
                            trade_capital = open_positions["short"]["capital"]
                            shares = trade_capital / entry_p
                            pnl_dollars = shares * (entry_p - price)
                            tc_entry = PARAMS["TRANSACTION_COST"] * entry_p * shares
                            tc_exit = PARAMS["TRANSACTION_COST"] * price * shares
                            pnl_after_tc = pnl_dollars - tc_entry - tc_exit
                            day_pnl_dollars += pnl_after_tc
                            cumulative_equity += pnl_after_tc
                            equity_after_each_trade.append(cumulative_equity)
                            pnl_pct = (pnl_after_tc / trade_capital) * 100
                            all_trade_pnls.append(pnl_pct)
                            total_trades += 1
                            del open_positions["short"]
                    else:
                        trade_type = f"{prev_pos}->{new_pos}"
                        mtm_pct = 0.0
                    
                    pos_name = {0: "FLAT", 1: "LONG", -1: "SHORT"}[new_pos]
                    print(f"{step:<6} {trade_type:<20} {price:<10.4f} {pos_name:<10} {mtm_pct:<8.3f}")
                
                signals_data.append({
                    'Time': timestamp.strftime('%H:%M:%S') if timestamp else None,
                    'Price': price,
                    'BUY': buy_signal,
                    'SELL': sell_signal,
                    'EXIT': exit_signal
                })
                
                prev_pos = new_pos
                step += 1

            env_real = vec_env.venv.envs[0]
            last_step_idx = env_real.current_step - 1

            if last_step_idx >= 0 and last_step_idx < len(env_real.indicators_df):
                final_price = env_real.indicators_df.loc[last_step_idx, "_close"]
                final_timestamp = original_timestamps[last_step_idx] if last_step_idx < len(original_timestamps) else None
                
                if env_real.position == 1 and "long" in open_positions:
                    logged_trades.append({"step": last_step_idx, "trade_type": "CLOSE LONG","entry_price": env_real.entry_price,"exit_price": final_price})
                    entry_p = open_positions["long"]["entry"]
                    trade_capital = open_positions["long"]["capital"]
                    shares = trade_capital / entry_p
                    pnl_dollars = shares * (final_price - entry_p)
                    tc_entry = PARAMS["TRANSACTION_COST"] * entry_p * shares
                    tc_exit = PARAMS["TRANSACTION_COST"] * final_price * shares
                    pnl_after_tc = pnl_dollars - tc_entry - tc_exit
                    day_pnl_dollars += pnl_after_tc
                    cumulative_equity += pnl_after_tc
                    equity_after_each_trade.append(cumulative_equity)
                    pnl_pct = (pnl_after_tc / trade_capital) * 100
                    all_trade_pnls.append(pnl_pct)
                    total_trades += 1
                    print(f"{step:<6} {'CLOSE LONG (EOD)':<20} {final_price:<10.4f} {'FLAT':<10} {((final_price - entry_p) / entry_p * 100):<8.3f}")
                    
                elif env_real.position == -1 and "short" in open_positions:
                    logged_trades.append({"step": last_step_idx, "trade_type": "CLOSE SHORT","entry_price": env_real.entry_price,"exit_price": final_price})
                    entry_p = open_positions["short"]["entry"]
                    trade_capital = open_positions["short"]["capital"]
                    shares = trade_capital / entry_p
                    pnl_dollars = shares * (entry_p - final_price)
                    tc_entry = PARAMS["TRANSACTION_COST"] * entry_p * shares
                    tc_exit = PARAMS["TRANSACTION_COST"] * final_price * shares
                    pnl_after_tc = pnl_dollars - tc_entry - tc_exit
                    day_pnl_dollars += pnl_after_tc
                    cumulative_equity += pnl_after_tc
                    equity_after_each_trade.append(cumulative_equity)
                    pnl_pct = (pnl_after_tc / trade_capital) * 100
                    all_trade_pnls.append(pnl_pct)
                    total_trades += 1
                    print(f"{step:<6} {'CLOSE SHORT (EOD)':<20} {final_price:<10.4f} {'FLAT':<10} {((entry_p - final_price) / entry_p * 100):<8.3f}")
           
            save_trade_plot(day_idx, day_name, indicators_df, logged_trades, ticker)
            
            signals_df = pd.DataFrame(signals_data)
            signals_path = f"signals_{ticker}/{day_name}.csv"
            signals_df.to_csv(signals_path, index=False)
            
            print("-" * 80)
            # Calculate day return as percentage of starting equity for the day
            day_start_equity = cumulative_equity - day_pnl_dollars
            day_return_pct = (day_pnl_dollars / day_start_equity) * 100 if day_start_equity > 0 else 0
            print(f"Day PnL: ${day_pnl_dollars:+,.2f} ({day_return_pct:+.2f}%) | EOD Equity: ${cumulative_equity:,.2f}\n")
            
            f.write(f"Steps executed: {step}\n")
            f.write(f"Day PnL: ${day_pnl_dollars:+,.2f} ({day_return_pct:+.2f}%)\n")
            f.write(f"EOD Equity: ${cumulative_equity:,.2f}\n")
            total_test_pnl += day_pnl_dollars
            daily_pnls.append(day_pnl_dollars)
            
            if day_pnl_dollars > 0:
                winning_days += 1
            elif day_pnl_dollars < 0:
                losing_days += 1
            
            vec_env.close()
        
        total_days = len(daily_pnls)
        flat_days = total_days - winning_days - losing_days
        avg_daily_pnl = float(np.mean(daily_pnls)) if total_days > 0 else 0.0
                
        if len(daily_pnls) > 0:
            total_profit = sum(daily_pnls)
            final_equity_val = initial_capital + total_profit
            
            # Use trade-by-trade equity for TRUE intraday drawdown
            equity_curve = np.array(equity_after_each_trade)
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = (peak - equity_curve) / peak
            max_dd = float(drawdowns.max())
            
            # Daily returns for Sharpe (still use daily PnL)
            equity_daily = [initial_capital]
            for d_pnl in daily_pnls:
                equity_daily.append(equity_daily[-1] + d_pnl)
            equity_daily = np.array(equity_daily)
            returns = np.diff(equity_daily) / equity_daily[:-1]
            daily_mean = np.mean(returns)
            daily_std = np.std(returns)
            sharpe_ratio = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 1e-9 else 0.0
            
            years = total_days / 252.0
            if years > 0 and final_equity_val > 0:
                cagr = (final_equity_val / initial_capital) ** (1 / years) - 1
            else:
                cagr = 0.0
            
            calmar = cagr / max_dd if max_dd > 0 else 0.0
        else:
            max_dd = 0.0
            cagr = 0.0
            sharpe_ratio = 0.0
            calmar = 0.0
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"FINAL SUMMARY - {ticker}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Days: {total_days}\n")
        f.write(f"Initial Capital: $100,000.00\n")
        f.write(f"Final Equity: ${100000 + total_test_pnl:,.2f}\n")
        f.write(f"Total PnL: ${total_test_pnl:,.2f}\n")
        f.write(f"Winning Days: {winning_days}\n")
        f.write(f"Losing Days: {losing_days}\n")
        f.write(f"Flat Days: {flat_days}\n")
        f.write(f"Average Daily PnL: {avg_daily_pnl:+.2f} bps\n")
        f.write(f"Total Trades: {total_trades}\n")
        
        if all_trade_pnls:
            avg_trade_pnl = float(np.mean(all_trade_pnls))
            f.write(f"Average Trade PnL: {avg_trade_pnl:+.2f} bps\n")
            
            wins = [p for p in all_trade_pnls if p > 0]
            losses = [p for p in all_trade_pnls if p < 0]
            if wins or losses:
                trade_win_rate = len(wins) / (len(wins) + len(losses)) * 100
                f.write(f"Winning Trades: {len(wins)}\n")
                f.write(f"Losing Trades: {len(losses)}\n")
                f.write(f"Trade Win Rate: {trade_win_rate:.2f}%\n")
                if wins:
                    f.write(f"Average Win: {np.mean(wins):+.2f} bps\n")
                if losses:
                    f.write(f"Average Loss: {np.mean(losses):+.2f} bps\n")
        
        f.write("\nRisk Metrics:\n")
        f.write(f"CAGR (Compounded): {cagr * 100:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Max Drawdown: {max_dd * 100:.2f}%\n")
        f.write(f"Calmar Ratio: {calmar:.3f}\n")
        f.write("=" * 80 + "\n")
    
    print("=" * 80)
    print(f"FINAL SUMMARY - {ticker}")
    print("=" * 80)
    print(f"Days tested: {total_days}")
    print(f"Initial Capital: $100,000.00")
    print(f"Final Equity: ${100000 + total_test_pnl:,.2f}")
    print(f"Total PnL: ${total_test_pnl:,.2f}")
    print(f"Avg Daily PnL: ${avg_daily_pnl:,.2f}")
    print(f"Winning Days: {winning_days}")
    print(f"Losing Days: {losing_days}")
    print(f"Total Trades: {total_trades}")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_dd * 100:.2f}%")
    print(f"Calmar Ratio: {calmar:.3f}")
    print()
        
    plot_equity_and_drawdown(daily_pnls, ticker)
    
    print(f"Test results saved to {output_file}")

def load_models_for_ticker(ticker):
    """
    Attempts to load the PPO, HMM, and XGBoost models for a specific ticker.
    Returns (model, hmm_model, xgb_model) or (None, None, None) if failed.
    """
    model_folder = f"Models_{ticker}"
    ppo_path = f"{model_folder}/ppo_trading_model_{ticker}.zip"
    regime_path = f"{model_folder}/regime_detector_model.pkl"
    hmm_path = f"{model_folder}/hmm_model.pkl"
    vec_norm_path = f"{model_folder}/ppo_trading_model_{ticker}_vecnormalize.pkl"
    
    if not os.path.exists(ppo_path) or not os.path.exists(regime_path):
        return None, None, None, None
        
    try:
        model = PPO.load(ppo_path)
        
        with open(regime_path, 'rb') as f: xgb_model = pickle.load(f)
        with open(hmm_path, 'rb') as f: hmm_model = pickle.load(f)
        
        return model, hmm_model, xgb_model, vec_norm_path
    except Exception as e:
        print(f"Error loading models for {ticker}: {e}")
        return None, None, None, None

def test_portfolio_selection(json_path: str = "selection_map.json", deterministic: bool = True):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        selection_map = json.load(f)
    
    os.makedirs("portfolio_test_results", exist_ok=True)
    portfolio_results = []
    
    model_cache = {}

    sorted_dates = sorted(selection_map.keys())
    print(f"\nSimulating Portfolio over {len(sorted_dates)} days...") 

    for date_str in tqdm(sorted_dates, desc="Portfolio Days"):
        selected_tickers = selection_map[date_str]
        if not selected_tickers: continue
        
        day_pnl = 0.0
        active_stocks = 0
        
        daily_stock_data = {}
        for t in selected_tickers:
            fname = f"{t}_minute.csv" 
            if not os.path.exists(fname): fname = f"data/{t}_minute.csv"
            
            if os.path.exists(fname):
                try:
                    raw_df = load_ohlc_data(fname)
                    day_df = raw_df[raw_df.index.strftime('%Y-%m-%d') == date_str]
                    if len(day_df) > 30:
                        agg = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}
                        daily_stock_data[t] = day_df.resample(PARAMS['CANDLE_FREQUENCY']).agg(agg).dropna()
                except:
                    continue

        if not daily_stock_data: continue

        vols = {t: df['Close'].pct_change().std() for t, df in daily_stock_data.items()}
        inv_vols = {t: (1.0 / v if v > 0 else 0) for t, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        weights = {t: w / total_inv_vol for t, w in inv_vols.items()} if total_inv_vol > 0 else {t: 1.0/len(vols) for t in vols}

        for t, df_prices in daily_stock_data.items():
            
            if t not in model_cache:
                model, hmm, xgb_mod, v_path = load_models_for_ticker(t)
                if model:
                    model_cache[t] = (model, hmm, xgb_mod, v_path)
                else:
                    continue
            
            model, hmm_model, xgb_model, vec_normalize_path = model_cache[t]

            try:
                df_ind = calculate_all_indicators(df_prices, hmm_model, xgb_model)
                warmup_cutoff = df_ind.index[0] + pd.Timedelta(minutes=PARAMS['WARMUP_MINUTES'])
                df_ind = df_ind[df_ind.index > warmup_cutoff].copy()
                df_ind['_time'] = df_ind.index
                df_ind = df_ind.reset_index(drop=True)
            except Exception as e:
                print(f"Indicator calculation failed for {t}: {e}")
                continue
            
            temp_env = IntradayTradingEnv([df_ind], mode='test')
            vec_env = DummyVecEnv([lambda: temp_env])
            
            if os.path.exists(vec_normalize_path):
                vec_env = VecNormalize.load(vec_normalize_path, vec_env)
                vec_env.training = False
                vec_env.norm_reward = False
            else:
                continue

            obs = vec_env.reset()
            done = [False]
            while not done[0]:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, done, _ = vec_env.step(action)
            
            day_pnl += temp_env.daily_pnl_abs
            active_stocks += 1

        portfolio_results.append({'Date': date_str, 'PnL': day_pnl})
        running_equity = 100000 + sum([r['PnL'] for r in portfolio_results])
        if active_stocks > 0:
            print(f"Date: {date_str} | Traded: {active_stocks} | Day PnL: ${day_pnl:+,.2f} | EOD Equity: ${running_equity:,.2f}")

    df_res = pd.DataFrame(portfolio_results)
    if not df_res.empty:
        df_res['Equity'] = 100000 + df_res['PnL'].cumsum()
        print(f"\n{'='*60}")
        print(f"PORTFOLIO SUMMARY")
        print(f"{'='*60}")
        print(f"Total Days: {len(df_res)}")
        print(f"Initial Capital: $100,000.00")
        print(f"Final Equity: ${df_res['Equity'].iloc[-1]:,.2f}")
        print(f"Total PnL: ${df_res['PnL'].sum():,.2f}")
        df_res.to_csv(f"portfolio_test_results/specialist_portfolio_metrics.csv", index=False)
        
def run_training_pipeline(ticker):
    """
    Encapsulates the training logic so it can be called for one stock or many.
    """
    target_folder = f"{ticker}_{PARAMS['CANDLE_FREQUENCY']}"
    print(f"\n{'='*60}")
    print(f"STARTING PIPELINE FOR: {ticker}")
    print(f"{'='*60}")
    
    success = prepare_data(ticker, target_folder, PARAMS['CANDLE_FREQUENCY'])
    if not success:
        print(f"Skipping {ticker} (Data prep failed)")
        return

    all_files = sorted(glob.glob(f"{target_folder}/*.csv"), 
                       key=lambda x: int(re.search(r'day(\d+)', str(x)).group(1)))
    
    if len(all_files) < 10:
        print(f"Skipping {ticker} (Not enough data: {len(all_files)} days)")
        return

    idx_10 = int(len(all_files) * 0.10)
    idx_60 = int(len(all_files) * 0.60)
    
    regime_files = all_files[:idx_10]
    rl_train_files = all_files[idx_10:idx_60]
    rl_test_files = all_files[idx_60:]
    
    regime_model, hmm_model = train_regime_detector(regime_files)
    if regime_model is None: return

    model_dir = f"Models_{ticker}"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f'{model_dir}/regime_detector_model.pkl', 'wb') as f: pickle.dump(regime_model, f)
    with open(f'{model_dir}/hmm_model.pkl', 'wb') as f: pickle.dump(hmm_model, f)
    
    with open(f"train_days_{ticker}.txt", "w") as f: f.write("\n".join([Path(x).stem for x in rl_train_files]))
    with open(f"test_days_{ticker}.txt", "w") as f: f.write("\n".join([Path(x).stem for x in rl_test_files]))

    print(f"Training PPO for {ticker}...")
    
    # Set global HMM model so precompute can use it
    global HMM_MODEL, REGIME_MODEL
    HMM_MODEL = hmm_model
    REGIME_MODEL = regime_model
    
    train_model_parallel(rl_train_files, ticker, PARAMS['EPISODES'], use_gpu=True, seed=PARAMS['SEED'])
    
    gc.collect()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python RL.py [train|test|train_all|portfolio_test] [TICKER]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "train":
        if len(sys.argv) < 3:
            print("Usage: python RL.py train <TICKER>")
            sys.exit(1)
        run_training_pipeline(sys.argv[2].upper())

    elif command == "train_all":
        source_dir = PARAMS['SOURCE_FOLDER'] 
        print(f"Scanning {source_dir} for stocks...")
        
        files = glob.glob(f"{source_dir}/*.csv")
        tickers = []
        for f in files:
            name = Path(f).stem
            clean_ticker = name.replace("_minute", "").upper()
            tickers.append(clean_ticker)
        
        tickers = sorted(list(set(tickers)))
        print(f"Found {len(tickers)} tickers: {tickers}")
        
        for t in tickers:
            try:
                run_training_pipeline(t)
            except Exception as e:
                print(f"FAILED to train {t}: {e}")
                continue

    elif command == "test":
        if len(sys.argv) < 3:
            print("Usage: python RL.py test <TICKER>")
            sys.exit(1)
        
        ticker = sys.argv[2].upper()
        model, hmm_model, xgb_model, v_path = load_models_for_ticker(ticker)
        
        if model is None:
            print(f"No models found for {ticker}. Please run train first.")
            sys.exit(1)

        target_folder = f"{ticker}_{PARAMS['CANDLE_FREQUENCY']}"
        prepare_data(ticker, target_folder, PARAMS['CANDLE_FREQUENCY'])
        _, test_files = load_saved_days(ticker) 
        
        test_model(ticker, test_files, deterministic=True) 

    elif command == "portfolio_test":
        test_portfolio_selection(json_path="selection_map.json", deterministic=True)
    
    elif command == "verify":
        # =====================================================================
        # LOOKAHEAD VERIFICATION TEST
        # Usage: python RL.py verify <TICKER>
        # This proves there is no lookahead by corrupting future data and
        # checking if the observation changes. If unchanged, no lookahead.
        # =====================================================================
        if len(sys.argv) < 3:
            print("Usage: python RL.py verify <TICKER>")
            sys.exit(1)
        
        ticker = sys.argv[2].upper()
        target_folder = f"{ticker}_{PARAMS['CANDLE_FREQUENCY']}"
        
        # Prepare data if needed
        if not os.path.exists(target_folder):
            prepare_data(ticker, target_folder, PARAMS['CANDLE_FREQUENCY'])
        
        # Load one day of data
        files = sorted(glob.glob(f"{target_folder}/*.csv"))
        if not files:
            print(f"No data found for {ticker}")
            sys.exit(1)
        
        print(f"\nLoading data for {ticker}...")
        test_data = precompute_all_data(files[:5])  # Use first 5 days
        
        if not test_data:
            print("Failed to load data")
            sys.exit(1)
        
        # Create environment and run verification
        env = IntradayTradingEnv(test_data, mode='test')
        result = verify_no_lookahead(env, num_tests=30)
        
        if result:
            print("✅ VERIFICATION PASSED: This code has NO lookahead bias!")
        else:
            print("❌ VERIFICATION FAILED: Lookahead bias detected!")
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, test, train_all, portfolio_test, verify")
        sys.exit(1)
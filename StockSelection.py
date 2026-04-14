import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

K = 10
START_DATE = "2015-04-02" 
END_DATE = "2025-08-06"   

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = "selection_map.json"

TICKERS = [
    "ABB.NS","ADANIENSOL.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS",
    "ADANIPOWER.NS","AMBUJACEM.NS","APOLLOHOSP.NS","ASIANPAINT.NS","DMART.NS",
    "AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BANKBARODA.NS",
    "BEL.NS","BPCL.NS","BHARTIARTL.NS","BOSCHLTD.NS","BRITANNIA.NS",
    "CGPOWER.NS","CANBK.NS","CHOLAFIN.NS","CIPLA.NS","COALINDIA.NS",
    "DLF.NS","DABUR.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS",
    "GAIL.NS","GODREJCP.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS",
    "HDFCLIFE.NS","HAVELLS.NS","HEROMOTOCO.NS","HINDALCO.NS","HAL.NS",
    "HINDUNILVR.NS","ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS","ITC.NS",
    "INDHOTEL.NS","IOC.NS","IRFC.NS","INDUSINDBK.NS","INFY.NS",
    "JSWENERGY.NS","JSWSTEEL.NS","JINDALSTEL.NS","JIOFIN.NS","KOTAKBANK.NS",
    "LTIM.NS","LT.NS","LICI.NS","M&M.NS","MARUTI.NS","NTPC.NS",
    "NESTLEIND.NS","ONGC.NS","PIDILITIND.NS","PFC.NS","POWERGRID.NS",
    "PNB.NS","RECLTD.NS","RELIANCE.NS","SBILIFE.NS","MOTHERSON.NS",
    "SHREECEM.NS","SHRIRAMFIN.NS","SIEMENS.NS","SBIN.NS","SUNPHARMA.NS",
    "TVSMOTOR.NS","TCS.NS","TATACONSUM.NS","TATAMOTORS.NS",
    "TATAPOWER.NS","TATASTEEL.NS","TECHM.NS","TITAN.NS","TRENT.NS",
    "ULTRACEMCO.NS","UNITDSPR.NS","VBL.NS","VEDL.NS","WIPRO.NS","ZYDUSLIFE.NS"
]

WEIGHTS = {
    "momentum_long": 0.30,
    "momentum_medium": 0.20,
    "volatility":    0.30,
    "liquidity":     0.10,
    "natr":          0.05, 
    "beta":          0.05
}

def download_data():
    print("Downloading market data...")
    dl_start = "2014-01-01"
    
    if not (DATA_DIR / "MARKET_NSEI.csv").exists():
        try:
            df = yf.download("^NSEI", start=dl_start, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[["Close"]].rename(columns={"Close": "close"})
                df.index.name = "datetime"
                df.to_csv(DATA_DIR / "MARKET_NSEI.csv")
        except Exception as e:
            print(f"Error downloading Market Data: {e}")

    for t in tqdm(TICKERS, desc="Downloading Stocks"):
        clean_name = t.replace('.', '_')
        fname = DATA_DIR / f"{clean_name}.csv"
        
        if fname.exists(): continue

        try:
            df = yf.download(t, start=dl_start, progress=False)
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
                
            df.columns = [c.capitalize() for c in df.columns]
            req_cols = ["Open","High","Low","Close","Volume"]
            
            if not all(col in df.columns for col in req_cols): continue
                
            df = df[req_cols]
            df.columns = df.columns.str.lower()
            df.index.name = "datetime"
            df.to_csv(fname)
            
        except: pass

def load_data():
    universe = {}
    for t in TICKERS:
        clean_name = t.replace('.', '_')
        f = DATA_DIR / f"{clean_name}.csv"
        if f.exists():
            try:
                df = pd.read_csv(f, parse_dates=["datetime"], index_col="datetime")
                df = df[~df.index.duplicated(keep='last')].sort_index()
                universe[t] = df
            except: pass
            
    try:
        market_df = pd.read_csv(DATA_DIR / "MARKET_NSEI.csv", parse_dates=["datetime"], index_col="datetime")
        market_df = market_df[~market_df.index.duplicated(keep='last')].sort_index()
    except:
        market_df = pd.DataFrame()
        print("Warning: Market data could not be loaded.")
        
    return universe, market_df

def calculate_ticker_metrics(df, market_df, ticker):
    df = df.copy()
    
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    
    df['momentum_long'] = df['close'] / df['close'].shift(60) - 1
    df['momentum_medium'] = df['close'] / df['close'].shift(20) - 1    
    df['volatility'] = df['ret'].rolling(60).std() * np.sqrt(252)    
    df['liquidity'] = (df['close'] * df['volume']).rolling(20).mean()
    
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift(1)).abs()
    l_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

    atr = tr.rolling(14).mean()
    df['natr'] = (atr / df['close']) * 100
    
    # Calculate Beta
    if not market_df.empty:
        mkt_ret = np.log(market_df['close'] / market_df['close'].shift(1))
        aligned_mkt = mkt_ret.reindex(df.index)
        
        rolling_cov = df['ret'].rolling(60).cov(aligned_mkt)
        rolling_mkt_var = aligned_mkt.rolling(60).var()
        df['beta'] = rolling_cov / (rolling_mkt_var + 1e-9)
    else:
        df['beta'] = 0.0
    
    metric_cols = list(WEIGHTS.keys())
    return df[metric_cols]

def generate_selection_map():
    download_data()
    universe, market_df = load_data()
    
    if not universe:
        print("Error: No stock data found.")
        return

    print("Computing vectorized metrics for all stocks...")
    
    all_metrics = []
    
    for ticker, df in tqdm(universe.items(), desc="Processing Stocks"):
        try:
            metrics_df = calculate_ticker_metrics(df, market_df, ticker)
            metrics_df['ticker'] = ticker
            all_metrics.append(metrics_df)
        except Exception as e:
            continue
            
    if not all_metrics:
        print("No metrics calculated.")
        return

    full_df = pd.concat(all_metrics)
    
    mask = (full_df.index >= pd.Timestamp(START_DATE)) & (full_df.index <= pd.Timestamp(END_DATE))
    full_df = full_df.loc[mask]
    
    print(f"Scoring & Ranking {len(full_df)} records...")
    
    def zscore(x):
        if x.std() == 0: return x * 0
        return (x - x.mean()) / (x.std() + 1e-9)

    grouped = full_df.groupby('datetime')
    
    scored_df = full_df.copy()
    
    for col in WEIGHTS.keys():
        scored_df[f'z_{col}'] = grouped[col].transform(zscore)
        scored_df[f'z_{col}'] = scored_df[f'z_{col}'].clip(-3, 3)

    scored_df['total_score'] = 0.0
    for col, weight in WEIGHTS.items():
        scored_df['total_score'] += scored_df[f'z_{col}'] * weight
        
    top_picks = scored_df.reset_index()
    top_picks = top_picks.sort_values(['datetime', 'total_score'], ascending=[True, False])
    top_picks = top_picks.groupby('datetime').head(K)
    
    selection_map = {}
    
    unique_dates = top_picks['datetime'].unique()
    
    print("Saving selections...")
    
    # Pre-calculate next market dates for efficiency
    market_dates = market_df.index
    # Create mapping: current_date -> next_valid_date
    next_date_map = {}
    if not market_df.empty:
        # iterate and map i to i+1
        for i in range(len(market_dates) - 1):
            next_date_map[market_dates[i]] = market_dates[i+1]
    
    count = 0
    for d in unique_dates:
        day_picks = top_picks[top_picks['datetime'] == d]
        selected = [t.replace('.NS', '') for t in day_picks['ticker'].tolist()]
        
        if selected:
            ts = pd.Timestamp(d)
            
            # Look up next market day, fallback to BusinessDay(1)
            next_bday = next_date_map.get(ts)
            if next_bday is None:
                next_bday = ts + pd.offsets.BusinessDay(1)
                
            next_d_str = next_bday.strftime("%Y-%m-%d")           
            selection_map[next_d_str] = selected
            count += 1
            
    with open(OUTPUT_FILE, "w") as f:
        json.dump(selection_map, f, indent=4)
        
    print(f"\nSaved selection map to {OUTPUT_FILE}")
    print(f"Days processed: {count}")

if __name__ == "__main__":
    generate_selection_map()
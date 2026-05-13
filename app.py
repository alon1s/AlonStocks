import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import requests
from io import StringIO
import concurrent.futures
import time
import random
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIG & CLOUD
# ==========================================
st.set_page_config(page_title="AlonStocks Pro", layout="wide", page_icon="🏦")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    
    /* Global font + RTL layout */
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .stApp, .reportview-container, .main { direction: rtl; text-align: right; }
    /* helper class to force LTR where needed */
    .ltr { direction: ltr !important; text-align: left !important; }
    .main { background: #0a0e1a; }
    
    .metric-card {
        background: linear-gradient(135deg, #0f1729 0%, #1a2444 100%);
        border: 1px solid #2a3a6a;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        box-shadow: 0 4px 24px rgba(0,150,255,0.08);
    }
    .signal-buy { color: #00ff88; font-weight: 700; }
    .signal-sell { color: #ff4466; font-weight: 700; }
    .signal-hold { color: #ffaa00; font-weight: 700; }
    
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85em;
        font-weight: 700;
    }
    
    .stDataFrame { font-family: 'Space Mono', monospace !important; font-size: 0.8em; }
    
    div[data-testid="stExpander"] {
        background: #0f1729;
        border: 1px solid #1e2d52;
        border-radius: 10px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Auto-detect Hebrew text and force RTL direction on matching elements
st.markdown("""
<script>
(function(){
    const hebRe = /[\u0590-\u05FF]/;
    function setRTL(){
        document.querySelectorAll('body *').forEach(el=>{
            try{
                if(el.children.length===0 && hebRe.test(el.innerText)){
                    el.style.direction = 'rtl';
                    el.style.textAlign = 'right';
                }
            }catch(e){}
        });
    }
    setRTL();
    const obs = new MutationObserver(()=>setRTL());
    obs.observe(document.body, {childList:true, subtree:true, characterData:true});
})();
</script>
""", unsafe_allow_html=True)

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

conn = st.connection("gsheets", type=GSheetsConnection)

def safe_conn_read(worksheet, ttl=0, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            df = conn.read(worksheet=worksheet, ttl=ttl)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"safe_conn_read error for '{worksheet}' (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(backoff ** attempt)
    return pd.DataFrame()


def normalize_portfolio_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')

    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    rename_map = {}
    for col in cleaned.columns:
        key = str(col).strip().lower().replace(' ', '')
        if key in {'ticker', 'symbol'}:
            rename_map[col] = 'Ticker'
        elif key in {'quantity', 'qty', 'shares'}:
            rename_map[col] = 'Quantity'
        elif key in {'purchaseprice', 'buyprice', 'costbasis', 'avgprice', 'averageprice'}:
            rename_map[col] = 'PurchasePrice'
        elif key == 'index' and 'Ticker' not in rename_map.values():
            rename_map[col] = 'Ticker'

    cleaned = cleaned.rename(columns=rename_map)

    if 'Ticker' not in cleaned.columns and cleaned.index.name is not None:
        cleaned = cleaned.reset_index()
        cleaned.columns = [str(col).strip() for col in cleaned.columns]

    if 'Ticker' not in cleaned.columns:
        first_col = cleaned.columns[0]
        cleaned = cleaned.rename(columns={first_col: 'Ticker'})

    if 'Quantity' not in cleaned.columns:
        cleaned['Quantity'] = 0.0
    if 'PurchasePrice' not in cleaned.columns:
        cleaned['PurchasePrice'] = 0.0

    cleaned = cleaned[['Ticker', 'Quantity', 'PurchasePrice']]
    cleaned = cleaned.dropna(subset=['Ticker'])
    cleaned['Ticker'] = cleaned['Ticker'].astype(str).str.strip()
    cleaned = cleaned[cleaned['Ticker'] != '']
    cleaned['Quantity'] = pd.to_numeric(cleaned['Quantity'], errors='coerce').fillna(0.0)
    cleaned['PurchasePrice'] = pd.to_numeric(cleaned['PurchasePrice'], errors='coerce').fillna(0.0)
    return cleaned.set_index('Ticker')


def load_local_portfolio_fallback():
    local_path = os.path.join(os.path.dirname(__file__), 'portfolio_data.csv')
    if os.path.exists(local_path):
        try:
            return normalize_portfolio_df(pd.read_csv(local_path))
        except Exception as e:
            print(f"Local portfolio fallback failed: {e}")
    return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')

def load_cloud_portfolio():
    try:
        df = safe_conn_read("Portfolio", ttl=0)
        normalized = normalize_portfolio_df(df)
        if normalized.empty:
            return load_local_portfolio_fallback()
        return normalized
    except Exception as e:
        print(f"load_cloud_portfolio error: {e}")
        return load_local_portfolio_fallback()

def save_cloud_portfolio(df):
    try:
        payload = df.reset_index()
        payload.columns = [str(col).strip() for col in payload.columns]
        conn.update(worksheet="Portfolio", data=payload)
        try:
            local_path = os.path.join(os.path.dirname(__file__), 'portfolio_data.csv')
            payload.to_csv(local_path, index=False)
        except Exception as e:
            print(f"save_cloud_portfolio local cache error: {e}")
        st.cache_data.clear()
        return True
    except Exception as e:
        print(f"save_cloud_portfolio error: {e}")
        return False

def log_activity(ticker, action, qty, price, notes=""):
    try:
        new_log = pd.DataFrame([{
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": ticker, "Action": action,
            "Quantity": float(qty), "Price": float(price), "Notes": notes
        }])
        try:
            existing = safe_conn_read("Activity", ttl=0)
        except:
            existing = pd.DataFrame()
        updated = pd.concat([existing, new_log], ignore_index=True) if not existing.empty else new_log
        conn.update(worksheet="Activity", data=updated)
    except:
        pass

# Always hydrate from the cloud sheet so the app reflects DB changes on every rerun.
st.session_state.portfolio = load_cloud_portfolio()

CONFIG_FILE = "config.json"
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}

if 'config' not in st.session_state:
    st.session_state.config = load_config()

# ==========================================
# 2. ADVANCED DATA ENGINE
# ==========================================
@st.cache_data(ttl=600)
def get_usd_ils():
    try:
        return yf.Ticker("USDILS=X").fast_info['lastPrice']
    except:
        return 3.75

@st.cache_data(ttl=3600)
def get_global_tickers():
    tickers = set()
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        resp = requests.get(url, headers=session.headers, timeout=10)
        table = pd.read_html(StringIO(resp.text))[0]
        tickers.update([t.replace('.', '-') for t in table['Symbol'].tolist()])
    except:
        pass
    return list(tickers)

def compute_advanced_indicators(hist):
    """Compute 20+ technical indicators"""
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    volume = hist['Volume']
    
    indicators = {}
    
    # Moving Averages
    for p in [5, 10, 20, 50, 100, 200]:
        if len(close) >= p:
            indicators[f'sma{p}'] = close.rolling(p).mean().iloc[-1]
            indicators[f'ema{p}'] = close.ewm(span=p, adjust=False).mean().iloc[-1]
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    indicators['rsi'] = (100 - (100 / (1 + (gain / loss)))).iloc[-1]
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    indicators['macd'] = macd_line.iloc[-1]
    indicators['macd_signal'] = signal_line.iloc[-1]
    indicators['macd_hist'] = (macd_line - signal_line).iloc[-1]
    indicators['macd_crossover'] = (macd_line.iloc[-1] > signal_line.iloc[-1]) and (macd_line.iloc[-2] <= signal_line.iloc[-2])
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = (sma20 + 2 * std20).iloc[-1]
    bb_lower = (sma20 - 2 * std20).iloc[-1]
    bb_mid = sma20.iloc[-1]
    indicators['bb_upper'] = bb_upper
    indicators['bb_lower'] = bb_lower
    indicators['bb_pct'] = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    indicators['bb_width'] = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0
    
    # Stochastic Oscillator
    if len(close) >= 14:
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k = 100 * (close - low14) / (high14 - low14)
        d = k.rolling(3).mean()
        indicators['stoch_k'] = k.iloc[-1]
        indicators['stoch_d'] = d.iloc[-1]
    else:
        indicators['stoch_k'] = 50
        indicators['stoch_d'] = 50
    
    # ATR (Average True Range) - Volatility
    if len(hist) >= 14:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        indicators['atr'] = tr.rolling(14).mean().iloc[-1]
        indicators['atr_pct'] = indicators['atr'] / close.iloc[-1] * 100
    else:
        indicators['atr'] = 0
        indicators['atr_pct'] = 0
    
    # Volume analysis
    if len(volume) >= 20:
        avg_vol = volume.rolling(20).mean()
        indicators['vol_ratio'] = (volume.iloc[-1] / avg_vol.iloc[-1]) if avg_vol.iloc[-1] > 0 else 1
        indicators['vol_trend'] = volume.rolling(5).mean().iloc[-1] / volume.rolling(20).mean().iloc[-1] if avg_vol.iloc[-1] > 0 else 1
    else:
        indicators['vol_ratio'] = 1
        indicators['vol_trend'] = 1
    
    # OBV (On Balance Volume)
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    obv_sma = obv.rolling(20).mean()
    indicators['obv_trend'] = 1 if obv.iloc[-1] > obv_sma.iloc[-1] else -1
    
    # Fibonacci levels
    high_1y = close.max()
    low_1y = close.min()
    diff = high_1y - low_1y
    indicators['fib_236'] = high_1y - diff * 0.236
    indicators['fib_382'] = high_1y - diff * 0.382
    indicators['fib_500'] = high_1y - diff * 0.5
    indicators['fib_618'] = high_1y - diff * 0.618
    indicators['fib_786'] = high_1y - diff * 0.786
    
    # Price relative to 52-week range
    indicators['pct_from_high'] = ((close.iloc[-1] - high_1y) / high_1y) * 100
    indicators['pct_from_low'] = ((close.iloc[-1] - low_1y) / low_1y) * 100
    
    # Momentum
    if len(close) >= 20:
        indicators['mom_1m'] = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100
    else:
        indicators['mom_1m'] = 0
    if len(close) >= 63:
        indicators['mom_3m'] = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100
    else:
        indicators['mom_3m'] = indicators.get('mom_1m', 0)
    if len(close) >= 126:
        indicators['mom_6m'] = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100
    else:
        indicators['mom_6m'] = indicators.get('mom_3m', 0)
    
    # Trend strength
    sma20_val = close.rolling(20).mean().iloc[-1]
    sma50_val = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20_val
    sma200_val = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50_val
    curr = close.iloc[-1]
    
    if sma20_val > sma50_val > sma200_val and curr > sma20_val:
        indicators['trend'] = "Strong Uptrend"
        indicators['trend_score'] = 5
    elif sma20_val > sma50_val and curr > sma20_val:
        indicators['trend'] = "Moderate Uptrend"
        indicators['trend_score'] = 3
    elif sma20_val < sma50_val < sma200_val and curr < sma20_val:
        indicators['trend'] = "Strong Downtrend"
        indicators['trend_score'] = 1
    elif sma20_val < sma50_val and curr < sma20_val:
        indicators['trend'] = "Moderate Downtrend"
        indicators['trend_score'] = 2
    else:
        indicators['trend'] = "Consolidation"
        indicators['trend_score'] = 3
    
    indicators['sma20'] = sma20_val
    indicators['sma50'] = sma50_val
    indicators['sma200'] = sma200_val
    
    return indicators

def compute_composite_score(ind, info):
    """Compute a 0-100 composite quant score"""
    score = 50
    reasons = []
    
    # Trend (max 20 pts)
    ts = ind.get('trend_score', 3)
    score += (ts - 3) * 4
    if ts >= 4: reasons.append("✅ Strong uptrend")
    elif ts <= 2: reasons.append("❌ Downtrend detected")
    
    # RSI (max ±10 pts)
    rsi = ind.get('rsi', 50)
    if 40 < rsi < 60: score += 5; reasons.append("✅ RSI neutral zone")
    elif rsi < 35: score += 8; reasons.append("✅ Oversold - buy opportunity")
    elif rsi > 70: score -= 8; reasons.append("⚠️ Overbought - caution")
    elif rsi > 80: score -= 12; reasons.append("🚨 Extremely overbought")
    
    # MACD (max ±8 pts)
    if ind.get('macd_crossover'): score += 8; reasons.append("✅ MACD bullish crossover")
    elif ind.get('macd', 0) > ind.get('macd_signal', 0): score += 4; reasons.append("✅ MACD positive")
    else: score -= 4; reasons.append("⚠️ MACD negative")
    
    # Volume (max ±6 pts)
    vr = ind.get('vol_ratio', 1)
    if vr > 1.5 and ind.get('trend_score', 3) > 3: score += 6; reasons.append("✅ High volume uptrend")
    elif vr > 1.5 and ind.get('trend_score', 3) < 3: score -= 6; reasons.append("⚠️ High volume downtrend")
    
    # Momentum (max ±8 pts)
    m1 = ind.get('mom_1m', 0)
    m3 = ind.get('mom_3m', 0)
    if m1 > 5 and m3 > 10: score += 8; reasons.append("✅ Strong momentum")
    elif m1 > 0 and m3 > 0: score += 4
    elif m1 < -10: score -= 8; reasons.append("⚠️ Negative momentum")
    
    # Fundamentals (max ±10 pts)
    pe = info.get('trailingPE', 0) or 0
    if 0 < pe < 15: score += 8; reasons.append("✅ Low P/E (value)")
    elif 15 <= pe < 25: score += 4
    elif pe > 50: score -= 6; reasons.append("⚠️ High P/E")
    
    growth = (info.get('revenueGrowth', 0) or 0) * 100
    if growth > 20: score += 6; reasons.append("✅ High revenue growth")
    elif growth > 10: score += 3
    elif growth < 0: score -= 6; reasons.append("⚠️ Revenue decline")
    
    # Analyst recommendation
    rec = info.get('recommendationKey', 'none')
    if rec in ['strong_buy', 'buy']: score += 5; reasons.append("✅ Analyst: Buy")
    elif rec in ['sell', 'strong_sell']: score -= 5; reasons.append("⚠️ Analyst: Sell")
    
    return min(100, max(0, score)), reasons

def get_signal(score, rsi, macd, macd_signal, trend_score):
    """Generate trading signal"""
    if score >= 70 and rsi < 65 and trend_score >= 4:
        return "STRONG BUY", "#00ff88"
    elif score >= 60 and trend_score >= 3:
        return "BUY", "#44ff99"
    elif score <= 30 and trend_score <= 2:
        return "STRONG SELL", "#ff2244"
    elif score <= 40:
        return "SELL", "#ff6688"
    elif 45 <= score <= 55:
        return "HOLD", "#ffaa00"
    elif score > 55:
        return "WATCH", "#44aaff"
    else:
        return "AVOID", "#ff8844"

@st.cache_data(ttl=900)
def fetch_expert_data(tickers_to_fetch):
    data = {}
    valid_list = [t for t in tickers_to_fetch if isinstance(t, str) and t.strip()]
    
    def fetch_single(t):
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="2y")
            if len(hist) < 50:
                return None
            info = stock.info
            curr = hist['Close'].iloc[-1]
            
            ind = compute_advanced_indicators(hist)
            score, reasons = compute_composite_score(ind, info)
            signal, signal_color = get_signal(
                score, ind['rsi'], ind['macd'], ind['macd_signal'], ind['trend_score']
            )
            
            # Earnings surprise
            try:
                earnings = stock.earnings_history
                recent_surprise = earnings['surprisePercent'].iloc[-1] if earnings is not None and not earnings.empty else 0
            except:
                recent_surprise = 0
            
            # Short interest
            short_pct = (info.get('shortPercentOfFloat') or 0) * 100
            
            # Institutional ownership
            inst_own = (info.get('institutionsPercentHeld') or 0) * 100
            
            return t, {
                'price': curr,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'pe': info.get('trailingPE', 0) or 0,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'peg': info.get('pegRatio', 0) or 0,
                'ps': info.get('priceToSalesTrailing12Months', 0) or 0,
                'pb': info.get('priceToBook', 0) or 0,
                'beta': info.get('beta', 1.0) or 1.0,
                'div': (info.get('dividendYield', 0) or 0) * 100,
                'market_cap': info.get('marketCap', 0) or 0,
                'revenue': info.get('totalRevenue', 0) or 0,
                'gross_margin': (info.get('grossMargins', 0) or 0) * 100,
                'profit_margin': (info.get('profitMargins', 0) or 0) * 100,
                'roe': (info.get('returnOnEquity', 0) or 0) * 100,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'current_ratio': info.get('currentRatio', 0) or 0,
                'growth_yoy': (info.get('revenueGrowth', 0) or 0) * 100,
                'earnings_growth': (info.get('earningsGrowth', 0) or 0) * 100,
                'analyst': info.get('recommendationKey', 'none'),
                'target_price': info.get('targetMeanPrice', curr) or curr,
                'target_upside': ((info.get('targetMeanPrice', curr) or curr) - curr) / curr * 100,
                'short_pct': short_pct,
                'inst_own': inst_own,
                'earnings_surprise': recent_surprise,
                'currency': "ILS" if str(t).endswith(".TA") else "USD",
                **ind,
                'score': score,
                'reasons': reasons,
                'signal': signal,
                'signal_color': signal_color,
            }
        except Exception as e:
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        results = executor.map(fetch_single, valid_list)
        for r in results:
            if r:
                data[r[0]] = r[1]
    return data

# ==========================================
# ML FORECAST ENGINE
# ==========================================
@st.cache_data(ttl=1800)
def ml_forecast(ticker, days=14):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        if len(hist) < 252:
            return None, None
        
        df = hist.copy()
        df['ret_1'] = df['Close'].pct_change(1)
        df['ret_5'] = df['Close'].pct_change(5)
        df['ret_20'] = df['Close'].pct_change(20)
        df['vol_20'] = df['Close'].pct_change().rolling(20).std()
        df['sma20'] = df['Close'].rolling(20).mean()
        df['sma50'] = df['Close'].rolling(50).mean()
        df['ratio_20_50'] = df['sma20'] / df['sma50']
        
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss)))
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        for lag in [1, 2, 3, 5, 10]:
            df[f'lag_{lag}'] = df['Close'].shift(lag)
        
        df['target'] = df['Close'].shift(-days)
        df = df.dropna()
        
        feature_cols = ['ret_1', 'ret_5', 'ret_20', 'vol_20', 'ratio_20_50', 'rsi', 'vol_ratio',
                        'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10']
        
        X = df[feature_cols]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train_s, y_train)
        
        # Predict future
        last_row = X.iloc[-1:].copy()
        last_scaled = scaler.transform(last_row)
        pred_rf = rf.predict(last_scaled)[0]
        pred_gb = gb.predict(last_scaled)[0]
        pred_ensemble = (pred_rf * 0.6 + pred_gb * 0.4)
        
        curr_price = df['Close'].iloc[-1]
        
        # Feature importance
        importance = dict(zip(feature_cols, rf.feature_importances_))
        
        return pred_ensemble, {
            'current': curr_price,
            'predicted': pred_ensemble,
            'pct_change': (pred_ensemble - curr_price) / curr_price * 100,
            'rf_pred': pred_rf,
            'gb_pred': pred_gb,
            'importance': importance,
            'days': days
        }
    except Exception as e:
        return None, None

# ==========================================
# CORRELATION & PORTFOLIO ANALYTICS
# ==========================================
@st.cache_data(ttl=1800)
def compute_portfolio_analytics(tickers):
    try:
        data = yf.download(tickers, period="1y", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            closes = data['Close']
        else:
            closes = data[['Close']] if 'Close' in data.columns else data
        
        returns = closes.pct_change().dropna()
        corr = returns.corr()
        
        # Sharpe ratio (annualized, assuming rf=5%)
        sharpe = {}
        for col in returns.columns:
            if returns[col].std() > 0:
                sharpe[col] = (returns[col].mean() * 252 - 0.05) / (returns[col].std() * np.sqrt(252))
        
        # Max drawdown
        max_dd = {}
        for col in closes.columns:
            roll_max = closes[col].cummax()
            drawdown = (closes[col] - roll_max) / roll_max
            max_dd[col] = drawdown.min() * 100
        
        # VaR 95%
        var95 = {}
        for col in returns.columns:
            var95[col] = np.percentile(returns[col].dropna(), 5) * 100
        
        return corr, sharpe, max_dd, var95
    except:
        return None, None, None, None

# ==========================================
# 3. SIDEBAR
# ==========================================
st.sidebar.header("💰 קופת מזומנים")
n_usd = st.sidebar.number_input("מזומן בדולר $", value=float(st.session_state.config['cash_usd']))
n_ils = st.sidebar.number_input("מזומן בשקל ₪", value=float(st.session_state.config['cash_ils']))
if n_usd != st.session_state.config['cash_usd'] or n_ils != st.session_state.config['cash_ils']:
    st.session_state.config = {"cash_usd": n_usd, "cash_ils": n_ils}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(st.session_state.config, f)

st.sidebar.markdown("---")
st.sidebar.subheader("🛒 ביצוע טרייד")
with st.sidebar.form("trade_form", clear_on_submit=True):
    t_in = st.text_input("סימול מניה").upper()
    act_in = st.selectbox("פעולה", ["Buy", "Sell"])
    q_in = st.number_input("כמות", min_value=0.0, step=0.1)
    p_in = st.number_input("מחיר", min_value=0.0, step=0.01)
    notes_in = st.text_area("הערות")
    if st.form_submit_button("עדכן לענן 🚀"):
        if t_in and q_in > 0:
            if act_in == "Buy":
                if t_in in st.session_state.portfolio.index:
                    oq, op = st.session_state.portfolio.loc[t_in, ['Quantity', 'PurchasePrice']]
                    nq = oq + q_in
                    st.session_state.portfolio.loc[t_in] = [nq, ((oq*op)+(q_in*p_in))/nq]
                else:
                    st.session_state.portfolio.loc[t_in] = [q_in, p_in]
            else:
                if t_in in st.session_state.portfolio.index:
                    nq = max(0, st.session_state.portfolio.loc[t_in, 'Quantity'] - q_in)
                    if nq == 0:
                        st.session_state.portfolio = st.session_state.portfolio.drop(t_in)
                    else:
                        st.session_state.portfolio.loc[t_in, 'Quantity'] = nq
            if save_cloud_portfolio(st.session_state.portfolio):
                log_activity(t_in, act_in, q_in, p_in, notes_in)
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 בדיקת מניה ספציפית")
quick_ticker = st.sidebar.text_input("הכנס סימול לניתוח מהיר").upper()

if st.sidebar.button("🔄 רענון תיק מה-DB"):
    st.session_state.portfolio = load_cloud_portfolio()
    st.cache_data.clear()
    st.rerun()

# ==========================================
# 4. MAIN TABS
# ==========================================
usd_ils_rate = get_usd_ils()
p_tickers = [t for t in st.session_state.portfolio.index if isinstance(t, str) and t.strip()]

tabs = st.tabs([
    "💼 התיק שלי",
    "📊 ניתוח טכני מעמיק",
    "🤖 AI Forecast",
    "🔥 סורק הזדמנויות",
    "⚖️ ניתוח סיכונים",
    "🎯 אסטרטגיות",
    "📜 יומן פעולות"
])
t_port, t_tech, t_ai, t_scan, t_risk, t_strat, t_journal = tabs

m_data = fetch_expert_data(p_tickers + ['SPY', 'QQQ', 'VIX'])

# ==========================================
# TAB 1: PORTFOLIO
# ==========================================
with t_port:
    st.subheader("💼 תיק השקעות - סקירה מקיפה")
    
    stock_val_usd = 0
    rows = []
    sector_weights = {}

    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]
            qty = row['Quantity']
            bp = row['PurchasePrice']
            v_u = (d['price'] * qty / (usd_ils_rate if d['currency'] == "ILS" else 1))
            stock_val_usd += v_u
            sector_weights[d['sector']] = sector_weights.get(d['sector'], 0) + v_u
            pl_pct = ((d['price'] - bp) / bp) * 100
            rows.append({
                "Ticker": t, "Sector": d['sector'], "Qty": qty,
                "Price": d['price'], "Buy@": bp,
                "P&L %": round(pl_pct, 2),
                "P&L $": round((d['price'] - bp) * qty, 2),
                "Value $": round(v_u, 2),
                "Score": d.get('score', 50),
                "Signal": d.get('signal', 'HOLD'),
                "RSI": round(d.get('rsi', 50), 1),
                "Trend": d.get('trend', '-'),
                "Target $": round(d.get('target_price', d['price']), 2),
                "Upside %": round(d.get('target_upside', 0), 1),
            })

    total_usd = stock_val_usd + n_usd + (n_ils / usd_ils_rate)
    total_ils = total_usd * usd_ils_rate
    profit_usd = total_usd - 7000
    profit_ils = total_ils - 25500

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 שווי כולל ($)", f"${total_usd:,.2f}")
    col2.metric("📈 רווח מול $7,000", f"${profit_usd:,.2f}", f"{(profit_usd/7000)*100:.1f}%")
    col3.metric("📈 רווח מול ₪25,500", f"₪{profit_ils:,.2f}", f"{(profit_ils/25500)*100:.1f}%")
    col4.metric("💱 שע\"ח USD/ILS", f"₪{usd_ils_rate:.3f}")

    if rows:
        df_rows = pd.DataFrame(rows)
        
        def color_signal(val):
            colors = {'STRONG BUY': 'color: #00ff88; font-weight: bold',
                     'BUY': 'color: #44dd88', 'HOLD': 'color: #ffaa00',
                     'WATCH': 'color: #44aaff', 'SELL': 'color: #ff6688',
                     'STRONG SELL': 'color: #ff2244; font-weight: bold'}
            return colors.get(val, '')
        
        def color_pl(val):
            try:
                return 'color: #00ff88' if float(val) >= 0 else 'color: #ff4466'
            except:
                return ''
        
        styled = df_rows.style.map(color_signal, subset=['Signal']) \
                              .map(color_pl, subset=['P&L %', 'P&L $', 'Upside %']) \
                              .format({'Price': '${:.2f}', 'Buy@': '${:.2f}', 'Value $': '${:,.2f}',
                                      'P&L %': '{:.2f}%', 'P&L $': '${:,.2f}',
                                      'Target $': '${:.2f}', 'Upside %': '{:.1f}%'})
        st.dataframe(styled, use_container_width=True)
        
        # Sector Pie
        if sector_weights:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig_pie = px.pie(
                    values=list(sector_weights.values()),
                    names=list(sector_weights.keys()),
                    title="פיזור סקטוריאלי",
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_p2:
                # Score bar chart
                fig_score = go.Figure(go.Bar(
                    x=[r['Ticker'] for r in rows],
                    y=[r['Score'] for r in rows],
                    marker_color=[
                        '#00ff88' if s >= 70 else '#44aaff' if s >= 55 else '#ffaa00' if s >= 45 else '#ff4466'
                        for s in [r['Score'] for r in rows]
                    ],
                    text=[r['Signal'] for r in rows],
                    textposition='auto'
                ))
                fig_score.update_layout(title="Quant Score לכל מניה", paper_bgcolor='rgba(0,0,0,0)',
                                        font_color='white', yaxis_range=[0, 100])
                st.plotly_chart(fig_score, use_container_width=True)

# ==========================================
# TAB 2: DEEP TECHNICAL ANALYSIS
# ==========================================
with t_tech:
    st.subheader("📊 ניתוח טכני מעמיק - 20+ אינדיקטורים")
    
    analyze_ticker = st.selectbox("בחר מניה לניתוח:", p_tickers + ([quick_ticker] if quick_ticker else []))
    period_choice = st.select_slider("תקופה:", options=["3mo", "6mo", "1y", "2y"], value="1y")
    
    if analyze_ticker:
        with st.spinner(f"טוען נתונים ל-{analyze_ticker}..."):
            stock = yf.Ticker(analyze_ticker)
            hist = stock.history(period=period_choice)
            
            if not hist.empty:
                d = m_data.get(analyze_ticker, {})
                
                # Summary row
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("מחיר", f"${d.get('price', hist['Close'].iloc[-1]):,.2f}")
                c2.metric("RSI", f"{d.get('rsi', 50):.1f}", 
                         "Overbought" if d.get('rsi', 50) > 70 else "Oversold" if d.get('rsi', 50) < 30 else "Neutral")
                c3.metric("Score", f"{d.get('score', 50)}/100")
                c4.metric("Signal", d.get('signal', 'N/A'))
                c5.metric("יעד אנליסטים", f"${d.get('target_price', 0):,.2f}", f"{d.get('target_upside', 0):.1f}%")
                
                # MAIN CHART with indicators
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    row_heights=[0.5, 0.17, 0.17, 0.16],
                                    subplot_titles=[f"{analyze_ticker} - מחיר", "RSI", "MACD", "Volume"])
                
                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=hist.index, open=hist['Open'], high=hist['High'],
                    low=hist['Low'], close=hist['Close'], name="Price",
                    increasing_line_color='#00ff88', decreasing_line_color='#ff4466'
                ), row=1, col=1)
                
                # Bollinger Bands
                sma20 = hist['Close'].rolling(20).mean()
                std20 = hist['Close'].rolling(20).std()
                fig.add_trace(go.Scatter(x=hist.index, y=sma20+2*std20, name='BB Upper',
                                         line=dict(color='rgba(100,100,255,0.4)', dash='dot'), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=sma20-2*std20, name='BB Lower',
                                         line=dict(color='rgba(100,100,255,0.4)', dash='dot'),
                                         fill='tonexty', fillcolor='rgba(100,100,255,0.04)', showlegend=False), row=1, col=1)
                
                # SMAs
                for period, color in [(20,'#ffaa00'), (50,'#44aaff'), (200,'#ff6644')]:
                    if len(hist) >= period:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(period).mean(),
                                                  name=f'SMA{period}', line=dict(color=color, width=1.5)), row=1, col=1)
                
                # Fibonacci levels
                high_1y = hist['Close'].max()
                low_1y = hist['Close'].min()
                diff = high_1y - low_1y
                for level, pct in [(0.382, '#ff8844'), (0.618, '#44ff88')]:
                    fib_val = high_1y - diff * level
                    fig.add_hline(y=fib_val, line_dash="dash", line_color=pct,
                                  annotation_text=f"Fib {level}", row=1, col=1)
                
                # RSI
                delta = hist['Close'].diff()
                gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
                rsi_series = 100 - (100 / (1 + (gain / loss)))
                fig.add_trace(go.Scatter(x=hist.index, y=rsi_series, name='RSI',
                                         line=dict(color='#aa44ff', width=1.5)), row=2, col=1)
                fig.add_hline(y=70, line_color='red', line_dash='dash', row=2, col=1)
                fig.add_hline(y=30, line_color='green', line_dash='dash', row=2, col=1)
                
                # MACD
                ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
                ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                macd_sig = macd.ewm(span=9, adjust=False).mean()
                macd_hist_vals = macd - macd_sig
                fig.add_trace(go.Scatter(x=hist.index, y=macd, name='MACD', line=dict(color='#44aaff', width=1.5)), row=3, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=macd_sig, name='Signal', line=dict(color='#ff8844', width=1.5)), row=3, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=macd_hist_vals, name='Histogram',
                                     marker_color=['#00ff88' if v >= 0 else '#ff4466' for v in macd_hist_vals]), row=3, col=1)
                
                # Volume
                fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume',
                                     marker_color='rgba(100,150,255,0.4)'), row=4, col=1)
                
                fig.update_layout(
                    height=900, paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1729',
                    font_color='white', xaxis_rangeslider_visible=False,
                    legend=dict(orientation='h', y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # All indicators table
                st.markdown("### 📋 טבלת אינדיקטורים מלאה")
                if d:
                    ind_data = {
                        "📈 מגמה": d.get('trend', '-'),
                        "RSI (14)": f"{d.get('rsi', 0):.1f}",
                        "MACD": f"{d.get('macd', 0):.3f}",
                        "MACD Signal": f"{d.get('macd_signal', 0):.3f}",
                        "Stoch %K": f"{d.get('stoch_k', 0):.1f}",
                        "Stoch %D": f"{d.get('stoch_d', 0):.1f}",
                        "BB %": f"{d.get('bb_pct', 0)*100:.1f}%",
                        "ATR %": f"{d.get('atr_pct', 0):.2f}%",
                        "Vol Ratio": f"{d.get('vol_ratio', 1):.2f}x",
                        "Momentum 1M": f"{d.get('mom_1m', 0):.1f}%",
                        "Momentum 3M": f"{d.get('mom_3m', 0):.1f}%",
                        "Momentum 6M": f"{d.get('mom_6m', 0):.1f}%",
                        "P/E": f"{d.get('pe', 0):.1f}",
                        "Forward P/E": f"{d.get('forward_pe', 0):.1f}",
                        "PEG": f"{d.get('peg', 0):.2f}",
                        "P/S": f"{d.get('ps', 0):.2f}",
                        "P/B": f"{d.get('pb', 0):.2f}",
                        "ROE": f"{d.get('roe', 0):.1f}%",
                        "Gross Margin": f"{d.get('gross_margin', 0):.1f}%",
                        "Profit Margin": f"{d.get('profit_margin', 0):.1f}%",
                        "Revenue Growth": f"{d.get('growth_yoy', 0):.1f}%",
                        "Debt/Equity": f"{d.get('debt_to_equity', 0):.2f}",
                        "Short %": f"{d.get('short_pct', 0):.1f}%",
                        "Inst. Own %": f"{d.get('inst_own', 0):.1f}%",
                        "Analyst Target": f"${d.get('target_price', 0):.2f}",
                        "Upside": f"{d.get('target_upside', 0):.1f}%",
                    }
                    ind_df = pd.DataFrame(list(ind_data.items()), columns=['Indicator', 'Value'])
                    col_i1, col_i2 = st.columns(2)
                    half = len(ind_df) // 2
                    col_i1.dataframe(ind_df.iloc[:half], use_container_width=True, hide_index=True)
                    col_i2.dataframe(ind_df.iloc[half:], use_container_width=True, hide_index=True)
                
                # Reasons for score
                st.markdown(f"### 🧠 ניתוח AI - למה Score={d.get('score', 0)}")
                for reason in d.get('reasons', []):
                    if "✅" in reason:
                        st.success(reason)
                    elif "⚠️" in reason or "🚨" in reason:
                        st.warning(reason)
                    else:
                        st.info(reason)

# ==========================================
# TAB 3: AI FORECAST
# ==========================================
with t_ai:
    st.subheader("🤖 תחזית AI - Random Forest + Gradient Boosting Ensemble")
    
    forecast_ticker = st.selectbox("מניה לתחזית:", p_tickers + ([quick_ticker] if quick_ticker else []), key='forecast_sel')
    forecast_days = st.slider("ימי תחזית:", 7, 30, 14)
    
    if st.button("🚀 הרץ מודל AI"):
        with st.spinner("מאמן מודל על 5 שנות היסטוריה..."):
            pred, details = ml_forecast(forecast_ticker, forecast_days)
            
            if details:
                c1, c2, c3 = st.columns(3)
                c1.metric("מחיר נוכחי", f"${details['current']:,.2f}")
                c2.metric(f"תחזית ל-{forecast_days} ימים", f"${details['predicted']:,.2f}",
                         f"{details['pct_change']:.1f}%")
                direction = "📈 עלייה צפויה" if details['pct_change'] > 0 else "📉 ירידה צפויה"
                c3.metric("כיוון", direction)
                
                # Individual model predictions
                st.markdown("#### תחזיות מודלים בודדים:")
                col_m1, col_m2 = st.columns(2)
                col_m1.info(f"🌲 Random Forest: ${details['rf_pred']:,.2f}")
                col_m2.info(f"⚡ Gradient Boost: ${details['gb_pred']:,.2f}")
                
                # Feature importance chart
                imp_df = pd.DataFrame(list(details['importance'].items()), 
                                      columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
                fig_imp = go.Figure(go.Bar(
                    x=imp_df['Importance'], y=imp_df['Feature'],
                    orientation='h', marker_color='#44aaff'
                ))
                fig_imp.update_layout(title="חשיבות פיצ'רים (Feature Importance)",
                                       paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
                
                st.warning("⚠️ אזהרה: תחזיות ML הן הסתברותיות בלבד ואינן ערובה לתשואה. תמיד בצע Due Diligence עצמאי.")
            else:
                st.error("לא היה מספיק דאטה לאמן את המודל.")

# ==========================================
# TAB 4: OPPORTUNITY SCANNER
# ==========================================
with t_scan:
    st.subheader("🔥 סורק הזדמנויות מתקדם")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        scan_universe = st.selectbox("יקום סריקה:", [
            "Mega Watchlist (18 מניות)",
            "S&P 500 Sample (40 אקראיים)",
            "Tech Giants",
            "Value Plays",
            "High Growth",
        ])
    with col_s2:
        min_score = st.slider("Quant Score מינימלי:", 50, 90, 60)
    
    if st.button("🔍 הרץ סריקה מתקדמת"):
        with st.spinner("סורק שוק..."):
            
            if scan_universe == "Mega Watchlist (18 מניות)":
                scan_list = ['AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','INTC',
                             'BA','PEP','COST','AVGO','DIS','UBER','ABNB','SPOT']
            elif scan_universe == "S&P 500 Sample (40 אקראיים)":
                all_t = get_global_tickers()
                scan_list = random.sample(all_t, min(40, len(all_t)))
            elif scan_universe == "Tech Giants":
                scan_list = ['AAPL','MSFT','NVDA','META','GOOGL','AMZN','AMD','INTC','AVGO','QCOM','TSM','ASML']
            elif scan_universe == "Value Plays":
                scan_list = ['BRK-B','JPM','BAC','WFC','C','PEP','KO','JNJ','PG','MRK','ABBV','CVX','XOM']
            else:
                scan_list = ['NVDA','META','AMZN','TSLA','CRM','SNOW','DDOG','PLTR','RBLX','COIN','SQ','SHOP']
            
            scan_data = fetch_expert_data(scan_list)
            
            # Filter and sort
            results = [(t, d) for t, d in scan_data.items() if d.get('score', 0) >= min_score]
            results.sort(key=lambda x: x[1].get('score', 0), reverse=True)
            
            st.markdown(f"### נמצאו {len(results)} מניות מעל ציון {min_score}")
            
            # Category columns
            col_buy, col_value, col_growth, col_swing = st.columns(4)
            
            with col_buy:
                st.markdown("### 🟢 STRONG BUY")
                for t, d in results:
                    if d.get('signal') in ['STRONG BUY', 'BUY']:
                        with st.container():
                            st.success(f"""**{t}** | Score: {d['score']}/100
📊 RSI: {d['rsi']:.0f} | P/E: {d['pe']:.1f}
📈 Trend: {d['trend']}
💹 Upside: {d.get('target_upside', 0):.1f}%""")
            
            with col_value:
                st.markdown("### 💎 ערך")
                for t, d in results:
                    if 0 < d.get('pe', 99) < 20 and d.get('score', 0) >= 55:
                        st.info(f"""**{t}** | P/E: {d['pe']:.1f}
P/B: {d.get('pb', 0):.1f} | ROE: {d.get('roe', 0):.1f}%
Score: {d['score']}/100""")
            
            with col_growth:
                st.markdown("### 🚀 צמיחה")
                for t, d in results:
                    if d.get('growth_yoy', 0) > 15 and d.get('score', 0) >= 55:
                        st.warning(f"""**{t}** | Growth: {d['growth_yoy']:.1f}%
Margin: {d.get('gross_margin', 0):.1f}%
Score: {d['score']}/100""")
            
            with col_swing:
                st.markdown("### ⚡ סווינג")
                for t, d in results:
                    rsi = d.get('rsi', 50)
                    trend = d.get('trend', '')
                    macd_co = d.get('macd_crossover', False)
                    if rsi < 45 and 'Up' in trend:
                        st.error(f"""**{t}** | RSI: {rsi:.0f}
MACD Cross: {'✅' if macd_co else '❌'}
Score: {d['score']}/100""")
            
            # Full table
            st.markdown("### 📋 טבלת תוצאות מלאה")
            if results:
                table_data = []
                for t, d in results:
                    table_data.append({
                        'Ticker': t, 'Score': d['score'], 'Signal': d['signal'],
                        'Price': f"${d['price']:.2f}", 'RSI': round(d['rsi'], 1),
                        'Trend': d['trend'], 'P/E': round(d['pe'], 1),
                        'Growth %': round(d.get('growth_yoy', 0), 1),
                        'Upside %': round(d.get('target_upside', 0), 1),
                        'Sector': d['sector'], 'MACD ✅': '✅' if d.get('macd_crossover') else '',
                        'Beta': round(d.get('beta', 1), 2),
                        'Short %': round(d.get('short_pct', 0), 1),
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# ==========================================
# TAB 5: RISK ANALYSIS
# ==========================================
with t_risk:
    st.subheader("⚖️ ניתוח סיכונים מתקדם")
    
    if p_tickers:
        with st.spinner("מחשב מתאמים ומדדי סיכון..."):
            corr, sharpe, max_dd, var95 = compute_portfolio_analytics(p_tickers)
        
        # Risk metrics per stock
        st.markdown("### 📊 מדדי סיכון לכל מניה")
        risk_rows = []
        for t in p_tickers:
            d = m_data.get(t, {})
            risk_rows.append({
                'Ticker': t,
                'Beta': round(d.get('beta', 1), 2),
                'ATR %': round(d.get('atr_pct', 0), 2),
                'Sharpe': round(sharpe.get(t, 0), 2) if sharpe else 0,
                'Max DD %': round(max_dd.get(t, 0), 1) if max_dd else 0,
                'VaR 95%': round(var95.get(t, 0), 2) if var95 else 0,
                'Short %': round(d.get('short_pct', 0), 1),
                'Debt/Eq': round(d.get('debt_to_equity', 0), 2),
                'RSI': round(d.get('rsi', 50), 1),
            })
        
        risk_df = pd.DataFrame(risk_rows)
        
        def color_risk(val, col):
            if col == 'Beta':
                return 'color: #ff4466' if val > 1.5 else 'color: #ffaa00' if val > 1.2 else 'color: #00ff88'
            if col == 'Sharpe':
                return 'color: #00ff88' if val > 1 else 'color: #ffaa00' if val > 0 else 'color: #ff4466'
            return ''
        
        st.dataframe(risk_df, use_container_width=True)
        
        # Correlation matrix
        if corr is not None and len(p_tickers) > 1:
            st.markdown("### 🔗 מטריצת מתאמים (Correlation Matrix)")
            fig_corr = px.imshow(
                corr,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                text_auto='.2f',
                title="מתאם בין המניות בתיק (0 = גיוון מושלם, 1 = תנועה זהה)"
            )
            fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
            st.info("💡 מתאם נמוך בין מניות = פיזור סיכון טוב יותר. מתאם גבוה = הן זזות ביחד.")
        
        # Portfolio Beta
        if rows:
            total_v = sum(r['Value $'] for r in rows)
            port_beta = sum(m_data[r['Ticker']].get('beta', 1) * r['Value $'] / total_v for r in rows if r['Ticker'] in m_data)
            
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("Beta של התיק הכולל", f"{port_beta:.2f}",
                         "⚡ תנודתי" if port_beta > 1.2 else "🛡️ מאוזן")
            if sharpe:
                avg_sharpe = np.mean(list(sharpe.values()))
                col_b2.metric("Sharpe Ratio ממוצע", f"{avg_sharpe:.2f}",
                             "✅ טוב" if avg_sharpe > 1 else "⚠️ מתקבל" if avg_sharpe > 0 else "❌ חלש")
            if max_dd:
                worst_dd = min(max_dd.values())
                worst_ticker = min(max_dd, key=max_dd.get)
                col_b3.metric("ירידה מקסימלית גרועה ביותר", f"{worst_dd:.1f}% ({worst_ticker})")
    else:
        st.info("הוסף מניות לתיק כדי לראות ניתוח סיכונים.")

# ==========================================
# TAB 6: STRATEGIES
# ==========================================
with t_strat:
    st.subheader("🎯 אסטרטגיות מסחר מתקדמות")
    
    strat_tabs = st.tabs(["📅 טווחי זמן", "🔄 ממוצע עלות (DCA)", "📐 מחשבון פוזיציה", "🎓 מדריך אסטרטגיות"])
    
    with strat_tabs[0]:
        st.markdown("### ⏱️ יעדי מחיר לפי טווח זמן")
        for t in p_tickers:
            if t not in m_data:
                continue
            d = m_data[t]
            p = d['price']
            atr = d.get('atr_pct', 2) / 100
            growth = max(0, d.get('growth_yoy', 10)) / 100
            
            with st.expander(f"📊 {t} - ${p:.2f} | Signal: {d.get('signal', 'N/A')}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ימים (1-5)", f"${p*(1+atr*1.5):.2f}", f"+{atr*150:.1f}%")
                c2.metric("שבועות (2-4)", f"${p*(1+atr*3):.2f}", f"+{atr*300:.1f}%")
                c3.metric("חודשים (1-3)", f"${p*(1+growth*0.25):.2f}", f"+{growth*25:.1f}%")
                c4.metric("שנה", f"${p*(1+growth):.2f}", f"+{growth*100:.1f}%")
                
                # Support/Resistance from Fibonacci
                st.markdown(f"**רמות פיבונאצ'י (תמיכה/התנגדות):**")
                cols = st.columns(5)
                for i, (level, key) in enumerate([(0.236,'fib_236'), (0.382,'fib_382'), (0.5,'fib_500'), (0.618,'fib_618'), (0.786,'fib_786')]):
                    cols[i].metric(f"Fib {level}", f"${d.get(key, 0):.2f}")
    
    with strat_tabs[1]:
        st.markdown("### 🔄 מחשבון DCA - ממוצע עלות")
        dca_ticker = st.selectbox("מניה:", p_tickers, key='dca')
        dca_monthly = st.number_input("השקעה חודשית ($):", value=100.0)
        dca_months = st.slider("מספר חודשים:", 6, 60, 12)
        
        if dca_ticker in m_data:
            curr_p = m_data[dca_ticker]['price']
            growth_est = max(0, m_data[dca_ticker].get('growth_yoy', 8)) / 100
            
            total_invested = dca_monthly * dca_months
            shares_accum = 0
            monthly_price = curr_p
            rows_dca = []
            
            for month in range(1, dca_months + 1):
                monthly_price *= (1 + growth_est / 12)
                shares_accum += dca_monthly / monthly_price
                val = shares_accum * monthly_price
                rows_dca.append({'חודש': month, 'מחיר': round(monthly_price, 2),
                                  'מניות מצטברות': round(shares_accum, 3), 'שווי': round(val, 2)})
            
            dca_df = pd.DataFrame(rows_dca)
            final_val = dca_df['שווי'].iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("סה\"כ הושקע", f"${total_invested:,.2f}")
            c2.metric("שווי צפוי", f"${final_val:,.2f}")
            c3.metric("רווח צפוי", f"${final_val-total_invested:,.2f}", f"{(final_val/total_invested-1)*100:.1f}%")
            
            fig_dca = go.Figure()
            fig_dca.add_trace(go.Scatter(x=dca_df['חודש'], y=dca_df['שווי'], name='שווי', fill='tozeroy', line=dict(color='#00ff88')))
            fig_dca.add_trace(go.Scatter(x=dca_df['חודש'], y=[dca_monthly * m for m in dca_df['חודש']], name='הושקע', line=dict(color='#ffaa00', dash='dash')))
            fig_dca.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', title="DCA - שווי מצטבר לעומת השקעה")
            st.plotly_chart(fig_dca, use_container_width=True)
    
    with strat_tabs[2]:
        st.markdown("### 📐 מחשבון גודל פוזיציה (Position Sizing)")
        ps_account = st.number_input("גודל תיק ($):", value=float(total_usd) if 'total_usd' in dir() else 7000.0)
        ps_risk_pct = st.slider("סיכון מקסימלי לעסקה (%):", 0.5, 5.0, 1.0, 0.5)
        ps_ticker_sel = st.selectbox("מניה:", p_tickers + ([quick_ticker] if quick_ticker else []), key='ps_sel')
        ps_stop_pct = st.slider("Stop Loss (%):", 2.0, 20.0, 8.0, 0.5)
        
        if ps_ticker_sel in m_data:
            d = m_data[ps_ticker_sel]
            curr_p = d['price']
            risk_per_share = curr_p * (ps_stop_pct / 100)
            max_risk = ps_account * (ps_risk_pct / 100)
            shares_to_buy = int(max_risk / risk_per_share)
            position_size = shares_to_buy * curr_p
            stop_loss_price = curr_p * (1 - ps_stop_pct / 100)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("מניות לקנות", str(shares_to_buy))
            c2.metric("גודל פוזיציה", f"${position_size:,.2f}", f"{position_size/ps_account*100:.1f}% מהתיק")
            c3.metric("Stop Loss", f"${stop_loss_price:.2f}")
            c4.metric("סיכון מקסימלי", f"${max_risk:.2f}")
            
            st.info(f"""
**📊 ניתוח הפוזיציה:**
- קנה **{shares_to_buy}** מניות ב-${curr_p:.2f} = ${position_size:,.2f}
- שים Stop Loss ב-${stop_loss_price:.2f} (-{ps_stop_pct}%)
- אם ה-Stop יופעל, תפסיד ${max_risk:.2f} ({ps_risk_pct}% מהתיק) 
- יעד אנליסטים: ${d.get('target_price', curr_p):.2f} | Upside: {d.get('target_upside', 0):.1f}%
- **Risk/Reward Ratio: 1:{d.get('target_upside', 0)/ps_stop_pct:.1f}**
""")
    
    with strat_tabs[3]:
        st.markdown("""
### 🎓 מדריך אסטרטגיות מסחר

#### 1. 📈 Trend Following (מסחר לפי מגמה)
**מתי?** כשה-SMA20 > SMA50 > SMA200 והמחיר מעל SMA20
**כיצד?** קנה בפולבק לרמת SMA20 | Stop Loss: מתחת ל-SMA50
**Quant Score:** > 65 | Trend: Strong Uptrend

#### 2. ⚡ RSI Swing Trading
**מתי?** RSI יורד מתחת ל-35 במגמה עולה (תיקון זמני)
**כיצד?** קנה כש-RSI חוזר מעל 40 | Target: +5-8%
**Quant Score:** > 55 | RSI: 30-45

#### 3. 💎 Value Investing
**מתי?** P/E < 15, P/B < 2, ROE > 15%
**כיצד?** קנה ואחזק 1-3 שנים | Target: +30-50%
**Quant Score:** > 60 | Growth: > 10%

#### 4. 🚀 Growth Momentum
**מתי?** Revenue Growth > 20%, Earnings Growth > 15%
**כיצד?** קנה בברייקאאוט | Stop Loss: -12%
**Quant Score:** > 70 | Trend: Strong Uptrend

#### 5. 🔄 MACD Crossover
**מתי?** MACD חוצה את קו ה-Signal מלמטה למעלה
**כיצד?** קנה ביום הקרוסאובר | Stop: -7%
**Quant Score:** > 58 + MACD Crossover = True
        """)

# ==========================================
# TAB 7: JOURNAL
# ==========================================
with t_journal:
    st.subheader("📜 יומן פעולות")
    
    try:
        logs = safe_conn_read("Activity", ttl=0)
        if logs is not None and not logs.empty:
            logs_sorted = logs.sort_values("Date", ascending=False)
            st.dataframe(logs_sorted, width='stretch')
            
            # Stats
            buys = logs[logs['Action'] == 'Buy']
            sells = logs[logs['Action'] == 'Sell']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("סה\"כ פעולות", len(logs))
            c2.metric("קניות", len(buys))
            c3.metric("מכירות", len(sells))
            
            if len(logs) > 1:
                fig_act = px.bar(logs, x='Date', y='Quantity', color='Action',
                                color_map={'Buy': '#00ff88', 'Sell': '#ff4466'},
                                title="היסטוריית פעולות")
                fig_act.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_act, use_container_width=True)
        else:
            st.info("אין פעולות עדיין.")
    except:
        st.info("ריק.")
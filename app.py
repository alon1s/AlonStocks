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
# PAGE CONFIG - mobile-first
# ==========================================
st.set_page_config(
    page_title="AlonStocks Pro",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    /* ── Base ────────────────────────────────── */
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .main { background: #0a0e1a; }
    [data-testid="stSidebar"] { background: #0d1220; }

    /* CRITICAL: prevent iOS double-tap zoom on inputs */
    input, select, textarea, button { font-size: 16px !important; }

    /* Tighter main container padding on mobile */
    .main .block-container {
        padding: 0.6rem 0.75rem 4rem !important;
        max-width: 100% !important;
    }

    /* ── Metrics ─────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0f1729, #1a2444);
        border: 1px solid #2a3a6a;
        border-radius: 12px;
        padding: 10px 12px !important;
    }
    [data-testid="stMetric"] label {
        font-size: 0.72rem !important;
        color: #7a8ab0 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        font-family: 'Space Mono', monospace !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

    /* ── Buttons ─────────────────────────────── */
    .stButton > button {
        min-height: 48px !important;
        font-size: 1rem !important;
        border-radius: 10px !important;
        width: 100% !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        transition: transform 0.1s;
    }
    .stButton > button:active { transform: scale(0.97); }

    /* ── Tabs ────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        flex-wrap: nowrap !important;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 10px !important;
        font-size: 0.82rem !important;
        white-space: nowrap !important;
        min-width: 44px;
    }

    /* ── DataFrames ──────────────────────────── */
    .stDataFrame { font-family: 'Space Mono', monospace !important; font-size: 0.75em; }

    /* ── Expanders ───────────────────────────── */
    div[data-testid="stExpander"] {
        background: #0f1729; border: 1px solid #1e2d52;
        border-radius: 10px; margin-bottom: 8px;
    }

    /* ── Stock card (portfolio) ──────────────── */
    .stock-card {
        background: linear-gradient(135deg, #0f1729, #131c38);
        border: 1px solid #1e2d52;
        border-radius: 14px;
        padding: 14px 16px;
        margin: 6px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        cursor: pointer;
        -webkit-tap-highlight-color: transparent;
    }
    .stock-card:active { background: #1a2444; }
    .stock-ticker  { font-size: 1.1rem; font-weight: 800; font-family: 'Space Mono', monospace; color: #fff; }
    .stock-name    { font-size: 0.68rem; color: #7a8ab0; margin-top: 2px; }
    .stock-price   { font-size: 1rem; font-weight: 700; font-family: 'Space Mono', monospace; color: #fff; }
    .stock-pl      { font-size: 0.9rem; font-weight: 700; margin-top: 2px; }
    .stock-signal  { font-size: 0.78rem; font-weight: 700; text-align: right; }
    .stock-score   { font-size: 0.68rem; color: #7a8ab0; text-align: right; margin-top: 2px; }

    /* ── Market ticker strip ─────────────────── */
    .ticker-strip {
        background: linear-gradient(90deg, #0a0e1a, #0f1729, #0a0e1a);
        border: 1px solid #1e2d52; border-radius: 10px;
        padding: 10px 12px; margin-bottom: 10px;
        display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
    }
    .ticker-item { display: flex; flex-direction: column; align-items: center; min-width: 62px; }
    .ticker-name  { font-size: 0.6rem; color: #7a8ab0; font-family: 'Space Mono', monospace; }
    .ticker-price { font-size: 0.88rem; font-weight: 700; font-family: 'Space Mono', monospace; }
    .ticker-chg   { font-size: 0.7rem; font-family: 'Space Mono', monospace; }

    /* ── Sentiment bar ───────────────────────── */
    .sentiment-bar {
        height: 10px; border-radius: 5px;
        background: linear-gradient(90deg,#ff2244,#ff8844,#ffaa00,#44ff88,#00ff88);
        margin: 6px 0 2px; position: relative;
    }
    .sentiment-marker {
        position: absolute; top: -5px;
        width: 20px; height: 20px; border-radius: 50%;
        background: white; border: 3px solid #0a0e1a;
        transform: translateX(-50%);
    }

    /* ── Quick stat row ──────────────────────── */
    .quick-stat {
        background: #0f1729; border: 1px solid #1e2d52;
        border-radius: 10px; padding: 10px;
        text-align: center; margin: 4px 0;
    }
    .quick-stat-label { font-size: 0.65rem; color: #7a8ab0; text-transform: uppercase; }
    .quick-stat-value { font-size: 1rem; font-weight: 700; font-family: 'Space Mono', monospace; }

    /* ── Section header ──────────────────────── */
    .section-header {
        font-size: 0.7rem; font-weight: 700; color: #7a8ab0;
        text-transform: uppercase; letter-spacing: 0.1em;
        padding: 4px 0; margin: 12px 0 6px;
        border-bottom: 1px solid #1e2d52;
    }

    /* ── Mobile-specific overrides ───────────── */
    @media (max-width: 768px) {
        .main .block-container { padding: 0.4rem 0.5rem 5rem !important; }
        [data-testid="stMetricValue"] { font-size: 1rem !important; }
        .stTabs [data-baseweb="tab"] { padding: 7px 8px !important; font-size: 0.78rem !important; }
        /* Forms: full width, spaced */
        [data-testid="stForm"] { padding: 0 !important; }
        .stSelectbox, .stTextInput, .stNumberInput { margin-bottom: 4px !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# GOOGLE SHEETS + SESSION STATE
# ==========================================
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

conn = st.connection("gsheets", type=GSheetsConnection)

def load_cloud_portfolio():
    try:
        df = conn.read(worksheet="Portfolio", ttl=0)
        if df is None or df.empty:
            return pd.DataFrame(columns=['Ticker','Quantity','PurchasePrice']).set_index('Ticker')
        df = df.dropna(subset=['Ticker'])
        df = df[df['Ticker'].astype(str).str.strip() != ""]
        return df.set_index('Ticker')
    except:
        return pd.DataFrame(columns=['Ticker','Quantity','PurchasePrice']).set_index('Ticker')

def save_cloud_portfolio(df):
    try:
        conn.update(worksheet="Portfolio", data=df.reset_index())
        st.cache_data.clear()
        return True
    except:
        return False

def load_cloud_config():
    try:
        df = conn.read(worksheet="Config", ttl=0)
        if df is None or df.empty:
            return {"cash_usd": 86.67, "cash_ils": 0.0, "initial_investment": 7000.0}
        cfg = dict(zip(df['Key'].astype(str), df['Value'].astype(float)))
        return {
            "cash_usd": cfg.get("cash_usd", 86.67),
            "cash_ils": cfg.get("cash_ils", 0.0),
            "initial_investment": cfg.get("initial_investment", 7000.0),
        }
    except:
        if os.path.exists("config.json"):
            with open("config.json") as f:
                return json.load(f)
        return {"cash_usd": 86.67, "cash_ils": 0.0, "initial_investment": 7000.0}

def save_cloud_config(cfg):
    try:
        rows = [{"Key": k, "Value": v} for k, v in cfg.items()]
        conn.update(worksheet="Config", data=pd.DataFrame(rows))
        with open("config.json", "w") as f:
            json.dump(cfg, f)
    except:
        with open("config.json", "w") as f:
            json.dump(cfg, f)

def load_watchlist():
    try:
        df = conn.read(worksheet="Watchlist", ttl=0)
        if df is None or df.empty:
            return pd.DataFrame(columns=['Ticker','Notes','AlertHigh','AlertLow'])
        return df.dropna(subset=['Ticker'])
    except:
        return pd.DataFrame(columns=['Ticker','Notes','AlertHigh','AlertLow'])

def save_watchlist(df):
    try:
        conn.update(worksheet="Watchlist", data=df)
    except:
        pass

def log_activity(ticker, action, qty, price, notes=""):
    try:
        new_log = pd.DataFrame([{
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": ticker, "Action": action,
            "Quantity": float(qty), "Price": float(price), "Notes": notes
        }])
        try:
            existing = conn.read(worksheet="Activity", ttl=0)
        except:
            existing = pd.DataFrame()
        updated = pd.concat([existing, new_log], ignore_index=True) if not existing.empty else new_log
        conn.update(worksheet="Activity", data=updated)
    except:
        pass

# Session state init
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_cloud_portfolio()
if 'config' not in st.session_state:
    st.session_state.config = load_cloud_config()
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# ==========================================
# TICKER UNIVERSES
# ==========================================
NASDAQ_100 = [
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AVGO','COST','NFLX',
    'AMD','CSCO','QCOM','INTC','INTU','AMGN','CMCSA','AMAT','TXN','BKNG',
    'VRTX','ISRG','ADP','SBUX','ADI','GILD','REGN','MU','LRCX','KLAC',
    'PANW','SNPS','CDNS','MELI','CRWD','FTNT','PCAR','ORLY','CTAS','NXPI',
    'ADSK','MRVL','PAYX','ROST','WDAY','ODFL','EA','ABNB','CHTR','ZS',
    'FAST','DDOG','IDXX','ILMN','LULU','MCHP','MDLZ','TTD','VRSK','TEAM',
    'ANSS','CEG','CPRT','DASH','EBAY','SMCI','ON','ENPH','KDP','BKR'
]

TA35_TICKERS = [
    'TEVA.TA','ICL.TA','NICE.TA','CHECK.TA','ESLT.TA','BCOM.TA',
    'LUMI.TA','HARL.TA','PAZA.TA','DSCT.TA','PHOE.TA','FIBR.TA',
    'AZRG.TA','ELAL.TA','MZTF.TA','SPNS.TA','MNRT.TA','RTEN.TA'
]

DIVIDEND_STOCKS = [
    'JNJ','PG','KO','PEP','MRK','ABBV','CVX','XOM','T','VZ',
    'MMM','IBM','MO','PM','O','MAIN','REALTY','WPC','IIPR','AGNC',
    'EPD','MMP','ET','MPLX','PAA','WMB','OKE','KMI','ENB','TRP'
]

ETF_UNIVERSE = [
    'SPY','QQQ','DIA','IWM','VTI','VOO','ARKK','XLK','XLF','XLE',
    'XLV','XLI','XLY','XLP','XLU','XLRE','GLD','SLV','USO','TLT',
    'HYG','LQD','EEM','EFA','VNQ','SCHD','NOBL','DGRO','VIG','DVY'
]

MARKET_TICKERS = {
    'SPY': 'S&P 500', 'QQQ': 'NASDAQ', 'DIA': 'Dow Jones',
    'IWM': 'Russell 2K', '^VIX': 'VIX', 'BTC-USD': 'Bitcoin',
    'GLD': 'Gold', 'USO': 'Oil ETF',
}

# ==========================================
# DATA ENGINE
# ==========================================
@st.cache_data(ttl=600)
def get_usd_ils():
    try:
        return yf.Ticker("USDILS=X").fast_info['lastPrice']
    except:
        return 3.75

@st.cache_data(ttl=300)
def get_market_overview():
    results = {}
    tickers = list(MARKET_TICKERS.keys())
    try:
        raw = yf.download(tickers, period='5d', progress=False, auto_adjust=True)
        closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        for t in tickers:
            try:
                series = closes[t].dropna() if t in closes.columns else None
                if series is not None and len(series) >= 2:
                    today = series.iloc[-1]
                    prev  = series.iloc[-2]
                    chg   = (today - prev) / prev * 100
                    results[t] = {
                        'name': MARKET_TICKERS[t],
                        'price': float(today),
                        'change_pct': float(chg),
                        'color': '#00ff88' if chg >= 0 else '#ff4466',
                        'arrow': '▲' if chg >= 0 else '▼',
                    }
            except:
                pass
    except:
        pass
    return results

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        resp = requests.get(url, headers=session.headers, timeout=10)
        table = pd.read_html(StringIO(resp.text))[0]
        return [t.replace('.', '-') for t in table['Symbol'].tolist()]
    except:
        return ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','V','UNH']

def compute_advanced_indicators(hist):
    close  = hist['Close']
    high   = hist['High']
    low    = hist['Low']
    volume = hist['Volume']
    ind    = {}

    for p in [5, 10, 20, 50, 100, 200]:
        if len(close) >= p:
            ind[f'sma{p}'] = close.rolling(p).mean().iloc[-1]
            ind[f'ema{p}'] = close.ewm(span=p, adjust=False).mean().iloc[-1]

    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    ind['rsi'] = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    ind['macd']          = macd_line.iloc[-1]
    ind['macd_signal']   = signal_line.iloc[-1]
    ind['macd_hist']     = (macd_line - signal_line).iloc[-1]
    ind['macd_crossover'] = bool(
        (macd_line.iloc[-1] > signal_line.iloc[-1]) and
        (macd_line.iloc[-2] <= signal_line.iloc[-2])
    )

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = (sma20 + 2*std20).iloc[-1]
    bb_lower = (sma20 - 2*std20).iloc[-1]
    bb_mid   = sma20.iloc[-1]
    ind['bb_upper'] = bb_upper
    ind['bb_lower'] = bb_lower
    ind['bb_pct']   = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    ind['bb_width'] = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0

    if len(close) >= 14:
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        k = 100 * (close - low14) / (high14 - low14 + 1e-9)
        d = k.rolling(3).mean()
        ind['stoch_k'] = k.iloc[-1]
        ind['stoch_d'] = d.iloc[-1]
    else:
        ind['stoch_k'] = ind['stoch_d'] = 50

    if len(hist) >= 14:
        tr = pd.concat([high - low,
                        (high - close.shift()).abs(),
                        (low  - close.shift()).abs()], axis=1).max(axis=1)
        ind['atr']     = tr.rolling(14).mean().iloc[-1]
        ind['atr_pct'] = ind['atr'] / close.iloc[-1] * 100
    else:
        ind['atr'] = ind['atr_pct'] = 0

    if len(volume) >= 20:
        avg_vol = volume.rolling(20).mean()
        ind['vol_ratio'] = float(volume.iloc[-1] / avg_vol.iloc[-1]) if avg_vol.iloc[-1] > 0 else 1.0
        ind['vol_trend'] = float(volume.rolling(5).mean().iloc[-1] / avg_vol.iloc[-1]) if avg_vol.iloc[-1] > 0 else 1.0
    else:
        ind['vol_ratio'] = ind['vol_trend'] = 1.0

    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    obv_sma = obv.rolling(20).mean()
    ind['obv_trend'] = 1 if len(obv) >= 20 and obv.iloc[-1] > obv_sma.iloc[-1] else -1

    high_1y = close.max(); low_1y = close.min(); diff = high_1y - low_1y
    ind['fib_236'] = high_1y - diff * 0.236
    ind['fib_382'] = high_1y - diff * 0.382
    ind['fib_500'] = high_1y - diff * 0.500
    ind['fib_618'] = high_1y - diff * 0.618
    ind['fib_786'] = high_1y - diff * 0.786
    ind['pct_from_high'] = ((close.iloc[-1] - high_1y) / high_1y) * 100
    ind['pct_from_low']  = ((close.iloc[-1] - low_1y)  / low_1y)  * 100

    for n, days in [('mom_1m',20),('mom_3m',63),('mom_6m',126),('mom_1y',252)]:
        ind[n] = ((close.iloc[-1] / close.iloc[-days]) - 1)*100 if len(close) >= days else 0

    sma20v = close.rolling(20).mean().iloc[-1] if len(close)>=20 else close.iloc[-1]
    sma50v = close.rolling(50).mean().iloc[-1] if len(close)>=50 else sma20v
    sma200v= close.rolling(200).mean().iloc[-1] if len(close)>=200 else sma50v
    curr = close.iloc[-1]
    ind['sma20'] = sma20v; ind['sma50'] = sma50v; ind['sma200'] = sma200v

    if sma20v > sma50v > sma200v and curr > sma20v:
        ind['trend'] = "Strong Uptrend";  ind['trend_score'] = 5
    elif sma20v > sma50v and curr > sma20v:
        ind['trend'] = "Moderate Uptrend";ind['trend_score'] = 4
    elif sma20v < sma50v < sma200v and curr < sma20v:
        ind['trend'] = "Strong Downtrend";ind['trend_score'] = 1
    elif sma20v < sma50v and curr < sma20v:
        ind['trend'] = "Moderate Downtrend";ind['trend_score'] = 2
    else:
        ind['trend'] = "Consolidation"; ind['trend_score'] = 3

    # Support / Resistance (recent pivot highs/lows)
    if len(close) >= 20:
        recent = close.iloc[-60:] if len(close) >= 60 else close
        ind['support']    = float(recent.rolling(10).min().dropna().iloc[-1])
        ind['resistance'] = float(recent.rolling(10).max().dropna().iloc[-1])
    else:
        ind['support']    = float(low.iloc[-1])
        ind['resistance'] = float(high.iloc[-1])

    return ind

def compute_composite_score(ind, info):
    score = 50; reasons = []
    ts = ind.get('trend_score', 3)
    score += (ts - 3) * 4
    if ts >= 5: reasons.append("✅ Strong uptrend (SMA stack bullish)")
    elif ts <= 2: reasons.append("❌ Downtrend detected")

    rsi = ind.get('rsi', 50)
    if   rsi < 30:        score += 10; reasons.append("✅ Oversold RSI < 30")
    elif rsi < 40:        score +=  6; reasons.append("✅ Oversold zone RSI < 40")
    elif 40 <= rsi <= 60: score +=  4; reasons.append("✅ RSI neutral zone")
    elif rsi > 80:        score -= 12; reasons.append("🚨 Extremely overbought RSI > 80")
    elif rsi > 70:        score -=  8; reasons.append("⚠️ Overbought RSI > 70")

    if ind.get('macd_crossover'): score += 8; reasons.append("✅ MACD bullish crossover !")
    elif ind.get('macd', 0) > ind.get('macd_signal', 0): score += 3; reasons.append("✅ MACD positive")
    else: score -= 4; reasons.append("⚠️ MACD below signal")

    vr = ind.get('vol_ratio', 1)
    if   vr > 2.0 and ts >= 4: score +=  8; reasons.append("✅ Surge volume in uptrend")
    elif vr > 1.5 and ts >= 4: score +=  5; reasons.append("✅ High volume uptrend")
    elif vr > 1.5 and ts <= 2: score -=  6; reasons.append("⚠️ High volume downtrend")

    m1 = ind.get('mom_1m', 0); m3 = ind.get('mom_3m', 0)
    if   m1 > 10 and m3 > 20: score += 10; reasons.append("✅ Very strong momentum")
    elif m1 >  5 and m3 > 10: score +=  6; reasons.append("✅ Strong momentum")
    elif m1 >  0 and m3 >  0: score +=  3
    elif m1 < -15:             score -= 10; reasons.append("⚠️ Very negative momentum")
    elif m1 < -8:              score -=  6; reasons.append("⚠️ Negative momentum")

    pe = info.get('trailingPE', 0) or 0
    if   0 < pe < 12: score += 10; reasons.append("✅ Very low P/E (deep value)")
    elif 0 < pe < 20: score +=  6; reasons.append("✅ Low P/E (value)")
    elif 20 <= pe < 35: score += 2
    elif pe > 60:     score -=  8; reasons.append("⚠️ Very high P/E > 60")
    elif pe > 40:     score -=  4; reasons.append("⚠️ High P/E")

    growth = (info.get('revenueGrowth', 0) or 0) * 100
    if   growth > 30: score += 8; reasons.append("✅ Revenue growth > 30%")
    elif growth > 15: score += 5; reasons.append("✅ Good revenue growth")
    elif growth >  5: score += 2
    elif growth <  0: score -= 6; reasons.append("⚠️ Revenue decline")

    roe = (info.get('returnOnEquity', 0) or 0) * 100
    if   roe > 25: score += 5; reasons.append("✅ Excellent ROE > 25%")
    elif roe > 15: score += 3
    elif roe <  0: score -= 4; reasons.append("⚠️ Negative ROE")

    rec = info.get('recommendationKey', 'none')
    if rec in ['strong_buy','buy']:     score += 5; reasons.append("✅ Analyst consensus: Buy")
    elif rec in ['sell','strong_sell']: score -= 6; reasons.append("⚠️ Analyst consensus: Sell")

    # OBV confirmation
    if ind.get('obv_trend', 0) == 1 and ts >= 4:
        score += 3; reasons.append("✅ OBV confirms uptrend")

    # Stoch oversold
    sk = ind.get('stoch_k', 50)
    if sk < 20: score += 4; reasons.append("✅ Stochastic oversold")
    elif sk > 85: score -= 4; reasons.append("⚠️ Stochastic overbought")

    return min(100, max(0, int(score))), reasons

def get_signal(score, rsi, macd, macd_signal, trend_score):
    if score >= 75 and rsi < 65 and trend_score >= 5:
        return "STRONG BUY", "#00ff88"
    elif score >= 65 and trend_score >= 4:
        return "BUY", "#44ff99"
    elif score >= 55 and trend_score >= 3:
        return "WATCH", "#44aaff"
    elif score <= 25 and trend_score <= 1:
        return "STRONG SELL", "#ff2244"
    elif score <= 38:
        return "SELL", "#ff6688"
    elif 43 <= score <= 57:
        return "HOLD", "#ffaa00"
    elif score > 57:
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
            hist  = stock.history(period="2y")
            if len(hist) < 50:
                return None
            info  = stock.info
            curr  = hist['Close'].iloc[-1]
            ind   = compute_advanced_indicators(hist)
            score, reasons = compute_composite_score(ind, info)
            signal, signal_color = get_signal(
                score, ind['rsi'], ind['macd'], ind['macd_signal'], ind['trend_score']
            )
            try:
                earnings = stock.earnings_history
                recent_surprise = float(earnings['surprisePercent'].iloc[-1]) if earnings is not None and not earnings.empty else 0
            except:
                recent_surprise = 0
            short_pct = (info.get('shortPercentOfFloat') or 0) * 100
            inst_own  = (info.get('institutionsPercentHeld') or 0) * 100
            div_yield = (info.get('dividendYield') or 0) * 100

            return t, {
                'price': float(curr),
                'sector': info.get('sector','Unknown'),
                'industry': info.get('industry','Unknown'),
                'name': info.get('longName', t),
                'pe': info.get('trailingPE', 0) or 0,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'peg': info.get('pegRatio', 0) or 0,
                'ps': info.get('priceToSalesTrailing12Months', 0) or 0,
                'pb': info.get('priceToBook', 0) or 0,
                'beta': info.get('beta', 1.0) or 1.0,
                'div': div_yield,
                'market_cap': info.get('marketCap', 0) or 0,
                'revenue': info.get('totalRevenue', 0) or 0,
                'gross_margin': (info.get('grossMargins', 0) or 0) * 100,
                'profit_margin': (info.get('profitMargins', 0) or 0) * 100,
                'roe': (info.get('returnOnEquity', 0) or 0) * 100,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'current_ratio': info.get('currentRatio', 0) or 0,
                'growth_yoy': (info.get('revenueGrowth', 0) or 0) * 100,
                'earnings_growth': (info.get('earningsGrowth', 0) or 0) * 100,
                'analyst': info.get('recommendationKey','none'),
                'target_price': info.get('targetMeanPrice', curr) or curr,
                'target_upside': ((info.get('targetMeanPrice', curr) or curr) - curr) / curr * 100,
                'target_low': info.get('targetLowPrice', curr) or curr,
                'target_high': info.get('targetHighPrice', curr) or curr,
                'num_analysts': info.get('numberOfAnalystOpinions', 0) or 0,
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
        except:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(fetch_single, valid_list):
            if r:
                data[r[0]] = r[1]
    return data

@st.cache_data(ttl=1800)
def ml_forecast(ticker, days=14):
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period="5y")
        if len(hist) < 252:
            return None, None
        df = hist.copy()
        df['ret_1']     = df['Close'].pct_change(1)
        df['ret_5']     = df['Close'].pct_change(5)
        df['ret_20']    = df['Close'].pct_change(20)
        df['vol_20']    = df['Close'].pct_change().rolling(20).std()
        df['sma20']     = df['Close'].rolling(20).mean()
        df['sma50']     = df['Close'].rolling(50).mean()
        df['ratio_20_50'] = df['sma20'] / df['sma50']
        delta = df['Close'].diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
        df['rsi']       = 100 - (100 / (1 + (gain / loss)))
        df['vol_ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)
        for lag in [1,2,3,5,10]:
            df[f'lag_{lag}'] = df['Close'].shift(lag)
        df['target'] = df['Close'].shift(-days)
        df = df.dropna()
        feat = ['ret_1','ret_5','ret_20','vol_20','ratio_20_50','rsi','vol_ratio',
                'lag_1','lag_2','lag_3','lag_5','lag_10']
        X = df[feat]; y = df['target']
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        gb = GradientBoostingRegressor(n_estimators=150, random_state=42)
        gb.fit(X_tr_s, y_tr)
        last_s   = sc.transform(X.iloc[-1:])
        pred_rf  = rf.predict(last_s)[0]
        pred_gb  = gb.predict(last_s)[0]
        ensemble = pred_rf * 0.6 + pred_gb * 0.4
        curr     = df['Close'].iloc[-1]
        # Confidence: std of test predictions vs actual
        test_preds = rf.predict(X_te_s)
        mae = np.mean(np.abs(test_preds - y_te.values))
        confidence = max(0, min(100, 100 - (mae / curr * 100 * 5)))
        return ensemble, {
            'current': float(curr), 'predicted': float(ensemble),
            'pct_change': float((ensemble - curr) / curr * 100),
            'rf_pred': float(pred_rf), 'gb_pred': float(pred_gb),
            'importance': dict(zip(feat, rf.feature_importances_)),
            'days': days, 'confidence': float(confidence),
            'mae': float(mae),
        }
    except:
        return None, None

@st.cache_data(ttl=1800)
def compute_portfolio_analytics(tickers):
    try:
        raw  = yf.download(list(tickers) + ['SPY'], period="1y", auto_adjust=True, progress=False)
        closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        returns = closes.pct_change().dropna()
        corr    = returns[[c for c in returns.columns if c in tickers]].corr() if len(tickers) > 1 else None
        sharpe, max_dd, var95, spy_corr = {}, {}, {}, {}
        for col in tickers:
            if col not in returns.columns:
                continue
            r = returns[col]
            if r.std() > 0:
                sharpe[col] = float((r.mean()*252 - 0.05) / (r.std()*np.sqrt(252)))
            roll_max = closes[col].cummax()
            max_dd[col]  = float(((closes[col] - roll_max) / roll_max).min() * 100)
            var95[col]   = float(np.percentile(r.dropna(), 5) * 100)
            if 'SPY' in returns.columns:
                spy_corr[col] = float(r.corr(returns['SPY']))
        spy_ret_1y = float((closes['SPY'].iloc[-1] / closes['SPY'].iloc[0] - 1) * 100) if 'SPY' in closes.columns else 0
        return corr, sharpe, max_dd, var95, spy_corr, spy_ret_1y
    except:
        return None, {}, {}, {}, {}, 0

@st.cache_data(ttl=1800)
def get_stock_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news
        return news[:6] if news else []
    except:
        return []

@st.cache_data(ttl=3600)
def get_earnings_calendar(ticker):
    try:
        stock = yf.Ticker(ticker)
        cal   = stock.calendar
        hist  = stock.earnings_history
        return cal, hist
    except:
        return None, None

@st.cache_data(ttl=300)
def get_benchmark_comparison(tickers):
    """Returns 1Y performance of each ticker vs SPY"""
    try:
        all_t = list(tickers) + ['SPY', 'QQQ']
        raw   = yf.download(all_t, period='1y', auto_adjust=True, progress=False)
        closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        results = {}
        for t in all_t:
            if t in closes.columns:
                series = closes[t].dropna()
                if len(series) > 1:
                    results[t] = float((series.iloc[-1]/series.iloc[0] - 1)*100)
        return results, closes
    except:
        return {}, pd.DataFrame()

def compute_market_sentiment(market_data):
    vix = market_data.get('^VIX', {}).get('price', 20)
    spy_chg = market_data.get('SPY', {}).get('change_pct', 0)
    qqq_chg = market_data.get('QQQ', {}).get('change_pct', 0)
    btc_chg = market_data.get('BTC-USD', {}).get('change_pct', 0)

    vix_score = 100 if vix < 12 else 80 if vix < 16 else 60 if vix < 20 else 40 if vix < 25 else 20 if vix < 30 else 5
    trend_score = 50 + (spy_chg + qqq_chg) * 3
    trend_score = max(0, min(100, trend_score))
    btc_score   = 70 if btc_chg > 3 else 55 if btc_chg > 0 else 40 if btc_chg > -3 else 25

    score = int(vix_score * 0.5 + trend_score * 0.3 + btc_score * 0.2)
    score = max(0, min(100, score))

    if score >= 80: label, color = "Extreme Greed 🤑", "#00ff88"
    elif score >= 60: label, color = "Greed 😊", "#88ff44"
    elif score >= 40: label, color = "Neutral 😐", "#ffaa00"
    elif score >= 20: label, color = "Fear 😰", "#ff8844"
    else: label, color = "Extreme Fear 😱", "#ff2244"
    return score, label, color

# ==========================================
# SIDEBAR
# ==========================================
usd_ils_rate = get_usd_ils()
market_data  = get_market_overview()

with st.sidebar:
    st.markdown("### 📊 Market Pulse")
    for t, d in list(market_data.items())[:4]:
        sign = "+" if d['change_pct'] >= 0 else ""
        clr  = d['color']
        st.markdown(
            f"**{d['name']}** — ${d['price']:,.2f} "
            f"<span style='color:{clr}'>{sign}{d['change_pct']:.2f}%</span>",
            unsafe_allow_html=True
        )
    st.markdown("---")

    st.subheader("💰 קופת מזומנים")
    n_usd = st.number_input("מזומן בדולר $", value=float(st.session_state.config['cash_usd']))
    n_ils = st.number_input("מזומן בשקל ₪",  value=float(st.session_state.config['cash_ils']))
    init_inv = st.number_input("השקעה ראשונית ($)", value=float(st.session_state.config.get('initial_investment', 7000)))
    if (n_usd != st.session_state.config['cash_usd'] or
        n_ils != st.session_state.config['cash_ils'] or
        init_inv != st.session_state.config.get('initial_investment', 7000)):
        st.session_state.config = {"cash_usd": n_usd, "cash_ils": n_ils, "initial_investment": init_inv}
        save_cloud_config(st.session_state.config)

    st.markdown("---")
    st.subheader("🛒 ביצוע טרייד")
    with st.form("trade_form", clear_on_submit=True):
        t_in    = st.text_input("סימול מניה").upper()
        act_in  = st.selectbox("פעולה", ["Buy","Sell"])
        q_in    = st.number_input("כמות", min_value=0.0, step=0.1)
        p_in    = st.number_input("מחיר", min_value=0.0, step=0.01)
        notes_in= st.text_area("הערות")
        if st.form_submit_button("עדכן לענן 🚀"):
            if t_in and q_in > 0:
                if act_in == "Buy":
                    if t_in in st.session_state.portfolio.index:
                        oq, op = st.session_state.portfolio.loc[t_in, ['Quantity','PurchasePrice']]
                        nq = oq + q_in
                        st.session_state.portfolio.loc[t_in] = [nq, ((oq*op)+(q_in*p_in))/nq]
                    else:
                        st.session_state.portfolio.loc[t_in] = [q_in, p_in]
                else:
                    if t_in in st.session_state.portfolio.index:
                        nq = max(0, st.session_state.portfolio.loc[t_in,'Quantity'] - q_in)
                        if nq == 0:
                            st.session_state.portfolio = st.session_state.portfolio.drop(t_in)
                        else:
                            st.session_state.portfolio.loc[t_in,'Quantity'] = nq
                if save_cloud_portfolio(st.session_state.portfolio):
                    log_activity(t_in, act_in, q_in, p_in, notes_in)
                    st.rerun()

    st.markdown("---")
    st.subheader("🔎 ניתוח מהיר")
    quick_ticker = st.text_input("הכנס סימול").upper()

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Updated: {datetime.now().strftime('%H:%M')}")

# ==========================================
# MAIN DATA
# ==========================================
p_tickers = [t for t in st.session_state.portfolio.index if isinstance(t, str) and t.strip()]
all_analyze = list(set(p_tickers + ['SPY','QQQ','^VIX']))
if quick_ticker:
    all_analyze.append(quick_ticker)
m_data = fetch_expert_data(tuple(all_analyze))

# ==========================================
# TABS
# ==========================================
tabs = st.tabs([
    "🌍 שוק", "💼 תיק", "📊 טכני", "🤖 AI",
    "🔥 סורק", "⚖️ סיכון", "🎯 אסטרטגיות",
    "📰 חדשות", "📜 יומן"
])
t_mkt, t_port, t_tech, t_ai, t_scan, t_risk, t_strat, t_news, t_journal = tabs

# ==========================================
# TAB 1: MARKET OVERVIEW
# ==========================================
with t_mkt:
    # ── Ticker strip ──────────────────────────────────────────────
    strip_html = '<div class="ticker-strip">'
    for t, d in market_data.items():
        sign = "+" if d['change_pct'] >= 0 else ""
        strip_html += (
            f'<div class="ticker-item">'
            f'<span class="ticker-name">{d["name"]}</span>'
            f'<span class="ticker-price" style="color:{d["color"]}">${d["price"]:,.0f}</span>'
            f'<span class="ticker-chg" style="color:{d["color"]}">{sign}{d["change_pct"]:.1f}%</span>'
            f'</div>'
        )
    strip_html += "</div>"
    st.markdown(strip_html, unsafe_allow_html=True)

    # ── Sentiment ─────────────────────────────────────────────────
    sent_score, sent_label, sent_color = compute_market_sentiment(market_data)
    pct = sent_score / 100
    st.markdown(
        f'<div style="margin:4px 0 2px;font-size:0.82rem;font-weight:700;">'
        f'שוק: <span style="color:{sent_color}">{sent_label}</span> ({sent_score}/100)'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="sentiment-bar">'
        f'<div class="sentiment-marker" style="left:{pct*100:.0f}%"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.6rem;color:#7a8ab0;margin-bottom:10px">'
        f'<span>Fear</span><span>Neutral</span><span>Greed</span></div>',
        unsafe_allow_html=True
    )

    # ── Indices grid — 2 columns ──────────────────────────────────
    st.markdown('<div class="section-header">📈 מדדים עולמיים</div>', unsafe_allow_html=True)
    mkt_items = list(market_data.items())
    for i in range(0, len(mkt_items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(mkt_items):
                tick, d = mkt_items[i + j]
                delta_str = f"{'+' if d['change_pct']>=0 else ''}{d['change_pct']:.2f}%"
                col.metric(f"{d['arrow']} {d['name']}", f"${d['price']:,.2f}", delta_str)

    st.markdown("---")

    # ── Benchmark bar chart ───────────────────────────────────────
    bench_returns, bench_closes = get_benchmark_comparison(tuple(p_tickers[:8]))
    if bench_returns:
        st.markdown('<div class="section-header">📊 ביצועים 1Y מול SPY</div>', unsafe_allow_html=True)
        tickers_for_chart = [t for t in list(p_tickers[:8]) + ['SPY','QQQ'] if t in bench_returns]
        perf_vals  = [bench_returns[t] for t in tickers_for_chart]
        fig_bench  = go.Figure(go.Bar(
            x=tickers_for_chart, y=perf_vals,
            marker_color=['#00ff88' if v >= 0 else '#ff4466' for v in perf_vals],
            text=[f"{v:.1f}%" for v in perf_vals], textposition='auto'
        ))
        fig_bench.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=280, margin=dict(l=10,r=10,t=10,b=30)
        )
        st.plotly_chart(fig_bench, use_container_width=True,
                        config={"displayModeBar": False})

    # ── Normalized line chart ─────────────────────────────────────
    if p_tickers and not bench_closes.empty:
        st.markdown('<div class="section-header">📈 תיק vs SPY — מנורמל</div>', unsafe_allow_html=True)
        fig_vs = go.Figure()
        for t in p_tickers[:5]:
            if t in bench_closes.columns:
                s = bench_closes[t].dropna()
                if len(s) > 1:
                    n = s / s.iloc[0] * 100
                    fig_vs.add_trace(go.Scatter(x=n.index, y=n.values, name=t, mode='lines', line=dict(width=1.5)))
        if 'SPY' in bench_closes.columns:
            sp = bench_closes['SPY'].dropna()
            sn = sp / sp.iloc[0] * 100
            fig_vs.add_trace(go.Scatter(x=sn.index, y=sn.values, name='SPY',
                                        line=dict(dash='dash', color='#ffaa00', width=2)))
        fig_vs.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=300, margin=dict(l=10,r=10,t=10,b=30),
            legend=dict(orientation='h', y=-0.15, font=dict(size=10)),
            hovermode='x unified'
        )
        st.plotly_chart(fig_vs, use_container_width=True, config={"displayModeBar": False})

    # ── Top movers list ───────────────────────────────────────────
    if p_tickers:
        st.markdown('<div class="section-header">🏆 מניות מהתיק</div>', unsafe_allow_html=True)
        movers = sorted(
            [(t, m_data[t]) for t in p_tickers if t in m_data],
            key=lambda x: x[1].get('mom_1m', 0), reverse=True
        )
        for t, d in movers:
            m1  = d.get('mom_1m', 0)
            clr = '#00ff88' if m1 >= 0 else '#ff4466'
            sig = d.get('signal','')
            sig_clr = {'STRONG BUY':'#00ff88','BUY':'#44dd88','WATCH':'#44aaff',
                       'HOLD':'#ffaa00','SELL':'#ff6688','STRONG SELL':'#ff2244'}.get(sig,'#ffaa00')
            st.markdown(
                f'<div class="stock-card">'
                f'<div><div class="stock-ticker">{t}</div>'
                f'<div class="stock-name">{d.get("sector","")[:18]}</div></div>'
                f'<div style="text-align:center">'
                f'<div class="stock-price">${d["price"]:,.2f}</div>'
                f'<div class="stock-pl" style="color:{clr}">{"+"+str(round(m1,1)) if m1>=0 else round(m1,1)}% 1M</div></div>'
                f'<div><div class="stock-signal" style="color:{sig_clr}">{sig}</div>'
                f'<div class="stock-score">Score {d.get("score",50)}</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

# ==========================================
# TAB 2: PORTFOLIO
# ==========================================
with t_port:
    # ── Build portfolio data ───────────────────────────────────────
    stock_val_usd = 0; rows = []; sector_weights = {}; cost_basis = 0

    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d   = m_data[t]
            qty = row['Quantity']; bp = row['PurchasePrice']
            v_u = d['price'] * qty / (usd_ils_rate if d['currency']=="ILS" else 1)
            stock_val_usd += v_u
            cost_basis    += bp * qty / (usd_ils_rate if d['currency']=="ILS" else 1)
            sector_weights[d['sector']] = sector_weights.get(d['sector'],0) + v_u
            pl_pct = ((d['price'] - bp) / bp) * 100
            pl_usd = (d['price'] - bp) * qty
            div_annual = v_u * d.get('div',0) / 100
            rows.append({
                "Ticker": t, "Name": d.get('name',t)[:20],
                "Sector": d['sector'], "Qty": qty,
                "Price": d['price'], "Buy@": bp,
                "P&L %": round(pl_pct,2), "P&L $": round(pl_usd,2),
                "Value $": round(v_u,2),
                "Annual Div $": round(div_annual,2),
                "Score": d.get('score',50),
                "Signal": d.get('signal','HOLD'),
                "RSI": round(d.get('rsi',50),1),
                "Trend": d.get('trend','-'),
                "Target $": round(d.get('target_price',d['price']),2),
                "Upside %": round(d.get('target_upside',0),1),
                "Beta": round(d.get('beta',1),2),
                "Analyst": d.get('analyst','none'),
            })

    total_usd        = stock_val_usd + n_usd + (n_ils / usd_ils_rate)
    total_ils        = total_usd * usd_ils_rate
    init_inv         = st.session_state.config.get('initial_investment', 7000)
    profit_usd       = total_usd - init_inv
    total_div        = sum(r['Annual Div $'] for r in rows)
    total_return_pct = (profit_usd / init_inv * 100) if init_inv > 0 else 0

    # ── Summary metrics — 2×2 grid ────────────────────────────────
    r1c1, r1c2 = st.columns(2)
    r1c1.metric("💵 שווי כולל", f"${total_usd:,.0f}",
                f"{'+'if profit_usd>=0 else ''}{profit_usd:,.0f}")
    r1c2.metric("📈 תשואה", f"{total_return_pct:.1f}%",
                f"{'+'if profit_usd>=0 else ''}${profit_usd:,.0f}")
    r2c1, r2c2 = st.columns(2)
    r2c1.metric("💱 USD/ILS", f"₪{usd_ils_rate:.3f}")
    r2c2.metric("💸 דיב. שנתי", f"${total_div:.0f}")

    # ── Quick Trade — exposed in main area for mobile ─────────────
    with st.expander("⚡ ביצוע טרייד מהיר", expanded=False):
        with st.form("quick_trade_main", clear_on_submit=True):
            qc1, qc2 = st.columns(2)
            qt_ticker = qc1.text_input("סימול").upper()
            qt_action = qc2.selectbox("פעולה", ["Buy","Sell"])
            qc3, qc4 = st.columns(2)
            qt_qty    = qc3.number_input("כמות", min_value=0.0, step=0.1)
            qt_price  = qc4.number_input("מחיר $", min_value=0.0, step=0.01)
            if st.form_submit_button("✅ שלח עסקה", use_container_width=True):
                if qt_ticker and qt_qty > 0:
                    if qt_action == "Buy":
                        if qt_ticker in st.session_state.portfolio.index:
                            oq, op = st.session_state.portfolio.loc[qt_ticker, ['Quantity','PurchasePrice']]
                            nq = oq + qt_qty
                            st.session_state.portfolio.loc[qt_ticker] = [nq, ((oq*op)+(qt_qty*qt_price))/nq]
                        else:
                            st.session_state.portfolio.loc[qt_ticker] = [qt_qty, qt_price]
                    else:
                        if qt_ticker in st.session_state.portfolio.index:
                            nq = max(0, st.session_state.portfolio.loc[qt_ticker,'Quantity'] - qt_qty)
                            if nq == 0:
                                st.session_state.portfolio = st.session_state.portfolio.drop(qt_ticker)
                            else:
                                st.session_state.portfolio.loc[qt_ticker,'Quantity'] = nq
                    if save_cloud_portfolio(st.session_state.portfolio):
                        log_activity(qt_ticker, qt_action, qt_qty, qt_price)
                        st.rerun()

    # ── Stock cards — mobile-first view ──────────────────────────
    if rows:
        st.markdown('<div class="section-header">מניות בתיק</div>', unsafe_allow_html=True)
        sig_colors = {
            'STRONG BUY':'#00ff88','BUY':'#44dd88','WATCH':'#44aaff',
            'HOLD':'#ffaa00','AVOID':'#ff8844','SELL':'#ff6688','STRONG SELL':'#ff2244'
        }
        for r in sorted(rows, key=lambda x: x['P&L %'], reverse=True):
            pl_clr  = '#00ff88' if r['P&L %'] >= 0 else '#ff4466'
            s_clr   = sig_colors.get(r['Signal'], '#ffaa00')
            pl_sign = '+' if r['P&L %'] >= 0 else ''
            st.markdown(
                f'<div class="stock-card">'
                f'  <div>'
                f'    <div class="stock-ticker">{r["Ticker"]}</div>'
                f'    <div class="stock-name">{r["Sector"][:16]}</div>'
                f'  </div>'
                f'  <div style="text-align:center">'
                f'    <div class="stock-price">${r["Price"]:,.2f}</div>'
                f'    <div class="stock-pl" style="color:{pl_clr}">{pl_sign}{r["P&L %"]:.1f}%&nbsp;&nbsp;${r["P&L $"]:+,.0f}</div>'
                f'  </div>'
                f'  <div style="text-align:right">'
                f'    <div class="stock-signal" style="color:{s_clr}">{r["Signal"]}</div>'
                f'    <div class="stock-score">RSI {r["RSI"]} · {r["Score"]}/100</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Detailed table (collapsed) ─────────────────────────────
        df_rows = pd.DataFrame(rows)
        with st.expander("📋 טבלה מלאה"):
            def color_signal(val):
                return {'STRONG BUY':'color:#00ff88;font-weight:bold','BUY':'color:#44dd88',
                        'HOLD':'color:#ffaa00','WATCH':'color:#44aaff',
                        'SELL':'color:#ff6688','STRONG SELL':'color:#ff2244;font-weight:bold'}.get(val,'')
            def color_pl(val):
                try: return 'color:#00ff88' if float(val)>=0 else 'color:#ff4466'
                except: return ''
            styled = df_rows.style\
                .map(color_signal, subset=['Signal'])\
                .map(color_pl, subset=['P&L %','P&L $','Upside %'])\
                .format({'Price':'${:.2f}','Buy@':'${:.2f}','Value $':'${:,.2f}',
                         'P&L %':'{:.2f}%','P&L $':'${:,.2f}','Annual Div $':'${:.2f}',
                         'Target $':'${:.2f}','Upside %':'{:.1f}%'})
            st.dataframe(styled, use_container_width=True)
            csv = df_rows.to_csv(index=False)
            st.download_button("📥 ייצוא CSV", csv, "portfolio.csv", "text/csv")

        col_p1,col_p2 = st.columns(2)
        with col_p1:
            if sector_weights:
                fig_pie = px.pie(values=list(sector_weights.values()),
                                 names=list(sector_weights.keys()),
                                 title="פיזור סקטוריאלי",
                                 color_discrete_sequence=px.colors.sequential.Plasma)
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                      height=280, margin=dict(l=0,r=0,t=30,b=0),
                                      legend=dict(font=dict(size=9)))
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar":False})
        with col_p2:
            fig_score = go.Figure(go.Bar(
                x=[r['Ticker'] for r in rows],
                y=[r['Score']  for r in rows],
                marker_color=['#00ff88' if s>=70 else '#44aaff' if s>=55 else '#ffaa00' if s>=45 else '#ff4466'
                              for s in [r['Score'] for r in rows]],
                text=[r['Signal'] for r in rows], textposition='auto'
            ))
            fig_score.update_layout(title="Quant Score", paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white', yaxis_range=[0,100],
                                    height=280, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar":False})

    else:
        st.info("הוסף מניות לתיק דרך הסיידבר.")

# ==========================================
# TAB 3: DEEP TECHNICAL ANALYSIS
# ==========================================
with t_tech:
    st.subheader("📊 ניתוח טכני מעמיק - 25+ אינדיקטורים")

    selectable = p_tickers + ([quick_ticker] if quick_ticker else [])
    analyze_ticker = st.selectbox("בחר מניה:", selectable) if selectable else None

    if not analyze_ticker and not quick_ticker:
        analyze_ticker = st.text_input("הכנס סימול לניתוח:").upper() or None

    period_choice = st.select_slider("תקופה:", options=["1mo","3mo","6mo","1y","2y"], value="1y")

    if analyze_ticker:
        with st.spinner(f"טוען {analyze_ticker}..."):
            stock = yf.Ticker(analyze_ticker)
            hist  = stock.history(period=period_choice)

        if not hist.empty:
            d = m_data.get(analyze_ticker, {})

            # 3-col top metrics (readable on phone)
            c1,c2,c3 = st.columns(3)
            c1.metric("מחיר", f"${d.get('price', hist['Close'].iloc[-1]):,.2f}")
            rsi_v = d.get('rsi',50)
            c2.metric("RSI", f"{rsi_v:.0f}",
                      "OB🔴" if rsi_v>70 else "OS🟢" if rsi_v<30 else "Neutral")
            c3.metric("Signal", d.get('signal','N/A'))
            c4,c5,c6 = st.columns(3)
            c4.metric("Score", f"{d.get('score',50)}/100")
            c5.metric("Target", f"${d.get('target_price',0):,.2f}", f"{d.get('target_upside',0):.1f}%")
            c6.metric("Beta", f"{d.get('beta',1):.2f}")

            # Main chart
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                row_heights=[0.5,0.17,0.17,0.16],
                                subplot_titles=[f"{analyze_ticker}", "RSI", "MACD", "Volume"])

            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'],
                low=hist['Low'], close=hist['Close'], name="Price",
                increasing_line_color='#00ff88', decreasing_line_color='#ff4466'
            ), row=1, col=1)

            sma20 = hist['Close'].rolling(20).mean()
            std20 = hist['Close'].rolling(20).std()
            fig.add_trace(go.Scatter(x=hist.index, y=sma20+2*std20, name='BB Upper',
                                     line=dict(color='rgba(100,100,255,0.4)',dash='dot'), showlegend=False), row=1,col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=sma20-2*std20, name='BB Lower',
                                     line=dict(color='rgba(100,100,255,0.4)',dash='dot'),
                                     fill='tonexty', fillcolor='rgba(100,100,255,0.05)', showlegend=False), row=1,col=1)

            for period, color in [(20,'#ffaa00'),(50,'#44aaff'),(200,'#ff6644')]:
                if len(hist) >= period:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(period).mean(),
                                             name=f'SMA{period}', line=dict(color=color,width=1.5)), row=1,col=1)

            # Support / Resistance
            sup = d.get('support', hist['Low'].min())
            res = d.get('resistance', hist['High'].max())
            fig.add_hline(y=sup, line_dash="dot", line_color="#44ff88",
                          annotation_text=f"Support ${sup:.2f}", row=1, col=1)
            fig.add_hline(y=res, line_dash="dot", line_color="#ff4466",
                          annotation_text=f"Resistance ${res:.2f}", row=1, col=1)

            # Fibonacci
            h1y = hist['Close'].max(); l1y = hist['Close'].min(); diff = h1y - l1y
            for lv, col_fib in [(0.382,'#ff8844'),(0.618,'#44ff88')]:
                fig.add_hline(y=h1y-diff*lv, line_dash="dash", line_color=col_fib,
                              annotation_text=f"Fib {lv}", row=1, col=1)

            # RSI
            delta = hist['Close'].diff()
            gain  = delta.clip(lower=0).ewm(com=13,adjust=False).mean()
            loss  = -delta.clip(upper=0).ewm(com=13,adjust=False).mean()
            rsi_s = 100 - (100/(1+(gain/loss)))
            fig.add_trace(go.Scatter(x=hist.index, y=rsi_s, name='RSI',
                                     line=dict(color='#aa44ff',width=1.5)), row=2,col=1)
            fig.add_hline(y=70, line_color='red',   line_dash='dash', row=2,col=1)
            fig.add_hline(y=30, line_color='green', line_dash='dash', row=2,col=1)
            fig.add_hline(y=50, line_color='gray',  line_dash='dot',  row=2,col=1)

            # MACD
            ema12    = hist['Close'].ewm(span=12,adjust=False).mean()
            ema26    = hist['Close'].ewm(span=26,adjust=False).mean()
            macd     = ema12 - ema26
            macd_sig = macd.ewm(span=9,adjust=False).mean()
            macd_h   = macd - macd_sig
            fig.add_trace(go.Scatter(x=hist.index, y=macd,     name='MACD',   line=dict(color='#44aaff',width=1.5)), row=3,col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=macd_sig, name='Signal', line=dict(color='#ff8844',width=1.5)), row=3,col=1)
            fig.add_trace(go.Bar(x=hist.index, y=macd_h, name='Hist',
                                 marker_color=['#00ff88' if v>=0 else '#ff4466' for v in macd_h]), row=3,col=1)

            # Volume
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume',
                                 marker_color='rgba(100,150,255,0.4)'), row=4,col=1)

            fig.update_layout(
                height=650, paper_bgcolor='#0a0e1a', plot_bgcolor='#0f1729',
                font_color='white', xaxis_rangeslider_visible=False,
                margin=dict(l=10,r=10,t=30,b=10),
                legend=dict(orientation='h', y=1.02, font=dict(size=9))
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False, "scrollZoom": True})

            # Score reasons — shown above the fold on mobile
            if d:
                st.markdown(
                    f'<div class="section-header">🧠 ניתוח Score {d.get("score",0)}/100 · {d.get("trend","-")}</div>',
                    unsafe_allow_html=True
                )
                for reason in d.get('reasons',[]):
                    if "✅" in reason: st.success(reason)
                    elif "⚠️" in reason or "🚨" in reason: st.warning(reason)

            # Indicators table in expander
            with st.expander("📋 כל האינדיקטורים"):
                if d:
                    ind_data = {
                        "מגמה": d.get('trend','-'),
                        "RSI": f"{d.get('rsi',0):.1f}",
                        "MACD Cross": "✅" if d.get('macd_crossover') else "❌",
                        "Stoch %K": f"{d.get('stoch_k',0):.1f}",
                        "BB %": f"{d.get('bb_pct',0)*100:.1f}%",
                        "ATR %": f"{d.get('atr_pct',0):.2f}%",
                        "Vol Ratio": f"{d.get('vol_ratio',1):.2f}x",
                        "OBV": "▲" if d.get('obv_trend',0)==1 else "▼",
                        "Support": f"${d.get('support',0):.2f}",
                        "Resistance": f"${d.get('resistance',0):.2f}",
                        "Mom 1M": f"{d.get('mom_1m',0):.1f}%",
                        "Mom 3M": f"{d.get('mom_3m',0):.1f}%",
                        "Mom 6M": f"{d.get('mom_6m',0):.1f}%",
                        "P/E": f"{d.get('pe',0):.1f}",
                        "Fwd P/E": f"{d.get('forward_pe',0):.1f}",
                        "PEG": f"{d.get('peg',0):.2f}",
                        "P/B": f"{d.get('pb',0):.2f}",
                        "ROE": f"{d.get('roe',0):.1f}%",
                        "Gross Margin": f"{d.get('gross_margin',0):.1f}%",
                        "Net Margin": f"{d.get('profit_margin',0):.1f}%",
                        "Rev Growth": f"{d.get('growth_yoy',0):.1f}%",
                        "Debt/Eq": f"{d.get('debt_to_equity',0):.2f}",
                        "Short %": f"{d.get('short_pct',0):.1f}%",
                        "Inst %": f"{d.get('inst_own',0):.1f}%",
                        "Target": f"${d.get('target_price',0):.2f}",
                        "Upside": f"{d.get('target_upside',0):.1f}%",
                        "Div Yield": f"{d.get('div',0):.2f}%",
                    }
                    ind_df = pd.DataFrame(list(ind_data.items()), columns=['Indicator','Value'])
                    c_i1, c_i2 = st.columns(2)
                    half = len(ind_df)//2
                    c_i1.dataframe(ind_df.iloc[:half], use_container_width=True, hide_index=True)
                    c_i2.dataframe(ind_df.iloc[half:], use_container_width=True, hide_index=True)

# ==========================================
# TAB 4: AI FORECAST
# ==========================================
with t_ai:
    st.subheader("🤖 תחזית AI - Random Forest + Gradient Boosting Ensemble")

    selectable_ai = p_tickers + ([quick_ticker] if quick_ticker else [])
    if selectable_ai:
        forecast_ticker = st.selectbox("מניה לתחזית:", selectable_ai, key='fc_sel')
    else:
        forecast_ticker = st.text_input("הכנס סימול:", key='fc_inp').upper()

    forecast_days = st.slider("ימי תחזית:", 7, 60, 14)

    if forecast_ticker and st.button("🚀 הרץ מודל AI"):
        with st.spinner("מאמן מודל על 5 שנות נתונים..."):
            pred, details = ml_forecast(forecast_ticker, forecast_days)

        if details:
            c1,c2 = st.columns(2)
            c1.metric("מחיר נוכחי", f"${details['current']:,.2f}")
            c2.metric(f"תחזית {forecast_days}d", f"${details['predicted']:,.2f}",
                      f"{details['pct_change']:.1f}%")
            c3,c4 = st.columns(2)
            direction = "📈 עלייה" if details['pct_change']>0 else "📉 ירידה"
            c3.metric("כיוון", direction)
            c4.metric("Confidence", f"{details['confidence']:.0f}%")

            col_m1,col_m2 = st.columns(2)
            col_m1.info(f"🌲 Random Forest: ${details['rf_pred']:,.2f}")
            col_m2.info(f"⚡ Gradient Boost: ${details['gb_pred']:,.2f}")

            # Range visualization
            curr = details['current']
            pred_v = details['predicted']
            confidence_range = curr * details.get('mae',0) / curr * 2 if details.get('mae') else abs(pred_v-curr)*0.3
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=['Now', f'+{forecast_days}d'],
                y=[curr, pred_v],
                mode='lines+markers',
                line=dict(color='#44aaff', width=3),
                name='Forecast'
            ))
            fig_pred.add_trace(go.Scatter(
                x=['Now', f'+{forecast_days}d', f'+{forecast_days}d', 'Now'],
                y=[curr, pred_v+confidence_range, pred_v-confidence_range, curr],
                fill='toself', fillcolor='rgba(68,170,255,0.1)',
                line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'
            ))
            fig_pred.update_layout(title=f"{forecast_ticker} Price Forecast",
                                   paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=280,
                                   margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar":False})

            with st.expander("📊 Feature Importance"):
                imp_df = pd.DataFrame(list(details['importance'].items()),
                                      columns=['Feature','Importance']).sort_values('Importance', ascending=True)
                fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'],
                                           orientation='h', marker_color='#44aaff'))
                fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=320,
                                      margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar":False})

            st.warning("⚠️ תחזיות ML הן הסתברותיות בלבד ואינן ערובה לתשואה. תמיד בצע Due Diligence עצמאי.")
        else:
            st.error("לא מספיק דאטה לאמן את המודל (נדרשות לפחות 252 נקודות / ~1 שנה).")

# ==========================================
# TAB 5: SCANNER
# ==========================================
with t_scan:
    st.subheader("🔥 סורק הזדמנויות מתקדם")

    sc1,sc2 = st.columns(2)
    with sc1:
        scan_universe = st.selectbox("יקום:", [
            "Mega Watchlist (18)", "NASDAQ 100 (40)",
            "S&P 500 (40)", "Tech Giants",
            "Value Plays", "High Growth", "Dividend Stocks",
            "ETFs", "Israeli Stocks (TA)"
        ])
    with sc2:
        min_score = st.slider("Score מינימלי:", 40, 90, 60)

    with st.expander("🔽 פילטרים נוספים"):
        sf1,sf2 = st.columns(2)
        sector_filter = sf1.selectbox("סקטור:", ["הכל","Technology","Healthcare",
                                                   "Financial Services","Consumer Cyclical",
                                                   "Energy","Industrials","Real Estate"])
        max_pe = sf2.number_input("P/E מקס (0=ללא):", value=0, min_value=0)
        sf3,sf4 = st.columns(2)
        min_growth = sf3.number_input("Growth% מינ:", value=0)
        min_div    = sf4.number_input("Div% מינ:", value=0.0, step=0.5)

    # normalize sector filter value
    if sector_filter == "הכל": sector_filter = "כל הסקטורים"

    if st.button("🔍 הרץ סריקה"):
        with st.spinner("סורק..."):
            if scan_universe == "Mega Watchlist (18)":
                scan_list = ['AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','INTC',
                             'BA','PEP','COST','AVGO','DIS','UBER','ABNB','SPOT']
            elif scan_universe == "NASDAQ 100 (Sample 40)":
                scan_list = random.sample(NASDAQ_100, min(40, len(NASDAQ_100)))
            elif scan_universe == "S&P 500 (Sample 40)":
                scan_list = random.sample(get_sp500_tickers(), 40)
            elif scan_universe == "Tech Giants":
                scan_list = ['AAPL','MSFT','NVDA','META','GOOGL','AMZN','AMD','INTC','AVGO','QCOM','TSM','ASML','CRM','ORCL','ADBE']
            elif scan_universe == "Value Plays":
                scan_list = ['BRK-B','JPM','BAC','WFC','C','PEP','KO','JNJ','PG','MRK','ABBV','CVX','XOM','V','MA']
            elif scan_universe == "High Growth":
                scan_list = ['NVDA','META','AMZN','TSLA','CRM','SNOW','DDOG','PLTR','RBLX','COIN','SQ','SHOP','NET','CRWD']
            elif scan_universe == "Dividend Stocks":
                scan_list = DIVIDEND_STOCKS
            elif scan_universe == "ETFs":
                scan_list = ETF_UNIVERSE
            else:
                scan_list = TA35_TICKERS

            scan_data = fetch_expert_data(tuple(scan_list))

            results = []
            for t, d in scan_data.items():
                if d.get('score',0) < min_score: continue
                if sector_filter != "כל הסקטורים" and d.get('sector','') != sector_filter: continue
                if max_pe > 0 and (d.get('pe',0) <= 0 or d.get('pe',0) > max_pe): continue
                if d.get('growth_yoy',0) < min_growth: continue
                if d.get('div',0) < min_div: continue
                results.append((t, d))
            results.sort(key=lambda x: x[1].get('score',0), reverse=True)

            st.markdown(f"### נמצאו **{len(results)}** מניות")

            if results:
                # Mobile-friendly: cards sorted by score
                sig_clr_map = {
                    'STRONG BUY':'#00ff88','BUY':'#44dd88','WATCH':'#44aaff',
                    'HOLD':'#ffaa00','SELL':'#ff6688','STRONG SELL':'#ff2244','AVOID':'#ff8844'
                }
                for t, d in results[:20]:  # cap at 20 for mobile performance
                    sc   = d.get('score',50)
                    sc_color = '#00ff88' if sc>=70 else '#44aaff' if sc>=55 else '#ffaa00'
                    sig  = d.get('signal','')
                    s_cl = sig_clr_map.get(sig,'#ffaa00')
                    up   = d.get('target_upside',0)
                    st.markdown(
                        f'<div class="stock-card">'
                        f'  <div>'
                        f'    <div class="stock-ticker">{t}</div>'
                        f'    <div class="stock-name">{d.get("sector","")[:16]}</div>'
                        f'    <div style="font-size:0.65rem;color:#7a8ab0;margin-top:2px">'
                        f'      P/E {d["pe"]:.0f} · Gr {d.get("growth_yoy",0):.0f}% · RSI {d["rsi"]:.0f}'
                        f'    </div>'
                        f'  </div>'
                        f'  <div style="text-align:center">'
                        f'    <div class="stock-price">${d["price"]:,.2f}</div>'
                        f'    <div style="font-size:0.72rem;color:#44aaff">▲ {up:.1f}% upside</div>'
                        f'  </div>'
                        f'  <div style="text-align:right">'
                        f'    <div class="stock-signal" style="color:{s_cl}">{sig}</div>'
                        f'    <div style="font-size:0.85rem;font-weight:700;color:{sc_color}">{sc}/100</div>'
                        f'    <div style="font-size:0.65rem;color:#7a8ab0">{"MACD✅" if d.get("macd_crossover") else ""}</div>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with st.expander("📋 טבלה מלאה"):
                    table_data = [{'Ticker':t,'Score':d['score'],'Signal':d['signal'],
                                   'Price':f"${d['price']:.2f}",'RSI':round(d['rsi'],1),
                                   'Trend':d['trend'],'P/E':round(d['pe'],1),
                                   'Growth%':round(d.get('growth_yoy',0),1),
                                   'Div%':round(d.get('div',0),2),
                                   'Upside%':round(d.get('target_upside',0),1)}
                                  for t,d in results]
                    tbl_df = pd.DataFrame(table_data)
                    st.dataframe(tbl_df, use_container_width=True)
                    st.download_button("📥 ייצוא", tbl_df.to_csv(index=False), "scan.csv", "text/csv")

# ==========================================
# TAB 6: RISK ANALYSIS
# ==========================================
with t_risk:
    st.subheader("⚖️ ניתוח סיכונים מתקדם")

    if p_tickers:
        with st.spinner("מחשב..."):
            corr, sharpe, max_dd, var95, spy_corr, spy_ret_1y = compute_portfolio_analytics(tuple(p_tickers))

        risk_rows = []
        for t in p_tickers:
            d = m_data.get(t,{})
            risk_rows.append({
                'Ticker':t, 'Beta':round(d.get('beta',1),2),
                'ATR%':round(d.get('atr_pct',0),2),
                'Sharpe':round(sharpe.get(t,0),2),
                'Max DD%':round(max_dd.get(t,0),1),
                'VaR 95%':round(var95.get(t,0),2),
                'SPY Corr':round(spy_corr.get(t,0),2),
                'Short%':round(d.get('short_pct',0),1),
                'Debt/Eq':round(d.get('debt_to_equity',0),2),
                'Inst%':round(d.get('inst_own',0),1),
            })

        risk_df = pd.DataFrame(risk_rows)
        st.dataframe(risk_df, use_container_width=True)

        if rows:
            total_v   = sum(r['Value $'] for r in rows)
            port_beta = sum(m_data[r['Ticker']].get('beta',1)*r['Value $']/total_v
                           for r in rows if r['Ticker'] in m_data) if total_v > 0 else 1
            avg_sharpe = float(np.mean(list(sharpe.values()))) if sharpe else 0
            worst_dd   = min(max_dd.values()) if max_dd else 0
            worst_t    = min(max_dd, key=max_dd.get) if max_dd else '-'

            # 2×2 risk metrics
            rk1,rk2 = st.columns(2)
            rk1.metric("Beta", f"{port_beta:.2f}", "⚡ תנודתי" if port_beta>1.2 else "🛡️ מאוזן")
            rk2.metric("Sharpe", f"{avg_sharpe:.2f}", "✅" if avg_sharpe>1 else "⚠️" if avg_sharpe>0 else "❌")
            rk3,rk4 = st.columns(2)
            rk3.metric("Max Drawdown", f"{worst_dd:.1f}%", worst_t)
            rk4.metric("SPY 1Y", f"{spy_ret_1y:.1f}%")

        if corr is not None and len(p_tickers)>1:
            with st.expander("🔗 מטריצת מתאמים"):
                fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r',
                                     zmin=-1, zmax=1, text_auto='.2f')
                fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=380,
                                       margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar":False})
                st.caption(f"SPY 1Y Return: {spy_ret_1y:.1f}% · מתאם נמוך = גיוון טוב")

        if rows and total_v > 0:
            with st.expander("🎲 Monte Carlo Simulation (1Y)"):
                n_sim  = 500; n_days = 252
                avg_ret = float(np.mean([m_data[r['Ticker']].get('mom_1m',0)/100/20
                                         for r in rows if r['Ticker'] in m_data]))
                avg_vol = float(np.mean([m_data[r['Ticker']].get('atr_pct',2)/100/np.sqrt(20)
                                         for r in rows if r['Ticker'] in m_data]))
                sims = np.array([
                    np.prod(1 + np.random.normal(avg_ret, avg_vol, n_days)) * total_v
                    for _ in range(n_sim)
                ])
                p5  = float(np.percentile(sims, 5))
                p50 = float(np.percentile(sims, 50))
                p95 = float(np.percentile(sims, 95))
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(x=sims, nbinsx=40, marker_color='rgba(68,170,255,0.6)'))
                for val, clr, lbl in [(p5,'#ff4466','5%'),(p50,'#ffaa00','50%'),(p95,'#00ff88','95%')]:
                    fig_mc.add_vline(x=val, line_dash='dash', line_color=clr,
                                     annotation_text=f"${val:,.0f}")
                fig_mc.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                     height=280, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_mc, use_container_width=True, config={"displayModeBar":False})
                mc1,mc2,mc3 = st.columns(3)
                mc1.metric("Worst 5%",  f"${p5:,.0f}",  f"{(p5/total_v-1)*100:.1f}%")
                mc2.metric("Median",    f"${p50:,.0f}", f"{(p50/total_v-1)*100:.1f}%")
                mc3.metric("Best 95%",  f"${p95:,.0f}", f"{(p95/total_v-1)*100:.1f}%")
    else:
        st.info("הוסף מניות לתיק.")

# ==========================================
# TAB 7: STRATEGIES
# ==========================================
with t_strat:
    st.subheader("🎯 אסטרטגיות מסחר מתקדמות")

    strat_tabs = st.tabs(["📅 יעדי מחיר","🔄 DCA","📐 Position Sizing","⚖️ Kelly Criterion","🎓 מדריך"])

    with strat_tabs[0]:
        st.markdown("### ⏱️ יעדי מחיר לפי טווח זמן")
        for t in p_tickers:
            if t not in m_data: continue
            d = m_data[t]; p = d['price']
            atr = d.get('atr_pct',2)/100; growth = max(0, d.get('growth_yoy',8))/100
            with st.expander(f"📊 {t} — ${p:.2f} | {d.get('signal','N/A')} | Score:{d.get('score',50)}"):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("1-5 ימים", f"${p*(1+atr*1.5):.2f}", f"+{atr*150:.1f}%")
                c2.metric("2-4 שבועות", f"${p*(1+atr*3):.2f}", f"+{atr*300:.1f}%")
                c3.metric("1-3 חודשים", f"${p*(1+growth*0.25):.2f}", f"+{growth*25:.1f}%")
                c4.metric("1 שנה", f"${p*(1+growth):.2f}", f"+{growth*100:.1f}%")
                st.markdown("**פיבונאצ'י:**")
                fc1,fc2,fc3,fc4,fc5 = st.columns(5)
                for col_w, (lv,key) in zip([fc1,fc2,fc3,fc4,fc5],
                    [(0.236,'fib_236'),(0.382,'fib_382'),(0.5,'fib_500'),(0.618,'fib_618'),(0.786,'fib_786')]):
                    col_w.metric(f"Fib {lv}", f"${d.get(key,0):.2f}")

    with strat_tabs[1]:
        st.markdown("### 🔄 מחשבון DCA")
        dca_ticker  = st.selectbox("מניה:", p_tickers, key='dca') if p_tickers else None
        dca_monthly = st.number_input("השקעה חודשית ($):", value=100.0)
        dca_months  = st.slider("חודשים:", 6, 120, 24)
        dca_growth  = st.slider("צמיחה שנתית משוערת (%):", -10, 30, 8)

        if dca_ticker and dca_ticker in m_data:
            curr_p    = m_data[dca_ticker]['price']
            growth_r  = dca_growth / 100
            total_inv = dca_monthly * dca_months
            shares    = 0; monthly_p = curr_p; rows_dca = []
            for month in range(1, dca_months+1):
                monthly_p *= (1 + growth_r/12)
                shares    += dca_monthly / monthly_p
                val        = shares * monthly_p
                rows_dca.append({'חודש':month,'מחיר':round(monthly_p,2),
                                  'מניות':round(shares,3),'שווי':round(val,2)})
            dca_df   = pd.DataFrame(rows_dca)
            final_v  = dca_df['שווי'].iloc[-1]
            c1,c2,c3 = st.columns(3)
            c1.metric("סה\"כ הושקע", f"${total_inv:,.2f}")
            c2.metric("שווי צפוי",   f"${final_v:,.2f}")
            c3.metric("רווח צפוי",   f"${final_v-total_inv:,.2f}", f"{(final_v/total_inv-1)*100:.1f}%")
            fig_dca = go.Figure()
            fig_dca.add_trace(go.Scatter(x=dca_df['חודש'], y=dca_df['שווי'], name='שווי', fill='tozeroy', line=dict(color='#00ff88')))
            fig_dca.add_trace(go.Scatter(x=dca_df['חודש'], y=[dca_monthly*m for m in dca_df['חודש']], name='הושקע', line=dict(color='#ffaa00',dash='dash')))
            fig_dca.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', title="DCA Growth")
            st.plotly_chart(fig_dca, use_container_width=True)

    with strat_tabs[2]:
        st.markdown("### 📐 Position Sizing Calculator")
        ps_account  = st.number_input("גודל תיק ($):", value=float(total_usd) if 'total_usd' in dir() else 7000.0)
        ps_risk_pct = st.slider("סיכון לעסקה (%):", 0.5, 5.0, 1.0, 0.5)
        sel_ps = p_tickers + ([quick_ticker] if quick_ticker else [])
        ps_sel = st.selectbox("מניה:", sel_ps, key='ps_sel') if sel_ps else None
        ps_stop = st.slider("Stop Loss (%):", 2.0, 25.0, 8.0, 0.5)
        ps_target = st.slider("Take Profit (%):", 5.0, 50.0, 20.0, 1.0)

        if ps_sel and ps_sel in m_data:
            d = m_data[ps_sel]; curr_p = d['price']
            risk_share = curr_p * (ps_stop/100)
            max_risk   = ps_account * (ps_risk_pct/100)
            shares     = int(max_risk / risk_share)
            pos_size   = shares * curr_p
            stop_p     = curr_p * (1 - ps_stop/100)
            target_p   = curr_p * (1 + ps_target/100)
            rr_ratio   = ps_target / ps_stop

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("מניות לקנות", str(shares))
            c2.metric("גודל פוזיציה", f"${pos_size:,.2f}", f"{pos_size/ps_account*100:.1f}% מהתיק")
            c3.metric("Stop Loss",  f"${stop_p:.2f}", f"-{ps_stop}%")
            c4.metric("Take Profit", f"${target_p:.2f}", f"+{ps_target}%")

            risk_color = "#00ff88" if rr_ratio >= 2 else "#ffaa00" if rr_ratio >= 1.5 else "#ff4466"
            st.markdown(f"**Risk/Reward Ratio: <span style='color:{risk_color}'>1:{rr_ratio:.1f}</span>** {'✅ Good' if rr_ratio>=2 else '⚠️ Marginal' if rr_ratio>=1.5 else '❌ Poor'}",
                        unsafe_allow_html=True)
            st.info(f"קנה **{shares}** מניות @ ${curr_p:.2f} = ${pos_size:,.2f}\n"
                    f"Stop: ${stop_p:.2f} | Target: ${target_p:.2f} | Max Loss: ${max_risk:.2f}")

    with strat_tabs[3]:
        st.markdown("### ⚖️ Kelly Criterion - גודל פוזיציה אופטימלי")
        kelly_winrate = st.slider("אחוז עסקאות מנצחות (%):", 30, 80, 55)
        kelly_avg_win = st.slider("רווח ממוצע בניצחון (%):", 5, 50, 15)
        kelly_avg_loss= st.slider("הפסד ממוצע בהפסד (%):", 3, 25, 8)

        p_w = kelly_winrate/100; p_l = 1-p_w
        b   = kelly_avg_win/kelly_avg_loss
        kelly_full = p_w - p_l/b
        kelly_half = kelly_full/2  # half-Kelly for safety

        c1,c2,c3 = st.columns(3)
        c1.metric("Full Kelly %",  f"{kelly_full*100:.1f}%")
        c2.metric("Half Kelly %",  f"{kelly_half*100:.1f}% (Recommended)")
        c3.metric("Expected Value",f"{(p_w*kelly_avg_win - p_l*kelly_avg_loss):.1f}% per trade")

        if kelly_full > 0:
            st.success(f"✅ השקע **{kelly_half*100:.1f}%** מהתיק בכל עסקה (Half-Kelly למינימום סיכון)")
        else:
            st.error("❌ Kelly שלילי — Edge שלילי! שפר את הביצועים לפני שמשקיעים.")

    with strat_tabs[4]:
        st.markdown("""
### 🎓 מדריך אסטרטגיות מסחר

#### 1. 📈 Trend Following
**מתי?** SMA20 > SMA50 > SMA200, מחיר מעל SMA20, Score > 65
**כיצד?** קנה בפולבק ל-SMA20 | Stop: מתחת ל-SMA50

#### 2. ⚡ RSI Swing Trading
**מתי?** RSI < 35 במגמה עולה | Target: RSI חוזר ל-55+
**Score:** > 55 | **Stop:** -7%

#### 3. 💎 Value Investing
**מתי?** P/E < 15, P/B < 2, ROE > 15%, Debt/Eq < 0.5
**כיצד?** קנה ואחזק 1-3 שנים

#### 4. 🚀 Growth Momentum
**מתי?** Revenue Growth > 20%, Earnings Growth > 15%, Strong Uptrend
**כיצד?** קנה בברייקאאוט | Stop: -12%

#### 5. 🔄 MACD Crossover
**מתי?** MACD חוצה Signal מלמטה | Score > 58
**Stop:** -7% | **Target:** +12-15%

#### 6. 📉 Oversold Bounce
**מתי?** RSI < 30, Stoch < 20, BB% < 0.1 (מחיר ליד הגבול התחתון)
**Stop:** -5% | **Target:** חזרה לSMA20

#### 7. 💸 Dividend Growth
**מתי?** Div > 3%, Payout Ratio < 70%, Profit Margin > 10%
**אסטרטגיה:** קנה ואחזק, השקע מחדש דיבידנדים (DRIP)
        """)

# ==========================================
# TAB 8: NEWS & EARNINGS
# ==========================================
with t_news:
    st.subheader("📰 חדשות ויומן רווחים")

    news_tabs = st.tabs(["📰 חדשות","📅 Earnings Calendar","👁️ רשימת מעקב"])

    with news_tabs[0]:
        st.markdown("### חדשות אחרונות לפי מניה")
        news_ticker = st.selectbox("בחר מניה:", p_tickers + ([quick_ticker] if quick_ticker else []), key='news_sel') if (p_tickers or quick_ticker) else None
        if not news_ticker:
            news_ticker = st.text_input("הכנס סימול:", key='news_inp').upper() or None

        if news_ticker:
            with st.spinner(f"טוען חדשות ל-{news_ticker}..."):
                news = get_stock_news(news_ticker)
            if news:
                for item in news:
                    try:
                        ts = datetime.fromtimestamp(item.get('providerPublishTime',0))
                        title = item.get('title','')
                        publisher = item.get('publisher','')
                        link = item.get('link','#')
                        with st.expander(f"📰 {title[:80]}"):
                            st.markdown(f"**{publisher}** — {ts.strftime('%Y-%m-%d %H:%M')}")
                            st.markdown(f"[קרא כתבה מלאה]({link})")
                    except:
                        pass
            else:
                st.info("אין חדשות זמינות.")

    with news_tabs[1]:
        st.markdown("### 📅 יומן רווחים קרובים")
        for t in p_tickers:
            d = m_data.get(t,{})
            with st.expander(f"📊 {t} — {d.get('name',t)[:30]}"):
                with st.spinner("טוען..."):
                    cal, hist_earn = get_earnings_calendar(t)
                col1_e, col2_e = st.columns(2)
                with col1_e:
                    if cal:
                        try:
                            if isinstance(cal, dict):
                                earn_date = cal.get('Earnings Date','N/A')
                                est_eps   = cal.get('EPS Estimate','N/A')
                                rev_est   = cal.get('Revenue Estimate','N/A')
                                st.markdown(f"""
**Earnings Date:** {earn_date}
**EPS Estimate:** {est_eps}
**Revenue Estimate:** {rev_est}
                                """)
                            else:
                                st.dataframe(cal.T if hasattr(cal,'T') else cal, use_container_width=True)
                        except:
                            st.info("לא ניתן לטעון נתוני לוח שנה.")
                    else:
                        st.info("אין נתוני לוח שנה.")
                with col2_e:
                    if hist_earn is not None and not hist_earn.empty:
                        st.markdown("**Earnings History (Last 4Q):**")
                        try:
                            cols_show = ['epsActual','epsEstimate','surprisePercent']
                            show_cols = [c for c in cols_show if c in hist_earn.columns]
                            if show_cols:
                                st.dataframe(hist_earn[show_cols].tail(4), use_container_width=True)
                        except:
                            st.dataframe(hist_earn.tail(4), use_container_width=True)
                    else:
                        st.info("אין היסטוריית רווחים.")

    with news_tabs[2]:
        st.markdown("### 👁️ רשימת מעקב (Watchlist)")
        wl = st.session_state.watchlist

        # Add to watchlist
        with st.form("add_watchlist"):
            col_w1,col_w2,col_w3,col_w4 = st.columns(4)
            wl_ticker = col_w1.text_input("Ticker").upper()
            wl_notes  = col_w2.text_input("הערות")
            wl_high   = col_w3.number_input("Alert High $", value=0.0)
            wl_low    = col_w4.number_input("Alert Low $", value=0.0)
            if st.form_submit_button("➕ הוסף"):
                if wl_ticker:
                    new_row = pd.DataFrame([{'Ticker':wl_ticker,'Notes':wl_notes,
                                             'AlertHigh':wl_high,'AlertLow':wl_low}])
                    wl = pd.concat([wl, new_row], ignore_index=True).drop_duplicates('Ticker')
                    st.session_state.watchlist = wl
                    save_watchlist(wl)
                    st.rerun()

        if not wl.empty:
            wl_tickers = [t for t in wl['Ticker'].tolist() if isinstance(t,str) and t.strip()]
            wl_data    = fetch_expert_data(tuple(wl_tickers)) if wl_tickers else {}

            # Alerts
            alerts = []
            for _, row_wl in wl.iterrows():
                t = str(row_wl['Ticker'])
                if t in wl_data:
                    curr_p = wl_data[t]['price']
                    try:
                        ah = float(row_wl['AlertHigh'])
                        if ah > 0 and curr_p >= ah:
                            alerts.append(f"🔴 {t} @ ${curr_p:.2f} >= High Alert ${ah:.2f}")
                    except: pass
                    try:
                        al = float(row_wl['AlertLow'])
                        if al > 0 and curr_p <= al:
                            alerts.append(f"🟢 {t} @ ${curr_p:.2f} <= Low Alert ${al:.2f}")
                    except: pass

            if alerts:
                st.markdown("#### 🚨 Alerts Active!")
                for a in alerts:
                    if "🔴" in a: st.warning(a)
                    else: st.success(a)

            # Watchlist table
            wl_rows = []
            for _, row_wl in wl.iterrows():
                t = str(row_wl['Ticker'])
                d_wl = wl_data.get(t, {})
                wl_rows.append({
                    'Ticker': t,
                    'Notes': row_wl.get('Notes',''),
                    'Price': f"${d_wl.get('price',0):.2f}" if d_wl else 'N/A',
                    'Signal': d_wl.get('signal','N/A') if d_wl else 'N/A',
                    'Score': d_wl.get('score','-') if d_wl else '-',
                    'RSI': round(d_wl.get('rsi',50),1) if d_wl else '-',
                    'Trend': d_wl.get('trend','-') if d_wl else '-',
                    'Upside%': round(d_wl.get('target_upside',0),1) if d_wl else '-',
                    'Alert High': row_wl.get('AlertHigh',0),
                    'Alert Low':  row_wl.get('AlertLow',0),
                })
            wl_df = pd.DataFrame(wl_rows)
            st.dataframe(wl_df, use_container_width=True, hide_index=True)

            # Remove
            remove_t = st.selectbox("הסר מניה:", ['—']+wl_tickers)
            if remove_t != '—' and st.button("🗑️ הסר"):
                wl = wl[wl['Ticker'] != remove_t]
                st.session_state.watchlist = wl
                save_watchlist(wl)
                st.rerun()
        else:
            st.info("הרשימה ריקה. הוסף מניות למעקב.")

# ==========================================
# TAB 9: JOURNAL
# ==========================================
with t_journal:
    st.subheader("📜 יומן פעולות")
    try:
        logs = conn.read(worksheet="Activity", ttl=0)
        if logs is not None and not logs.empty:
            logs_sorted = logs.sort_values("Date", ascending=False)
            st.dataframe(logs_sorted, use_container_width=True)

            buys  = logs[logs['Action']=='Buy']
            sells = logs[logs['Action']=='Sell']
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("סה\"כ פעולות", len(logs))
            c2.metric("קניות", len(buys))
            c3.metric("מכירות", len(sells))
            if len(buys) > 0:
                c4.metric("סה\"כ הושקע (קניות)", f"${(buys['Quantity']*buys['Price']).sum():,.2f}")

            if len(logs) > 1:
                fig_act = px.bar(logs.sort_values("Date"), x='Date', y='Quantity', color='Action',
                                 color_discrete_map={'Buy':'#00ff88','Sell':'#ff4466'},
                                 title="היסטוריית פעולות")
                fig_act.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_act, use_container_width=True)

            csv_journal = logs_sorted.to_csv(index=False)
            st.download_button("📥 ייצוא יומן", csv_journal, "journal.csv", "text/csv")
        else:
            st.info("אין פעולות עדיין.")
    except:
        st.info("לא ניתן לטעון יומן פעולות.")

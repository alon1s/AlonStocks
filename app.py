import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import requests
from io import StringIO
import concurrent.futures
import random
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AlonStocks Pro",
    layout="wide",
    page_icon="◆",
    initial_sidebar_state="collapsed"
)

# ==========================================
# DESIGN SYSTEM
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Instrument+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink: #0A0B0E;
        --panel: #14161C;
        --panel-2: #1A1D25;
        --line: #262A35;
        --line-soft: #1C1F28;
        --text: #ECECEE;
        --text-dim: #8B8F9C;
        --text-faint: #565A67;
        --gold: #C9A24B;
        --gold-soft: rgba(201,162,75,0.12);
        --gain: #55A87F;
        --gain-soft: rgba(85,168,127,0.12);
        --loss: #C1584F;
        --loss-soft: rgba(193,88,79,0.12);
        --info: #6B93BE;
        --info-soft: rgba(107,147,190,0.12);
    }

    html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif; }
    .main { background: var(--ink); }
    [data-testid="stAppViewContainer"] { background: var(--ink); }
    [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
    [data-testid="stHeader"] { background: transparent; }

    input, select, textarea, button { font-size: 16px !important; }

    .main .block-container {
        padding: 1.1rem 0.9rem 112px !important;
        max-width: 100% !important;
    }

    /* ── RTL ── */
    [data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li,
    [data-testid="stMarkdown"] h1, [data-testid="stMarkdown"] h2, [data-testid="stMarkdown"] h3,
    .stAlert p, .stSuccess p, .stWarning p, .stError p, .stInfo p,
    [data-testid="stWidgetLabel"] p, [data-testid="stMetric"] label,
    [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span,
    [data-testid="stFormSubmitButton"] button, .section-title, .row-name, .row-sub {
        direction: rtl !important; text-align: right !important; unicode-bidi: embed !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
    .stDataFrame, .stDataEditor, .js-plotly-plot, pre, code,
    .row-ticker, .row-price, .hero-value { direction: ltr !important; text-align: left !important; }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px !important;
    }
    [data-testid="stMetric"] label {
        font-size: 0.7rem !important; color: var(--text-dim) !important;
        letter-spacing: 0.01em; font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important; font-weight: 600 !important;
        font-family: 'IBM Plex Mono', monospace !important; color: var(--text) !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

    /* ── Buttons ── */
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        min-height: 46px !important; font-size: 0.92rem !important;
        border-radius: 7px !important; width: 100% !important;
        font-weight: 600 !important; background: transparent !important;
        color: var(--gold) !important; border: 1px solid var(--gold) !important;
        transition: all 0.12s;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background: var(--gold-soft) !important; }
    .stButton > button:active { transform: scale(0.98); }

    /* ── Inputs ── */
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea, [data-baseweb="select"] {
        background: var(--panel) !important; border: 1px solid var(--line) !important;
        color: var(--text) !important; border-radius: 6px !important;
    }

    /* ── DataFrames ── */
    .stDataFrame { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.75em; }

    /* ── Expanders ── */
    div[data-testid="stExpander"] {
        background: var(--panel); border: 1px solid var(--line);
        border-radius: 8px; margin-bottom: 8px;
    }

    /* ── Hero ── */
    .hero-label { font-size: 0.74rem; color: var(--text-dim); margin-bottom: 2px; }
    .hero-value {
        font-family: 'Fraunces', serif; font-weight: 500; font-size: 2.5rem;
        color: var(--text); line-height: 1.1; letter-spacing: -0.01em;
    }
    .hero-change { font-family: 'IBM Plex Mono', monospace; font-size: 0.92rem; margin-top: 4px; }

    /* ── Section title (structural, not decorative) ── */
    .section-title {
        font-size: 0.82rem; font-weight: 600; color: var(--text);
        border-right: 3px solid var(--gold); padding: 3px 10px 3px 0;
        margin: 22px 0 10px;
    }
    .section-sub { font-size: 0.72rem; color: var(--text-faint); margin: -6px 0 12px; padding-right: 13px; }

    /* ── Ledger row (replaces boxed cards) ── */
    .ledger-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px 4px; border-bottom: 1px solid var(--line-soft);
    }
    .ledger-row:hover { background: var(--panel); }
    .row-ticker { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.98rem; color: var(--text); }
    .row-name { font-size: 0.68rem; color: var(--text-faint); margin-top: 2px; }
    .row-price { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 0.9rem; color: var(--text); }
    .row-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; margin-top: 2px; }
    .row-tag { font-size: 0.76rem; font-weight: 600; display: flex; align-items: center; gap: 5px; justify-content: flex-end; }
    .row-caption { font-size: 0.68rem; color: var(--text-faint); margin-top: 2px; max-width: 220px; }
    .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }

    /* ── Alert / attention strip ── */
    .attn-strip {
        border-right: 3px solid var(--gold); background: var(--panel);
        border-radius: 6px; padding: 10px 12px; margin-bottom: 8px;
    }
    .attn-strip .row-name { color: var(--text-dim); }

    /* ── Ticker strip (indices) ── */
    .idx-strip { display: flex; flex-wrap: wrap; gap: 0; margin-bottom: 14px; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .idx-item { flex: 1; min-width: 90px; padding: 10px 12px; border-left: 1px solid var(--line); }
    .idx-item:last-child { border-left: none; }
    .idx-name { font-size: 0.62rem; color: var(--text-faint); }
    .idx-price { font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem; font-weight: 600; color: var(--text); margin-top: 2px; }
    .idx-chg { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; margin-top: 1px; }

    /* ── Sentiment bar ── */
    .sentiment-bar { height: 4px; border-radius: 2px; background: linear-gradient(90deg,#C1584F,#C9A24B,#55A87F); margin: 10px 0 4px; position: relative; }
    .sentiment-marker { position: absolute; top: -4px; width: 12px; height: 12px; border-radius: 50%; background: var(--text); border: 2px solid var(--ink); transform: translateX(-50%); }

    /* ── News item ── */
    .news-item { padding: 10px 4px; border-bottom: 1px solid var(--line-soft); }
    .news-title { font-size: 0.82rem; color: var(--text); }
    .news-meta { font-size: 0.66rem; color: var(--text-faint); margin-top: 3px; font-family: 'IBM Plex Mono', monospace; }

    /* ── Bottom nav (mobile) ── */
    @media (max-width: 900px) {
        [data-testid="stTabs"] > div:first-child > div[data-baseweb="tab-list"] {
            position: fixed !important; bottom: 0 !important; left: 0 !important; right: 0 !important; top: auto !important;
            z-index: 9999 !important; background: var(--panel) !important;
            border-top: 1px solid var(--line) !important; border-bottom: none !important;
            padding: 8px 4px env(safe-area-inset-bottom, 10px) !important; margin: 0 !important;
            display: flex !important; justify-content: space-around !important;
            box-shadow: 0 -8px 24px rgba(0,0,0,0.4) !important; gap: 0 !important; overflow-x: visible !important;
        }
        [data-testid="stTabs"] > div:first-child div[data-baseweb="tab"] {
            flex: 1 !important; flex-direction: column !important; align-items: center !important;
            justify-content: center !important; padding: 4px 2px !important; font-size: 0.62rem !important;
            white-space: normal !important; text-align: center !important; line-height: 1.25 !important;
            border-bottom: none !important; min-width: 0 !important; color: var(--text-dim) !important;
        }
        [data-testid="stTabs"] > div:first-child div[aria-selected="true"][data-baseweb="tab"] {
            color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important;
        }
        [data-testid="stTabs"] [data-testid="stTabs"] div[data-baseweb="tab-list"] {
            position: static !important; box-shadow: none !important; border-top: none !important;
            padding: 0 !important; justify-content: flex-start !important; overflow-x: auto !important; gap: 2px !important;
        }
        [data-testid="stTabs"] > div > div[data-baseweb="tab-panel"] { padding-bottom: 92px !important; }
        [data-testid="stTabs"] [data-testid="stTabs"] div[data-baseweb="tab"] {
            flex-direction: row !important; padding: 6px 10px !important; font-size: 0.76rem !important;
        }
        .hero-value { font-size: 2.1rem; }
    }
</style>
""", unsafe_allow_html=True)

SIGNAL_COLOR = {
    'STRONG BUY': 'var(--gain)', 'BUY': 'var(--gain)', 'WATCH': 'var(--info)',
    'HOLD': 'var(--text-dim)', 'SELL': 'var(--loss)', 'STRONG SELL': 'var(--loss)', 'AVOID': 'var(--loss)',
}
SIGNAL_LABEL_HE = {
    'STRONG BUY': 'קנייה חזקה', 'BUY': 'קנייה', 'WATCH': 'מעקב', 'HOLD': 'החזק',
    'SELL': 'מכירה', 'STRONG SELL': 'מכירה חזקה', 'AVOID': 'הימנע',
}

# ==========================================
# GOOGLE SHEETS + SESSION STATE
# ==========================================
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

conn = st.connection("gsheets", type=GSheetsConnection)

def load_cloud_portfolio():
    defaults = pd.DataFrame([
        {'Ticker': 'MSFT', 'Quantity': 7.4156, 'PurchasePrice': 371.17},
        {'Ticker': 'VOO', 'Quantity': 4.5496, 'PurchasePrice': 683.57},
        {'Ticker': 'META', 'Quantity': 3.0, 'PurchasePrice': 559.56},
        {'Ticker': 'ESLT', 'Quantity': 1.0, 'PurchasePrice': 780.25},
        {'Ticker': 'MU', 'Quantity': 2.0, 'PurchasePrice': 993.89},
    ]).set_index('Ticker')
    try:
        df = conn.read(worksheet="Portfolio", ttl=0)
        if df is None or df.empty:
            raise ValueError("Portfolio worksheet is empty")
        df = df.dropna(subset=['Ticker'])
        df = df[df['Ticker'].astype(str).str.strip() != ""]
        return df.set_index('Ticker')
    except Exception:
        if os.path.exists("portfolio_data.csv"):
            try:
                local = pd.read_csv("portfolio_data.csv")
                required = {'Ticker', 'Quantity', 'PurchasePrice'}
                if required.issubset(local.columns):
                    local = local.dropna(subset=['Ticker'])
                    local = local[local['Ticker'].astype(str).str.strip() != ""]
                    return local.set_index('Ticker')
            except (OSError, ValueError, pd.errors.ParserError):
                pass
        return defaults

def save_cloud_portfolio(df):
    try:
        conn.update(worksheet="Portfolio", data=df.reset_index())
        st.cache_data.clear()
        return True
    except Exception:
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
    except Exception:
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
    except Exception:
        with open("config.json", "w") as f:
            json.dump(cfg, f)

def load_watchlist():
    try:
        df = conn.read(worksheet="Watchlist", ttl=0)
        if df is None or df.empty:
            return pd.DataFrame(columns=['Ticker','Notes','AlertHigh','AlertLow'])
        return df.dropna(subset=['Ticker'])
    except Exception:
        return pd.DataFrame(columns=['Ticker','Notes','AlertHigh','AlertLow'])

def save_watchlist(df):
    try:
        conn.update(worksheet="Watchlist", data=df)
    except Exception:
        pass

def apply_trade(portfolio, ticker, action, qty, price):
    """Buy/sell against the in-memory portfolio. Returns updated df."""
    if action == "Buy":
        if ticker in portfolio.index:
            oq, op = portfolio.loc[ticker, ['Quantity', 'PurchasePrice']]
            nq = oq + qty
            portfolio.loc[ticker] = [nq, ((oq * op) + (qty * price)) / nq]
        else:
            portfolio.loc[ticker] = [qty, price]
    else:
        if ticker in portfolio.index:
            nq = max(0, portfolio.loc[ticker, 'Quantity'] - qty)
            if nq == 0:
                portfolio = portfolio.drop(ticker)
            else:
                portfolio.loc[ticker, 'Quantity'] = nq
    return portfolio

def clean_portfolio_editor(data):
    required = ['Ticker', 'Quantity', 'PurchasePrice']
    cleaned = data.reindex(columns=required).copy()
    cleaned['Ticker'] = cleaned['Ticker'].fillna('').astype(str).str.strip().str.upper()
    cleaned['Quantity'] = pd.to_numeric(cleaned['Quantity'], errors='coerce').fillna(0)
    cleaned['PurchasePrice'] = pd.to_numeric(cleaned['PurchasePrice'], errors='coerce').fillna(0)
    cleaned = cleaned[(cleaned['Ticker'] != '') & (cleaned['Quantity'] > 0) & (cleaned['PurchasePrice'] > 0)]
    cleaned = cleaned.drop_duplicates('Ticker', keep='last').set_index('Ticker')
    return cleaned[['Quantity', 'PurchasePrice']]

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_cloud_portfolio()
if 'config' not in st.session_state:
    st.session_state.config = load_cloud_config()
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# ==========================================
# TICKER UNIVERSES
# ==========================================
CORE_UNIVERSE = [
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','AVGO','JPM','V',
    'MA','UNH','JNJ','PG','KO','PEP','COST','HD','WMT','XOM',
    'CVX','LLY','ABBV','MRK','BAC','ADBE','CRM','NFLX','AMD','QCOM',
    'ORCL','DIS','NKE','MCD','SBUX','TXN','INTC','CSCO','ABT','TMO',
]

NASDAQ_100 = [
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AVGO','COST','NFLX',
    'AMD','CSCO','QCOM','INTC','INTU','AMGN','CMCSA','AMAT','TXN','BKNG',
    'VRTX','ISRG','ADP','SBUX','ADI','GILD','REGN','MU','LRCX','KLAC',
    'PANW','SNPS','CDNS','MELI','CRWD','FTNT','PCAR','ORLY','CTAS','NXPI',
]

TA35_TICKERS = [
    'TEVA.TA','ICL.TA','NICE.TA','CHECK.TA','ESLT.TA','BCOM.TA',
    'LUMI.TA','HARL.TA','PAZA.TA','DSCT.TA','PHOE.TA','FIBR.TA',
    'AZRG.TA','ELAL.TA','MZTF.TA','SPNS.TA','MNRT.TA','RTEN.TA'
]

DIVIDEND_STOCKS = [
    'JNJ','PG','KO','PEP','MRK','ABBV','CVX','XOM','T','VZ',
    'MMM','IBM','MO','PM','O','MAIN','WPC','IIPR','AGNC',
    'EPD','MMP','ET','MPLX','PAA','WMB','OKE','KMI','ENB','TRP'
]

ETF_UNIVERSE = [
    'SPY','QQQ','DIA','IWM','VTI','VOO','ARKK','XLK','XLF','XLE',
    'XLV','XLI','XLY','XLP','XLU','XLRE','GLD','SLV','USO','TLT',
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
        rate = float(yf.Ticker("USDILS=X").fast_info['lastPrice'])
        if np.isfinite(rate) and rate > 0:
            return rate
    except Exception:
        pass
    try:
        response = requests.get(
            "https://api.frankfurter.app/latest?from=USD&to=ILS",
            headers=session.headers, timeout=8,
        )
        response.raise_for_status()
        rate = float(response.json()['rates']['ILS'])
        if np.isfinite(rate) and rate > 0:
            return rate
    except (OSError, ValueError, KeyError, requests.RequestException):
        pass
    return None

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
                    prev = series.iloc[-2]
                    chg = (today - prev) / prev * 100
                    results[t] = {
                        'name': MARKET_TICKERS[t], 'price': float(today), 'change_pct': float(chg),
                        'color': 'var(--gain)' if chg >= 0 else 'var(--loss)',
                        'arrow': '▲' if chg >= 0 else '▼',
                    }
            except Exception:
                pass
    except Exception:
        pass
    return results

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        resp = requests.get(url, headers=session.headers, timeout=10)
        table = pd.read_html(StringIO(resp.text))[0]
        return [t.replace('.', '-') for t in table['Symbol'].tolist()]
    except Exception:
        return ['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','V','UNH']

def compute_advanced_indicators(hist):
    close, high, low, volume = hist['Close'], hist['High'], hist['Low'], hist['Volume']
    ind = {}

    for p in [5, 10, 20, 50, 100, 200]:
        if len(close) >= p:
            ind[f'sma{p}'] = close.rolling(p).mean().iloc[-1]
            ind[f'ema{p}'] = close.ewm(span=p, adjust=False).mean().iloc[-1]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    ind['rsi'] = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    ind['macd'] = macd_line.iloc[-1]
    ind['macd_signal'] = signal_line.iloc[-1]
    ind['macd_hist'] = (macd_line - signal_line).iloc[-1]
    ind['macd_crossover'] = bool(
        (macd_line.iloc[-1] > signal_line.iloc[-1]) and (macd_line.iloc[-2] <= signal_line.iloc[-2])
    )

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper, bb_lower, bb_mid = (sma20 + 2*std20).iloc[-1], (sma20 - 2*std20).iloc[-1], sma20.iloc[-1]
    ind['bb_upper'], ind['bb_lower'] = bb_upper, bb_lower
    ind['bb_pct'] = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    ind['bb_width'] = (bb_upper - bb_lower) / bb_mid if bb_mid > 0 else 0

    if len(close) >= 14:
        low14, high14 = low.rolling(14).min(), high.rolling(14).max()
        k = 100 * (close - low14) / (high14 - low14 + 1e-9)
        ind['stoch_k'] = k.iloc[-1]
        ind['stoch_d'] = k.rolling(3).mean().iloc[-1]
    else:
        ind['stoch_k'] = ind['stoch_d'] = 50

    if len(hist) >= 14:
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        ind['atr'] = tr.rolling(14).mean().iloc[-1]
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

    high_1y, low_1y = close.max(), close.min()
    diff = high_1y - low_1y
    for lv, key in [(0.236,'fib_236'),(0.382,'fib_382'),(0.5,'fib_500'),(0.618,'fib_618'),(0.786,'fib_786')]:
        ind[key] = high_1y - diff * lv
    ind['pct_from_high'] = ((close.iloc[-1] - high_1y) / high_1y) * 100
    ind['pct_from_low'] = ((close.iloc[-1] - low_1y) / low_1y) * 100

    for n, days in [('mom_1m',20),('mom_3m',63),('mom_6m',126),('mom_1y',252)]:
        ind[n] = ((close.iloc[-1] / close.iloc[-days]) - 1)*100 if len(close) >= days else 0

    if len(close) >= 2:
        ind['day_change_pct'] = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
    else:
        ind['day_change_pct'] = 0.0

    sma20v = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
    sma50v = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20v
    sma200v = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma50v
    curr = close.iloc[-1]
    ind['sma20'], ind['sma50'], ind['sma200'] = sma20v, sma50v, sma200v

    if sma20v > sma50v > sma200v and curr > sma20v:
        ind['trend'], ind['trend_score'] = "Strong Uptrend", 5
    elif sma20v > sma50v and curr > sma20v:
        ind['trend'], ind['trend_score'] = "Moderate Uptrend", 4
    elif sma20v < sma50v < sma200v and curr < sma20v:
        ind['trend'], ind['trend_score'] = "Strong Downtrend", 1
    elif sma20v < sma50v and curr < sma20v:
        ind['trend'], ind['trend_score'] = "Moderate Downtrend", 2
    else:
        ind['trend'], ind['trend_score'] = "Consolidation", 3

    if len(close) >= 20:
        recent = close.iloc[-60:] if len(close) >= 60 else close
        ind['support'] = float(recent.rolling(10).min().dropna().iloc[-1])
        ind['resistance'] = float(recent.rolling(10).max().dropna().iloc[-1])
    else:
        ind['support'], ind['resistance'] = float(low.iloc[-1]), float(high.iloc[-1])

    return ind

def compute_composite_score(ind, info):
    score, reasons = 50, []
    ts = ind.get('trend_score', 3)
    score += (ts - 3) * 4
    if ts >= 5: reasons.append("מגמת עלייה חזקה")
    elif ts <= 2: reasons.append("מגמת ירידה")

    rsi = ind.get('rsi', 50)
    if rsi < 30: score += 10; reasons.append("RSI במכירת יתר")
    elif rsi < 40: score += 6; reasons.append("אזור מכירת יתר")
    elif 40 <= rsi <= 60: score += 4
    elif rsi > 80: score -= 12; reasons.append("RSI בקניית יתר קיצונית")
    elif rsi > 70: score -= 8; reasons.append("קניית יתר")

    if ind.get('macd_crossover'): score += 8; reasons.append("חציית MACD חיובית")
    elif ind.get('macd', 0) > ind.get('macd_signal', 0): score += 3
    else: score -= 4; reasons.append("MACD מתחת לסיגנל")

    vr = ind.get('vol_ratio', 1)
    if vr > 2.0 and ts >= 4: score += 8; reasons.append("נפח חריג במגמת עלייה")
    elif vr > 1.5 and ts >= 4: score += 5
    elif vr > 1.5 and ts <= 2: score -= 6; reasons.append("נפח גבוה במגמת ירידה")

    m1, m3 = ind.get('mom_1m', 0), ind.get('mom_3m', 0)
    if m1 > 10 and m3 > 20: score += 10; reasons.append("מומנטום חזק מאוד")
    elif m1 > 5 and m3 > 10: score += 6; reasons.append("מומנטום חיובי")
    elif m1 > 0 and m3 > 0: score += 3
    elif m1 < -15: score -= 10; reasons.append("מומנטום שלילי חד")
    elif m1 < -8: score -= 6

    pe = info.get('trailingPE', 0) or 0
    if 0 < pe < 12: score += 10; reasons.append("P/E נמוך מאוד — ערך")
    elif 0 < pe < 20: score += 6; reasons.append("P/E נמוך")
    elif 20 <= pe < 35: score += 2
    elif pe > 60: score -= 8; reasons.append("P/E גבוה מאוד")
    elif pe > 40: score -= 4

    growth = (info.get('revenueGrowth', 0) or 0) * 100
    if growth > 30: score += 8; reasons.append("צמיחת הכנסות > 30%")
    elif growth > 15: score += 5
    elif growth > 5: score += 2
    elif growth < 0: score -= 6; reasons.append("ירידה בהכנסות")

    roe = (info.get('returnOnEquity', 0) or 0) * 100
    if roe > 25: score += 5; reasons.append("ROE מצוין")
    elif roe > 15: score += 3
    elif roe < 0: score -= 4

    rec = info.get('recommendationKey', 'none')
    if rec in ['strong_buy','buy']: score += 5; reasons.append("קונצנזוס אנליסטים: קנייה")
    elif rec in ['sell','strong_sell']: score -= 6; reasons.append("קונצנזוס אנליסטים: מכירה")

    if ind.get('obv_trend', 0) == 1 and ts >= 4: score += 3

    sk = ind.get('stoch_k', 50)
    if sk < 20: score += 4
    elif sk > 85: score -= 4

    return min(100, max(0, int(score))), reasons

def get_signal(score, rsi, trend_score):
    if score >= 75 and rsi < 65 and trend_score >= 5: return "STRONG BUY"
    elif score >= 65 and trend_score >= 4: return "BUY"
    elif score >= 55 and trend_score >= 3: return "WATCH"
    elif score <= 25 and trend_score <= 1: return "STRONG SELL"
    elif score <= 38: return "SELL"
    elif 43 <= score <= 57: return "HOLD"
    elif score > 57: return "WATCH"
    else: return "AVOID"

def valuation_tag(d):
    pe, peg, upside = d.get('pe', 0), d.get('peg', 0), d.get('target_upside', 0)
    if pe > 0 and pe < 18 and (not peg or peg < 1.5) and upside > 8:
        return "מתומחר בזול", 'var(--gain)'
    if pe > 45 or (peg and peg > 3):
        return "מתומחר ביוקר", 'var(--loss)'
    return "תמחור הוגן", 'var(--text-dim)'

def playbook_line(d):
    """Rules-based, one-line read on what the position's chart/valuation is saying — not investment advice."""
    sig, rsi = d.get('signal', 'HOLD'), d.get('rsi', 50)
    price, res, sup = d.get('price', 0), d.get('resistance', 0), d.get('support', 0)
    parts = []
    if sig in ('STRONG BUY', 'BUY'):
        parts.append(f"מגמה חיובית — תמיכה קרובה ${sup:.2f}")
    elif sig in ('STRONG SELL', 'SELL'):
        parts.append("חולשה טכנית — שקול stop הדוק יותר")
    elif rsi > 70:
        parts.append("קניית יתר — ייתכן קירור בטווח הקצר")
    elif rsi < 30:
        parts.append("מכירת יתר — אזור התאוששות אפשרי")
    else:
        parts.append("ללא איתות ברור כרגע")
    if res and price >= res * 0.98:
        parts.append(f"קרוב להתנגדות ${res:.2f}")
    elif sup and price <= sup * 1.02:
        parts.append(f"קרוב לתמיכה ${sup:.2f}")
    return " · ".join(parts)

@st.cache_data(ttl=900)
def fetch_expert_data(tickers_to_fetch):
    data = {}
    valid_list = list(dict.fromkeys(t for t in tickers_to_fetch if isinstance(t, str) and t.strip()))
    if not valid_list:
        return data

    try:
        raw = yf.download(valid_list, period="2y", auto_adjust=True, progress=False, threads=False)
    except Exception:
        return data

    def fetch_single(t):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(-1):
                    return None
                hist = raw.xs(t, axis=1, level=-1).dropna(how='all')
            else:
                hist = raw.dropna(how='all') if len(valid_list) == 1 else pd.DataFrame()
            if len(hist) < 50:
                return None
            info = {}
            curr = hist['Close'].iloc[-1]
            ind = compute_advanced_indicators(hist)
            score, reasons = compute_composite_score(ind, info)
            signal = get_signal(score, ind['rsi'], ind['trend_score'])
            div_yield = (info.get('dividendYield') or 0) * 100

            return t, {
                'price': float(curr), 'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'), 'name': info.get('longName', t),
                'pe': info.get('trailingPE', 0) or 0, 'forward_pe': info.get('forwardPE', 0) or 0,
                'peg': info.get('pegRatio', 0) or 0, 'ps': info.get('priceToSalesTrailing12Months', 0) or 0,
                'pb': info.get('priceToBook', 0) or 0, 'beta': info.get('beta', 1.0) or 1.0,
                'div': div_yield, 'market_cap': info.get('marketCap', 0) or 0,
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
                'target_low': info.get('targetLowPrice', curr) or curr,
                'target_high': info.get('targetHighPrice', curr) or curr,
                'num_analysts': info.get('numberOfAnalystOpinions', 0) or 0,
                'short_pct': (info.get('shortPercentOfFloat') or 0) * 100,
                'inst_own': (info.get('institutionsPercentHeld') or 0) * 100,
                'currency': "ILS" if str(t).endswith(".TA") else "USD",
                **ind, 'score': score, 'reasons': reasons, 'signal': signal,
                'signal_color': SIGNAL_COLOR.get(signal, 'var(--text-dim)'),
            }
        except Exception:
            return None

    for ticker in valid_list:
        result = fetch_single(ticker)
        if result:
            data[result[0]] = result[1]
    return data

@st.cache_data(ttl=1800)
def compute_portfolio_analytics(tickers):
    try:
        raw = yf.download(list(tickers) + ['SPY'], period="1y", auto_adjust=True, progress=False)
        closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
        returns = closes.pct_change().dropna()
        corr = returns[[c for c in returns.columns if c in tickers]].corr() if len(tickers) > 1 else None
        sharpe, max_dd, var95, spy_corr = {}, {}, {}, {}
        for col in tickers:
            if col not in returns.columns:
                continue
            r = returns[col]
            if r.std() > 0:
                sharpe[col] = float((r.mean()*252 - 0.05) / (r.std()*np.sqrt(252)))
            roll_max = closes[col].cummax()
            max_dd[col] = float(((closes[col] - roll_max) / roll_max).min() * 100)
            var95[col] = float(np.percentile(r.dropna(), 5) * 100)
            if 'SPY' in returns.columns:
                spy_corr[col] = float(r.corr(returns['SPY']))
        spy_ret_1y = float((closes['SPY'].iloc[-1] / closes['SPY'].iloc[0] - 1) * 100) if 'SPY' in closes.columns else 0
        return corr, sharpe, max_dd, var95, spy_corr, spy_ret_1y
    except Exception:
        return None, {}, {}, {}, {}, 0

@st.cache_data(ttl=1800)
def get_stock_news(ticker, limit=6):
    try:
        news = yf.Ticker(ticker).news
        return news[:limit] if news else []
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_next_earnings(ticker):
    """Returns (date, eps_estimate) for the next scheduled earnings report, or (None, None)."""
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if isinstance(cal, dict):
            ed = cal.get('Earnings Date')
            if isinstance(ed, list) and ed:
                ed = ed[0]
            return ed, cal.get('EPS Estimate')
        if cal is not None and hasattr(cal, 'empty') and not cal.empty:
            if 'Earnings Date' in cal.index:
                val = cal.loc['Earnings Date']
                ed = val.iloc[0] if hasattr(val, 'iloc') else val
                return ed, None
            col = cal.iloc[:, 0]
            return (col.get('Earnings Date'), col.get('EPS Estimate')) if hasattr(col, 'get') else (None, None)
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=3600)
def get_recent_earnings(ticker):
    """Returns dict with last reported EPS actual/estimate/surprise, or None."""
    try:
        hist_earn = yf.Ticker(ticker).earnings_history
        if hist_earn is not None and not hist_earn.empty:
            last = hist_earn.iloc[-1]
            idx = hist_earn.index[-1]
            return {
                'date': idx if hasattr(idx, 'strftime') else None,
                'eps_actual': last.get('epsActual'),
                'eps_estimate': last.get('epsEstimate'),
                'surprise_pct': last.get('surprisePercent'),
            }
    except Exception:
        pass
    return None

def build_earnings_board(tickers):
    upcoming, recent = [], []
    for t in tickers:
        ed, eps_est = get_next_earnings(t)
        if ed is not None:
            try:
                ed_ts = pd.Timestamp(ed).tz_localize(None) if pd.Timestamp(ed).tzinfo else pd.Timestamp(ed)
                days_out = (ed_ts.normalize() - pd.Timestamp.now().normalize()).days
                if days_out >= -1:
                    upcoming.append({'ticker': t, 'date': ed_ts, 'days_out': days_out, 'eps_est': eps_est})
            except Exception:
                pass
        rec = get_recent_earnings(t)
        if rec and rec.get('date') is not None:
            recent.append({'ticker': t, **rec})
    upcoming.sort(key=lambda x: x['days_out'])
    recent.sort(key=lambda x: x['date'], reverse=True)
    return upcoming, recent[:8]

def compute_market_sentiment(market_data):
    vix = market_data.get('^VIX', {}).get('price', 20)
    spy_chg = market_data.get('SPY', {}).get('change_pct', 0)
    qqq_chg = market_data.get('QQQ', {}).get('change_pct', 0)
    btc_chg = market_data.get('BTC-USD', {}).get('change_pct', 0)

    vix_score = 100 if vix < 12 else 80 if vix < 16 else 60 if vix < 20 else 40 if vix < 25 else 20 if vix < 30 else 5
    trend_score = max(0, min(100, 50 + (spy_chg + qqq_chg) * 3))
    btc_score = 70 if btc_chg > 3 else 55 if btc_chg > 0 else 40 if btc_chg > -3 else 25

    score = max(0, min(100, int(vix_score * 0.5 + trend_score * 0.3 + btc_score * 0.2)))
    if score >= 80: label, color = "תיאבון סיכון גבוה", 'var(--gain)'
    elif score >= 60: label, color = "אופטימי", 'var(--gain)'
    elif score >= 40: label, color = "ניטרלי", 'var(--gold)'
    elif score >= 20: label, color = "חשש", 'var(--loss)'
    else: label, color = "פחד קיצוני", 'var(--loss)'
    return score, label, color

# ==========================================
# PORTFOLIO SCREENSHOT IMPORT (best-effort OCR)
# ==========================================
TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')
NUMBER_PATTERN = re.compile(r'\d+(?:[.,]\d+)?')
EXCLUDE_WORDS = {
    'THE','AND','FOR','QTY','AVG','USD','ILS','BUY','SELL','NEW','ETF','ALL','TOP',
    'LOW','HIGH','DAY','ASK','BID','NAV','CASH','TOTAL','VALUE','PRICE','SHARES',
    'COST','GAIN','LOSS','TODAY','YTD','MKT','CAP',
}

def parse_portfolio_image(image):
    """Best-effort extraction of ticker/qty/price rows from a broker screenshot.
    Always requires manual review before it touches the real portfolio."""
    if not OCR_AVAILABLE:
        return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice'])
    image = image.convert('RGB')
    image.thumbnail((1800, 1800))
    text = pytesseract.image_to_string(image, config='--psm 6')
    normalized = text.upper().replace('—', '-').replace('–', '-')
    known_tickers = ['MSFT', 'VOO', 'META', 'ESLT', 'MU', 'NUVB']
    rows = []
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    for line_number, line in enumerate(lines):
        line = line.strip()
        tickers_found = [t for t in known_tickers if re.search(rf'\b{t}\b', line)]
        if not tickers_found:
            tickers_found = [w for w in TICKER_PATTERN.findall(line) if w not in EXCLUDE_WORDS]
        context = ' '.join(lines[line_number:line_number + 8])
        numbers_found = [n.replace(',', '') for n in NUMBER_PATTERN.findall(context)]
        if not tickers_found or not numbers_found:
            continue
        nums = []
        for n in numbers_found:
            try:
                nums.append(float(n))
            except ValueError:
                pass
        if not nums:
            continue
        qty_match = re.search(r'(?:QTY|QUANTITY|כמות)\D{0,20}(\d+(?:\.\d+)?)', context, re.IGNORECASE)
        price_match = re.search(r'(?:AVG|AVERAGE|BUY|PURCHASE|קניה|קנייה)\D{0,30}(\d[\d,.]*)', context, re.IGNORECASE)
        qty = float(qty_match.group(1).replace(',', '')) if qty_match else min(nums, key=lambda n: abs(n - round(n)))
        price = float(price_match.group(1).replace(',', '')) if price_match else (nums[1] if len(nums) > 1 else 0.0)
        if price <= 0 or qty <= 0:
            continue
        rows.append({'Ticker': tickers_found[0], 'Quantity': qty, 'PurchasePrice': price})
    return pd.DataFrame(rows).drop_duplicates('Ticker', keep='first')

# ==========================================
# SHARED UI HELPERS
# ==========================================
def ledger_row(ticker, subtitle, price_str, sub_str, sub_color, tag_text, tag_color, caption=None):
    cap_html = f'<div class="row-caption">{caption}</div>' if caption else ''
    return (
        f'<div class="ledger-row">'
        f'  <div><div class="row-ticker">{ticker}</div><div class="row-name">{subtitle}</div></div>'
        f'  <div style="text-align:center"><div class="row-price">{price_str}</div>'
        f'    <div class="row-sub" style="color:{sub_color}">{sub_str}</div></div>'
        f'  <div style="text-align:right"><div class="row-tag" style="color:{tag_color}">'
        f'    <span class="dot" style="background:{tag_color}"></span>{tag_text}</div>{cap_html}</div>'
        f'</div>'
    )

def render_ledger(rows_html):
    st.markdown('<div>' + ''.join(rows_html) + '</div>', unsafe_allow_html=True)

def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
usd_ils_rate = get_usd_ils()
if usd_ils_rate is None:
    st.error("לא ניתן לקבל כרגע שער USD/ILS חי. נסה/י לרענן בעוד רגע.")
    st.stop()
market_data = get_market_overview()

with st.sidebar:
    st.markdown("**AlonStocks Pro**")
    st.caption("מסוף מסחר אישי")
    st.markdown("---")

    st.markdown("**דופק שוק**")
    for t, d in list(market_data.items())[:4]:
        sign = "+" if d['change_pct'] >= 0 else ""
        st.markdown(
            f"{d['name']} — ${d['price']:,.2f} "
            f"<span style='color:{d['color']};font-family:IBM Plex Mono, monospace'>{sign}{d['change_pct']:.2f}%</span>",
            unsafe_allow_html=True
        )
    st.markdown("---")

    st.markdown("**קופת מזומנים**")
    n_usd = st.number_input("מזומן בדולר $", value=float(st.session_state.config['cash_usd']))
    n_ils = st.number_input("מזומן בשקל ₪", value=float(st.session_state.config['cash_ils']))
    init_inv = st.number_input("השקעה ראשונית ($)", value=float(st.session_state.config.get('initial_investment', 7000)))
    if (n_usd != st.session_state.config['cash_usd'] or n_ils != st.session_state.config['cash_ils']
            or init_inv != st.session_state.config.get('initial_investment', 7000)):
        st.session_state.config = {"cash_usd": n_usd, "cash_ils": n_ils, "initial_investment": init_inv}
        save_cloud_config(st.session_state.config)

    st.markdown("---")
    quick_ticker = st.text_input("חיפוש מהיר — סימול").upper()

    st.markdown("---")
    if st.button("רענן נתונים"):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"עדכון אחרון: {datetime.now().strftime('%H:%M')}")
    st.markdown("---")
    st.caption("המידע מוצג להמחשה טכנית בלבד ואינו מהווה ייעוץ השקעות.")

# ==========================================
# MAIN DATA
# ==========================================
p_tickers = [t for t in st.session_state.portfolio.index if isinstance(t, str) and t.strip()]
watch_tickers = [t for t in st.session_state.watchlist['Ticker'].tolist() if isinstance(t, str) and t.strip()] \
    if not st.session_state.watchlist.empty else []
all_analyze = list(set(p_tickers + watch_tickers + ['SPY', 'QQQ', '^VIX']))
if quick_ticker:
    all_analyze.append(quick_ticker)
m_data = fetch_expert_data(tuple(all_analyze))
core_data = {}

# ==========================================
# TABS
# ==========================================
tabs = st.tabs(["היום", "תיק", "טכני", "הזדמנויות", "רווחים וחדשות", "סיכון וכלים"])
t_brief, t_port, t_tech, t_scan, t_earn, t_risk = tabs

# ==========================================
# TAB: BRIEF (HOME)
# ==========================================
with t_brief:
    stock_val_usd, cost_basis = 0.0, 0.0
    day_pl_usd = 0.0
    holding_rows = []
    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]
            qty, bp = row['Quantity'], row['PurchasePrice']
            fx = usd_ils_rate if d['currency'] == "ILS" else 1
            v_u = d['price'] * qty / fx
            stock_val_usd += v_u
            cost_basis += bp * qty / fx
            day_pl_usd += d.get('day_change_pct', 0) / 100 * v_u
            holding_rows.append((t, d, qty, bp))

    total_usd = stock_val_usd + n_usd + (n_ils / usd_ils_rate)
    init_inv_val = st.session_state.config.get('initial_investment', 7000)
    profit_usd = total_usd - init_inv_val

    st.markdown('<div class="hero-label">שווי תיק כולל</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-value">${total_usd:,.0f}</div>', unsafe_allow_html=True)
    day_color = 'var(--gain)' if day_pl_usd >= 0 else 'var(--loss)'
    tot_color = 'var(--gain)' if profit_usd >= 0 else 'var(--loss)'
    st.markdown(
        f'<div class="hero-change">'
        f'<span style="color:{day_color}">{"+" if day_pl_usd>=0 else ""}{day_pl_usd:,.0f}$ היום</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'<span style="color:{tot_color}">{"+" if profit_usd>=0 else ""}{profit_usd:,.0f}$ סה"כ</span>'
        f'</div>', unsafe_allow_html=True
    )

    sent_score, sent_label, sent_color = compute_market_sentiment(market_data)
    st.markdown(
        f'<div style="margin-top:16px;font-size:0.8rem;color:var(--text-dim)">'
        f'סנטימנט שוק: <span style="color:{sent_color};font-weight:600">{sent_label}</span> ({sent_score}/100)</div>'
        f'<div class="sentiment-bar"><div class="sentiment-marker" style="left:{sent_score}%"></div></div>',
        unsafe_allow_html=True
    )

    # ── Needs attention ──
    attention = []
    for t, d, qty, bp in holding_rows:
        reasons = []
        if d.get('signal') in ('STRONG BUY', 'STRONG SELL'):
            reasons.append(SIGNAL_LABEL_HE[d['signal']])
        if d.get('rsi', 50) > 75:
            reasons.append("RSI קיצוני למעלה")
        elif d.get('rsi', 50) < 25:
            reasons.append("RSI קיצוני למטה")
        res, sup, price = d.get('resistance', 0), d.get('support', 0), d.get('price', 0)
        if res and price >= res * 0.98:
            reasons.append("נוגע בהתנגדות")
        if sup and price <= sup * 1.02:
            reasons.append("נוגע בתמיכה")
        if reasons:
            attention.append((t, d, reasons))

    if attention:
        section("דורש תשומת לב", "פוזיציות עם איתות טכני משמעותי כרגע")
        for t, d, reasons in attention[:6]:
            st.markdown(
                f'<div class="attn-strip"><div class="row-ticker">{t}</div>'
                f'<div class="row-name">{" · ".join(reasons)} — ${d["price"]:,.2f}</div></div>',
                unsafe_allow_html=True
            )

    # ── Earnings this week (holdings + watchlist) ──
    brief_tickers = []
    if brief_tickers:
        with st.spinner("בודק תאריכי דוחות..."):
            upcoming, recent = build_earnings_board(tuple(brief_tickers))
        soon = [u for u in upcoming if u['days_out'] <= 7]
        if soon:
            section("דוחות רווחים השבוע", "עלולים להגביר תנודתיות בפוזיציה")
            rows_html = []
            for u in soon:
                when = "היום" if u['days_out'] == 0 else f"בעוד {u['days_out']} ימים"
                eps = f"תחזית EPS ${u['eps_est']:.2f}" if u.get('eps_est') else "—"
                rows_html.append(ledger_row(
                    u['ticker'], eps, when, u['date'].strftime('%d.%m'), 'var(--text-faint)',
                    "קרוב", 'var(--gold)'
                ))
            render_ledger(rows_html)

    # ── Undervalued & momentum picks from the core universe ──
    if core_data:
        undervalued = sorted(
            [(t, d) for t, d in core_data.items() if valuation_tag(d)[0] == "מתומחר בזול"],
            key=lambda x: x[1]['score'], reverse=True
        )[:5]
        momentum = sorted(
            [(t, d) for t, d in core_data.items() if d.get('signal') in ('STRONG BUY', 'BUY')],
            key=lambda x: x[1]['score'], reverse=True
        )[:5]

        if undervalued:
            section("מניות במחיר אטרקטיבי", "מתוך רשימת המניות הגדולות שנסרקה ברקע")
            rows_html = [
                ledger_row(
                    t, d.get('sector', '')[:20], f"${d['price']:,.2f}",
                    f"P/E {d['pe']:.0f}" if d['pe'] else "—", 'var(--text-faint)',
                    f"{d['score']}/100", SIGNAL_COLOR.get(d['signal'], 'var(--text-dim)'),
                    caption=playbook_line(d)
                ) for t, d in undervalued
            ]
            render_ledger(rows_html)

        if momentum:
            section("מומנטום חיובי כרגע")
            rows_html = [
                ledger_row(
                    t, d.get('sector', '')[:20], f"${d['price']:,.2f}",
                    f"{'+' if d.get('mom_1m',0)>=0 else ''}{d.get('mom_1m',0):.1f}% / חודש",
                    'var(--gain)' if d.get('mom_1m',0)>=0 else 'var(--loss)',
                    SIGNAL_LABEL_HE.get(d['signal'], d['signal']), SIGNAL_COLOR.get(d['signal'], 'var(--text-dim)'),
                ) for t, d in momentum
            ]
            render_ledger(rows_html)

# ==========================================
# TAB: PORTFOLIO
# ==========================================
with t_port:
    section("עריכה מהירה של התיק", "שנה/י כמות או מחיר, הוסף/י שורה חדשה, או מחק/י שורה ואז שמור/י.")
    editor_source = st.session_state.portfolio.reset_index()
    edited_portfolio = st.data_editor(
        editor_source, num_rows="dynamic", use_container_width=True,
        column_config={
            'Ticker': st.column_config.TextColumn('סימול', required=True),
            'Quantity': st.column_config.NumberColumn('כמות', min_value=0, step=0.0001, format='%.4f'),
            'PurchasePrice': st.column_config.NumberColumn('מחיר קנייה', min_value=0, step=0.01, format='%.2f'),
        }, hide_index=True, key='portfolio_editor'
    )
    if st.button("שמור שינויים בתיק", type="primary", use_container_width=True):
        cleaned = clean_portfolio_editor(edited_portfolio)
        st.session_state.portfolio = cleaned
        if save_cloud_portfolio(cleaned):
            st.success("התיק נשמר")
            st.rerun()
        st.error("לא ניתן לשמור כרגע. בדוק/י את חיבור Google Sheets.")

    stock_val_usd, rows, sector_weights, cost_basis = 0.0, [], {}, 0.0

    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]
            qty, bp = row['Quantity'], row['PurchasePrice']
            fx = usd_ils_rate if d['currency'] == "ILS" else 1
            v_u = d['price'] * qty / fx
            stock_val_usd += v_u
            cost_basis += bp * qty / fx
            sector_weights[d['sector']] = sector_weights.get(d['sector'], 0) + v_u
            pl_pct = ((d['price'] - bp) / bp) * 100
            pl_usd = (d['price'] - bp) * qty
            div_annual = v_u * d.get('div', 0) / 100
            rows.append({
                "Ticker": t, "Name": d.get('name', t)[:20], "Sector": d['sector'], "Qty": qty,
                "Price": d['price'], "Buy@": bp, "P&L %": round(pl_pct, 2), "P&L $": round(pl_usd, 2),
                "Value $": round(v_u, 2), "Annual Div $": round(div_annual, 2), "Score": d.get('score', 50),
                "Signal": d.get('signal', 'HOLD'), "RSI": round(d.get('rsi', 50), 1), "Trend": d.get('trend', '-'),
                "Target $": round(d.get('target_price', d['price']), 2), "Upside %": round(d.get('target_upside', 0), 1),
                "Beta": round(d.get('beta', 1), 2), "Analyst": d.get('analyst', 'none'),
            })

    total_usd = stock_val_usd + n_usd + (n_ils / usd_ils_rate)
    init_inv_val = st.session_state.config.get('initial_investment', 7000)
    profit_usd = total_usd - init_inv_val
    total_div = sum(r['Annual Div $'] for r in rows)
    total_return_pct = (profit_usd / init_inv_val * 100) if init_inv_val > 0 else 0

    r1c1, r1c2 = st.columns(2)
    r1c1.metric("שווי כולל", f"${total_usd:,.0f}", f"{'+' if profit_usd>=0 else ''}{profit_usd:,.0f}")
    r1c2.metric("תשואה", f"{total_return_pct:.1f}%", f"{'+' if profit_usd>=0 else ''}${profit_usd:,.0f}")
    r2c1, r2c2 = st.columns(2)
    r2c1.metric("USD/ILS", f"₪{usd_ils_rate:.3f}")
    r2c2.metric("דיבידנד שנתי", f"${total_div:.0f}")

    with st.expander("פעולה מהירה: קנייה או מכירה"):
        with st.form("quick_trade_main", clear_on_submit=True):
            qc1, qc2 = st.columns([3, 2])
            qt_ticker = qc1.text_input("סימול מניה", placeholder="AAPL / TSLA / MSFT").upper()
            qt_action = qc2.selectbox("פעולה", ["קנייה", "מכירה"])
            qc3, qc4 = st.columns(2)
            qt_qty = qc3.number_input("כמות", min_value=0.0, step=0.1, value=1.0)
            qt_price = qc4.number_input("מחיר $", min_value=0.0, step=0.01, value=0.0)
            if st.form_submit_button(f"בצע {qt_action}", use_container_width=True):
                action_str = "Buy" if qt_action == "קנייה" else "Sell"
                if qt_ticker and qt_qty > 0:
                    st.session_state.portfolio = apply_trade(st.session_state.portfolio, qt_ticker, action_str, qt_qty, qt_price)
                    if save_cloud_portfolio(st.session_state.portfolio):
                        st.success(f"{qt_action} של {qt_ticker} בוצעה")
                        st.rerun()

    section("ייבוא מתמונת מסך", "צלם/י צילום מסך של האפליקציה של הברוקר — נזהה סימולים, כמויות ומחירים באופן ראשוני. תמיד תוכל/י לערוך לפני שמירה.")
    if not OCR_AVAILABLE:
        st.info("זיהוי תמונות אינו זמין בסביבה הנוכחית. יש להוסיף pytesseract ו-Pillow ל-requirements.txt ו-tesseract-ocr ל-packages.txt.")
    else:
        uploaded_img = st.file_uploader("העלה/י צילום מסך", type=["png", "jpg", "jpeg"])
        if uploaded_img is not None:
            img = Image.open(uploaded_img)
            st.image(img, use_container_width=True)
            if st.button("זהה מניות מהתמונה"):
                with st.spinner("מזהה טקסט בתמונה..."):
                    extracted = parse_portfolio_image(img)
                st.session_state['_ocr_extracted'] = extracted

        if '_ocr_extracted' in st.session_state and not st.session_state['_ocr_extracted'].empty:
            st.caption("בדוק/י ותקן/י לפני הוספה לתיק — הזיהוי הוא ראשוני בלבד.")
            edited = st.data_editor(
                st.session_state['_ocr_extracted'], num_rows="dynamic", use_container_width=True,
                key="ocr_editor"
            )
            if st.button("אשר והוסף לתיק", use_container_width=True):
                for _, r in edited.iterrows():
                    tk = str(r.get('Ticker', '')).strip().upper()
                    if tk and r.get('Quantity', 0) > 0:
                        st.session_state.portfolio = apply_trade(
                            st.session_state.portfolio, tk, "Buy", float(r['Quantity']), float(r.get('PurchasePrice', 0) or 0)
                        )
                if save_cloud_portfolio(st.session_state.portfolio):
                    del st.session_state['_ocr_extracted']
                    st.success("התיק עודכן")
                    st.rerun()

    if rows:
        section("מניות בתיק")
        rows_html = []
        for r in sorted(rows, key=lambda x: x['P&L %'], reverse=True):
            pl_color = 'var(--gain)' if r['P&L %'] >= 0 else 'var(--loss)'
            sig_color = SIGNAL_COLOR.get(r['Signal'], 'var(--text-dim)')
            pl_sign = '+' if r['P&L %'] >= 0 else ''
            rows_html.append(ledger_row(
                r['Ticker'], r['Sector'][:18], f"${r['Price']:,.2f}",
                f"{pl_sign}{r['P&L %']:.1f}%  ${r['P&L $']:+,.0f}", pl_color,
                SIGNAL_LABEL_HE.get(r['Signal'], r['Signal']), sig_color,
                caption=f"RSI {r['RSI']} · Score {r['Score']}/100"
            ))
        render_ledger(rows_html)

        section("עריכת פוזיציה")
        with st.expander("שינוי / מחיקת מניה מהתיק"):
            edit_choices = [r['Ticker'] for r in rows]
            ed_col1, _ = st.columns([2, 1])
            edit_sel = ed_col1.selectbox("בחר מניה:", edit_choices, key='edit_sel')
            cur_row = st.session_state.portfolio.loc[edit_sel] if edit_sel in st.session_state.portfolio.index else None
            cur_qty = float(cur_row['Quantity']) if cur_row is not None else 0.0
            cur_price = float(cur_row['PurchasePrice']) if cur_row is not None else 0.0
            with st.form("edit_position_form"):
                ep1, ep2 = st.columns(2)
                new_qty = ep1.number_input("כמות חדשה", min_value=0.0, step=0.1, value=cur_qty)
                new_price = ep2.number_input("מחיר קנייה ממוצע $", min_value=0.0, step=0.01, value=cur_price)
                upd, dlt = st.columns(2)
                if upd.form_submit_button("עדכן", use_container_width=True):
                    if new_qty > 0:
                        st.session_state.portfolio.loc[edit_sel] = [new_qty, new_price]
                    else:
                        st.session_state.portfolio = st.session_state.portfolio.drop(edit_sel)
                    if save_cloud_portfolio(st.session_state.portfolio):
                        st.success(f"{edit_sel} עודכן"); st.rerun()
                if dlt.form_submit_button("מחק לחלוטין", use_container_width=True):
                    if edit_sel in st.session_state.portfolio.index:
                        st.session_state.portfolio = st.session_state.portfolio.drop(edit_sel)
                        if save_cloud_portfolio(st.session_state.portfolio):
                            st.success(f"{edit_sel} נמחק"); st.rerun()

        df_rows = pd.DataFrame(rows)
        with st.expander("טבלה מלאה"):
            def color_signal(val):
                return {'STRONG BUY':'color:var(--gain);font-weight:bold','BUY':'color:var(--gain)',
                        'HOLD':'color:var(--text-dim)','WATCH':'color:var(--info)',
                        'SELL':'color:var(--loss)','STRONG SELL':'color:var(--loss);font-weight:bold'}.get(val, '')
            def color_pl(val):
                try: return 'color:var(--gain)' if float(val) >= 0 else 'color:var(--loss)'
                except Exception: return ''
            styled = df_rows.style.map(color_signal, subset=['Signal']).map(color_pl, subset=['P&L %', 'P&L $', 'Upside %']) \
                .format({'Price': '${:.2f}', 'Buy@': '${:.2f}', 'Value $': '${:,.2f}', 'P&L %': '{:.2f}%',
                          'P&L $': '${:,.2f}', 'Annual Div $': '${:.2f}', 'Target $': '${:.2f}', 'Upside %': '{:.1f}%'})
            st.dataframe(styled, use_container_width=True)
            st.download_button("ייצוא CSV", df_rows.to_csv(index=False), "portfolio.csv", "text/csv")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if sector_weights:
                fig_pie = px.pie(values=list(sector_weights.values()), names=list(sector_weights.keys()),
                                  title="פיזור סקטוריאלי",
                                  color_discrete_sequence=['#C9A24B','#55A87F','#6B93BE','#C1584F','#8B8F9C','#8C7239'])
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ECECEE', height=280,
                                       margin=dict(l=0, r=0, t=30, b=0), legend=dict(font=dict(size=9)))
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        with col_p2:
            fig_score = go.Figure(go.Bar(
                x=[r['Ticker'] for r in rows], y=[r['Score'] for r in rows],
                marker_color=['#55A87F' if s >= 70 else '#6B93BE' if s >= 55 else '#C9A24B' if s >= 45 else '#C1584F'
                              for s in [r['Score'] for r in rows]],
                text=[SIGNAL_LABEL_HE.get(r['Signal'], r['Signal']) for r in rows], textposition='auto'
            ))
            fig_score.update_layout(title="ציון כמותי", paper_bgcolor='rgba(0,0,0,0)', font_color='#ECECEE',
                                     yaxis_range=[0, 100], height=280, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("הוסף/י מניות לתיק דרך הטופס למעלה או ייבוא מתמונה.")

# ==========================================
# TAB: TECHNICAL
# ==========================================
with t_tech:
    section("ניתוח טכני מעמיק")
    selectable = p_tickers + ([quick_ticker] if quick_ticker else [])
    analyze_ticker = st.selectbox("בחר/י מניה:", selectable) if selectable else None
    if not analyze_ticker and not quick_ticker:
        analyze_ticker = st.text_input("הכנס/י סימול לניתוח:").upper() or None
    period_choice = st.select_slider("תקופה:", options=["1mo", "3mo", "6mo", "1y", "2y"], value="1y")

    if analyze_ticker:
        with st.spinner(f"טוען {analyze_ticker}..."):
            stock = yf.Ticker(analyze_ticker)
            hist = stock.history(period=period_choice)

        if not hist.empty:
            d = m_data.get(analyze_ticker, {})
            c1, c2, c3 = st.columns(3)
            c1.metric("מחיר", f"${d.get('price', hist['Close'].iloc[-1]):,.2f}")
            rsi_v = d.get('rsi', 50)
            c2.metric("RSI", f"{rsi_v:.0f}", "קניית יתר" if rsi_v > 70 else "מכירת יתר" if rsi_v < 30 else "ניטרלי")
            c3.metric("איתות", SIGNAL_LABEL_HE.get(d.get('signal', ''), d.get('signal', 'N/A')))
            c4, c5, c6 = st.columns(3)
            c4.metric("ציון", f"{d.get('score', 50)}/100")
            c5.metric("יעד", f"${d.get('target_price', 0):,.2f}", f"{d.get('target_upside', 0):.1f}%")
            c6.metric("Beta", f"{d.get('beta', 1):.2f}")

            if d:
                st.markdown(f'<div style="margin-top:8px;font-size:0.82rem;color:var(--text-dim)">{playbook_line(d)}</div>', unsafe_allow_html=True)

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.17, 0.17, 0.16],
                                 subplot_titles=[analyze_ticker, "RSI", "MACD", "Volume"])

            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'],
                                          close=hist['Close'], name="Price",
                                          increasing_line_color='#55A87F', decreasing_line_color='#C1584F'), row=1, col=1)

            sma20 = hist['Close'].rolling(20).mean()
            std20 = hist['Close'].rolling(20).std()
            fig.add_trace(go.Scatter(x=hist.index, y=sma20+2*std20, line=dict(color='rgba(107,147,190,0.35)', dash='dot'), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=sma20-2*std20, line=dict(color='rgba(107,147,190,0.35)', dash='dot'),
                                      fill='tonexty', fillcolor='rgba(107,147,190,0.05)', showlegend=False), row=1, col=1)

            for period, color in [(20, '#C9A24B'), (50, '#6B93BE'), (200, '#C1584F')]:
                if len(hist) >= period:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(period).mean(),
                                              name=f'SMA{period}', line=dict(color=color, width=1.4)), row=1, col=1)

            sup, res = d.get('support', hist['Low'].min()), d.get('resistance', hist['High'].max())
            fig.add_hline(y=sup, line_dash="dot", line_color="#55A87F", annotation_text=f"תמיכה ${sup:.2f}", row=1, col=1)
            fig.add_hline(y=res, line_dash="dot", line_color="#C1584F", annotation_text=f"התנגדות ${res:.2f}", row=1, col=1)

            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
            rsi_s = 100 - (100 / (1 + (gain / loss)))
            fig.add_trace(go.Scatter(x=hist.index, y=rsi_s, name='RSI', line=dict(color='#C9A24B', width=1.4)), row=2, col=1)
            fig.add_hline(y=70, line_color='#C1584F', line_dash='dash', row=2, col=1)
            fig.add_hline(y=30, line_color='#55A87F', line_dash='dash', row=2, col=1)

            ema12, ema26 = hist['Close'].ewm(span=12, adjust=False).mean(), hist['Close'].ewm(span=26, adjust=False).mean()
            macd, macd_sig = ema12 - ema26, (ema12 - ema26).ewm(span=9, adjust=False).mean()
            macd_h = macd - macd_sig
            fig.add_trace(go.Scatter(x=hist.index, y=macd, name='MACD', line=dict(color='#6B93BE', width=1.4)), row=3, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=macd_sig, name='Signal', line=dict(color='#C9A24B', width=1.4)), row=3, col=1)
            fig.add_trace(go.Bar(x=hist.index, y=macd_h, name='Hist', marker_color=['#55A87F' if v >= 0 else '#C1584F' for v in macd_h]), row=3, col=1)

            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color='rgba(107,147,190,0.35)'), row=4, col=1)

            fig.update_layout(height=620, paper_bgcolor='#0A0B0E', plot_bgcolor='#14161C', font_color='#ECECEE',
                               xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                               legend=dict(orientation='h', y=1.02, font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})

            if d.get('reasons'):
                with st.expander(f"למה ציון {d.get('score', 0)}/100 — {d.get('trend', '-')}"):
                    for reason in d['reasons']:
                        st.markdown(f"· {reason}")

            with st.expander("כל האינדיקטורים"):
                if d:
                    ind_data = {
                        "מגמה": d.get('trend', '-'), "RSI": f"{d.get('rsi', 0):.1f}",
                        "חציית MACD": "כן" if d.get('macd_crossover') else "לא",
                        "Stoch %K": f"{d.get('stoch_k', 0):.1f}", "BB %": f"{d.get('bb_pct', 0)*100:.1f}%",
                        "ATR %": f"{d.get('atr_pct', 0):.2f}%", "יחס נפח": f"{d.get('vol_ratio', 1):.2f}x",
                        "תמיכה": f"${d.get('support', 0):.2f}", "התנגדות": f"${d.get('resistance', 0):.2f}",
                        "מומנטום חודש": f"{d.get('mom_1m', 0):.1f}%", "מומנטום 3 חודשים": f"{d.get('mom_3m', 0):.1f}%",
                        "P/E": f"{d.get('pe', 0):.1f}", "P/E עתידי": f"{d.get('forward_pe', 0):.1f}",
                        "PEG": f"{d.get('peg', 0):.2f}", "P/B": f"{d.get('pb', 0):.2f}", "ROE": f"{d.get('roe', 0):.1f}%",
                        "מרווח גולמי": f"{d.get('gross_margin', 0):.1f}%", "צמיחת הכנסות": f"{d.get('growth_yoy', 0):.1f}%",
                        "חוב/הון": f"{d.get('debt_to_equity', 0):.2f}", "יעד אנליסטים": f"${d.get('target_price', 0):.2f}",
                        "תשואת דיבידנד": f"{d.get('div', 0):.2f}%",
                    }
                    ind_df = pd.DataFrame(list(ind_data.items()), columns=['אינדיקטור', 'ערך'])
                    c_i1, c_i2 = st.columns(2)
                    half = len(ind_df) // 2
                    c_i1.dataframe(ind_df.iloc[:half], use_container_width=True, hide_index=True)
                    c_i2.dataframe(ind_df.iloc[half:], use_container_width=True, hide_index=True)

# ==========================================
# TAB: OPPORTUNITIES
# ==========================================
with t_scan:
    section("הזדמנויות במניות הגדולות", "נסרק אוטומטית ברקע — עודכן לאחרונה עם רענון הנתונים")
    if core_data:
        core_sorted = sorted(core_data.items(), key=lambda x: x[1]['score'], reverse=True)
        rows_html = []
        for t, d in core_sorted[:12]:
            val_tag, val_color = valuation_tag(d)
            rows_html.append(ledger_row(
                t, d.get('sector', '')[:20], f"${d['price']:,.2f}", val_tag, val_color,
                SIGNAL_LABEL_HE.get(d['signal'], d['signal']), SIGNAL_COLOR.get(d['signal'], 'var(--text-dim)'),
                caption=playbook_line(d)
            ))
        render_ledger(rows_html)

    section("סריקה מותאמת אישית")
    sc1, sc2 = st.columns(2)
    with sc1:
        scan_universe = st.selectbox("יקום:", [
            "Mega Watchlist (18)", "NASDAQ 100", "S&P 500 (מדגם 40)", "Tech Giants",
            "Value Plays", "High Growth", "Dividend Stocks", "ETFs", "Israeli Stocks (TA)"
        ])
    with sc2:
        min_score = st.slider("ציון מינימלי:", 40, 90, 60)

    with st.expander("פילטרים נוספים"):
        sf1, sf2 = st.columns(2)
        sector_filter = sf1.selectbox("סקטור:", ["הכל", "Technology", "Healthcare", "Financial Services",
                                                    "Consumer Cyclical", "Energy", "Industrials", "Real Estate"])
        max_pe = sf2.number_input("P/E מקסימלי (0=ללא):", value=0, min_value=0)
        sf3, sf4 = st.columns(2)
        min_growth = sf3.number_input("צמיחה % מינ':", value=0)
        min_div = sf4.number_input("דיבידנד % מינ':", value=0.0, step=0.5)

    if st.button("הרץ סריקה"):
        with st.spinner("סורק..."):
            if scan_universe == "Mega Watchlist (18)":
                scan_list = ['AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','INTC',
                             'BA','PEP','COST','AVGO','DIS','UBER','ABNB','SPOT']
            elif scan_universe == "NASDAQ 100":
                scan_list = random.sample(NASDAQ_100, min(40, len(NASDAQ_100)))
            elif scan_universe == "S&P 500 (מדגם 40)":
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
                if d.get('score', 0) < min_score: continue
                if sector_filter != "הכל" and d.get('sector', '') != sector_filter: continue
                if max_pe > 0 and (d.get('pe', 0) <= 0 or d.get('pe', 0) > max_pe): continue
                if d.get('growth_yoy', 0) < min_growth: continue
                if d.get('div', 0) < min_div: continue
                results.append((t, d))
            results.sort(key=lambda x: x[1].get('score', 0), reverse=True)

            st.markdown(f"נמצאו **{len(results)}** מניות")
            if results:
                rows_html = []
                for t, d in results[:20]:
                    val_tag, val_color = valuation_tag(d)
                    rows_html.append(ledger_row(
                        t, f"{d.get('sector','')[:16]} · P/E {d['pe']:.0f} · RSI {d['rsi']:.0f}",
                        f"${d['price']:,.2f}", val_tag, val_color,
                        f"{d['score']}/100", SIGNAL_COLOR.get(d['signal'], 'var(--text-dim)'),
                    ))
                render_ledger(rows_html)

                with st.expander("טבלה מלאה"):
                    table_data = [{'Ticker': t, 'Score': d['score'], 'Signal': d['signal'], 'Price': f"${d['price']:.2f}",
                                   'RSI': round(d['rsi'], 1), 'Trend': d['trend'], 'P/E': round(d['pe'], 1),
                                   'Growth%': round(d.get('growth_yoy', 0), 1), 'Div%': round(d.get('div', 0), 2),
                                   'Upside%': round(d.get('target_upside', 0), 1)} for t, d in results]
                    tbl_df = pd.DataFrame(table_data)
                    st.dataframe(tbl_df, use_container_width=True)
                    st.download_button("ייצוא", tbl_df.to_csv(index=False), "scan.csv", "text/csv")

# ==========================================
# TAB: EARNINGS & NEWS
# ==========================================
with t_earn:
    inner = st.tabs(["דוחות רווחים", "חדשות", "רשימת מעקב"])
    tab_earn, tab_news, tab_watch = inner

    with tab_earn:
        board_tickers = list(set(p_tickers + watch_tickers))
        if board_tickers and st.button("טען דוחות", key="load_earnings"):
            with st.spinner("טוען לוח דוחות..."):
                upcoming, recent = build_earnings_board(tuple(board_tickers))

            section("דוחות קרובים", "ממוינים לפי הקרוב ביותר")
            if upcoming:
                rows_html = []
                for u in upcoming[:15]:
                    when = "היום" if u['days_out'] == 0 else f"בעוד {u['days_out']} ימים" if u['days_out'] > 0 else "פורסם"
                    eps = f"תחזית EPS ${u['eps_est']:.2f}" if u.get('eps_est') else "אין תחזית"
                    tag_color = 'var(--gold)' if u['days_out'] <= 7 else 'var(--text-dim)'
                    rows_html.append(ledger_row(u['ticker'], eps, u['date'].strftime('%d.%m.%Y'), when, tag_color, when, tag_color))
                render_ledger(rows_html)
            else:
                st.info("אין תאריכי דוחות קרובים ידועים למניות שלך.")

            section("דוחות אחרונים שפורסמו", "בפועל מול תחזית האנליסטים")
            if recent:
                rows_html = []
                for r in recent:
                    surprise = r.get('surprise_pct')
                    surprise_str = f"{'+' if surprise and surprise>=0 else ''}{surprise:.1f}%" if surprise is not None else "—"
                    surprise_color = 'var(--gain)' if (surprise or 0) >= 0 else 'var(--loss)'
                    actual = r.get('eps_actual'); est = r.get('eps_estimate')
                    sub = f"בפועל ${actual:.2f} מול תחזית ${est:.2f}" if actual is not None and est is not None else "—"
                    date_str = r['date'].strftime('%d.%m.%Y') if hasattr(r['date'], 'strftime') else "—"
                    rows_html.append(ledger_row(r['ticker'], sub, date_str, "הפתעת EPS", 'var(--text-faint)',
                                                 surprise_str, surprise_color))
                render_ledger(rows_html)
            else:
                st.info("אין נתוני דוחות אחרונים זמינים.")
        else:
            st.info("הוסף/י מניות לתיק או לרשימת המעקב כדי לראות דוחות רווחים.")

    with tab_news:
        news_ticker = st.selectbox("בחר/י מניה:", p_tickers + ([quick_ticker] if quick_ticker else []), key='news_sel') \
            if (p_tickers or quick_ticker) else None
        if not news_ticker:
            news_ticker = st.text_input("הכנס/י סימול:", key='news_inp').upper() or None
        if news_ticker and st.button("טען חדשות", key="load_news"):
            with st.spinner(f"טוען חדשות ל-{news_ticker}..."):
                news = get_stock_news(news_ticker)
            if news:
                for item in news:
                    try:
                        ts = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                        st.markdown(
                            f'<div class="news-item"><div class="news-title">'
                            f'<a href="{item.get("link","#")}" target="_blank" style="color:var(--text);text-decoration:none">{item.get("title","")}</a>'
                            f'</div><div class="news-meta">{item.get("publisher","")} · {ts.strftime("%d.%m.%Y %H:%M")}</div></div>',
                            unsafe_allow_html=True
                        )
                    except Exception:
                        pass
            else:
                st.info("אין חדשות זמינות.")

    with tab_watch:
        wl = st.session_state.watchlist
        with st.form("add_watchlist"):
            col_w1, col_w2, col_w3, col_w4 = st.columns(4)
            wl_ticker = col_w1.text_input("Ticker").upper()
            wl_notes = col_w2.text_input("הערות")
            wl_high = col_w3.number_input("התראה מעל $", value=0.0)
            wl_low = col_w4.number_input("התראה מתחת $", value=0.0)
            if st.form_submit_button("הוסף"):
                if wl_ticker:
                    new_row = pd.DataFrame([{'Ticker': wl_ticker, 'Notes': wl_notes, 'AlertHigh': wl_high, 'AlertLow': wl_low}])
                    wl = pd.concat([wl, new_row], ignore_index=True).drop_duplicates('Ticker')
                    st.session_state.watchlist = wl
                    save_watchlist(wl)
                    st.rerun()

        if not wl.empty:
            wl_tickers_list = [t for t in wl['Ticker'].tolist() if isinstance(t, str) and t.strip()]
            wl_data = fetch_expert_data(tuple(wl_tickers_list)) if wl_tickers_list else {}

            alerts = []
            for _, row_wl in wl.iterrows():
                t = str(row_wl['Ticker'])
                if t in wl_data:
                    curr_p = wl_data[t]['price']
                    try:
                        ah = float(row_wl['AlertHigh'])
                        if ah > 0 and curr_p >= ah:
                            alerts.append((t, f"${curr_p:.2f} ≥ יעד עליון ${ah:.2f}", 'var(--gain)'))
                    except Exception: pass
                    try:
                        al = float(row_wl['AlertLow'])
                        if al > 0 and curr_p <= al:
                            alerts.append((t, f"${curr_p:.2f} ≤ יעד תחתון ${al:.2f}", 'var(--loss)'))
                    except Exception: pass

            if alerts:
                section("התראות פעילות")
                for t, msg, color in alerts:
                    st.markdown(f'<div class="attn-strip" style="border-color:{color}"><div class="row-ticker">{t}</div><div class="row-name">{msg}</div></div>', unsafe_allow_html=True)

            section("רשימת מעקב")
            rows_html = []
            for _, row_wl in wl.iterrows():
                t = str(row_wl['Ticker'])
                d_wl = wl_data.get(t, {})
                if d_wl:
                    rows_html.append(ledger_row(
                        t, row_wl.get('Notes', '') or d_wl.get('sector', ''), f"${d_wl.get('price', 0):,.2f}",
                        f"RSI {d_wl.get('rsi', 50):.0f} · {d_wl.get('trend','-')}", 'var(--text-faint)',
                        SIGNAL_LABEL_HE.get(d_wl.get('signal',''), d_wl.get('signal','—')),
                        SIGNAL_COLOR.get(d_wl.get('signal'), 'var(--text-dim)')
                    ))
                else:
                    rows_html.append(ledger_row(t, row_wl.get('Notes', ''), "—", "אין נתונים", 'var(--text-faint)', "—", 'var(--text-dim)'))
            render_ledger(rows_html)

            remove_t = st.selectbox("הסר/י מניה:", ['—'] + wl_tickers_list)
            if remove_t != '—' and st.button("הסר מרשימת המעקב"):
                wl = wl[wl['Ticker'] != remove_t]
                st.session_state.watchlist = wl
                save_watchlist(wl)
                st.rerun()
        else:
            st.info("הרשימה ריקה. הוסף/י מניות למעקב למעלה.")

# ==========================================
# TAB: RISK & TOOLS
# ==========================================
with t_risk:
    inner = st.tabs(["ניתוח סיכון", "יעדי מחיר", "DCA", "גודל פוזיציה"])
    tab_r, tab_targets, tab_dca, tab_sizing = inner

    with tab_r:
        if p_tickers:
            with st.spinner("מחשב..."):
                corr, sharpe, max_dd, var95, spy_corr, spy_ret_1y = compute_portfolio_analytics(tuple(p_tickers))

            risk_rows = []
            for t in p_tickers:
                d = m_data.get(t, {})
                risk_rows.append({
                    'Ticker': t, 'Beta': round(d.get('beta', 1), 2), 'ATR%': round(d.get('atr_pct', 0), 2),
                    'Sharpe': round(sharpe.get(t, 0), 2), 'Max DD%': round(max_dd.get(t, 0), 1),
                    'VaR 95%': round(var95.get(t, 0), 2), 'SPY Corr': round(spy_corr.get(t, 0), 2),
                    'Short%': round(d.get('short_pct', 0), 1), 'Debt/Eq': round(d.get('debt_to_equity', 0), 2),
                })
            st.dataframe(pd.DataFrame(risk_rows), use_container_width=True)

            rows = [{'Ticker': t, 'Value $': m_data[t]['price'] * st.session_state.portfolio.loc[t, 'Quantity']}
                    for t in p_tickers if t in m_data]
            total_v = sum(r['Value $'] for r in rows)
            if total_v > 0:
                port_beta = sum(m_data[r['Ticker']].get('beta', 1) * r['Value $'] / total_v for r in rows)
                avg_sharpe = float(np.mean(list(sharpe.values()))) if sharpe else 0
                worst_dd = min(max_dd.values()) if max_dd else 0
                worst_t = min(max_dd, key=max_dd.get) if max_dd else '-'

                rk1, rk2 = st.columns(2)
                rk1.metric("Beta", f"{port_beta:.2f}", "תנודתי" if port_beta > 1.2 else "מאוזן")
                rk2.metric("Sharpe", f"{avg_sharpe:.2f}", "טוב" if avg_sharpe > 1 else "חלש" if avg_sharpe > 0 else "שלילי")
                rk3, rk4 = st.columns(2)
                rk3.metric("Max Drawdown", f"{worst_dd:.1f}%", worst_t)
                rk4.metric("SPY שנה", f"{spy_ret_1y:.1f}%")

            if corr is not None and len(p_tickers) > 1:
                with st.expander("מטריצת מתאמים"):
                    fig_corr = px.imshow(corr, color_continuous_scale=[[0,'#C1584F'],[0.5,'#14161C'],[1,'#55A87F']],
                                          zmin=-1, zmax=1, text_auto='.2f')
                    fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ECECEE', height=380, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

            if total_v > 0:
                with st.expander("סימולציית Monte Carlo (שנה קדימה)"):
                    n_sim, n_days = 500, 252
                    avg_ret = float(np.mean([m_data[r['Ticker']].get('mom_1m', 0)/100/20 for r in rows]))
                    avg_vol = float(np.mean([m_data[r['Ticker']].get('atr_pct', 2)/100/np.sqrt(20) for r in rows]))
                    sims = np.array([np.prod(1 + np.random.normal(avg_ret, avg_vol, n_days)) * total_v for _ in range(n_sim)])
                    p5, p50, p95 = float(np.percentile(sims, 5)), float(np.percentile(sims, 50)), float(np.percentile(sims, 95))
                    fig_mc = go.Figure(go.Histogram(x=sims, nbinsx=40, marker_color='rgba(107,147,190,0.5)'))
                    for val, clr in [(p5, '#C1584F'), (p50, '#C9A24B'), (p95, '#55A87F')]:
                        fig_mc.add_vline(x=val, line_dash='dash', line_color=clr, annotation_text=f"${val:,.0f}")
                    fig_mc.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ECECEE', height=280, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_mc, use_container_width=True, config={"displayModeBar": False})
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("תרחיש גרוע (5%)", f"${p5:,.0f}")
                    mc2.metric("חציון", f"${p50:,.0f}")
                    mc3.metric("תרחיש טוב (95%)", f"${p95:,.0f}")
        else:
            st.info("הוסף/י מניות לתיק.")

    with tab_targets:
        for t in p_tickers:
            if t not in m_data: continue
            d = m_data[t]; p = d['price']
            atr = d.get('atr_pct', 2) / 100
            growth = max(0, d.get('growth_yoy', 8)) / 100
            with st.expander(f"{t} — ${p:.2f} · {SIGNAL_LABEL_HE.get(d.get('signal','N/A'), d.get('signal','N/A'))} · ציון {d.get('score', 50)}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("1–5 ימים", f"${p*(1+atr*1.5):.2f}", f"+{atr*150:.1f}%")
                c2.metric("2–4 שבועות", f"${p*(1+atr*3):.2f}", f"+{atr*300:.1f}%")
                c3.metric("1–3 חודשים", f"${p*(1+growth*0.25):.2f}", f"+{growth*25:.1f}%")
                c4.metric("שנה", f"${p*(1+growth):.2f}", f"+{growth*100:.1f}%")

    with tab_dca:
        dca_ticker = st.selectbox("מניה:", p_tickers, key='dca') if p_tickers else None
        dca_monthly = st.number_input("השקעה חודשית ($):", value=100.0)
        dca_months = st.slider("חודשים:", 6, 120, 24)
        dca_growth = st.slider("צמיחה שנתית משוערת (%):", -10, 30, 8)

        if dca_ticker and dca_ticker in m_data:
            curr_p = m_data[dca_ticker]['price']
            growth_r = dca_growth / 100
            total_inv = dca_monthly * dca_months
            shares, monthly_p, rows_dca = 0, curr_p, []
            for month in range(1, dca_months + 1):
                monthly_p *= (1 + growth_r / 12)
                shares += dca_monthly / monthly_p
                rows_dca.append({'חודש': month, 'שווי': round(shares * monthly_p, 2)})
            dca_df = pd.DataFrame(rows_dca)
            final_v = dca_df['שווי'].iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("סה\"כ הושקע", f"${total_inv:,.2f}")
            c2.metric("שווי צפוי", f"${final_v:,.2f}")
            c3.metric("רווח צפוי", f"${final_v-total_inv:,.2f}", f"{(final_v/total_inv-1)*100:.1f}%")
            fig_dca = go.Figure()
            fig_dca.add_trace(go.Scatter(x=dca_df['חודש'], y=dca_df['שווי'], fill='tozeroy', line=dict(color='#55A87F')))
            fig_dca.add_trace(go.Scatter(x=dca_df['חודש'], y=[dca_monthly*m for m in dca_df['חודש']], line=dict(color='#C9A24B', dash='dash')))
            fig_dca.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ECECEE', height=300, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_dca, use_container_width=True, config={"displayModeBar": False})

    with tab_sizing:
        ps_account = st.number_input("גודל תיק ($):", value=float(total_usd) if 'total_usd' in dir() else 7000.0)
        ps_risk_pct = st.slider("סיכון לעסקה (%):", 0.5, 5.0, 1.0, 0.5)
        sel_ps = p_tickers + ([quick_ticker] if quick_ticker else [])
        ps_sel = st.selectbox("מניה:", sel_ps, key='ps_sel') if sel_ps else None
        ps_stop = st.slider("Stop Loss (%):", 2.0, 25.0, 8.0, 0.5)
        ps_target = st.slider("Take Profit (%):", 5.0, 50.0, 20.0, 1.0)

        if ps_sel and ps_sel in m_data:
            curr_p = m_data[ps_sel]['price']
            risk_share = curr_p * (ps_stop / 100)
            max_risk = ps_account * (ps_risk_pct / 100)
            shares = int(max_risk / risk_share) if risk_share > 0 else 0
            pos_size = shares * curr_p
            stop_p, target_p = curr_p * (1 - ps_stop/100), curr_p * (1 + ps_target/100)
            rr_ratio = ps_target / ps_stop if ps_stop else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("מניות לקנייה", str(shares))
            c2.metric("גודל פוזיציה", f"${pos_size:,.2f}", f"{pos_size/ps_account*100:.1f}% מהתיק" if ps_account else "")
            c3.metric("Stop Loss", f"${stop_p:.2f}", f"-{ps_stop}%")
            c4.metric("Take Profit", f"${target_p:.2f}", f"+{ps_target}%")
            rr_color = 'var(--gain)' if rr_ratio >= 2 else 'var(--gold)' if rr_ratio >= 1.5 else 'var(--loss)'
            st.markdown(f"<span style='color:{rr_color};font-weight:600'>יחס סיכוי/סיכון 1:{rr_ratio:.1f}</span>", unsafe_allow_html=True)

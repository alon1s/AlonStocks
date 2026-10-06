import html
import json
import os
import random
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_gsheets import GSheetsConnection

warnings.filterwarnings('ignore')

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

TZ = ZoneInfo("Asia/Jerusalem")
RISK_FREE = 0.04

# ==========================================
# PAGE CONFIG + DESIGN SYSTEM
# ==========================================
st.set_page_config(page_title="AlonStocks Pro", layout="wide", page_icon="◆", initial_sidebar_state="collapsed")

INK, PANEL, LINE, TEXT, TEXT_DIM = '#0A0B0E', '#14161C', '#262A35', '#ECECEE', '#8B8F9C'
GOLD, GAIN, LOSS, INFO = '#C9A24B', '#55A87F', '#C1584F', '#6B93BE'

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Instrument+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    :root {
        --ink:#0A0B0E; --panel:#14161C; --panel-2:#1A1D25; --line:#262A35; --line-soft:#1C1F28;
        --text:#ECECEE; --text-dim:#8B8F9C; --text-faint:#565A67;
        --gold:#C9A24B; --gold-soft:rgba(201,162,75,0.12);
        --gain:#55A87F; --loss:#C1584F; --info:#6B93BE;
    }
    html, body, [class*="css"] { font-family:'Instrument Sans', sans-serif; }
    .main, [data-testid="stAppViewContainer"] { background:var(--ink); }
    [data-testid="stHeader"] { background:transparent; }
    input, select, textarea, button { font-size:16px !important; }
    .main .block-container { padding:1.1rem 0.9rem 112px !important; max-width:100% !important; }

    [data-testid="stMarkdown"] p, [data-testid="stMarkdown"] li,
    [data-testid="stMarkdown"] h1, [data-testid="stMarkdown"] h2, [data-testid="stMarkdown"] h3,
    .stAlert p, [data-testid="stWidgetLabel"] p, [data-testid="stMetric"] label,
    [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span,
    [data-testid="stFormSubmitButton"] button, [data-testid="stCaptionContainer"],
    .section-title, .section-sub, .row-name, .row-sub, .row-caption, .stamp, .hero-label {
        direction:rtl !important; text-align:right !important; unicode-bidi:embed !important;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"], .stDataFrame, .stDataEditor,
    .js-plotly-plot, pre, code, .row-ticker, .row-price, .hero-value {
        direction:ltr !important; text-align:left !important;
    }

    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px !important; }
    [data-testid="stMetric"] label { font-size:0.7rem !important; color:var(--text-dim) !important; font-weight:500; }
    [data-testid="stMetricValue"] { font-size:1.2rem !important; font-weight:600 !important; font-family:'IBM Plex Mono', monospace !important; color:var(--text) !important; }
    [data-testid="stMetricDelta"] { font-size:0.78rem !important; }

    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        min-height:46px !important; font-size:0.92rem !important; border-radius:7px !important; width:100% !important;
        font-weight:600 !important; background:transparent !important; color:var(--gold) !important;
        border:1px solid var(--gold) !important; transition:background 0.12s;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background:var(--gold-soft) !important; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-baseweb="select"] {
        background:var(--panel) !important; border:1px solid var(--line) !important; color:var(--text) !important; border-radius:6px !important;
    }
    .stDataFrame { font-family:'IBM Plex Mono', monospace !important; font-size:0.75em; }
    div[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:8px; margin-bottom:8px; }

    .hero-label { font-size:0.74rem; color:var(--text-dim); margin-bottom:2px; }
    .hero-value { font-family:'Fraunces', serif; font-weight:500; font-size:2.5rem; color:var(--text); line-height:1.1; letter-spacing:-0.01em; }
    .hero-change { font-family:'IBM Plex Mono', monospace; font-size:0.92rem; margin-top:4px; }
    .stamp { font-size:0.72rem; color:var(--text-faint); margin-top:4px; }

    .section-title { font-size:0.82rem; font-weight:600; color:var(--text); border-right:3px solid var(--gold); padding:3px 10px 3px 0; margin:22px 0 10px; }
    .section-sub { font-size:0.72rem; color:var(--text-faint); margin:-6px 0 12px; padding-right:13px; }

    .ledger-row { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:12px 4px; border-bottom:1px solid var(--line-soft); }
    .ledger-row.total { border-top:1px solid var(--line); border-bottom:none; }
    .row-ticker { font-family:'IBM Plex Mono', monospace; font-weight:600; font-size:0.98rem; color:var(--text); }
    .row-name { font-size:0.68rem; color:var(--text-faint); margin-top:2px; }
    .row-price { font-family:'IBM Plex Mono', monospace; font-weight:600; font-size:0.9rem; color:var(--text); }
    .row-sub { font-family:'IBM Plex Mono', monospace; font-size:0.76rem; margin-top:2px; direction:ltr !important; text-align:center !important; }
    .row-tag { font-size:0.8rem; font-weight:600; display:flex; align-items:center; gap:5px; justify-content:flex-end; font-family:'IBM Plex Mono', monospace; }
    .row-caption { font-size:0.66rem; color:var(--text-faint); margin-top:2px; max-width:240px; }
    .dot { width:7px; height:7px; border-radius:50%; display:inline-block; }

    .attn-strip { border-right:3px solid var(--gold); background:var(--panel); border-radius:6px; padding:10px 12px; margin-bottom:8px; }
    .attn-strip .row-name { color:var(--text-dim); font-size:0.74rem; }

    .idx-strip { display:flex; flex-wrap:wrap; margin-bottom:14px; border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    .idx-item { flex:1; min-width:90px; padding:10px 12px; border-left:1px solid var(--line); }
    .idx-item:last-child { border-left:none; }
    .idx-name { font-size:0.64rem; color:var(--text-faint); }
    .idx-price { font-family:'IBM Plex Mono', monospace; font-size:0.86rem; font-weight:600; color:var(--text); margin-top:2px; direction:ltr; }
    .idx-chg { font-family:'IBM Plex Mono', monospace; font-size:0.7rem; margin-top:1px; direction:ltr; }

    .range-wrap { direction:ltr; margin:2px 0 14px; }
    .range-bar { height:4px; border-radius:2px; background:linear-gradient(90deg,#C1584F,#C9A24B,#55A87F); position:relative; margin:8px 0 4px; }
    .range-marker { position:absolute; top:-4px; width:12px; height:12px; border-radius:50%; background:var(--text); border:2px solid var(--ink); transform:translateX(-50%); }
    .range-labels { display:flex; justify-content:space-between; font-family:'IBM Plex Mono', monospace; font-size:0.68rem; color:var(--text-faint); }

    .news-item { padding:10px 4px; border-bottom:1px solid var(--line-soft); }
    .news-title { font-size:0.84rem; color:var(--text); }
    .news-meta { font-size:0.66rem; color:var(--text-faint); margin-top:3px; font-family:'IBM Plex Mono', monospace; }

    @media (max-width: 900px) {
        [data-testid="stTabs"] > div:first-child > div[data-baseweb="tab-list"] {
            position:fixed !important; bottom:0 !important; left:0 !important; right:0 !important; top:auto !important;
            z-index:9999 !important; background:var(--panel) !important; border-top:1px solid var(--line) !important;
            padding:8px 4px env(safe-area-inset-bottom, 10px) !important; margin:0 !important;
            display:flex !important; justify-content:space-around !important; box-shadow:0 -8px 24px rgba(0,0,0,0.4) !important;
            gap:0 !important; overflow-x:visible !important;
        }
        [data-testid="stTabs"] > div:first-child div[data-baseweb="tab"] {
            flex:1 !important; justify-content:center !important; padding:4px 2px !important; font-size:0.66rem !important;
            white-space:normal !important; text-align:center !important; line-height:1.25 !important;
            border-bottom:none !important; min-width:0 !important; color:var(--text-dim) !important;
        }
        [data-testid="stTabs"] > div:first-child div[aria-selected="true"][data-baseweb="tab"] {
            color:var(--gold) !important; border-bottom:2px solid var(--gold) !important;
        }
        [data-testid="stTabs"] [data-testid="stTabs"] div[data-baseweb="tab-list"] {
            position:static !important; box-shadow:none !important; border-top:none !important;
            padding:0 !important; justify-content:flex-start !important; overflow-x:auto !important; gap:2px !important;
        }
        [data-testid="stTabs"] > div > div[data-baseweb="tab-panel"] { padding-bottom:92px !important; }
        [data-testid="stTabs"] [data-testid="stTabs"] div[data-baseweb="tab"] { padding:6px 10px !important; font-size:0.76rem !important; }
        .hero-value { font-size:2.1rem; }
    }
</style>
""", unsafe_allow_html=True)

SIGNAL_COLOR = {'STRONG BUY': GAIN, 'BUY': GAIN, 'WATCH': INFO, 'HOLD': TEXT_DIM,
                'SELL': LOSS, 'STRONG SELL': LOSS, 'AVOID': LOSS}
SIGNAL_LABEL_HE = {'STRONG BUY': 'קנייה חזקה', 'BUY': 'קנייה', 'WATCH': 'מעקב', 'HOLD': 'החזק',
                   'SELL': 'מכירה', 'STRONG SELL': 'מכירה חזקה', 'AVOID': 'הימנע'}
DAYS_HE = ['שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת', 'ראשון']
PERIODS = {"יום": "1d", "שבוע": "1w", "חודש": "1m", "מתחילת השנה": "ytd"}
CHART_BARS = {"חודש": 21, "3 חודשים": 63, "6 חודשים": 126, "שנה": 252, "שנתיים": 504}

MARKET_TICKERS = {'SPY': ('S&P 500', '$'), 'QQQ': ('NASDAQ 100', '$'), '^VIX': ('VIX', ''),
                  'USDILS=X': ('דולר/שקל', '₪'), 'GLD': ('זהב', '$'), 'BTC-USD': ('ביטקוין', '$')}

INFO_KEYS = ['longName', 'shortName', 'sector', 'quoteType', 'currency', 'trailingPE', 'forwardPE',
             'pegRatio', 'beta', 'marketCap', 'dividendRate', 'revenueGrowth', 'earningsGrowth',
             'returnOnEquity', 'profitMargins', 'debtToEquity', 'recommendationKey',
             'targetMeanPrice', 'numberOfAnalystOpinions']

NASDAQ_100 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'AVGO', 'COST', 'NFLX',
              'AMD', 'CSCO', 'QCOM', 'INTC', 'INTU', 'AMGN', 'CMCSA', 'AMAT', 'TXN', 'BKNG',
              'VRTX', 'ISRG', 'ADP', 'SBUX', 'ADI', 'GILD', 'REGN', 'MU', 'LRCX', 'KLAC',
              'PANW', 'SNPS', 'CDNS', 'MELI', 'CRWD', 'FTNT', 'PCAR', 'ORLY', 'CTAS', 'NXPI']
TA35_TICKERS = ['TEVA.TA', 'ICL.TA', 'NICE.TA', 'ESLT.TA', 'BEZQ.TA', 'LUMI.TA', 'HARL.TA', 'POLI.TA',
                'DSCT.TA', 'PHOE.TA', 'AZRG.TA', 'ELAL.TA', 'MZTF.TA', 'MGDL.TA', 'NVMI.TA', 'TSEM.TA']
SCAN_UNIVERSES = {
    "Mega caps": ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'GOOGL', 'NFLX', 'AMD', 'AVGO',
                  'COST', 'JPM', 'V', 'MA', 'LLY', 'UNH', 'WMT', 'XOM'],
    "NASDAQ 100 (40)": NASDAQ_100,
    "S&P 500 (מדגם 40)": None,
    "Tech": ['AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'AMZN', 'AMD', 'INTC', 'AVGO', 'QCOM', 'TSM', 'ASML', 'CRM', 'ORCL', 'ADBE'],
    "Value": ['BRK-B', 'JPM', 'BAC', 'WFC', 'C', 'PEP', 'KO', 'JNJ', 'PG', 'MRK', 'ABBV', 'CVX', 'XOM', 'V', 'MA'],
    "Growth": ['NVDA', 'META', 'AMZN', 'TSLA', 'CRM', 'SNOW', 'DDOG', 'PLTR', 'RBLX', 'COIN', 'SHOP', 'NET', 'CRWD'],
    "Dividend": ['JNJ', 'PG', 'KO', 'PEP', 'MRK', 'ABBV', 'CVX', 'XOM', 'T', 'VZ', 'IBM', 'MO', 'PM', 'O', 'EPD', 'ENB'],
    "ETFs": ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'GLD', 'TLT'],
    "ת״א (TA)": TA35_TICKERS,
}

# ==========================================
# FORMATTING
# ==========================================
def num(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default

def is_ils(t):
    return str(t).upper().endswith('.TA')

def money(v, sym='$', dec=2):
    return f"{'-' if v < 0 else ''}{sym}{abs(v):,.{dec}f}"

def signed_money(v, sym='$', dec=0):
    return f"{'+' if v >= 0 else '-'}{sym}{abs(v):,.{dec}f}"

def pct(v, dec=2):
    return f"{v:+.{dec}f}%"

def tone(v):
    return GAIN if v >= 0 else LOSS

def esc(s):
    return html.escape(str(s if s is not None else ''))

def fmt_qty(q):
    return f"{q:,.4f}".rstrip('0').rstrip('.')

def big(v):
    for div, suf in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M')):
        if abs(v) >= div:
            return f"${v / div:,.2f}{suf}"
    return f"${v:,.0f}" if v else "—"

def now_il():
    return datetime.now(TZ)

def to_ts(v):
    try:
        ts = pd.Timestamp(v)
        if pd.isna(ts):
            return None
        return ts.tz_convert(None) if ts.tzinfo else ts
    except (TypeError, ValueError):
        return None

def flash(kind, msg):
    st.session_state['_flash'] = (kind, msg)

def show_flash():
    item = st.session_state.pop('_flash', None)
    if item:
        getattr(st, item[0])(item[1])

# ==========================================
# STORAGE (Google Sheets, local file fallback)
# ==========================================
PORT_COLS = ['Ticker', 'Quantity', 'PurchasePrice']
WATCH_COLS = ['Ticker', 'Notes', 'AlertHigh', 'AlertLow']
TRADE_COLS = ['Date', 'Ticker', 'Action', 'Quantity', 'Price', 'Currency', 'Realized']
CASHFLOW_COLS = ['Date', 'Type', 'Amount', 'Currency', 'AmountUSD', 'Note']
CONFIG_DEFAULTS = {"cash_usd": 86.67, "cash_ils": 0.0, "initial_investment": 7000.0}
SAVE_MSG = {
    'cloud': ('success', 'נשמר ב-Google Sheets'),
    'local': ('warning', 'נשמר מקומית בלבד — אין חיבור ל-Google Sheets, והשינוי יימחק בהפעלה מחדש של האפליקציה בענן'),
    'failed': ('error', 'השמירה נכשלה — בדוק/י את חיבור Google Sheets ואת הרשאות העריכה של חשבון השירות'),
}

try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception:
    conn = None

http = requests.Session()
http.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

def load_table(ws, local_path, columns):
    """Returns (df, source) where source is 'cloud', 'local' or 'none'."""
    if conn is not None:
        try:
            df = conn.read(worksheet=ws, ttl=0)
            if df is not None:
                return df.dropna(how='all').dropna(axis=1, how='all'), 'cloud'
        except Exception:
            pass
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path), 'local'
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    return pd.DataFrame(columns=columns), 'none'

def persist(ws, df, local_path):
    """Write to Google Sheets (creating the worksheet if missing); fall back to a local file."""
    if conn is not None:
        try:
            conn.update(worksheet=ws, data=df)
            return 'cloud'
        except Exception:
            try:
                conn.create(worksheet=ws, data=df)
                return 'cloud'
            except Exception:
                pass
    try:
        df.to_csv(local_path, index=False)
        return 'local'
    except OSError:
        return 'failed'

def worst(*statuses):
    for s in ('failed', 'local'):
        if s in statuses:
            return s
    return 'cloud'

def clean_portfolio(data):
    df = data.reset_index() if 'Ticker' not in data.columns else data
    df = df.reindex(columns=PORT_COLS).copy()
    df['Ticker'] = df['Ticker'].fillna('').astype(str).str.strip().str.upper()
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0.0).astype(float)
    df['PurchasePrice'] = pd.to_numeric(df['PurchasePrice'], errors='coerce').fillna(0.0).astype(float)
    df = df[(df['Ticker'] != '') & (df['Ticker'] != 'NAN') & (df['Quantity'] > 0) & (df['PurchasePrice'] > 0)]
    return df.drop_duplicates('Ticker', keep='last').set_index('Ticker')[['Quantity', 'PurchasePrice']]

def clean_watchlist(data):
    df = data.reindex(columns=WATCH_COLS).copy()
    df['Ticker'] = df['Ticker'].fillna('').astype(str).str.strip().str.upper()
    df['Notes'] = df['Notes'].fillna('').astype(str).replace('nan', '')
    for c in ('AlertHigh', 'AlertLow'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype(float)
    df = df[(df['Ticker'] != '') & (df['Ticker'] != 'NAN')]
    return df.drop_duplicates('Ticker', keep='last').reset_index(drop=True)

def load_config():
    df, src = load_table('Config', 'config.csv', ['Key', 'Value'])
    cfg = dict(CONFIG_DEFAULTS)
    if not df.empty and {'Key', 'Value'} <= set(df.columns):
        for k, v in zip(df['Key'].astype(str).str.strip(), df['Value']):
            if k in cfg:
                cfg[k] = num(v, cfg[k])
    elif src == 'none' and os.path.exists('config.json'):
        try:
            with open('config.json') as f:
                cfg.update({k: num(v, cfg[k]) for k, v in json.load(f).items() if k in cfg})
        except (OSError, ValueError):
            pass
    return cfg

def save_config(cfg):
    return persist('Config', pd.DataFrame([{'Key': k, 'Value': float(v)} for k, v in cfg.items()]), 'config.csv')

def save_portfolio(df):
    return persist('Portfolio', df.reset_index(), 'portfolio_data.csv')

def save_watchlist(df):
    return persist('Watchlist', df, 'watchlist.csv')

def append_log(state_key, ws, path, row):
    df = pd.concat([st.session_state[state_key], pd.DataFrame([row])], ignore_index=True)
    st.session_state[state_key] = df
    return persist(ws, df, path)

DEFAULT_PORTFOLIO = clean_portfolio(pd.DataFrame([
    {'Ticker': 'MSFT', 'Quantity': 7.4156, 'PurchasePrice': 371.17},
    {'Ticker': 'VOO', 'Quantity': 4.5496, 'PurchasePrice': 683.57},
    {'Ticker': 'META', 'Quantity': 3.0, 'PurchasePrice': 559.56},
    {'Ticker': 'ESLT', 'Quantity': 1.0, 'PurchasePrice': 780.25},
    {'Ticker': 'MU', 'Quantity': 2.0, 'PurchasePrice': 993.89},
]))

if 'portfolio' not in st.session_state:
    raw_port, port_src = load_table('Portfolio', 'portfolio_data.csv', PORT_COLS)
    st.session_state.portfolio = clean_portfolio(raw_port) if port_src != 'none' else DEFAULT_PORTFOLIO.copy()
    st.session_state.storage_src = port_src
    st.session_state.config = load_config()
    st.session_state.watchlist = clean_watchlist(load_table('Watchlist', 'watchlist.csv', WATCH_COLS)[0])
    st.session_state.trades = load_table('Trades', 'trades.csv', TRADE_COLS)[0].reindex(columns=TRADE_COLS)
    st.session_state.cashflow = load_table('Cashflow', 'cashflow.csv', CASHFLOW_COLS)[0].reindex(columns=CASHFLOW_COLS)
    st.session_state.editor_ver = 0

# ==========================================
# MARKET DATA
# ==========================================
OHLC = ['Open', 'High', 'Low', 'Close']

@st.cache_data(ttl=300, show_spinner=False)
def download_history(tickers, period='2y'):
    """Daily OHLCV per ticker. TASE (.TA) prices arrive in agorot and are converted to shekels."""
    tickers = [t for t in tickers if t]
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}
    out = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(-1):
                    continue
                h = raw.xs(t, axis=1, level=-1)
            elif len(tickers) == 1:
                h = raw
            else:
                continue
            h = h.reindex(columns=OHLC + ['Volume']).dropna(subset=['Close']).copy()
            if len(h) < 2:
                continue
            if h.index.tz is not None:
                h.index = h.index.tz_localize(None)
            if is_ils(t) and not t.startswith('^'):
                h[OHLC] = h[OHLC] / 100
            h['Volume'] = h['Volume'].fillna(0)
            out[t] = h
        except Exception:
            continue
    return out

@st.cache_data(ttl=21600, show_spinner=False)
def get_infos(tickers):
    def one(t):
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        return t, {k: info.get(k) for k in INFO_KEYS}
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(one, tickers))

def compute_indicators(hist):
    close, high, low, vol = hist['Close'], hist['High'], hist['Low'], hist['Volume']
    curr = float(close.iloc[-1])
    ind = {}

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ind['rsi'] = num(rsi.iloc[-1], 100.0 if loss.iloc[-1] == 0 else 50.0)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    ind['macd'], ind['macd_signal'] = num(macd.iloc[-1]), num(sig.iloc[-1])
    ind['macd_crossover'] = bool(macd.iloc[-1] > sig.iloc[-1] and macd.iloc[-2] <= sig.iloc[-2])

    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    ind['stoch_k'] = num((100 * (close - low14) / (high14 - low14 + 1e-9)).iloc[-1], 50.0)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    ind['atr_pct'] = num(tr.rolling(14).mean().iloc[-1]) / curr * 100

    avg_vol = num(vol.rolling(20).mean().iloc[-1])
    ind['vol_ratio'] = num(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0
    obv = (np.sign(delta) * vol).fillna(0).cumsum()
    ind['obv_trend'] = 1 if len(obv) >= 20 and obv.iloc[-1] > obv.rolling(20).mean().iloc[-1] else -1

    ind['mom_1m'] = (curr / close.iloc[-22] - 1) * 100 if len(close) > 21 else 0.0
    ind['mom_3m'] = (curr / close.iloc[-64] - 1) * 100 if len(close) > 63 else 0.0

    def sma(p, fallback):
        return num(close.rolling(p).mean().iloc[-1], fallback) if len(close) >= p else fallback
    s20 = sma(20, curr)
    s50 = sma(50, s20)
    s200 = sma(200, s50)
    ind['sma20'], ind['sma50'], ind['sma200'] = s20, s50, s200

    if s20 > s50 > s200 and curr > s20:
        ind['trend'], ind['trend_score'] = "מגמת עלייה חזקה", 5
    elif s20 > s50 and curr > s20:
        ind['trend'], ind['trend_score'] = "מגמת עלייה מתונה", 4
    elif s20 < s50 < s200 and curr < s20:
        ind['trend'], ind['trend_score'] = "מגמת ירידה חזקה", 1
    elif s20 < s50 and curr < s20:
        ind['trend'], ind['trend_score'] = "מגמת ירידה מתונה", 2
    else:
        ind['trend'], ind['trend_score'] = "דשדוש", 3

    ind['support'] = num(low.iloc[-20:].min(), curr)
    ind['resistance'] = num(high.iloc[-20:].max(), curr)
    return ind

def compute_composite_score(ind, info):
    score, reasons = 50, []
    ts = ind['trend_score']
    score += (ts - 3) * 4
    if ts >= 5: reasons.append("מגמת עלייה חזקה")
    elif ts <= 2: reasons.append("מגמת ירידה")

    rsi = ind['rsi']
    if rsi < 30: score += 10; reasons.append("RSI במכירת יתר")
    elif rsi < 40: score += 6; reasons.append("אזור מכירת יתר")
    elif rsi <= 60: score += 4
    elif rsi > 80: score -= 12; reasons.append("RSI בקניית יתר קיצונית")
    elif rsi > 70: score -= 8; reasons.append("קניית יתר")

    if ind['macd_crossover']: score += 8; reasons.append("חציית MACD חיובית")
    elif ind['macd'] > ind['macd_signal']: score += 3
    else: score -= 4; reasons.append("MACD מתחת לקו האות")

    vr = ind['vol_ratio']
    if vr > 2.0 and ts >= 4: score += 8; reasons.append("נפח חריג במגמת עלייה")
    elif vr > 1.5 and ts >= 4: score += 5
    elif vr > 1.5 and ts <= 2: score -= 6; reasons.append("נפח גבוה במגמת ירידה")

    m1, m3 = ind['mom_1m'], ind['mom_3m']
    if m1 > 10 and m3 > 20: score += 10; reasons.append("מומנטום חזק מאוד")
    elif m1 > 5 and m3 > 10: score += 6; reasons.append("מומנטום חיובי")
    elif m1 > 0 and m3 > 0: score += 3
    elif m1 < -15: score -= 10; reasons.append("מומנטום שלילי חד")
    elif m1 < -8: score -= 6

    pe = num(info.get('trailingPE'))
    if 0 < pe < 12: score += 10; reasons.append("P/E נמוך מאוד")
    elif 0 < pe < 20: score += 6; reasons.append("P/E נמוך")
    elif 20 <= pe < 35: score += 2
    elif pe > 60: score -= 8; reasons.append("P/E גבוה מאוד")
    elif pe > 40: score -= 4

    growth = num(info.get('revenueGrowth')) * 100
    if growth > 30: score += 8; reasons.append("צמיחת הכנסות מעל 30%")
    elif growth > 15: score += 5
    elif growth > 5: score += 2
    elif growth < 0: score -= 6; reasons.append("ירידה בהכנסות")

    roe = num(info.get('returnOnEquity')) * 100
    if roe > 25: score += 5; reasons.append("ROE מצוין")
    elif roe > 15: score += 3
    elif roe < 0: score -= 4

    rec = info.get('recommendationKey') or 'none'
    if rec in ('strong_buy', 'buy'): score += 5; reasons.append("קונצנזוס אנליסטים: קנייה")
    elif rec in ('sell', 'strong_sell'): score -= 6; reasons.append("קונצנזוס אנליסטים: מכירה")

    if ind['obv_trend'] == 1 and ts >= 4: score += 3
    if ind['stoch_k'] < 20: score += 4
    elif ind['stoch_k'] > 85: score -= 4
    return min(100, max(0, int(score))), reasons

def get_signal(score, rsi, trend_score):
    if score >= 75 and rsi < 65 and trend_score >= 5: return "STRONG BUY"
    if score >= 65 and trend_score >= 4: return "BUY"
    if score >= 55 and trend_score >= 3: return "WATCH"
    if score <= 25 and trend_score <= 1: return "STRONG SELL"
    if score <= 38: return "SELL"
    if 43 <= score <= 57: return "HOLD"
    if score > 57: return "WATCH"
    return "AVOID"

def build_snapshot(t, hist, info):
    close = hist['Close']
    price, prev = float(close.iloc[-1]), float(close.iloc[-2])

    def base(n):
        return float(close.iloc[-n - 1]) if len(close) > n else float(close.iloc[0])
    before_year = close[close.index < pd.Timestamp(now_il().year, 1, 1)]
    bases = {'1d': prev, '1w': base(5), '1m': base(21), '3m': base(63),
             'ytd': float(before_year.iloc[-1]) if len(before_year) else float(close.iloc[0]),
             '1y': base(252)}
    last_year = hist.iloc[-252:]
    hi52 = num(last_year['High'].max(), price)
    lo52 = num(last_year['Low'].min(), price)

    ind = compute_indicators(hist) if len(hist) >= 30 else {}
    if ind:
        score, reasons = compute_composite_score(ind, info)
        signal = get_signal(score, ind['rsi'], ind['trend_score'])
    else:
        score, reasons, signal = 50, [], 'HOLD'

    scale = 0.01 if is_ils(t) else 1.0  # Yahoo quotes TASE targets/dividends in agorot
    quote_type = info.get('quoteType') or ''
    return {
        'ticker': t, 'name': info.get('longName') or info.get('shortName') or t,
        'sector': info.get('sector') or ('ETF' if quote_type == 'ETF' else '—'), 'quote_type': quote_type,
        'currency': 'ILS' if is_ils(t) else (info.get('currency') or 'USD'),
        'sym': '₪' if is_ils(t) else '$', 'price': price, 'prev': prev, 'day_chg': price - prev,
        'bases': bases, 'rets': {k: (price / v - 1) * 100 if v else 0.0 for k, v in bases.items()},
        'hi52': hi52, 'lo52': lo52, 'last_date': close.index[-1],
        'pe': num(info.get('trailingPE')), 'forward_pe': num(info.get('forwardPE')),
        'peg': num(info.get('pegRatio')), 'beta': num(info.get('beta'), None),
        'market_cap': num(info.get('marketCap')), 'div_rate': num(info.get('dividendRate')) * scale,
        'growth': num(info.get('revenueGrowth')) * 100, 'roe': num(info.get('returnOnEquity')) * 100,
        'margin': num(info.get('profitMargins')) * 100, 'debt_eq': num(info.get('debtToEquity')),
        'rec': info.get('recommendationKey') or '—', 'target': num(info.get('targetMeanPrice')) * scale,
        'n_analysts': int(num(info.get('numberOfAnalystOpinions'))),
        **ind, 'score': score, 'reasons': reasons, 'signal': signal,
    }

@st.cache_data(ttl=300, show_spinner=False)
def load_market(tickers, with_info=True):
    """Returns (histories, snapshots) for the given tickers."""
    hists = download_history(tuple(tickers), '2y')
    infos = get_infos(tuple(sorted(hists))) if with_info else {}
    snaps = {}
    for t, h in hists.items():
        try:
            snaps[t] = build_snapshot(t, h, infos.get(t, {}))
        except Exception:
            continue
    return hists, snaps

@st.cache_data(ttl=600, show_spinner=False)
def get_usd_ils_fallback():
    try:
        r = http.get("https://api.frankfurter.app/latest?from=USD&to=ILS", timeout=8)
        r.raise_for_status()
        rate = float(r.json()['rates']['ILS'])
        return rate if np.isfinite(rate) and rate > 0 else None
    except (OSError, ValueError, KeyError, requests.RequestException):
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_tickers():
    try:
        resp = http.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', timeout=10)
        return [t.replace('.', '-') for t in pd.read_html(StringIO(resp.text))[0]['Symbol'].tolist()]
    except Exception:
        return SCAN_UNIVERSES["Mega caps"]

def _fetch_earnings(t):
    out = {'next': None, 'eps_est': None, 'last': None}
    tk = yf.Ticker(t)
    try:
        cal = tk.calendar
        ed = est = None
        if isinstance(cal, dict):
            ed, est = cal.get('Earnings Date'), cal.get('Earnings Average', cal.get('EPS Estimate'))
        elif isinstance(cal, pd.DataFrame) and not cal.empty and 'Earnings Date' in cal.index:
            ed = cal.loc['Earnings Date'].iloc[0]
            est = cal.loc['EPS Estimate'].iloc[0] if 'EPS Estimate' in cal.index else None
        if isinstance(ed, (list, tuple)):
            ed = ed[0] if ed else None
        out['next'] = to_ts(ed) if ed is not None else None
        out['eps_est'] = num(est, None) if est is not None else None
    except Exception:
        pass
    try:
        eh = tk.earnings_history
        if isinstance(eh, pd.DataFrame) and not eh.empty:
            eh = eh.sort_index()
            last = eh.iloc[-1]
            act, est = num(last.get('epsActual'), None), num(last.get('epsEstimate'), None)
            surprise = (act - est) / abs(est) * 100 if act is not None and est not in (None, 0) else None
            out['last'] = {'date': to_ts(eh.index[-1]), 'actual': act, 'estimate': est, 'surprise': surprise}
    except Exception:
        pass
    return t, out

@st.cache_data(ttl=21600, show_spinner=False)
def get_earnings_many(tickers):
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(_fetch_earnings, tickers))

def earnings_board(tickers, snaps):
    skip = ('ETF', 'MUTUALFUND', 'INDEX', 'CRYPTOCURRENCY', 'CURRENCY')
    eligible = tuple(sorted(t for t in set(tickers) if snaps.get(t, {}).get('quote_type') not in skip))
    data = get_earnings_many(eligible)
    today = pd.Timestamp(now_il().date())
    upcoming, recent = [], []
    for t, e in data.items():
        if e['next'] is not None:
            days = (e['next'].normalize() - today).days
            if days >= 0:
                upcoming.append({'ticker': t, 'date': e['next'], 'days': days, 'eps_est': e['eps_est']})
        if e['last'] and e['last']['date'] is not None:
            recent.append({'ticker': t, **e['last']})
    upcoming.sort(key=lambda x: x['days'])
    recent.sort(key=lambda x: x['date'], reverse=True)
    return upcoming, recent

def parse_news(item):
    c = item.get('content') if isinstance(item.get('content'), dict) else item
    url = ((c.get('canonicalUrl') or {}).get('url') or (c.get('clickThroughUrl') or {}).get('url')
           or item.get('link') or '')
    publisher = (c.get('provider') or {}).get('displayName') or item.get('publisher') or ''
    ts = None
    try:
        if c.get('pubDate'):
            ts = pd.Timestamp(c['pubDate'])
        elif item.get('providerPublishTime'):
            ts = pd.Timestamp(int(item['providerPublishTime']), unit='s')
        if ts is not None:
            ts = (ts.tz_localize('UTC') if ts.tzinfo is None else ts).tz_convert(TZ)
    except (TypeError, ValueError):
        ts = None
    return {'title': c.get('title') or '', 'url': url, 'publisher': publisher, 'ts': ts}

@st.cache_data(ttl=1800, show_spinner=False)
def get_news(tickers, per=4):
    def one(t):
        try:
            items = yf.Ticker(t).news or []
        except Exception:
            items = []
        return [{**parse_news(i), 'ticker': t} for i in items[:per]]
    if not tickers:
        return []
    with ThreadPoolExecutor(max_workers=6) as ex:
        batches = list(ex.map(one, tickers))
    seen, news = set(), []
    for n in (n for b in batches for n in b):
        key = n['url'] or n['title']
        if n['title'] and key not in seen:
            seen.add(key)
            news.append(n)
    news.sort(key=lambda n: n['ts'].timestamp() if n['ts'] is not None else 0, reverse=True)
    return news

# ==========================================
# PORTFOLIO MATH
# ==========================================
def compute_positions(portfolio, snaps, rate):
    rows, missing = [], []
    for t, r in portfolio.iterrows():
        s = snaps.get(t)
        if s is None:
            missing.append(t)
            continue
        qty, avg = float(r['Quantity']), float(r['PurchasePrice'])
        k = 1 / rate if s['currency'] == 'ILS' else 1.0
        value, cost = s['price'] * qty * k, avg * qty * k
        row = {'ticker': t, 'name': s['name'], 'sector': s['sector'], 'quote_type': s['quote_type'],
               'qty': qty, 'avg': avg, 'sym': s['sym'], 'price': s['price'], 'value': value, 'cost': cost,
               'unreal': value - cost, 'unreal_pct': (s['price'] / avg - 1) * 100 if avg > 0 else 0.0,
               'day_pct': s['rets']['1d'], 'div_annual': s['div_rate'] * qty * k}
        for p, b in s['bases'].items():
            row[f'pl_{p}'] = (s['price'] - b) * qty * k
        rows.append(row)
    return rows, missing

def apply_trade(portfolio, ticker, action, qty, price):
    """Returns (new_portfolio, executed_qty, realized_pnl_in_quote_currency)."""
    p = portfolio.copy()
    if action == 'buy':
        if ticker in p.index:
            oq, op = float(p.loc[ticker, 'Quantity']), float(p.loc[ticker, 'PurchasePrice'])
            nq = oq + qty
            p.loc[ticker, ['Quantity', 'PurchasePrice']] = [nq, (oq * op + qty * price) / nq]
        else:
            p.loc[ticker] = [float(qty), float(price)]
        return p, qty, 0.0
    if ticker not in p.index:
        return p, 0.0, 0.0
    held, avg = float(p.loc[ticker, 'Quantity']), float(p.loc[ticker, 'PurchasePrice'])
    sold = min(qty, held)
    if held - sold <= 1e-9:
        p = p.drop(ticker)
    else:
        p.loc[ticker, 'Quantity'] = held - sold
    return p, sold, (price - avg) * sold

def risk_analytics(tickers, hists):
    cols = [t for t in tickers if t in hists]
    if not cols or 'SPY' not in hists:
        return pd.DataFrame(), None
    closes = pd.DataFrame({t: hists[t]['Close'] for t in cols + ['SPY']}).sort_index().ffill().iloc[-253:]
    rets = closes.pct_change(fill_method=None).iloc[1:]
    rows = []
    for t in cols:
        pair = pd.concat([rets[t], rets['SPY']], axis=1, keys=['r', 'm']).dropna()
        if len(pair) < 20:
            continue
        r, m = pair['r'], pair['m']
        vol = r.std() * np.sqrt(252)
        c = closes[t].dropna()
        rows.append({
            'סימול': t,
            'בטא': r.cov(m) / m.var() if m.var() > 0 else np.nan,
            'תנודתיות שנתית %': vol * 100,
            'שארפ': (r.mean() * 252 - RISK_FREE) / vol if vol > 0 else np.nan,
            'ירידה מקסימלית %': (c / c.cummax() - 1).min() * 100,
            'VaR 95% יומי': np.percentile(r, 5) * 100,
            'מתאם ל-S&P': r.corr(m),
        })
    corr = rets[cols].corr() if len(cols) > 1 else None
    return pd.DataFrame(rows), corr

def parse_tickers(text, limit=25):
    out = []
    for tok in re.split(r'[\s,;]+', (text or '').upper()):
        if re.fullmatch(r'[A-Z0-9.\-^=]{1,15}', tok) and tok not in out:
            out.append(tok)
    return out[:limit]

# ==========================================
# SCREENSHOT IMPORT (best-effort OCR)
# ==========================================
TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')
NUMBER_PATTERN = re.compile(r'\d+(?:[.,]\d+)?')
EXCLUDE_WORDS = {'THE', 'AND', 'FOR', 'QTY', 'AVG', 'USD', 'ILS', 'BUY', 'SELL', 'NEW', 'ETF', 'ALL', 'TOP',
                 'LOW', 'HIGH', 'DAY', 'ASK', 'BID', 'NAV', 'CASH', 'TOTAL', 'VALUE', 'PRICE', 'SHARES',
                 'COST', 'GAIN', 'LOSS', 'TODAY', 'YTD', 'MKT', 'CAP'}

def parse_portfolio_image(image, known_tickers):
    """Extracts ticker/qty/price rows from a broker screenshot. Always reviewed manually before saving."""
    image = image.convert('RGB')
    image.thumbnail((1800, 1800))
    text = pytesseract.image_to_string(image, config='--psm 6').upper()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for i, line in enumerate(lines):
        found = [t for t in known_tickers if re.search(rf'\b{re.escape(t)}\b', line)]
        found = found or [w for w in TICKER_PATTERN.findall(line) if w not in EXCLUDE_WORDS]
        context = ' '.join(lines[i:i + 8])
        nums = [num(n.replace(',', ''), None) for n in NUMBER_PATTERN.findall(context)]
        nums = [n for n in nums if n is not None]
        if not found or not nums:
            continue
        qty_m = re.search(r'(?:QTY|QUANTITY|כמות)\D{0,20}(\d+(?:\.\d+)?)', context)
        price_m = re.search(r'(?:AVG|AVERAGE|PURCHASE|קניה|קנייה)\D{0,30}(\d[\d,.]*)', context)
        qty = num(qty_m.group(1)) if qty_m else min(nums, key=lambda n: abs(n - round(n)))
        price = num(price_m.group(1).replace(',', '')) if price_m else (nums[1] if len(nums) > 1 else 0.0)
        if qty > 0 and price > 0:
            rows.append({'Ticker': found[0], 'Quantity': qty, 'PurchasePrice': price})
    return pd.DataFrame(rows, columns=PORT_COLS).drop_duplicates('Ticker', keep='first')

# ==========================================
# UI HELPERS
# ==========================================
def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)

def ledger_row(ticker, subtitle, price_str, sub_str, sub_color, tag_text, tag_color, caption=None, total=False):
    cap = f'<div class="row-caption">{esc(caption)}</div>' if caption else ''
    return (f'<div class="ledger-row{" total" if total else ""}">'
            f'<div><div class="row-ticker">{esc(ticker)}</div><div class="row-name">{esc(subtitle)}</div></div>'
            f'<div style="text-align:center"><div class="row-price">{esc(price_str)}</div>'
            f'<div class="row-sub" style="color:{sub_color}">{esc(sub_str)}</div></div>'
            f'<div style="text-align:right"><div class="row-tag" style="color:{tag_color}">'
            f'<span class="dot" style="background:{tag_color}"></span>{esc(tag_text)}</div>{cap}</div></div>')

def render_ledger(rows_html):
    st.markdown('<div>' + ''.join(rows_html) + '</div>', unsafe_allow_html=True)

def attn(ticker, msg, color=GOLD):
    st.markdown(f'<div class="attn-strip" style="border-color:{color}"><div class="row-ticker">{esc(ticker)}</div>'
                f'<div class="row-name">{esc(msg)}</div></div>', unsafe_allow_html=True)

def strip(items):
    """items: (label, value, sub, color) — color applies to sub, or to value when sub is empty."""
    cells = []
    for label, value, sub, color in items:
        vstyle = '' if sub else f' style="color:{color}"'
        sub_html = f'<div class="idx-chg" style="color:{color}">{esc(sub)}</div>' if sub else ''
        cells.append(f'<div class="idx-item"><div class="idx-name">{esc(label)}</div>'
                     f'<div class="idx-price"{vstyle}>{esc(value)}</div>{sub_html}</div>')
    st.markdown(f'<div class="idx-strip">{"".join(cells)}</div>', unsafe_allow_html=True)

def range_bar(lo, hi, price, sym):
    pos = 50.0 if hi <= lo else min(100.0, max(0.0, (price - lo) / (hi - lo) * 100))
    st.markdown('<div class="hero-label">טווח 52 שבועות</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="range-wrap"><div class="range-bar"><div class="range-marker" style="left:{pos:.1f}%"></div></div>'
                f'<div class="range-labels"><span>{esc(money(lo, sym))}</span><span>{esc(money(hi, sym))}</span></div></div>',
                unsafe_allow_html=True)

def render_news(items):
    if not items:
        st.info("אין כרגע כותרות זמינות.")
        return
    out = []
    for n in items:
        title = esc(n['title'])
        if n['url'].startswith('http'):
            title = f'<a href="{esc(n["url"])}" target="_blank" rel="noopener" style="color:var(--text);text-decoration:none">{title}</a>'
        when = n['ts'].strftime('%d.%m %H:%M') if n['ts'] is not None else ''
        out.append(f'<div class="news-item"><div class="news-title">{title}</div>'
                   f'<div class="news-meta">{esc(n["ticker"])} · {esc(n["publisher"])} · {esc(when)}</div></div>')
    st.markdown(''.join(out), unsafe_allow_html=True)

def dark_layout(fig, height):
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=PANEL, font_color=TEXT,
                      margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(gridcolor=LINE)
    fig.update_yaxes(gridcolor=LINE)
    return fig

def price_chart(hist, bars, cost=None, sym='$', skip_weekends=True):
    smas = {p: hist['Close'].rolling(p).mean().iloc[-bars:] for p in (20, 50, 200)}
    h = hist.iloc[-bars:]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=h.index, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'],
                                 name='מחיר', showlegend=False,
                                 increasing_line_color=GAIN, decreasing_line_color=LOSS), row=1, col=1)
    for p, color in ((20, GOLD), (50, INFO), (200, LOSS)):
        if smas[p].notna().any():
            fig.add_trace(go.Scatter(x=smas[p].index, y=smas[p], name=f'SMA{p}', line=dict(color=color, width=1.3)), row=1, col=1)
    if cost:
        fig.add_hline(y=cost, line_dash='dash', line_color=TEXT_DIM, row=1, col=1,
                      annotation_text=f'מחיר קנייה {sym}{cost:,.2f}', annotation_position='top left')
    fig.add_trace(go.Bar(x=h.index, y=h['Volume'], name='נפח', showlegend=False,
                         marker_color='rgba(107,147,190,0.35)'), row=2, col=1)
    dark_layout(fig, 460)
    fig.update_layout(legend=dict(orientation='h', y=1.05, font=dict(size=10)))
    fig.update_xaxes(rangeslider_visible=False)
    if skip_weekends:
        fig.update_xaxes(rangebreaks=[dict(bounds=['sat', 'mon'])])
    return fig

def render_stock_detail(t, s, hist, pos=None, key='detail'):
    sym = s['sym']
    st.markdown(f'<div class="row-ticker" style="font-size:1.35rem">{esc(t)}</div>'
                f'<div class="row-name">{esc(s["name"])} · {esc(s["sector"])}</div>'
                f'<div class="stamp">נכון לסגירה/מחיר אחרון של {s["last_date"]:%d.%m.%Y}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("מחיר", money(s['price'], sym), f"{pct(s['rets']['1d'])} ({signed_money(s['day_chg'], sym, 2)})")
    if pos:
        c2.metric("שווי הפוזיציה", money(pos['value']), f"{signed_money(pos['pl_1d'])} היום")
        c3, c4 = st.columns(2)
        c3.metric("רווח/הפסד מאז הקנייה", signed_money(pos['unreal']), pct(pos['unreal_pct']))
        c4.metric("מחיר קנייה ממוצע", money(pos['avg'], sym),
                  f"{fmt_qty(pos['qty'])} יח' · {pos['weight']:.1f}% מהתיק", delta_color='off')
    else:
        c2.metric("מתחילת השנה", pct(s['rets']['ytd']), signed_money(s['price'] - s['bases']['ytd'], sym, 2))

    strip([(lbl, pct(s['rets'][k], 1), '', tone(s['rets'][k])) for lbl, k in
           (("יום", '1d'), ("שבוע", '1w'), ("חודש", '1m'), ("3 חודשים", '3m'), ("מתחילת שנה", 'ytd'), ("שנה", '1y'))])
    range_bar(s['lo52'], s['hi52'], s['price'], sym)

    period = st.radio("טווח גרף", list(CHART_BARS), index=2, horizontal=True, key=f'{key}_period_{t}')
    weekend_trading = is_ils(t) or s['quote_type'] == 'CRYPTOCURRENCY' or t.endswith('-USD')
    st.plotly_chart(price_chart(hist, CHART_BARS[period], pos['avg'] if pos else None, sym, not weekend_trading),
                    width='stretch', config={"displayModeBar": False}, key=f'{key}_chart_{t}')

    if 'rsi' in s:
        st.markdown(f'<div class="stamp">{esc(s["trend"])} · RSI {s["rsi"]:.0f} · '
                    f'טווח 20 יום {esc(money(s["support"], sym))}–{esc(money(s["resistance"], sym))} · '
                    f'קריאה טכנית: {esc(SIGNAL_LABEL_HE[s["signal"]])} ({s["score"]}/100)</div>', unsafe_allow_html=True)
        if s['reasons']:
            with st.expander("על מה מבוססת הקריאה הטכנית"):
                for reason in s['reasons']:
                    st.markdown(f"· {reason}")

    with st.expander("נתונים פונדמנטליים"):
        upside = (s['target'] / s['price'] - 1) * 100 if s['target'] else None
        div_yield = s['div_rate'] / s['price'] * 100 if s['div_rate'] else 0.0
        f1, f2 = st.columns(2)
        f1.metric("P/E", f"{s['pe']:.1f}" if s['pe'] else "—")
        f2.metric("P/E עתידי", f"{s['forward_pe']:.1f}" if s['forward_pe'] else "—")
        f3, f4 = st.columns(2)
        f3.metric("שווי שוק", big(s['market_cap']))
        f4.metric("תשואת דיבידנד", f"{div_yield:.2f}%" if div_yield else "—")
        f5, f6 = st.columns(2)
        f5.metric("יעד אנליסטים", money(s['target'], sym) if s['target'] else "—",
                  pct(upside, 1) if upside is not None else None)
        f6.metric("המלצה", s['rec'].replace('_', ' '), f"{s['n_analysts']} אנליסטים" if s['n_analysts'] else None,
                  delta_color='off')
        if s['quote_type'] != 'ETF':
            st.caption(f"צמיחת הכנסות {s['growth']:.1f}% · שולי רווח {s['margin']:.1f}% · ROE {s['roe']:.1f}%")

    if s['quote_type'] not in ('ETF', 'CRYPTOCURRENCY', 'INDEX', 'CURRENCY'):
        e = get_earnings_many((t,)).get(t, {})
        if e.get('next') is not None:
            est = f" · תחזית EPS {money(e['eps_est'], sym)}" if e.get('eps_est') is not None else ''
            st.markdown(f'<div class="stamp">דוח רווחים הבא: {e["next"]:%d.%m.%Y}{esc(est)}</div>', unsafe_allow_html=True)

    with st.expander("חדשות"):
        render_news(get_news((t,), per=6))

def render_cash_editor(form_key):
    cfg = st.session_state.config
    with st.form(form_key):
        c1, c2 = st.columns(2)
        usd = c1.number_input("מזומן בדולר ($)", value=float(cfg['cash_usd']), step=10.0, format="%.2f")
        ils = c2.number_input("מזומן בשקל (₪)", value=float(cfg['cash_ils']), step=10.0, format="%.2f")
        inv = st.number_input("סך הפקדות נטו ($)", value=float(cfg['initial_investment']), step=100.0, format="%.2f",
                              help="כמה כסף הפקדת בסך הכול פחות משיכות. משמש לחישוב התשואה הכוללת.")
        if st.form_submit_button("שמור יתרות"):
            new_cfg = {'cash_usd': usd, 'cash_ils': ils, 'initial_investment': inv}
            st.session_state.config = new_cfg
            flash(*SAVE_MSG[save_config(new_cfg)])
            st.rerun()

def styled_table(df, pct_cols, fmt):
    def color(v):
        return f'color:{GAIN if v >= 0 else LOSS}' if isinstance(v, (int, float)) and np.isfinite(v) else ''
    return df.style.format(fmt, na_rep='—').map(color, subset=pct_cols)

# ==========================================
# LOAD DATA
# ==========================================
show_flash()
if st.session_state.storage_src != 'cloud':
    st.warning("אין חיבור ל-Google Sheets — מוצגים נתונים "
               + ("מקובץ מקומי" if st.session_state.storage_src == 'local' else "ברירת מחדל")
               + ". שינויים לא יישמרו בענן עד שהחיבור יחזור.")

with st.spinner("טוען נתוני שוק..."):
    _, market = load_market(tuple(MARKET_TICKERS), False)
    usd_ils = market['USDILS=X']['price'] if 'USDILS=X' in market else get_usd_ils_fallback()
if not usd_ils or usd_ils <= 0:
    st.error("לא ניתן לקבל כרגע שער דולר/שקל. רענן/י את הדף בעוד דקה.")
    st.stop()

portfolio = st.session_state.portfolio
watch = st.session_state.watchlist
cfg = st.session_state.config
p_tickers = list(portfolio.index)
w_tickers = watch['Ticker'].tolist()

with st.spinner("טוען את התיק..."):
    hists, snaps = load_market(tuple(sorted(set(p_tickers + w_tickers + ['SPY']))), True)

positions, missing = compute_positions(portfolio, snaps, usd_ils)
cash_total = cfg['cash_usd'] + cfg['cash_ils'] / usd_ils
stock_value = sum(r['value'] for r in positions)
total_value = stock_value + cash_total
for r in positions:
    r['weight'] = r['value'] / total_value * 100 if total_value else 0.0
pos_by_ticker = {r['ticker']: r for r in positions}
cost_total = sum(r['cost'] for r in positions)
unreal_total = sum(r['unreal'] for r in positions)
day_pl = sum(r['pl_1d'] for r in positions)
day_pct = day_pl / (total_value - day_pl) * 100 if total_value - day_pl > 0 else 0.0
net_deposits = cfg['initial_investment']
total_return = total_value - net_deposits
total_return_pct = total_return / net_deposits * 100 if net_deposits > 0 else 0.0

watch_alerts = []
for _, w in watch.iterrows():
    s = snaps.get(w['Ticker'])
    if not s:
        continue
    if w['AlertHigh'] > 0 and s['price'] >= w['AlertHigh']:
        watch_alerts.append((w['Ticker'], f"{money(s['price'], s['sym'])} — עבר את יעד העלייה {money(w['AlertHigh'], s['sym'])}", GAIN))
    if w['AlertLow'] > 0 and s['price'] <= w['AlertLow']:
        watch_alerts.append((w['Ticker'], f"{money(s['price'], s['sym'])} — ירד מתחת ליעד {money(w['AlertLow'], s['sym'])}", LOSS))

t_today, t_pos, t_search, t_watch, t_earn, t_manage = st.tabs(
    ["היום", "פוזיציות", "חיפוש", "מעקב", "דוחות וחדשות", "ניהול"])

# ==========================================
# TAB: TODAY
# ==========================================
with t_today:
    now = now_il()
    h1, h2 = st.columns([3, 1])
    h1.markdown(f'<div class="stamp">יום {DAYS_HE[now.weekday()]}, {now:%d.%m.%Y} · עודכן {now:%H:%M}</div>',
                unsafe_allow_html=True)
    if h2.button("רענן מחירים"):
        load_market.clear()
        download_history.clear()
        st.rerun()

    st.markdown('<div class="hero-label">שווי תיק כולל</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-value">${total_value:,.0f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-change"><span style="color:{tone(day_pl)}">{signed_money(day_pl)} ({pct(day_pct)}) היום</span>'
                f'&nbsp;&nbsp;·&nbsp;&nbsp;<span style="color:{tone(total_return)}">{signed_money(total_return)} ({pct(total_return_pct, 1)}) מאז ההפקדה</span></div>',
                unsafe_allow_html=True)
    spy_day = market.get('SPY', {}).get('rets', {}).get('1d')
    spy_txt = f"S&P 500 היום {pct(spy_day)} · " if spy_day is not None else ''
    st.markdown(f'<div class="stamp">{spy_txt}≈ ₪{total_value * usd_ils:,.0f} לפי שער {usd_ils:.3f}</div>', unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    m1.metric("שווי מניות", money(stock_value, '$', 0), f"{signed_money(day_pl)} היום")
    m2.metric("מזומן", money(cash_total, '$', 0), f"${cfg['cash_usd']:,.0f} + ₪{cfg['cash_ils']:,.0f}", delta_color='off')
    m3, m4 = st.columns(2)
    m3.metric("רווח/הפסד פתוח", signed_money(unreal_total), pct(unreal_total / cost_total * 100 if cost_total else 0))
    m4.metric("תשואה כוללת", pct(total_return_pct, 1), f"מול ${net_deposits:,.0f} הפקדות", delta_color='off')

    with st.expander("עדכון יתרות מזומן"):
        render_cash_editor('cash_form_today')

    if market:
        strip([(name, f"{sym}{market[t]['price']:,.2f}", pct(market[t]['rets']['1d']), tone(market[t]['rets']['1d']))
               for t, (name, sym) in MARKET_TICKERS.items() if t in market])

    if missing:
        st.warning(f"לא התקבלו נתונים עבור: {', '.join(missing)}. בדוק/י שהסימול נכון (מניות ת״א עם ‎.TA).")

    # ── Needs attention ──
    with st.spinner("בודק דוחות רווחים..."):
        upcoming, _ = earnings_board(p_tickers + w_tickers, snaps)
    attention = []
    for r in positions:
        s, notes = snaps[r['ticker']], []
        if abs(r['day_pct']) >= 3:
            notes.append(f"תנועה חדה היום {pct(r['day_pct'], 1)}")
        if s['price'] >= s['hi52'] * 0.98:
            notes.append("קרוב לשיא 52 שבועות")
        elif s['price'] <= s['lo52'] * 1.02:
            notes.append("קרוב לשפל 52 שבועות")
        if s.get('rsi', 50) >= 75:
            notes.append(f"RSI גבוה ({s['rsi']:.0f})")
        elif s.get('rsi', 50) <= 25:
            notes.append(f"RSI נמוך ({s['rsi']:.0f})")
        if r['unreal_pct'] <= -15:
            notes.append(f"{r['unreal_pct']:.1f}% מתחת למחיר הקנייה")
        if r['weight'] >= 35 and r['quote_type'] != 'ETF':
            notes.append(f"{r['weight']:.0f}% מהתיק בפוזיציה אחת")
        if notes:
            attention.append((r['ticker'], ' · '.join(notes), tone(r['day_pct'])))
    for u in upcoming:
        if u['days'] <= 7:
            when = "היום" if u['days'] == 0 else "מחר" if u['days'] == 1 else f"בעוד {u['days']} ימים"
            attention.append((u['ticker'], f"דוח רווחים {when} ({u['date']:%d.%m})", GOLD))
    attention += watch_alerts

    section("דורש תשומת לב")
    if attention:
        for t, msg, color in attention:
            attn(t, msg, color)
    else:
        st.caption("אין אירועים חריגים בתיק היום.")

    # ── Positions today ──
    if positions:
        section("הפוזיציות שלי היום", "שינוי יומי במחיר וברווח, ומשמאל — רווח/הפסד כולל מאז הקנייה")
        rows_html = [ledger_row(
            r['ticker'], r['name'][:26], money(r['price'], r['sym']),
            f"{pct(r['day_pct'])}  {signed_money(r['pl_1d'])}", tone(r['pl_1d']),
            pct(r['unreal_pct'], 1), tone(r['unreal']),
            caption=f"{signed_money(r['unreal'])} כולל · שווי ${r['value']:,.0f} · {r['weight']:.1f}%")
            for r in sorted(positions, key=lambda x: x['day_pct'], reverse=True)]
        rows_html.append(ledger_row(
            "סה״כ", f"{len(positions)} פוזיציות", money(stock_value, '$', 0),
            f"{pct(day_pct)}  {signed_money(day_pl)}", tone(day_pl),
            pct(unreal_total / cost_total * 100 if cost_total else 0, 1), tone(unreal_total),
            caption=f"{signed_money(unreal_total)} כולל", total=True))
        render_ledger(rows_html)

        # ── Period performance ──
        section("ביצועים לפי תקופה", "לפי הכמויות שמוחזקות כרגע, ללא מזומן")
        per_label = st.radio("תקופה", list(PERIODS), horizontal=True, key='today_period', label_visibility='collapsed')
        p = PERIODS[per_label]
        pl_period = sum(r[f'pl_{p}'] for r in positions)
        base_value = stock_value - pl_period
        spy_ret = market.get('SPY', {}).get('rets', {}).get(p)
        b1, b2 = st.columns(2)
        b1.metric(f"התיק — {per_label}", signed_money(pl_period), pct(pl_period / base_value * 100 if base_value > 0 else 0))
        b2.metric(f"S&P 500 — {per_label}", pct(spy_ret) if spy_ret is not None else "—")
        ordered = sorted(positions, key=lambda x: x[f'pl_{p}'])
        fig = go.Figure(go.Bar(
            x=[r[f'pl_{p}'] for r in ordered], y=[r['ticker'] for r in ordered], orientation='h',
            marker_color=[tone(r[f'pl_{p}']) for r in ordered],
            text=[f"{signed_money(r[f'pl_{p}'])} ({pct((r['price'] / snaps[r['ticker']]['bases'][p] - 1) * 100, 1)})" for r in ordered],
            textposition='auto'))
        dark_layout(fig, 70 + 38 * len(ordered))
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False}, key='period_chart')

    # ── Earnings calendar ──
    soon = [u for u in upcoming if u['days'] <= 14]
    if soon:
        section("דוחות רווחים בשבועיים הקרובים", "תאריכים לפי Yahoo Finance; ייתכנו שינויים")
        render_ledger([ledger_row(
            u['ticker'], f"תחזית EPS {u['eps_est']:.2f}" if u['eps_est'] is not None else "אין תחזית",
            f"{u['date']:%d.%m}", "היום" if u['days'] == 0 else f"בעוד {u['days']} ימים", TEXT_DIM,
            "בתיק" if u['ticker'] in pos_by_ticker else "במעקב", GOLD if u['days'] <= 7 else TEXT_DIM)
            for u in soon])

    st.caption("המידע מוצג לצורך מעקב אישי בלבד ואינו מהווה ייעוץ השקעות.")

# ==========================================
# TAB: POSITIONS
# ==========================================
with t_pos:
    if not positions:
        st.info("אין פוזיציות בתיק. הוסף/י דרך לשונית ניהול.")
    else:
        names = {r['ticker']: r['name'] for r in positions}
        choice = st.selectbox("תצוגה", ["all"] + [r['ticker'] for r in positions],
                              format_func=lambda x: "כל הפוזיציות" if x == "all" else f"{x} — {names[x]}")
        if choice == "all":
            c1, c2 = st.columns(2)
            c1.metric("שווי מניות", money(stock_value, '$', 0), f"{signed_money(day_pl)} היום")
            c2.metric("רווח/הפסד פתוח", signed_money(unreal_total), pct(unreal_total / cost_total * 100 if cost_total else 0))
            c3, c4 = st.columns(2)
            c3.metric("עלות כוללת", money(cost_total, '$', 0))
            c4.metric("דיבידנד שנתי משוער", money(sum(r['div_annual'] for r in positions), '$', 0))

            table = pd.DataFrame([{
                'סימול': r['ticker'], 'כמות': r['qty'], 'מחיר': money(r['price'], r['sym']),
                'יומי %': r['day_pct'], 'יומי $': r['pl_1d'], 'שבוע %': snaps[r['ticker']]['rets']['1w'],
                'חודש %': snaps[r['ticker']]['rets']['1m'], 'מחיר קנייה': money(r['avg'], r['sym']),
                'שווי $': r['value'], 'רווח/הפסד $': r['unreal'], 'רווח/הפסד %': r['unreal_pct'], 'משקל %': r['weight'],
            } for r in sorted(positions, key=lambda x: x['value'], reverse=True)])
            pct_cols = ['יומי %', 'יומי $', 'שבוע %', 'חודש %', 'רווח/הפסד $', 'רווח/הפסד %']
            fmt = {'כמות': '{:,.4f}', 'יומי %': '{:+.2f}%', 'יומי $': '{:+,.0f}', 'שבוע %': '{:+.2f}%',
                   'חודש %': '{:+.2f}%', 'שווי $': '{:,.0f}', 'רווח/הפסד $': '{:+,.0f}',
                   'רווח/הפסד %': '{:+.2f}%', 'משקל %': '{:.1f}%'}
            st.dataframe(styled_table(table, pct_cols, fmt), width='stretch', hide_index=True)
            st.download_button("ייצוא CSV", table.to_csv(index=False).encode('utf-8-sig'), "portfolio.csv", "text/csv")

            alloc = {r['ticker']: r['value'] for r in positions}
            if cash_total > 0:
                alloc['מזומן'] = cash_total
            fig_pie = px.pie(values=list(alloc.values()), names=list(alloc.keys()), hole=0.55,
                             color_discrete_sequence=[GOLD, GAIN, INFO, LOSS, TEXT_DIM, '#8C7239', '#3F6E58', '#4A6A8C'])
            fig_pie.update_traces(textinfo='label+percent')
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=TEXT, height=300,
                                  margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            section("פיזור התיק")
            st.plotly_chart(fig_pie, width='stretch', config={"displayModeBar": False}, key='alloc_pie')
        else:
            render_stock_detail(choice, snaps[choice], hists[choice], pos_by_ticker[choice], key='pos')

# ==========================================
# TAB: SEARCH
# ==========================================
with t_search:
    section("חיפוש והשוואה", "סימול אחד או כמה, מופרדים בפסיק או ברווח. מניות ת״א עם ‎.TA (למשל TEVA.TA)")
    with st.form('search_form'):
        query = st.text_input("סימולים", value=' '.join(st.session_state.get('search_list', [])),
                              placeholder="AAPL, NVDA, TEVA.TA")
        if st.form_submit_button("חפש"):
            st.session_state.search_list = parse_tickers(query)
            if not st.session_state.search_list:
                st.error("לא זוהו סימולים תקינים.")

    search_list = st.session_state.get('search_list', [])
    if search_list:
        with st.spinner("טוען..."):
            s_hists, s_snaps = load_market(tuple(sorted(search_list)), True)
        bad = [t for t in search_list if t not in s_snaps]
        good = [t for t in search_list if t in s_snaps]
        if bad:
            st.warning(f"לא נמצאו נתונים עבור: {', '.join(bad)}")
        if good:
            render_ledger([ledger_row(
                t, s_snaps[t]['name'][:26], money(s_snaps[t]['price'], s_snaps[t]['sym']),
                f"{pct(s_snaps[t]['rets']['1d'])} היום", tone(s_snaps[t]['rets']['1d']),
                f"{pct(s_snaps[t]['rets']['ytd'], 1)} השנה", tone(s_snaps[t]['rets']['ytd']),
                caption=f"שבוע {pct(s_snaps[t]['rets']['1w'], 1)} · חודש {pct(s_snaps[t]['rets']['1m'], 1)}"
                        + (" · בתיק" if t in pos_by_ticker else ""))
                for t in good])

            if len(good) > 1:
                with st.expander("טבלת השוואה"):
                    cmp_df = pd.DataFrame([{
                        'סימול': t, 'מחיר': money(s['price'], s['sym']), 'יומי %': s['rets']['1d'],
                        'שבוע %': s['rets']['1w'], 'חודש %': s['rets']['1m'], 'מתחילת שנה %': s['rets']['ytd'],
                        'שנה %': s['rets']['1y'], 'מיקום בטווח 52ש׳ %': (s['price'] - s['lo52']) / (s['hi52'] - s['lo52']) * 100 if s['hi52'] > s['lo52'] else 50.0,
                        'P/E': s['pe'] or np.nan, 'RSI': s.get('rsi', np.nan),
                    } for t, s in ((t, s_snaps[t]) for t in good)])
                    pcols = ['יומי %', 'שבוע %', 'חודש %', 'מתחילת שנה %', 'שנה %']
                    fmt = {c: '{:+.2f}%' for c in pcols} | {'מיקום בטווח 52ש׳ %': '{:.0f}%', 'P/E': '{:.1f}', 'RSI': '{:.0f}'}
                    st.dataframe(styled_table(cmp_df, pcols, fmt), width='stretch', hide_index=True)

            section("ניתוח מלא")
            pick = st.selectbox("בחר/י מניה", good, key='search_pick')
            render_stock_detail(pick, s_snaps[pick], s_hists[pick], pos_by_ticker.get(pick), key='search')
            if pick not in w_tickers and pick not in pos_by_ticker:
                if st.button(f"הוסף את {pick} לרשימת המעקב", key='search_add_watch'):
                    new_wl = clean_watchlist(pd.concat([watch, pd.DataFrame([{'Ticker': pick, 'Notes': '', 'AlertHigh': 0.0, 'AlertLow': 0.0}])],
                                                       ignore_index=True))
                    st.session_state.watchlist = new_wl
                    kind, msg = SAVE_MSG[save_watchlist(new_wl)]
                    flash(kind, f"{pick} נוסף לרשימת המעקב — {msg}")
                    st.rerun()

# ==========================================
# TAB: WATCHLIST
# ==========================================
with t_watch:
    with st.expander("הוספה או עדכון של מניה ברשימה", expanded=watch.empty):
        with st.form("add_watchlist", clear_on_submit=True):
            w1, w2 = st.columns(2)
            wl_ticker = w1.text_input("סימול").strip().upper()
            wl_notes = w2.text_input("הערה")
            w3, w4 = st.columns(2)
            wl_high = w3.number_input("התראה כשהמחיר מעל", min_value=0.0, value=0.0, step=1.0)
            wl_low = w4.number_input("התראה כשהמחיר מתחת", min_value=0.0, value=0.0, step=1.0)
            if st.form_submit_button("שמור ברשימה"):
                if not parse_tickers(wl_ticker):
                    st.error("הזן/י סימול תקין.")
                else:
                    new_wl = clean_watchlist(pd.concat([watch, pd.DataFrame([{'Ticker': wl_ticker, 'Notes': wl_notes,
                                                                             'AlertHigh': wl_high, 'AlertLow': wl_low}])],
                                                       ignore_index=True))
                    st.session_state.watchlist = new_wl
                    flash(*SAVE_MSG[save_watchlist(new_wl)])
                    st.rerun()

    if watch.empty:
        st.info("הרשימה ריקה. הוסף/י מניות שברצונך לעקוב אחריהן.")
    else:
        if watch_alerts:
            section("התראות מחיר שהופעלו")
            for t, msg, color in watch_alerts:
                attn(t, msg, color)

        section("רשימת מעקב")
        rows_html = []
        for _, w in watch.iterrows():
            s = snaps.get(w['Ticker'])
            alerts_txt = ' · '.join(x for x in (
                f"מעל {w['AlertHigh']:,.2f}" if w['AlertHigh'] > 0 else '',
                f"מתחת {w['AlertLow']:,.2f}" if w['AlertLow'] > 0 else '') if x)
            if s:
                rows_html.append(ledger_row(
                    w['Ticker'], w['Notes'] or s['name'][:26], money(s['price'], s['sym']),
                    f"{pct(s['rets']['1d'])} היום", tone(s['rets']['1d']),
                    f"{pct(s['rets']['1w'], 1)} שבוע", tone(s['rets']['1w']),
                    caption=f"חודש {pct(s['rets']['1m'], 1)}" + (f" · התראות: {alerts_txt}" if alerts_txt else '')))
            else:
                rows_html.append(ledger_row(w['Ticker'], w['Notes'], "—", "אין נתונים", TEXT_DIM, "—", TEXT_DIM))
        render_ledger(rows_html)

        available = [t for t in w_tickers if t in snaps]
        if available:
            section("ניתוח מניה מהרשימה")
            w_pick = st.selectbox("בחר/י מניה", available, key='watch_pick')
            render_stock_detail(w_pick, snaps[w_pick], hists[w_pick], pos_by_ticker.get(w_pick), key='watch')

        with st.expander("הסרה מהרשימה"):
            remove_t = st.selectbox("מניה להסרה", w_tickers, key='watch_remove')
            if st.button("הסר מרשימת המעקב"):
                new_wl = watch[watch['Ticker'] != remove_t].reset_index(drop=True)
                st.session_state.watchlist = new_wl
                flash(*SAVE_MSG[save_watchlist(new_wl)])
                st.rerun()

# ==========================================
# TAB: EARNINGS & NEWS
# ==========================================
with t_earn:
    tab_e, tab_n = st.tabs(["דוחות רווחים", "חדשות"])
    with tab_e:
        board = p_tickers + w_tickers
        if not board:
            st.info("הוסף/י מניות לתיק או לרשימת המעקב כדי לראות דוחות רווחים.")
        else:
            with st.spinner("טוען דוחות..."):
                upcoming_all, recent_all = earnings_board(board, snaps)
            section("דוחות קרובים", "מניות בתיק ובמעקב, ממוינות לפי תאריך. קרנות סל לא מפרסמות דוחות.")
            if upcoming_all:
                render_ledger([ledger_row(
                    u['ticker'], f"תחזית EPS {u['eps_est']:.2f}" if u['eps_est'] is not None else "אין תחזית",
                    f"{u['date']:%d.%m.%Y}", "היום" if u['days'] == 0 else f"בעוד {u['days']} ימים", TEXT_DIM,
                    "בתיק" if u['ticker'] in pos_by_ticker else "במעקב", GOLD if u['days'] <= 7 else TEXT_DIM)
                    for u in upcoming_all[:15]])
            else:
                st.info("אין תאריכי דוחות ידועים כרגע.")

            section("דוחות אחרונים", "EPS בפועל מול תחזית האנליסטים")
            if recent_all:
                render_ledger([ledger_row(
                    r['ticker'],
                    f"בפועל {r['actual']:.2f} מול תחזית {r['estimate']:.2f}" if r['actual'] is not None and r['estimate'] is not None else "—",
                    f"{r['date']:%d.%m.%Y}", "הפתעה", TEXT_DIM,
                    pct(r['surprise'], 1) if r['surprise'] is not None else "—", tone(r['surprise'] or 0))
                    for r in recent_all[:10]])
            else:
                st.info("אין נתוני דוחות אחרונים.")

    with tab_n:
        news_opts = ["all"] + p_tickers + [t for t in w_tickers if t not in p_tickers]
        news_sel = st.selectbox("מקור", news_opts, key='news_sel',
                                format_func=lambda x: "כל התיק והמעקב" if x == "all" else x)
        selected = tuple(sorted(news_opts[1:])) if news_sel == "all" else (news_sel,)
        if selected:
            with st.spinner("טוען חדשות..."):
                feed = get_news(selected, per=3 if news_sel == "all" else 10)
            render_news(feed[:25])

# ==========================================
# TAB: MANAGE
# ==========================================
with t_manage:
    tab_cash, tab_trade, tab_risk, tab_size, tab_scan = st.tabs(["מזומן", "עסקאות ועריכה", "סיכון", "גודל פוזיציה", "סורק"])

    with tab_cash:
        section("יתרות נוכחיות", "נשמר ב-Google Sheets ונטען אוטומטית בכניסה הבאה")
        render_cash_editor('cash_form_manage')

        section("הפקדה או משיכה", "מעדכן גם את יתרת המזומן וגם את סך ההפקדות, ונרשם ביומן")
        with st.form('cashflow_form', clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            cf_kind = c1.selectbox("פעולה", ["הפקדה", "משיכה"])
            cf_cur = c2.selectbox("מטבע", ["USD", "ILS"])
            cf_amt = c3.number_input("סכום", min_value=0.0, step=100.0, format="%.2f")
            cf_note = st.text_input("הערה (לא חובה)")
            if st.form_submit_button("רשום פעולה"):
                if cf_amt <= 0:
                    st.error("הזן/י סכום גדול מאפס.")
                else:
                    sign = 1 if cf_kind == "הפקדה" else -1
                    amt_usd = cf_amt if cf_cur == "USD" else cf_amt / usd_ils
                    new_cfg = dict(cfg)
                    new_cfg['cash_usd' if cf_cur == "USD" else 'cash_ils'] += sign * cf_amt
                    new_cfg['initial_investment'] += sign * amt_usd
                    st.session_state.config = new_cfg
                    s1 = save_config(new_cfg)
                    s2 = append_log('cashflow', 'Cashflow', 'cashflow.csv', {
                        'Date': now_il().strftime('%Y-%m-%d %H:%M'), 'Type': cf_kind, 'Amount': cf_amt,
                        'Currency': cf_cur, 'AmountUSD': round(sign * amt_usd, 2), 'Note': cf_note})
                    flash(*SAVE_MSG[worst(s1, s2)])
                    st.rerun()

        if not st.session_state.cashflow.empty:
            with st.expander("יומן הפקדות ומשיכות"):
                st.dataframe(st.session_state.cashflow.iloc[::-1], width='stretch', hide_index=True)

    with tab_trade:
        section("רישום קנייה או מכירה", "מעדכן את התיק, ואם תבחר/י — גם את יתרת המזומן")
        with st.form("trade_form", clear_on_submit=True):
            q1, q2 = st.columns([3, 2])
            tr_ticker = q1.text_input("סימול", placeholder="AAPL / TEVA.TA").strip().upper()
            tr_action = q2.selectbox("פעולה", ["קנייה", "מכירה"])
            q3, q4 = st.columns(2)
            tr_qty = q3.number_input("כמות", min_value=0.0, step=0.1, value=1.0, format="%.4f")
            tr_price = q4.number_input("מחיר ליחידה (0 = מחיר אחרון)", min_value=0.0, step=0.01, format="%.2f")
            tr_cash = st.checkbox("עדכן את יתרת המזומן", value=True)
            if st.form_submit_button("רשום עסקה"):
                action = 'buy' if tr_action == "קנייה" else 'sell'
                valid = bool(parse_tickers(tr_ticker)) and tr_qty > 0
                price = tr_price
                if valid and price <= 0:
                    h = download_history((tr_ticker,), '1mo').get(tr_ticker)
                    price = float(h['Close'].iloc[-1]) if h is not None else 0.0
                if not valid:
                    st.error("הזן/י סימול וכמות גדולה מאפס.")
                elif price <= 0:
                    st.error(f"לא נמצא מחיר עבור {tr_ticker} — הזן/י מחיר ידנית.")
                elif action == 'sell' and tr_ticker not in portfolio.index:
                    st.error(f"{tr_ticker} לא נמצאת בתיק.")
                else:
                    new_port, done, realized = apply_trade(portfolio, tr_ticker, action, tr_qty, price)
                    st.session_state.portfolio = new_port
                    st.session_state.editor_ver += 1
                    statuses = [save_portfolio(new_port)]
                    if tr_cash:
                        new_cfg = dict(cfg)
                        key = 'cash_ils' if is_ils(tr_ticker) else 'cash_usd'
                        new_cfg[key] += (-1 if action == 'buy' else 1) * done * price
                        st.session_state.config = new_cfg
                        statuses.append(save_config(new_cfg))
                    statuses.append(append_log('trades', 'Trades', 'trades.csv', {
                        'Date': now_il().strftime('%Y-%m-%d %H:%M'), 'Ticker': tr_ticker, 'Action': tr_action,
                        'Quantity': done, 'Price': round(price, 4), 'Currency': 'ILS' if is_ils(tr_ticker) else 'USD',
                        'Realized': round(realized, 2)}))
                    kind, msg = SAVE_MSG[worst(*statuses)]
                    sym = '₪' if is_ils(tr_ticker) else '$'
                    extra = f" · רווח ממומש {signed_money(realized, sym, 2)}" if action == 'sell' else ''
                    flash(kind, f"{tr_action} של {fmt_qty(done)} {tr_ticker} ב-{money(price, sym)} נרשמה{extra}. {msg}")
                    st.rerun()

        trades = st.session_state.trades
        if not trades.empty:
            with st.expander("יומן עסקאות"):
                realized_usd = pd.to_numeric(trades['Realized'], errors='coerce').fillna(0)
                is_ils_row = trades['Currency'].astype(str) == 'ILS'
                total_realized = float((realized_usd.where(~is_ils_row, realized_usd / usd_ils)).sum())
                st.metric("רווח ממומש מצטבר", signed_money(total_realized, '$', 2))
                st.dataframe(trades.iloc[::-1], width='stretch', hide_index=True)

        section("עריכה ישירה של התיק", "שנה/י כמות או מחיר קנייה, הוסף/י או מחק/י שורות ושמור/י. מחיר קנייה של מניות ת״א בשקלים.")
        edited = st.data_editor(
            portfolio.reset_index(), num_rows="dynamic", width='stretch', hide_index=True,
            key=f"portfolio_editor_{st.session_state.editor_ver}",
            column_config={
                'Ticker': st.column_config.TextColumn('סימול', required=True),
                'Quantity': st.column_config.NumberColumn('כמות', min_value=0.0, step=0.0001, format='%.4f'),
                'PurchasePrice': st.column_config.NumberColumn('מחיר קנייה', min_value=0.0, step=0.01, format='%.2f'),
            })
        if st.button("שמור שינויים בתיק"):
            cleaned = clean_portfolio(edited)
            st.session_state.portfolio = cleaned
            st.session_state.editor_ver += 1
            flash(*SAVE_MSG[save_portfolio(cleaned)])
            st.rerun()

        with st.expander("ייבוא מצילום מסך של הברוקר"):
            if not OCR_AVAILABLE:
                st.info("זיהוי טקסט לא זמין: יש להוסיף pytesseract ו-Pillow ל-requirements.txt ו-tesseract-ocr ל-packages.txt.")
            else:
                uploaded = st.file_uploader("צילום מסך", type=["png", "jpg", "jpeg"])
                if uploaded is not None:
                    img = Image.open(uploaded)
                    st.image(img, width='stretch')
                    if st.button("זהה מניות מהתמונה"):
                        with st.spinner("מזהה טקסט..."):
                            st.session_state['_ocr'] = parse_portfolio_image(img, p_tickers)
                ocr_df = st.session_state.get('_ocr')
                if ocr_df is not None:
                    if ocr_df.empty:
                        st.warning("לא זוהו שורות בתמונה. נסה/י צילום חד יותר או הוסף/י ידנית.")
                    else:
                        st.caption("בדוק/י ותקן/י לפני ההוספה — הזיהוי אינו מדויק תמיד. השורות יתווספו כקניות.")
                        ocr_edit = st.data_editor(ocr_df, num_rows="dynamic", width='stretch', key="ocr_editor")
                        if st.button("אשר והוסף לתיק"):
                            new_port = portfolio
                            for _, r in clean_portfolio(ocr_edit).iterrows():
                                new_port = apply_trade(new_port, r.name, 'buy', float(r['Quantity']), float(r['PurchasePrice']))[0]
                            st.session_state.portfolio = new_port
                            st.session_state.editor_ver += 1
                            del st.session_state['_ocr']
                            flash(*SAVE_MSG[save_portfolio(new_port)])
                            st.rerun()

    with tab_risk:
        if not positions:
            st.info("הוסף/י מניות לתיק.")
        else:
            risk_df, corr = risk_analytics([r['ticker'] for r in positions], hists)
            if risk_df.empty:
                st.info("אין מספיק היסטוריה לחישוב סיכון.")
            else:
                weights = {r['ticker']: r['value'] / stock_value for r in positions} if stock_value else {}
                rd = risk_df.set_index('סימול')
                port_beta = float(sum(rd.loc[t, 'בטא'] * w for t, w in weights.items() if t in rd.index and np.isfinite(rd.loc[t, 'בטא'])))
                top = max(positions, key=lambda x: x['weight'])
                k1, k2 = st.columns(2)
                k1.metric("בטא משוקלל לתיק", f"{port_beta:.2f}", "תנודתי מהשוק" if port_beta > 1.1 else "דומה לשוק או פחות", delta_color='off')
                k2.metric("הפוזיציה הגדולה", top['ticker'], f"{top['weight']:.1f}% מהתיק", delta_color='off')
                k3, k4 = st.columns(2)
                worst_row = rd['ירידה מקסימלית %'].idxmin()
                k3.metric("ירידה מקסימלית בשנה", f"{rd.loc[worst_row, 'ירידה מקסימלית %']:.1f}%", worst_row, delta_color='off')
                k4.metric("מזומן מתוך התיק", f"{cash_total / total_value * 100 if total_value else 0:.1f}%")
                st.dataframe(risk_df.style.format({c: '{:.2f}' for c in risk_df.columns if c != 'סימול'}, na_rep='—'),
                             width='stretch', hide_index=True)
                st.caption("מחושב על תשואות יומיות של השנה האחרונה מול SPY. VaR 95% — ההפסד היומי שנחצה בכ-5% מהימים.")
                if corr is not None:
                    with st.expander("מטריצת מתאמים"):
                        fig_corr = px.imshow(corr, color_continuous_scale=[[0, LOSS], [0.5, PANEL], [1, GAIN]],
                                             zmin=-1, zmax=1, text_auto='.2f')
                        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=TEXT, height=380,
                                               margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig_corr, width='stretch', config={"displayModeBar": False}, key='corr')

    with tab_size:
        section("מחשבון גודל פוזיציה", "כמה יחידות לקנות כך שפגיעה ב-Stop תעלה לכל היותר אחוז נתון מהתיק")
        ps_account = st.number_input("גודל תיק ($)", value=float(round(total_value, 2)), min_value=0.0)
        ps_risk = st.slider("סיכון מקסימלי לעסקה (%)", 0.5, 5.0, 1.0, 0.5)
        ps_ticker = st.text_input("סימול", value=p_tickers[0] if p_tickers else "", key='ps_ticker').strip().upper()
        ps_stop = st.slider("Stop Loss (%)", 2.0, 25.0, 8.0, 0.5)
        ps_target = st.slider("Take Profit (%)", 5.0, 50.0, 20.0, 1.0)
        if ps_ticker:
            ps_snap = snaps.get(ps_ticker) or load_market((ps_ticker,), False)[1].get(ps_ticker)
            if not ps_snap:
                st.warning(f"לא נמצא מחיר עבור {ps_ticker}.")
            else:
                cur_p = ps_snap['price']
                k = 1 / usd_ils if ps_snap['currency'] == 'ILS' else 1.0
                risk_per_share_usd = cur_p * ps_stop / 100 * k
                shares = int(ps_account * ps_risk / 100 / risk_per_share_usd) if risk_per_share_usd > 0 else 0
                pos_usd = shares * cur_p * k
                sym = ps_snap['sym']
                c1, c2 = st.columns(2)
                c1.metric("יחידות לקנייה", str(shares), f"מחיר {money(cur_p, sym)}", delta_color='off')
                c2.metric("גודל הפוזיציה", money(pos_usd), f"{pos_usd / ps_account * 100:.1f}% מהתיק" if ps_account else None, delta_color='off')
                c3, c4 = st.columns(2)
                c3.metric("Stop Loss", money(cur_p * (1 - ps_stop / 100), sym), f"-{ps_stop}%")
                c4.metric("Take Profit", money(cur_p * (1 + ps_target / 100), sym), f"+{ps_target}%")
                rr = ps_target / ps_stop
                st.markdown(f"<span style='color:{GAIN if rr >= 2 else GOLD if rr >= 1.5 else LOSS};font-weight:600'>"
                            f"יחס סיכוי/סיכון 1:{rr:.1f}</span>", unsafe_allow_html=True)

    with tab_scan:
        section("סורק מניות", "דירוג טכני-פונדמנטלי לפי כללים קבועים — נקודת פתיחה למחקר, לא המלצה")
        s1, s2 = st.columns(2)
        universe = s1.selectbox("קבוצה", list(SCAN_UNIVERSES))
        min_score = s2.slider("ציון מינימלי", 0, 90, 55)
        with st.expander("סינון נוסף"):
            f1, f2 = st.columns(2)
            max_pe = f1.number_input("P/E מקסימלי (0 = ללא)", value=0, min_value=0)
            min_growth = f2.number_input("צמיחת הכנסות מינימלית %", value=-100)
        if st.button("הרץ סריקה"):
            lst = SCAN_UNIVERSES[universe]
            if lst is None:
                pool = get_sp500_tickers()
                lst = random.sample(pool, min(40, len(pool)))
            st.session_state.scan_list = lst
        scan_list = st.session_state.get('scan_list')
        if scan_list:
            with st.spinner("סורק... (עד דקה בהרצה ראשונה)"):
                _, scan_snaps = load_market(tuple(sorted(scan_list)), True)
            results = sorted(
                [(t, d) for t, d in scan_snaps.items()
                 if d['score'] >= min_score
                 and (max_pe == 0 or 0 < d['pe'] <= max_pe)
                 and d['growth'] >= min_growth],
                key=lambda x: x[1]['score'], reverse=True)
            st.caption(f"{len(results)} מתוך {len(scan_snaps)} מניות עומדות בסינון")
            render_ledger([ledger_row(
                t, f"{d['name'][:22]} · P/E {d['pe']:.0f}" if d['pe'] else d['name'][:26],
                money(d['price'], d['sym']), f"{pct(d['rets']['1m'], 1)} חודש", tone(d['rets']['1m']),
                f"{d['score']}/100", SIGNAL_COLOR[d['signal']],
                caption=f"{d.get('trend', '—')} · {SIGNAL_LABEL_HE[d['signal']]}")
                for t, d in results[:20]])

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
# PAGE CONFIG + DESIGN SYSTEM (shadcn zinc tokens; base theme lives in .streamlit/config.toml)
# ==========================================
st.set_page_config(page_title="AlonStocks Pro", layout="wide", page_icon="◆", initial_sidebar_state="collapsed")

INK, PANEL, LINE, TEXT, TEXT_DIM = '#09090B', '#111113', '#27272A', '#FAFAFA', '#A1A1AA'
GOLD, GAIN, LOSS, INFO = '#F59E0B', '#22C55E', '#EF4444', '#60A5FA'
FONT = 'Geist, Heebo, sans-serif'

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;600;700&display=swap');
    :root {
        --background:#09090B; --card:#111113; --muted:#18181B; --border:#27272A; --ring:#3F3F46;
        --foreground:#FAFAFA; --muted-foreground:#A1A1AA; --faint:#71717A;
        --gain:#22C55E; --loss:#EF4444; --warn:#F59E0B; --info:#60A5FA; --radius:12px;
    }
    .main .block-container, [data-testid="stMainBlockContainer"] { padding:3.4rem 1rem 96px !important; max-width:880px !important; }
    [data-testid="stHeader"] { background:transparent; }
    input, select, textarea { font-size:16px !important; }
    * { font-variant-numeric:tabular-nums; }

    /* RTL for text, LTR for numbers and charts */
    [data-testid="stMarkdown"], .stAlert, [data-testid="stWidgetLabel"], [data-testid="stMetric"],
    [data-testid="stExpander"] summary, [data-testid="stCaptionContainer"], [data-testid="stRadio"],
    [data-testid="stCheckbox"], [data-testid="stForm"], [data-testid="stTextInput"], [data-testid="stNumberInput"],
    [data-testid="stSelectbox"] { direction:rtl; text-align:right; }
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stDataFrame"],
    [data-testid="stDataEditor"], .js-plotly-plot, .ltr, .num { direction:ltr !important; unicode-bidi:isolate; }
    [data-testid="stNumberInput"] input { direction:ltr; text-align:right; }
    [data-testid="stTextInput"] input { unicode-bidi:plaintext; text-align:right; }
    [data-testid="stFormSubmitButton"], [data-testid="stButton"], [data-testid="stDownloadButton"] { width:100%; }
    [data-testid="stMetricValue"] > div { text-align:right; }

    /* Metric = shadcn Card */
    [data-testid="stMetric"] { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:14px 16px !important; }
    [data-testid="stMetric"] label { color:var(--muted-foreground) !important; }
    [data-testid="stMetric"] label p { font-size:0.78rem !important; }
    [data-testid="stMetricValue"] { letter-spacing:-0.02em; }
    [data-testid="stMetricDelta"] { justify-content:flex-start; }

    /* Buttons = shadcn outline / default */
    .stButton > button, .stDownloadButton > button {
        min-height:42px; width:100%; font-weight:500; background:var(--background); color:var(--foreground);
        border:1px solid var(--border); transition:background-color .15s, border-color .15s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { background:var(--muted); border-color:var(--ring); color:var(--foreground); }
    .stFormSubmitButton > button { min-height:42px; width:100%; font-weight:500; background:var(--foreground); color:var(--background);
                                   border:1px solid var(--foreground); transition:background-color .15s; }
    .stFormSubmitButton > button:hover { background:#D4D4D8; color:var(--background); border-color:#D4D4D8; }
    [data-testid="stForm"] { background:var(--card); }
    [data-testid="stExpander"] details { background:var(--card); }

    /* Header */
    .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; direction:rtl; }
    .wordmark {
        font-weight:600; font-size:0.95rem; letter-spacing:-0.01em; direction:ltr;
        background:linear-gradient(110deg, #71717A 35%, #FAFAFA 50%, #71717A 65%); background-size:250% 100%;
        -webkit-background-clip:text; background-clip:text; color:transparent; animation:shine 6s linear infinite;
    }
    @keyframes shine { from { background-position:100% 0; } to { background-position:-150% 0; } }
    .live { display:inline-flex; align-items:center; gap:8px; font-size:0.74rem; color:var(--muted-foreground); }
    .live-dot { position:relative; width:7px; height:7px; border-radius:50%; background:var(--gain); flex:none; }
    .live-dot::after { content:''; position:absolute; inset:0; border-radius:50%; background:var(--gain); animation:ping 2s cubic-bezier(0,0,.2,1) infinite; }
    .live-dot.closed { background:var(--faint); }
    .live-dot.closed::after { display:none; }
    @keyframes ping { 75%, 100% { transform:scale(2.6); opacity:0; } }

    /* Hero */
    .hero { direction:rtl; }
    .hero-label { font-size:0.8rem; color:var(--muted-foreground); }
    .hero-value { font-size:3rem; font-weight:600; letter-spacing:-0.045em; line-height:1.05; margin-top:4px;
                  direction:ltr; text-align:right; animation:blurIn .8s cubic-bezier(.2,.7,.2,1) both; }
    @keyframes blurIn { from { opacity:0; filter:blur(12px); transform:translateY(6px); } to { opacity:1; filter:none; transform:none; } }
    .badges { display:flex; gap:6px; flex-wrap:wrap; margin-top:12px; }
    .badge { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--border); border-radius:6px;
             padding:3px 9px; font-size:0.76rem; color:var(--muted-foreground); background:var(--card); white-space:nowrap; }
    .badge .num { font-weight:600; }
    .stamp { font-size:0.74rem; color:var(--faint); margin-top:8px; direction:rtl; text-align:right; }

    /* Section headings */
    .section-title { font-size:0.95rem; font-weight:600; color:var(--foreground); margin:30px 0 2px; direction:rtl; text-align:right; letter-spacing:-0.01em; }
    .section-sub { font-size:0.78rem; color:var(--muted-foreground); margin:0 0 10px; direction:rtl; text-align:right; }

    /* List card */
    .ledger { border:1px solid var(--border); border-radius:var(--radius); background:var(--card); overflow:hidden; margin-top:10px; }
    .ledger-row { display:grid; grid-template-columns:minmax(0,1.25fr) minmax(0,1fr) minmax(0,1fr); align-items:center; gap:10px;
                  padding:12px 14px; border-bottom:1px solid var(--border); direction:rtl; transition:background-color .15s; }
    .ledger-row:last-child { border-bottom:none; }
    .ledger-row:hover { background:var(--muted); }
    .ledger-row.total { background:var(--muted); }
    .row-ticker { font-weight:600; font-size:0.95rem; direction:ltr; unicode-bidi:isolate; }
    .row-name { font-size:0.72rem; color:var(--muted-foreground); margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .row-name.latin { direction:ltr; text-align:right; }
    .row-mid { text-align:center; }
    .row-price { font-weight:500; font-size:0.9rem; direction:ltr; }
    .row-sub { font-size:0.75rem; margin-top:2px; direction:ltr; }
    .row-end { text-align:left; }
    .row-tag { font-size:0.9rem; font-weight:600; direction:ltr; }
    .row-caption { font-size:0.7rem; color:var(--faint); margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

    /* Attention items (shadcn Alert) */
    .attn-strip { display:flex; gap:12px; align-items:stretch; border:1px solid var(--border); border-radius:10px;
                  background:var(--card); padding:10px 12px; margin-bottom:6px; direction:rtl; }
    .attn-strip .bar { width:3px; border-radius:2px; flex:none; }
    .ledger .attn-strip { border:none; border-radius:0; margin:0; border-bottom:1px solid var(--border); }
    .ledger .attn-strip:last-child { border-bottom:none; }
    .attn-strip .row-name { color:var(--muted-foreground); font-size:0.8rem; white-space:normal; }

    /* Stat strip */
    .idx-strip { display:grid; grid-template-columns:repeat(auto-fit, minmax(104px, 1fr)); border:1px solid var(--border);
                 border-radius:var(--radius); overflow:hidden; background:var(--card); margin:14px 0 4px; direction:rtl; }
    .idx-item { padding:10px 12px; box-shadow:-1px 0 0 var(--border), 0 1px 0 var(--border); }
    .idx-name { font-size:0.7rem; color:var(--muted-foreground); }
    .idx-price { font-size:0.88rem; font-weight:600; margin-top:2px; direction:ltr; text-align:right; }
    .idx-chg { font-size:0.72rem; margin-top:1px; direction:ltr; text-align:right; }

    .range-wrap { direction:ltr; margin:6px 0 14px; }
    .range-bar { height:6px; border-radius:3px; background:var(--muted); position:relative; margin:8px 0 6px; }
    .range-fill { position:absolute; top:0; bottom:0; left:0; border-radius:3px; background:linear-gradient(90deg, #3F3F46, #A1A1AA); }
    .range-marker { position:absolute; top:-4px; width:14px; height:14px; border-radius:50%; background:var(--foreground);
                    border:3px solid var(--background); transform:translateX(-50%); }
    .range-labels { display:flex; justify-content:space-between; font-size:0.72rem; color:var(--faint); }

    .news-item { padding:12px 14px; border-bottom:1px solid var(--border); direction:rtl; }
    .news-item:last-child { border-bottom:none; }
    .news-title { font-size:0.86rem; unicode-bidi:plaintext; }
    .news-title a { color:var(--foreground); text-decoration:none; }
    .news-title a:hover { text-decoration:underline; }
    .news-meta { font-size:0.7rem; color:var(--faint); margin-top:4px; }
    .chip { display:inline-block; border:1px solid var(--border); border-radius:6px; padding:0 6px; font-size:0.68rem;
            color:var(--muted-foreground); margin-inline-start:6px; direction:ltr; unicode-bidi:isolate; }

    @media (prefers-reduced-motion: reduce) { .wordmark, .live-dot::after, .hero-value { animation:none; } }

    [data-testid="stButtonGroup"] { overflow-x:auto; scrollbar-width:none; width:100%; }
    [data-testid="stButtonGroup"] [role="radiogroup"] { width:100%; flex-wrap:nowrap !important; padding:1px; box-sizing:border-box; }
    [data-testid="stButtonGroup"] [role="radiogroup"] > * { flex:1 0 auto; display:flex; }
    [data-testid="stButtonGroup"] button { flex:1 1 auto; white-space:nowrap; }
    [data-testid="stButtonGroup"] button p { font-size:0.82rem; overflow:visible !important; text-overflow:clip !important; }

    /* Metric cards: 2 per row on phones, 4 per row on wide screens */
    [class*="st-key-metrics4"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; }
    [class*="st-key-metrics4"] [data-testid="stColumn"], [class*="st-key-metrics4"] [data-testid="column"] {
        flex:1 1 calc(50% - 5px) !important; width:calc(50% - 5px) !important; min-width:calc(50% - 5px) !important; }

    [class*="st-key-metrics4"] [data-testid="stMetric"] { min-height:108px; box-sizing:border-box; }

    /* Positions: cards on phones, full table on larger screens */
    .st-key-pos_cards { display:none; }
    @media (max-width: 700px) {
        .st-key-pos_cards { display:block; }
        .st-key-pos_table { display:none; }
    }

    /* Wide screens: use the room */
    @media (min-width: 1000px) {
        .main .block-container, [data-testid="stMainBlockContainer"] { max-width:1120px !important; }
        [class*="st-key-metrics4"] [data-testid="stColumn"], [class*="st-key-metrics4"] [data-testid="column"] {
            flex:1 1 calc(25% - 8px) !important; width:calc(25% - 8px) !important; min-width:calc(25% - 8px) !important; }
        .hero-value { font-size:3.4rem; }
    }

    /* Keep columns side by side on phones (2x2 metric grids, inline form fields) */
    [data-testid="stHorizontalBlock"] { flex-wrap:nowrap !important; gap:10px !important; direction:rtl; }
    [data-testid="stColumn"], [data-testid="column"] { min-width:0 !important; }

    /* Top-level tabs: underline style on desktop, frosted bottom nav on phones. Nested tabs stay inline. */
    [data-testid="stTabs"] [role="tablist"] { direction:rtl; }
    [data-testid="stTab"] p { font-size:0.9rem; }
    @media (max-width: 900px) {
        [data-testid="stHeader"] { background:rgba(9,9,11,.96) !important; }
        /* Main navigation: sticky bar at the top (the bottom corners are covered by Streamlit Cloud's own badges) */
        [data-testid="stTabs"] [role="tablist"] {
            position:sticky !important; top:3.6rem !important; z-index:999 !important;
            background:rgba(9,9,11,.92) !important; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
            border-bottom:1px solid var(--border) !important; padding:0 !important;
            margin:0 0 6px !important; display:flex !important; justify-content:space-around !important; gap:0 !important; overflow:visible !important;
        }
        [data-testid="stTabs"] [role="tablist"] > [role="tab"] {
            flex:1 1 0 !important; min-width:0 !important; justify-content:center !important; padding:12px 2px !important; margin:0 !important;
            color:var(--muted-foreground) !important; border:none !important;
        }
        [data-testid="stTabs"] [role="tablist"] > [role="tab"] p { font-size:0.8rem !important; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        [data-testid="stTabs"] [role="tablist"] > [role="tab"][aria-selected="true"] { color:var(--foreground) !important; box-shadow:inset 0 -2px 0 var(--foreground); }
        [data-testid="stTabs"] [role="tablist"] > [role="tab"][aria-selected="true"] p { font-weight:600; }
        [data-testid="stTabs"] [role="tablist"] > [role="tab"] > div:not([data-testid]),
        [data-testid="stTabs"] [role="tablist"] > div[data-baseweb="tab-highlight"],
        [data-testid="stTabs"] [role="tablist"] > div[data-baseweb="tab-border"] { display:none !important; }

        [data-testid="stTabPanel"] [data-testid="stTabs"] [role="tablist"], [role="tabpanel"] [data-testid="stTabs"] [role="tablist"] {
            position:static !important; background:transparent !important; backdrop-filter:none; -webkit-backdrop-filter:none;
            border-top:none !important; margin:0 !important; border-bottom:1px solid var(--border) !important; padding:0 !important;
            justify-content:flex-start !important; overflow-x:auto !important; gap:4px !important;
        }
        [data-testid="stTabPanel"] [data-testid="stTabs"] [role="tablist"] > [role="tab"], [role="tabpanel"] [data-testid="stTabs"] [role="tablist"] > [role="tab"] {
            flex:none !important; padding:8px 10px !important;
        }
        [data-testid="stTabPanel"] [data-testid="stTabs"] [role="tablist"] > [role="tab"][aria-selected="true"],
        [role="tabpanel"] [data-testid="stTabs"] [role="tablist"] > [role="tab"][aria-selected="true"] { box-shadow:inset 0 -2px 0 var(--foreground); }
        [data-testid="stTabPanel"] [data-testid="stTabs"] [role="tablist"] > [role="tab"] p { font-size:0.8rem !important; }

        [data-testid="stMetric"] { padding:12px 12px !important; }
        [data-testid="stMetricValue"] { font-size:1.15rem !important; }
        .hero-value { font-size:2.6rem; }
    }
</style>
""", unsafe_allow_html=True)

SIGNAL_COLOR = {'STRONG BUY': GAIN, 'BUY': GAIN, 'WATCH': INFO, 'HOLD': TEXT_DIM,
                'SELL': LOSS, 'STRONG SELL': LOSS, 'AVOID': LOSS}
SIGNAL_LABEL_HE = {'STRONG BUY': 'קנייה חזקה', 'BUY': 'קנייה', 'WATCH': 'מעקב', 'HOLD': 'החזק',
                   'SELL': 'מכירה', 'STRONG SELL': 'מכירה חזקה', 'AVOID': 'הימנע'}
DAYS_HE = ['שני', 'שלישי', 'רביעי', 'חמישי', 'שישי', 'שבת', 'ראשון']
PERIODS = {"יום": "1d", "שבוע": "1w", "חודש": "1m", "מתחילת השנה": "ytd"}
CHART_BARS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504}

MARKET_TICKERS = {'SPY': ('S&P 500', '$'), 'QQQ': ('NASDAQ 100', '$'), 'TA35.TA': ('ת״א 35', ''),
                  '^VIX': ('VIX', ''), 'USDILS=X': ('דולר/שקל', '₪'), 'GLD': ('זהב', '$')}
TASE_INDICES = {'TA35.TA', '^TA35.TA', 'TA90.TA', 'TA125.TA', '^TA125.TA'}  # index points, not agorot
PROXY_CANDIDATES = {'^TA125.TA': 'ת״א 125', 'TA125.TA': 'ת״א 125', 'TA35.TA': 'ת״א 35', '^TA35.TA': 'ת״א 35'}

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

# Hebrew (and common English) names for TASE stocks -> Yahoo symbol. Anything else resolves through Yahoo search.
TASE_NAMES = {
    'לאומי': 'LUMI.TA', 'בנק לאומי': 'LUMI.TA', 'הפועלים': 'POLI.TA', 'בנק הפועלים': 'POLI.TA',
    'דיסקונט': 'DSCT.TA', 'מזרחי': 'MZTF.TA', 'מזרחי טפחות': 'MZTF.TA', 'הבינלאומי': 'FIBI.TA',
    'טבע': 'TEVA.TA', 'אלביט': 'ESLT.TA', 'אלביט מערכות': 'ESLT.TA', 'נייס': 'NICE.TA', 'כיל': 'ICL.TA',
    'בזק': 'BEZQ.TA', 'אל על': 'ELAL.TA', 'עזריאלי': 'AZRG.TA', 'הפניקס': 'PHOE.TA', 'הראל': 'HARL.TA',
    'מגדל': 'MGDL.TA', 'כלל ביטוח': 'CLIS.TA', 'מנורה': 'MMHD.TA', 'מנורה מבטחים': 'MMHD.TA',
    'נובה': 'NVMI.TA', 'טאואר': 'TSEM.TA', 'בזן': 'ORL.TA', 'דלק': 'DLEKG.TA', 'קבוצת דלק': 'DLEKG.TA',
    'ניו מד': 'NWMD.TA', 'אנלייט': 'ENLT.TA', 'אורמת': 'ORA.TA', 'שופרסל': 'SAE.TA', 'שטראוס': 'STRS.TA',
    'ביג': 'BIG.TA', 'אמות': 'AMOT.TA', 'מליסרון': 'MLSR.TA', 'אלוני חץ': 'ALHE.TA', 'שיכון ובינוי': 'SKBN.TA',
    'קמטק': 'CAMT.TA', 'מטריקס': 'MTRX.TA', 'ישראכרט': 'ISCD.TA', 'אנרג׳יאן': 'ENOG.TA', 'שפיר': 'SPEN.TA',
    'פתאל': 'FTAL.TA', 'אלקו': 'ELCO.TA', 'קנון': 'KEN.TA', 'פורמולה': 'FORTY.TA', 'נקסט ויז׳ן': 'NXSN.TA',
    'ת״א 35': 'TA35.TA', 'תא 35': 'TA35.TA',
}
HEBREW = re.compile(r'[\u0590-\u05FF]')
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
    if round(abs(v), dec) == 0:
        return f"{sym}{0:,.{dec}f}"
    return f"{'+' if v > 0 else '-'}{sym}{abs(v):,.{dec}f}"

def pct(v, dec=2):
    return f"{v:+.{dec}f}%"

def iso(x):
    """Bidi-isolate a number inside Hebrew text so signs and symbols stay in place."""
    return f'\u2066{x}\u2069'

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
PRICING_COLS = ['Proxy', 'RefPrice', 'RefProxy']  # for holdings Yahoo can't price
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
    df = df.reindex(columns=PORT_COLS + PRICING_COLS).copy()
    df['Ticker'] = df['Ticker'].fillna('').astype(str).str.strip().str.upper()
    for c in ('Quantity', 'PurchasePrice', 'RefPrice', 'RefProxy'):
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype(float)
    df['Proxy'] = df['Proxy'].fillna('').astype(str).str.strip().str.upper().replace('NAN', '')
    df = df[(df['Ticker'] != '') & (df['Ticker'] != 'NAN') & (df['Quantity'] > 0) & (df['PurchasePrice'] > 0)]
    return df.drop_duplicates('Ticker', keep='last').set_index('Ticker')[['Quantity', 'PurchasePrice'] + PRICING_COLS]

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
            if is_ils(t) and t not in TASE_INDICES:
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

def synthetic_snapshot(t, row, hists):
    """Prices a holding Yahoo doesn't carry. With a proxy index and a reference price, the unit price
    follows the index's % moves from that reference; otherwise it holds the reference (or purchase) price.
    Returns (snapshot, history_or_None)."""
    ref_price = float(row['RefPrice']) or float(row['PurchasePrice'])
    proxy, ref_proxy = row['Proxy'], float(row['RefProxy'])
    ph = hists.get(proxy) if proxy else None
    number = t[:-3] if t.endswith('.TA') else t
    if ph is not None and float(row['RefPrice']) > 0 and ref_proxy > 0:
        h = ph.copy()
        h[OHLC] = h[OHLC] * (ref_price / ref_proxy)
        h['Volume'] = 0.0
        snap = build_snapshot(t, h, {'quoteType': 'ETF'})
        snap.update(name=f"הערכה לפי {PROXY_CANDIDATES.get(proxy, proxy)}", sector='—', synthetic='proxy',
                    proxy=proxy, ref_price=ref_price, number=number)
        return snap, h
    today = pd.Timestamp(now_il().date())
    h = pd.DataFrame({c: [ref_price, ref_price] for c in OHLC} | {'Volume': [0.0, 0.0]},
                     index=[today - pd.offsets.BDay(1), today])
    source = 'manual' if float(row['RefPrice']) > 0 else 'cost'
    snap = build_snapshot(t, h, {'quoteType': 'ETF'})
    snap.update(name="מחיר ידני" if source == 'manual' else "ללא מחיר שוק — לפי מחיר הקנייה", sector='—',
                synthetic=source, proxy='', ref_price=ref_price, number=number)
    return snap, None

def apply_trade(portfolio, ticker, action, qty, price):
    """Returns (new_portfolio, executed_qty, realized_pnl_in_quote_currency)."""
    p = portfolio.copy()
    if action == 'buy':
        if ticker in p.index:
            oq, op = float(p.loc[ticker, 'Quantity']), float(p.loc[ticker, 'PurchasePrice'])
            nq = oq + qty
            p.loc[ticker, 'Quantity'] = nq
            p.loc[ticker, 'PurchasePrice'] = (oq * op + qty * price) / nq
        else:
            new_row = {c: 0.0 for c in p.columns}
            new_row.update(Quantity=float(qty), PurchasePrice=float(price))
            if 'Proxy' in new_row:
                new_row['Proxy'] = ''
            p.loc[ticker] = pd.Series(new_row)
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

def _norm_he(text):
    text = text.replace("'", '׳').replace('"', '״').replace('בע״מ', '')
    return re.sub(r'\s+', ' ', text).strip()

TASE_NAMES_NORM = {_norm_he(k): v for k, v in TASE_NAMES.items()}
TICKER_RE = re.compile(r'[A-Z0-9.\-^=]{1,15}')

def can_be_tase(t):
    return '.' not in t and '=' not in t and not t.startswith('^') and not t.endswith('-USD')

def match_hebrew(text):
    """Hebrew company name -> TASE symbol, or None if unknown/ambiguous."""
    q = _norm_he(text)
    if q in TASE_NAMES_NORM:
        return TASE_NAMES_NORM[q]
    hits = {v for k, v in TASE_NAMES_NORM.items() if q in k or k in q}
    return hits.pop() if len(hits) == 1 else None

def resolve_token(tok, market='auto'):
    """market: 'auto' (as typed), 'us', or 'ta' (append .TA)."""
    tok = tok.strip()
    if not tok:
        return None
    if HEBREW.search(tok):
        return match_hebrew(tok)
    t = tok.upper()
    if not TICKER_RE.fullmatch(t):
        return None
    if market == 'ta' and can_be_tase(t):
        t += '.TA'
    return t

def parse_tickers(text, market='auto', limit=25):
    """Returns (symbols, unrecognized). Commas/newlines separate entries; Latin entries may also be space-separated."""
    out, bad = [], []
    for chunk in re.split(r'[,;\n]+', text or ''):
        chunk = chunk.strip()
        if not chunk:
            continue
        for part in ([chunk] if HEBREW.search(chunk) else chunk.split()):
            sym = resolve_token(part, market)
            if sym is None:
                bad.append(part)
            elif sym not in out:
                out.append(sym)
    return out[:limit], bad

def load_with_fallback(symbols, market='auto'):
    """Loads symbols; in auto mode, a bare symbol with no US data is retried on TASE (SYMBOL.TA)."""
    hists, snaps = load_market(tuple(sorted(symbols)), True)
    retry = {t: f"{t}.TA" for t in symbols if t not in snaps and market == 'auto' and can_be_tase(t)}
    if retry:
        h2, s2 = load_market(tuple(sorted(retry.values())), True)
        hists, snaps = {**hists, **h2}, {**snaps, **s2}
    resolved, bad = [], []
    for t in symbols:
        sym = t if t in snaps else retry.get(t)
        if sym in snaps:
            if sym not in resolved:
                resolved.append(sym)
        else:
            bad.append(t)
    return hists, snaps, resolved, bad

def resolve_single(text, market='auto'):
    """Returns (symbol, last_price). last_price is None when no market data exists for the symbol."""
    syms, _ = parse_tickers(text, market, 1)
    if not syms:
        return None, None
    t = syms[0]
    for cand in [t] + ([f"{t}.TA"] if market == 'auto' and can_be_tase(t) else []):
        h = download_history((cand,), '1mo').get(cand)
        if h is not None:
            return cand, float(h['Close'].iloc[-1])
    if t.isdigit():
        return f"{t}.TA", None  # TASE security number (מספר נייר) that Yahoo doesn't carry
    return t, None

@st.cache_data(ttl=86400, show_spinner=False)
def search_symbols(query, tase_only=False):
    """Company-name lookup. Hebrew queries use the built-in TASE list; others use Yahoo search."""
    q = (query or '').strip()
    if not q:
        return []
    if HEBREW.search(q):
        qn, seen, out = _norm_he(q), set(), []
        for k, v in TASE_NAMES_NORM.items():
            if (qn in k or k in qn) and v not in seen:
                seen.add(v)
                out.append({'symbol': v, 'name': k, 'exchange': 'בורסת ת״א'})
        return out
    try:
        quotes = yf.Search(q, max_results=10, news_count=0, lists_count=0, raise_errors=False).quotes or []
    except Exception:
        quotes = []
    out = []
    for x in quotes:
        sym = x.get('symbol')
        if not sym or x.get('quoteType') not in ('EQUITY', 'ETF', 'MUTUALFUND', 'INDEX'):
            continue
        if tase_only and not sym.endswith('.TA'):
            continue
        out.append({'symbol': sym, 'name': x.get('longname') or x.get('shortname') or sym,
                    'exchange': 'בורסת ת״א' if sym.endswith('.TA') else (x.get('exchDisp') or x.get('exchange') or '')})
    return out

MARKET_OPTIONS = {'auto': 'זיהוי אוטומטי', 'us': 'ארה״ב', 'ta': 'בורסת ת״א'}

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

def exchange_label(t):
    return 'ת״א' if is_ils(t) else None

def ledger_row(ticker, subtitle, price_str, sub_str, sub_color, tag_text, tag_color, caption=None, total=False):
    cap = f'<div class="row-caption">{esc(caption)}</div>' if caption else ''
    chip = f'<span class="chip">{exchange_label(ticker)}</span>' if exchange_label(ticker) else ''
    name_dir = '' if HEBREW.search(str(subtitle)) else ' latin'
    return (f'<div class="ledger-row{" total" if total else ""}">'
            f'<div><div><span class="row-ticker">{esc(ticker)}</span>{chip}</div><div class="row-name{name_dir}">{esc(subtitle)}</div></div>'
            f'<div class="row-mid"><div class="row-price">{esc(price_str)}</div>'
            f'<div class="row-sub" style="color:{sub_color}">{esc(sub_str)}</div></div>'
            f'<div class="row-end"><div class="row-tag" style="color:{tag_color}">{esc(tag_text)}</div>{cap}</div></div>')

def render_ledger(rows_html):
    if rows_html:
        st.markdown('<div class="ledger">' + ''.join(rows_html) + '</div>', unsafe_allow_html=True)

def attn(ticker, msg, color=GOLD):
    st.markdown(f'<div class="attn-strip"><span class="bar" style="background:{color}"></span>'
                f'<div><div class="row-ticker">{esc(ticker)}</div><div class="row-name">{esc(msg)}</div></div></div>',
                unsafe_allow_html=True)

def attn_list(items):
    rows = ''.join(f'<div class="attn-strip"><span class="bar" style="background:{c}"></span>'
                   f'<div><div class="row-ticker">{esc(t)}</div><div class="row-name">{esc(m)}</div></div></div>' for t, m, c in items)
    st.markdown(f'<div class="ledger">{rows}</div>', unsafe_allow_html=True)

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
    st.markdown(f'<div class="section-sub" style="margin:14px 0 0">טווח 52 שבועות · {pos:.0f}% מהשפל לשיא</div>'
                f'<div class="range-wrap"><div class="range-bar"><div class="range-fill" style="width:{pos:.1f}%"></div>'
                f'<div class="range-marker" style="left:{pos:.1f}%"></div></div>'
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
            title = f'<a href="{esc(n["url"])}" target="_blank" rel="noopener">{title}</a>'
        when = n['ts'].strftime('%d.%m %H:%M') if n['ts'] is not None else ''
        meta = ' · '.join(x for x in (esc(n['publisher']), esc(when)) if x)
        out.append(f'<div class="news-item"><div class="news-title">{title}</div>'
                   f'<div class="news-meta"><span class="chip" style="margin:0 0 0 6px">{esc(n["ticker"])}</span>{meta}</div></div>')
    st.markdown('<div class="ledger">' + ''.join(out) + '</div>', unsafe_allow_html=True)

def dark_layout(fig, height):
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family=FONT, color=TEXT_DIM, size=11), margin=dict(l=6, r=6, t=8, b=6),
                      hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(family=FONT, color=TEXT, size=12)))
    fig.update_xaxes(gridcolor='rgba(39,39,42,0.6)', zeroline=False, linecolor=LINE, showline=False)
    fig.update_yaxes(gridcolor='rgba(39,39,42,0.6)', zeroline=False)
    return fig

def hex_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"

def price_chart(hist, bars, cost=None, sym='$', candles=False):
    """Category x-axis: plots only real trading days, whatever the exchange calendar."""
    sma = {p: hist['Close'].rolling(p).mean().iloc[-bars:] for p in (50, 200)}
    h = hist.iloc[-bars:]
    x = h.index.strftime('%d.%m.%y')
    color = tone(h['Close'].iloc[-1] - h['Close'].iloc[0])
    lows = [float(h['Low' if candles else 'Close'].min())] + [float(v.min()) for v in sma.values() if v.notna().any()]
    highs = [float(h['High' if candles else 'Close'].max())] + [float(v.max()) for v in sma.values() if v.notna().any()]
    if cost:
        lows.append(cost); highs.append(cost)
    pad = (max(highs) - min(lows)) * 0.06 or max(highs) * 0.02
    floor, ceil = min(lows) - pad, max(highs) + pad
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.04)
    if candles:
        fig.add_trace(go.Candlestick(x=x, open=h['Open'], high=h['High'], low=h['Low'], close=h['Close'], name='מחיר',
                                     showlegend=False, increasing_line_color=GAIN, decreasing_line_color=LOSS,
                                     increasing_fillcolor=GAIN, decreasing_fillcolor=LOSS), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=[floor] * len(h), line=dict(width=0), hoverinfo='skip', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=h['Close'], name='מחיר', showlegend=False, fill='tonexty',
                                 fillcolor=hex_rgba(color, 0.10), line=dict(color=color, width=2),
                                 hovertemplate=f'{sym}%{{y:,.2f}}<extra></extra>'), row=1, col=1)
    for p, c in ((50, INFO), (200, GOLD)):
        if sma[p].notna().any():
            fig.add_trace(go.Scatter(x=x, y=sma[p].values, name=f'ממוצע {p}', line=dict(color=c, width=1.1, dash='dot'),
                                     hovertemplate=f'SMA{p} {sym}%{{y:,.2f}}<extra></extra>'), row=1, col=1)
    if cost:
        fig.add_hline(y=cost, line_dash='dash', line_color=TEXT_DIM, line_width=1, row=1, col=1,
                      annotation_text=f'מחיר קנייה {sym}{cost:,.2f}', annotation_position='top left',
                      annotation_font=dict(color=TEXT_DIM, size=11))
    fig.add_trace(go.Bar(x=x, y=h['Volume'], name='נפח', showlegend=False, marker_color='rgba(161,161,170,0.28)',
                         hovertemplate='%{y:,.0f}<extra>נפח</extra>'), row=2, col=1)
    dark_layout(fig, 440)
    fig.update_layout(hovermode='x unified', bargap=0.25,
                      legend=dict(orientation='h', y=1.06, x=1, xanchor='right', font=dict(size=11)))
    fig.update_xaxes(type='category', nticks=6, rangeslider_visible=False, showgrid=False)
    fig.update_yaxes(range=[floor, ceil], row=1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=2, col=1)
    return fig

def price_note(s, t):
    src = s.get('synthetic')
    if src == 'proxy':
        return (f"הערכה: עוקב אחר {PROXY_CANDIDATES.get(s['proxy'], s['proxy'])} ממחיר ייחוס {money(s['ref_price'], s['sym'])} "
                "· לא כולל דמי ניהול ופערי מעקב")
    if src == 'manual':
        return f"מחיר ידני {money(s['ref_price'], s['sym'])} — מתעדכן רק כשמעדכנים אותו"
    if src == 'cost':
        return "אין מחיר שוק — מוצג לפי מחיר הקנייה"
    return f"מחיר אחרון מ-{s['last_date']:%d.%m.%Y}" + (" · מחירים בשקלים" if is_ils(t) else "")

def render_stock_detail(t, s, hist, pos=None, key='detail'):
    sym = s['sym']
    chip = f'<span class="chip">{exchange_label(t)}</span>' if exchange_label(t) else ''
    sector = f" · {s['sector']}" if s['sector'] and s['sector'] != '—' else ''
    st.markdown(f'<div class="hero" style="margin-top:6px"><div><span class="row-ticker" style="font-size:1.4rem">{esc(t)}</span>{chip}</div>'
                f'<div class="row-name" style="font-size:0.8rem">{esc(s["name"])}{esc(sector)}</div>'
                f'<div class="stamp">{esc(price_note(s, t))}</div></div>', unsafe_allow_html=True)
    synth = s.get('synthetic')

    c1, c2 = st.columns(2)
    c1.metric("מחיר", money(s['price'], sym), f"{pct(s['rets']['1d'])} ({signed_money(s['day_chg'], sym, 2)})")
    if pos:
        c2.metric("שווי הפוזיציה", money(pos['value']), f"{signed_money(pos['pl_1d'])} היום",
                  delta_color='normal' if round(pos['pl_1d']) else 'off', delta_arrow='auto' if round(pos['pl_1d']) else 'off')
        c3, c4 = st.columns(2)
        c3.metric("רווח/הפסד מאז הקנייה", signed_money(pos['unreal']), pct(pos['unreal_pct']))
        c4.metric("מחיר קנייה ממוצע", money(pos['avg'], sym),
                  f"{iso(fmt_qty(pos['qty']))} יח׳ · {iso(format(pos['weight'], '.1f') + '%')} מהתיק", delta_color='off', delta_arrow='off')
    else:
        c2.metric("מתחילת השנה", pct(s['rets']['ytd']), signed_money(s['price'] - s['bases']['ytd'], sym, 2))

    if hist is None or synth in ('manual', 'cost'):
        st.caption("ל-Yahoo אין נתונים על נייר זה, ולכן אין גרף. אפשר לקבוע מעקב לפי מדד: ניהול → עסקאות ועריכה → ניירות ללא מחיר שוק.")
        return
    strip([(lbl, pct(s['rets'][k], 1), '', tone(s['rets'][k])) for lbl, k in
           (("יום", '1d'), ("שבוע", '1w'), ("חודש", '1m'), ("3 חודשים", '3m'), ("מתחילת שנה", 'ytd'), ("שנה", '1y'))])
    range_bar(s['lo52'], s['hi52'], s['price'], sym)

    g1, g2 = st.columns([5, 3])
    period = g1.segmented_control("טווח גרף", list(CHART_BARS), default="3M", key=f'{key}_period_{t}',
                                  label_visibility='collapsed') or "3M"
    style = g2.segmented_control("סוג גרף", ["קו", "נרות"], default="קו", key=f'{key}_style_{t}',
                                 label_visibility='collapsed') or "קו"
    st.plotly_chart(price_chart(hist, CHART_BARS[period], pos['avg'] if pos else None, sym, style == "נרות"),
                    width='stretch', config={"displayModeBar": False}, key=f'{key}_chart_{t}')
    if synth == 'proxy':
        return

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
        f3.metric("שווי שוק", big(s['market_cap']) if not is_ils(t) else (f"₪{s['market_cap'] / 1e9:,.2f}B" if s['market_cap'] else "—"))
        f4.metric("תשואת דיבידנד", f"{div_yield:.2f}%" if div_yield else "—")
        f5, f6 = st.columns(2)
        f5.metric("יעד אנליסטים", money(s['target'], sym) if s['target'] else "—",
                  pct(upside, 1) if upside is not None else None)
        f6.metric("המלצה", s['rec'].replace('_', ' '), f"{s['n_analysts']} אנליסטים" if s['n_analysts'] else None,
                  delta_color='off', delta_arrow='off')
        if s['quote_type'] != 'ETF':
            st.caption(f"צמיחת הכנסות {s['growth']:.1f}% · שולי רווח {s['margin']:.1f}% · ROE {s['roe']:.1f}%")
        if is_ils(t):
            st.caption("כיסוי הנתונים הפונדמנטליים למניות ת״א ב-Yahoo חלקי — ייתכנו שדות ריקים.")

    if s['quote_type'] not in ('ETF', 'CRYPTOCURRENCY', 'INDEX', 'CURRENCY', 'MUTUALFUND'):
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
        if st.form_submit_button("שמור יתרות", width='stretch'):
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
    hists, snaps = dict(hists), dict(snaps)  # copies: never mutate cached results
    needs_pricing = [t for t in p_tickers if t not in snaps]
    proxy_hists = {}
    if needs_pricing:
        proxy_hists = download_history(tuple(sorted(PROXY_CANDIDATES)), '2y')
    synthetic = {}
    for t in needs_pricing:
        snap, h = synthetic_snapshot(t, portfolio.loc[t], proxy_hists)
        snaps[t] = snap
        synthetic[t] = snap['synthetic']
        if h is not None:
            hists[t] = h
# one entry per index (first symbol that Yahoo actually returns data for)
available_proxies = {}
for sym_, label_ in PROXY_CANDIDATES.items():
    if sym_ in proxy_hists and label_ not in available_proxies.values():
        available_proxies[sym_] = label_

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
        watch_alerts.append((w['Ticker'], f"{iso(money(s['price'], s['sym']))} — מעל יעד ההתראה {iso(money(w['AlertHigh'], s['sym']))}", GAIN))
    if w['AlertLow'] > 0 and s['price'] <= w['AlertLow']:
        watch_alerts.append((w['Ticker'], f"{iso(money(s['price'], s['sym']))} — מתחת ליעד ההתראה {iso(money(w['AlertLow'], s['sym']))}", LOSS))

def wall_street_open(market):
    """Open only during NYSE hours on a day SPY actually printed a bar (so holidays read as closed)."""
    ny = datetime.now(ZoneInfo("America/New_York"))
    in_hours = ny.weekday() < 5 and (9, 30) <= (ny.hour, ny.minute) < (16, 0)
    traded_today = 'SPY' in market and market['SPY']['last_date'].date() == ny.date()
    return in_hours and traded_today

is_open = wall_street_open(market)
st.markdown(f'<div class="topbar"><span class="wordmark">AlonStocks Pro</span>'
            f'<span class="live"><span class="live-dot{"" if is_open else " closed"}"></span>'
            f'{"וול סטריט פתוחה" if is_open else "וול סטריט סגורה"} · {now_il():%H:%M}</span></div>', unsafe_allow_html=True)

t_today, t_pos, t_search, t_watch, t_earn, t_manage = st.tabs(
    ["היום", "פוזיציות", "חיפוש", "מעקב", "דוחות", "ניהול"])

# ==========================================
# TAB: TODAY
# ==========================================
with t_today:
    def badge(value, label, color):
        return f'<span class="badge"><span class="num ltr" style="color:{color}">{esc(value)}</span>{esc(label)}</span>'
    now = now_il()
    spy_day = market.get('SPY', {}).get('rets', {}).get('1d')
    badges = [badge(f"{signed_money(day_pl)} ({pct(day_pct)})", "היום", tone(day_pl)),
              badge(f"{signed_money(total_return)} ({pct(total_return_pct, 1)})", "מאז ההפקדה", tone(total_return))]
    if spy_day is not None:
        badges.append(badge(pct(spy_day), "S&P 500", tone(spy_day)))
    st.markdown(f'<div class="hero"><div class="hero-label">שווי תיק כולל</div>'
                f'<div class="hero-value">${total_value:,.0f}</div>'
                f'<div class="badges">{"".join(badges)}</div>'
                f'<div class="stamp">≈ ₪{total_value * usd_ils:,.0f} לפי שער {usd_ils:.3f} · יום {DAYS_HE[now.weekday()]} {now:%d.%m.%Y}</div></div>',
                unsafe_allow_html=True)
    with st.container(key='metrics4_today'):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("שווי מניות", money(stock_value, '$', 0), f"{signed_money(day_pl)} היום",
                  delta_color='normal' if round(day_pl) else 'off', delta_arrow='auto' if round(day_pl) else 'off')
        m2.metric("מזומן", money(cash_total, '$', 0), f"${cfg['cash_usd']:,.0f} + ₪{cfg['cash_ils']:,.0f}", delta_color='off', delta_arrow='off')
        m3.metric("רווח/הפסד פתוח", signed_money(unreal_total), pct(unreal_total / cost_total * 100 if cost_total else 0))
        m4.metric("תשואה כוללת", pct(total_return_pct, 1), f"מול ${net_deposits:,.0f} הפקדות", delta_color='off', delta_arrow='off')

    with st.expander("עדכון יתרות מזומן"):
        render_cash_editor('cash_form_today')

    if market:
        strip([(name, f"{sym}{market[t]['price']:,.2f}", pct(market[t]['rets']['1d']), tone(market[t]['rets']['1d']))
               for t, (name, sym) in MARKET_TICKERS.items() if t in market])

    unpriced = [t for t, src in synthetic.items() if src == 'cost']
    if unpriced:
        st.warning(f"ל-Yahoo אין מחיר עבור {', '.join(unpriced)}, ולכן הם מוצגים לפי מחיר הקנייה. "
                   "להגדרת מעקב לפי מדד או מחיר ידני: ניהול → עסקאות ועריכה → ניירות ללא מחיר שוק.")

    # ── Needs attention ──
    with st.spinner("בודק דוחות רווחים..."):
        upcoming, _ = earnings_board(p_tickers + w_tickers, snaps)
    attention = {}
    def flag(t, note, color):
        notes, first_color = attention.get(t, ([], color))
        attention[t] = (notes + [note], first_color)
    for r in positions:
        s_ = snaps[r['ticker']]
        if abs(r['day_pct']) >= 3:
            flag(r['ticker'], f"תנועה חדה היום {iso(pct(r['day_pct'], 1))}", tone(r['day_pct']))
        if s_['price'] >= s_['hi52'] * 0.98:
            flag(r['ticker'], "קרוב לשיא 52 שבועות", GAIN)
        elif s_['price'] <= s_['lo52'] * 1.02:
            flag(r['ticker'], "קרוב לשפל 52 שבועות", LOSS)
        if s_.get('rsi', 50) >= 75:
            flag(r['ticker'], f"RSI גבוה ({s_['rsi']:.0f})", GOLD)
        elif s_.get('rsi', 50) <= 25:
            flag(r['ticker'], f"RSI נמוך ({s_['rsi']:.0f})", GOLD)
        if r['unreal_pct'] <= -15:
            flag(r['ticker'], f"{iso(pct(r['unreal_pct'], 1))} מאז הקנייה", LOSS)
        if r['weight'] >= 35 and r['quote_type'] != 'ETF':
            flag(r['ticker'], f"{iso(format(r['weight'], '.0f') + '%')} מהתיק בפוזיציה אחת", GOLD)
    for u in upcoming:
        if u['days'] <= 7:
            when = "היום" if u['days'] == 0 else "מחר" if u['days'] == 1 else f"בעוד {u['days']} ימים"
            flag(u['ticker'], f"דוח רווחים {when} ({u['date']:%d.%m})", GOLD)
    for t, msg, color in watch_alerts:
        flag(t, msg, color)

    section("דורש תשומת לב", f"{len(attention)} סימולים" if attention else None)
    if attention:
        attn_list([(t, ' · '.join(notes), color) for t, (notes, color) in attention.items()])
    else:
        st.caption("אין אירועים חריגים בתיק היום.")

    # ── Positions today ──
    if positions:
        section("הפוזיציות שלי היום", "שינוי יומי במחיר וברווח, ומשמאל — רווח/הפסד כולל מאז הקנייה")
        rows_html = [ledger_row(
            r['ticker'], r['name'][:26], money(r['price'], r['sym']),
            f"{pct(r['day_pct'])}  {signed_money(r['pl_1d'])}", tone(r['pl_1d']),
            pct(r['unreal_pct'], 1), tone(r['unreal']),
            caption=f"{iso(signed_money(r['unreal']))} כולל")
            for r in sorted(positions, key=lambda x: x['day_pct'], reverse=True)]
        rows_html.append(ledger_row(
            "סה״כ", f"{len(positions)} פוזיציות", money(stock_value, '$', 0),
            f"{pct(day_pct)}  {signed_money(day_pl)}", tone(day_pl),
            pct(unreal_total / cost_total * 100 if cost_total else 0, 1), tone(unreal_total),
            caption=f"{iso(signed_money(unreal_total))} כולל", total=True))
        render_ledger(rows_html)

        # ── Period performance ──
        section("ביצועים לפי תקופה", "לפי הכמויות שמוחזקות כרגע, ללא מזומן")
        per_label = st.segmented_control("תקופה", list(PERIODS), default="יום", key='today_period', label_visibility='collapsed') or "יום"
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
            textposition='auto', marker_cornerradius=4, insidetextfont=dict(color='#09090B'), outsidetextfont=dict(color=TEXT),
            hovertemplate='%{y}: %{text}<extra></extra>'))
        dark_layout(fig, 60 + 40 * len(ordered))
        fig.update_layout(bargap=0.35)
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=True, zerolinecolor=LINE)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False}, key='period_chart')

    # ── Earnings calendar ──
    soon = [u for u in upcoming if u['days'] <= 14]
    if soon:
        section("דוחות רווחים בשבועיים הקרובים", "תאריכים לפי Yahoo Finance; ייתכנו שינויים")
        render_ledger([ledger_row(
            u['ticker'], f"תחזית EPS {u['eps_est']:.2f}" if u['eps_est'] is not None else "אין תחזית",
            f"{u['date']:%d.%m}", "היום" if u['days'] == 0 else f"בעוד {u['days']} ימים", GOLD if u['days'] <= 7 else TEXT_DIM,
            "בתיק" if u['ticker'] in pos_by_ticker else "במעקב", TEXT_DIM)
            for u in soon])

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    if st.button("רענן מחירים", key='refresh'):
        load_market.clear()
        download_history.clear()
        st.rerun()
    st.caption("מחירים מתעדכנים אוטומטית כל 5 דקות. המידע מוצג לצורך מעקב אישי בלבד ואינו מהווה ייעוץ השקעות.")

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
            with st.container(key='metrics4_pos'):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("שווי מניות", money(stock_value, '$', 0), f"{signed_money(day_pl)} היום")
                c2.metric("רווח/הפסד פתוח", signed_money(unreal_total), pct(unreal_total / cost_total * 100 if cost_total else 0))
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
            sorted_pos = sorted(positions, key=lambda x: x['value'], reverse=True)
            with st.container(key='pos_table'):
                st.dataframe(styled_table(table, pct_cols, fmt), width='stretch', hide_index=True)
            with st.container(key='pos_cards'):
                render_ledger([ledger_row(
                    r['ticker'], f"{r['qty']:,.4g} יח׳ · משקל {r['weight']:.1f}%", money(r['price'], r['sym']),
                    f"{pct(r['day_pct'])}  {signed_money(r['pl_1d'])}", tone(r['pl_1d']),
                    pct(r['unreal_pct'], 1), tone(r['unreal']),
                    caption=f"שווי {iso(money(r['value'], '$', 0))} · {iso(signed_money(r['unreal']))}")
                    for r in sorted_pos])
            st.download_button("ייצוא CSV", table.to_csv(index=False).encode('utf-8-sig'), "portfolio.csv", "text/csv")

            alloc = {r['ticker']: r['value'] for r in positions}
            if cash_total > 0:
                alloc['מזומן'] = cash_total
            fig_pie = px.pie(values=list(alloc.values()), names=list(alloc.keys()), hole=0.55,
                             color_discrete_sequence=['#FAFAFA', '#60A5FA', '#22C55E', '#F59E0B', '#A78BFA', '#A1A1AA', '#F472B6', '#52525B'])
            fig_pie.update_traces(textinfo='label+percent', marker=dict(line=dict(color=INK, width=2)),
                                  hovertemplate='%{label}: $%{value:,.0f} (%{percent})<extra></extra>')
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family=FONT, color=TEXT), height=300,
                                  margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                                  hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(family=FONT, color=TEXT)))
            section("פיזור התיק")
            st.plotly_chart(fig_pie, width='stretch', config={"displayModeBar": False}, key='alloc_pie')
        else:
            render_stock_detail(choice, snaps[choice], hists.get(choice), pos_by_ticker[choice], key='pos')

# ==========================================
# TAB: SEARCH
# ==========================================
def pick_lookup(sym):
    """Callback: runs before widgets are created, so it may set the search box's state."""
    current = st.session_state.get('search_list', [])
    st.session_state.search_list = [sym] + [x for x in current if x != sym]
    st.session_state.search_input = ', '.join(st.session_state.search_list)
    st.session_state.search_market_used = 'auto'
    st.session_state.search_unknown = []

with t_search:
    section("חיפוש מניות", "כמה סימולים או שמות בעברית, מופרדים בפסיק — למשל: AAPL, NVDA, לאומי, טבע")
    with st.form('search_form'):
        query = st.text_input("סימולים או שמות", key='search_input',
                              placeholder="AAPL, NVDA, לאומי, ESLT.TA", label_visibility='collapsed')
        s_market = st.segmented_control("שוק", list(MARKET_OPTIONS), format_func=MARKET_OPTIONS.get,
                                        default='auto', key='search_market') or 'auto'
        st.caption("זיהוי אוטומטי: מחפש קודם בארה״ב, ואם הסימול לא קיים שם — בבורסת ת״א. כדי לכפות ת״א בחר/י ״בורסת ת״א״.")
        if st.form_submit_button("חפש", width='stretch'):
            syms, unknown = parse_tickers(query, s_market)
            st.session_state.search_list = syms
            st.session_state.search_market_used = s_market
            st.session_state.search_unknown = unknown
            if not syms and not unknown:
                st.error("הזן/י לפחות סימול או שם אחד.")

    with st.expander("לא יודע/ת את הסימול? חיפוש לפי שם חברה"):
        with st.form('lookup_form'):
            lk_q = st.text_input("שם החברה", placeholder="Elbit / Teva / בנק הפועלים")
            lk_ta = st.checkbox("רק בורסת ת״א", value=False)
            if st.form_submit_button("מצא", width='stretch'):
                st.session_state.lookup = search_symbols(lk_q, lk_ta)
                st.session_state.lookup_done = bool(lk_q.strip())
        results = st.session_state.get('lookup', [])
        if st.session_state.get('lookup_done') and not results:
            st.info("לא נמצאו התאמות. נסה/י שם באנגלית (למשל Elbit Systems) — החיפוש בעברית מכסה רק את המניות הגדולות בת״א.")
        for r in results:
            c1, c2 = st.columns([4, 1], vertical_alignment='center')
            c1.markdown(f'<div><span class="row-ticker">{esc(r["symbol"])}</span><span class="chip">{esc(r["exchange"])}</span></div>'
                        f'<div class="row-name">{esc(r["name"])}</div>', unsafe_allow_html=True)
            c2.button("הצג", key=f'lookup_{r["symbol"]}', on_click=pick_lookup, args=(r['symbol'],))

    if st.session_state.get('search_unknown'):
        st.warning(f"לא זוהו: {', '.join(st.session_state.search_unknown)}. לשמות שלא מוכרים — השתמש/י בחיפוש לפי שם חברה.")

    search_list = st.session_state.get('search_list', [])
    if search_list:
        with st.spinner("טוען..."):
            s_hists, s_snaps, good, bad = load_with_fallback(search_list, st.session_state.get('search_market_used', 'auto'))
        if bad:
            st.warning(f"לא נמצאו נתונים עבור: {', '.join(bad)}")
        if good:
            render_ledger([ledger_row(
                t, s_snaps[t]['name'][:26], money(s_snaps[t]['price'], s_snaps[t]['sym']),
                f"{pct(s_snaps[t]['rets']['1d'])} היום", tone(s_snaps[t]['rets']['1d']),
                f"{pct(s_snaps[t]['rets']['ytd'], 1)}", tone(s_snaps[t]['rets']['ytd']),
                caption="השנה" + (" · בתיק" if t in pos_by_ticker else " · במעקב" if t in w_tickers else ""))
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
            pick = st.selectbox("בחר/י מניה", good, key='search_pick',
                                format_func=lambda x: f"{x} — {s_snaps[x]['name']}")
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
            wl_ticker = w1.text_input("סימול או שם", placeholder="NVDA / לאומי / POLI.TA")
            wl_notes = w2.text_input("הערה")
            wl_market = st.segmented_control("שוק", list(MARKET_OPTIONS), format_func=MARKET_OPTIONS.get,
                                             default='auto', key='wl_market') or 'auto'
            w3, w4 = st.columns(2)
            wl_high = w3.number_input("התראה כשהמחיר מעל", min_value=0.0, value=0.0, step=1.0)
            wl_low = w4.number_input("התראה כשהמחיר מתחת", min_value=0.0, value=0.0, step=1.0)
            st.caption("מחירי התראה של מניות ת״א — בשקלים.")
            if st.form_submit_button("שמור ברשימה", width='stretch'):
                sym, last = resolve_single(wl_ticker, wl_market)
                if sym is None:
                    st.error("לא זוהה סימול. הזן/י סימול (NVDA, LUMI.TA) או שם מוכר בעברית.")
                elif last is None:
                    st.error(f"לא נמצאו נתוני מסחר עבור {sym}. בדוק/י את הסימול או בחר/י את השוק הנכון.")
                else:
                    new_wl = clean_watchlist(pd.concat([watch, pd.DataFrame([{'Ticker': sym, 'Notes': wl_notes,
                                                                             'AlertHigh': wl_high, 'AlertLow': wl_low}])],
                                                       ignore_index=True))
                    st.session_state.watchlist = new_wl
                    kind, msg = SAVE_MSG[save_watchlist(new_wl)]
                    flash(kind, f"{sym} נשמר ברשימה — {msg}")
                    st.rerun()

    if watch.empty:
        st.info("הרשימה ריקה. הוסף/י מניות שברצונך לעקוב אחריהן.")
    else:
        if watch_alerts:
            section("התראות מחיר שהופעלו")
            attn_list(watch_alerts)

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
                    caption=f"חודש {iso(pct(s['rets']['1m'], 1))}" + (f" · התראה: {alerts_txt}" if alerts_txt else '')))
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
                    f"{u['date']:%d.%m.%Y}", "היום" if u['days'] == 0 else f"בעוד {u['days']} ימים", GOLD if u['days'] <= 7 else TEXT_DIM,
                    "בתיק" if u['ticker'] in pos_by_ticker else "במעקב", TEXT_DIM)
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
            if st.form_submit_button("רשום פעולה", width='stretch'):
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
            tr_input = q1.text_input("סימול או שם", placeholder="AAPL / לאומי / TEVA.TA")
            tr_action = q2.selectbox("פעולה", ["קנייה", "מכירה"])
            tr_market = st.segmented_control("שוק", list(MARKET_OPTIONS), format_func=MARKET_OPTIONS.get,
                                             default='auto', key='trade_market') or 'auto'
            q3, q4 = st.columns(2)
            tr_qty = q3.number_input("כמות", min_value=0.0, step=0.1, value=1.0, format="%.4f")
            tr_price = q4.number_input("מחיר ליחידה (0 = מחיר אחרון)", min_value=0.0, step=0.01, format="%.2f",
                                       help="במטבע המסחר: שקלים למניות ת״א, דולרים למניות ארה״ב.")
            tr_cash = st.checkbox("עדכן את יתרת המזומן", value=True)
            if st.form_submit_button("רשום עסקה", width='stretch'):
                action = 'buy' if tr_action == "קנייה" else 'sell'
                held = {t: t for t in portfolio.index} | {t[:-3]: t for t in portfolio.index if t.endswith('.TA')}
                typed, _ = parse_tickers(tr_input, tr_market, 1)
                if action == 'sell' and typed and tr_market == 'auto' and typed[0] in held:
                    tr_ticker, last = held[typed[0]], None
                else:
                    tr_ticker, last = resolve_single(tr_input, tr_market)
                valid = tr_ticker is not None and tr_qty > 0
                price = tr_price
                if valid and price <= 0:
                    if last is None:
                        h = download_history((tr_ticker,), '1mo').get(tr_ticker)
                        last = float(h['Close'].iloc[-1]) if h is not None else None
                    price = last or 0.0
                if not valid:
                    st.error("הזן/י סימול (או שם מוכר בעברית) וכמות גדולה מאפס.")
                elif price <= 0:
                    st.error(f"לא נמצא מחיר עבור {tr_ticker} — הזן/י מחיר ידנית.")
                elif action == 'sell' and tr_ticker not in portfolio.index:
                    st.error(f"{tr_ticker} לא נמצאת בתיק.")
                else:
                    new_port, done, realized = apply_trade(portfolio, tr_ticker, action, tr_qty, price)
                    if action == 'buy' and (tr_ticker in synthetic or (last is None and tr_price > 0 and tr_ticker not in snaps)):
                        # No market price for this holding: the price just paid becomes its reference price
                        px = new_port.loc[tr_ticker, 'Proxy']
                        new_port.loc[tr_ticker, 'RefPrice'] = float(price)
                        new_port.loc[tr_ticker, 'RefProxy'] = float(proxy_hists[px]['Close'].iloc[-1]) if px in proxy_hists else 0.0
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

        if synthetic:
            section("ניירות ללא מחיר שוק",
                    "ל-Yahoo אין נתונים עליהם. הזן/י את המחיר הנוכחי מהברוקר; אם הנייר עוקב אחרי מדד — בחר/י אותו, והמחיר יתעדכן לפי שינויי המדד מעכשיו.")
            if not available_proxies:
                st.caption("כרגע לא התקבלו מ-Yahoo נתוני מדדי ת״א, ולכן זמין רק מחיר ידני.")
            proxy_opts = [''] + list(available_proxies)
            for t in synthetic:
                row = portfolio.loc[t]
                with st.form(f'pricing_{t}'):
                    st.markdown(f'<div><span class="row-ticker">{esc(t)}</span></div>'
                                f'<div class="row-name" style="white-space:normal">{esc(price_note(snaps[t], t))}</div>', unsafe_allow_html=True)
                    f1, f2 = st.columns(2)
                    current = row['Proxy'] if row['Proxy'] in available_proxies else ''
                    sel = f1.selectbox("מעקב לפי", proxy_opts, index=proxy_opts.index(current), key=f'proxy_{t}',
                                       format_func=lambda x: available_proxies.get(x, 'ללא — מחיר ידני'))
                    unit = f2.number_input(f"מחיר נוכחי ליחידה ({'₪' if is_ils(t) else '$'})", min_value=0.0,
                                           value=float(round(snaps[t]['price'], 2)), step=0.01, format="%.2f", key=f'unit_{t}',
                                           help="בשקלים ולא באגורות: 38,000 אג׳ = 380 ₪.")
                    if st.form_submit_button("שמור מחיר", width='stretch'):
                        if unit <= 0:
                            st.error("הזן/י מחיר גדול מאפס.")
                        else:
                            new_port = portfolio.copy()
                            new_port.loc[t, 'Proxy'] = sel
                            new_port.loc[t, 'RefPrice'] = float(unit)
                            new_port.loc[t, 'RefProxy'] = float(proxy_hists[sel]['Close'].iloc[-1]) if sel else 0.0
                            st.session_state.portfolio = new_port
                            st.session_state.editor_ver += 1
                            kind, msg = SAVE_MSG[save_portfolio(new_port)]
                            how = f"עוקב אחר {available_proxies[sel]}" if sel else "מחיר ידני"
                            flash(kind, f"{t}: {how} ממחיר {money(unit, '₪' if is_ils(t) else '$')}. {msg}")
                            st.rerun()

        section("עריכה ישירה של התיק", "שנה/י כמות או מחיר קנייה, הוסף/י או מחק/י שורות ושמור/י. מחיר קנייה של מניות ת״א בשקלים.")
        edited = st.data_editor(
            portfolio.reset_index(), num_rows="dynamic", width='stretch', hide_index=True,
            column_order=('Ticker', 'Quantity', 'PurchasePrice'),
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
                k1.metric("בטא משוקלל לתיק", f"{port_beta:.2f}", "תנודתי מהשוק" if port_beta > 1.1 else "דומה לשוק או פחות", delta_color='off', delta_arrow='off')
                k2.metric("הפוזיציה הגדולה", top['ticker'], f"{top['weight']:.1f}% מהתיק", delta_color='off', delta_arrow='off')
                k3, k4 = st.columns(2)
                worst_row = rd['ירידה מקסימלית %'].idxmin()
                k3.metric("ירידה מקסימלית בשנה", f"{rd.loc[worst_row, 'ירידה מקסימלית %']:.1f}%", worst_row, delta_color='off', delta_arrow='off')
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
                c1.metric("יחידות לקנייה", str(shares), f"מחיר {money(cur_p, sym)}", delta_color='off', delta_arrow='off')
                c2.metric("גודל הפוזיציה", money(pos_usd), f"{pos_usd / ps_account * 100:.1f}% מהתיק" if ps_account else None, delta_color='off', delta_arrow='off')
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

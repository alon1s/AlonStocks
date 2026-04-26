import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import requests
from io import StringIO
import concurrent.futures
import time
import random

# ==========================================
# 1. הגדרות וחיבור לענן
# ==========================================
st.set_page_config(page_title="AlonStocks: Professional Quant", layout="wide", page_icon="📈")

# עקיפת חסימות של Yahoo
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

# חיבור לבסיס הנתונים
conn = st.connection("gsheets", type=GSheetsConnection)

def load_cloud_portfolio():
    try:
        df = conn.read(worksheet="Portfolio", ttl=0)
        if df is None or df.empty or 'Ticker' not in df.columns:
            return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')
        return df.set_index('Ticker')
    except:
        return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')

def save_cloud_portfolio(df):
    try:
        conn.update(worksheet="Portfolio", data=df.reset_index())
        return True
    except: return False

def log_activity(ticker, action, qty, price, notes=""):
    try:
        new_log = pd.DataFrame([{
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": ticker, "Action": action, "Quantity": float(qty),
            "Price": float(price), "Notes": notes
        }])
        try:
            existing = conn.read(worksheet="Activity", ttl=0)
        except: existing = pd.DataFrame()
        updated = pd.concat([existing, new_log], ignore_index=True) if not existing.empty else new_log
        conn.update(worksheet="Activity", data=updated)
    except: pass

# ניהול מזומן
CONFIG_FILE = "config.json"
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_cloud_portfolio()
if 'config' not in st.session_state:
    st.session_state.config = load_config()

# רשימת מניות למעקב
TA35 = ['ESLT.TA', 'NICE.TA', 'LUMI.TA', 'POLI.TA', 'ICL.TA', 'DSCT.TA']
WATCHLIST = list(set(['META', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'MU', 'SPY', 'QQQ', 'GOOGL', 'AMZN'] + list(st.session_state.portfolio.index) + TA35))
WATCHLIST.sort()

# ==========================================
# 2. מנוע נתונים (Deep Quant Engine)
# ==========================================
@st.cache_data(ttl=600)
def get_usd_ils():
    try: return yf.Ticker("USDILS=X", session=session).fast_info['lastPrice']
    except: return 3.75

@st.cache_data(ttl=3600)
def get_global_tickers():
    tickers = set()
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        resp = requests.get(url, headers=session.headers, timeout=10)
        table = pd.read_html(StringIO(resp.text))[0]
        tickers.update([t.replace('.', '-') for t in table['Symbol'].tolist()])
    except: pass
    tickers.update(TA35)
    return list(tickers)

@st.cache_data(ttl=1200)
def fetch_deep_data(tickers_to_fetch):
    data = {}
    def fetch_single(t):
        try:
            stock = yf.Ticker(t, session=session)
            hist = stock.history(period="1y") # היסטוריה ארוכה יותר לממוצעים
            if len(hist) < 20: return None
            info = stock.info
            curr = hist['Close'].iloc[-1]
            sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
            sma50 = hist['Close'].rolling(50).mean().iloc[-1]
            h52 = info.get('fiftyTwoWeekHigh', hist['Close'].max())
            
            # RSI חישוב
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            return t, {
                'price': curr, 'change': ((curr - hist['Close'].iloc[-2])/hist['Close'].iloc[-2])*100,
                'sector': info.get('sector', 'Unknown'), 'pe': info.get('trailingPE', 0),
                'beta': info.get('beta', 1.0), 'div': info.get('dividendYield', 0) or 0,
                'high_drop': ((curr - h52) / h52) * 100, 'sma50': sma50, 'sma200': sma200, 'rsi': rsi,
                'currency': "ILS" if str(t).endswith(".TA") else "USD"
            }
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_single, tickers_to_fetch)
        for r in results:
            if r: data[r[0]] = r[1]
    return data

usd_ils_rate = get_usd_ils()

# ==========================================
# 3. Sidebar: ביצוע פעולות
# ==========================================
st.sidebar.header("💰 ניהול מזומן")
c1, c2 = st.sidebar.columns(2)
n_usd = c1.number_input("Cash $", value=float(st.session_state.config['cash_usd']))
n_ils = c2.number_input("Cash ₪", value=float(st.session_state.config['cash_ils']))
if n_usd != st.session_state.config['cash_usd'] or n_ils != st.session_state.config['cash_ils']:
    st.session_state.config = {"cash_usd": n_usd, "cash_ils": n_ils}
    with open(CONFIG_FILE, 'w') as f: json.dump(st.session_state.config, f)

st.sidebar.markdown("---")
st.sidebar.subheader("🛒 ביצוע טרייד")
with st.sidebar.form("trade_form", clear_on_submit=True):
    t_in = st.text_input("Ticker Symbol").upper()
    act_in = st.selectbox("Action", ["Buy", "Sell"])
    q_in = st.number_input("Quantity", min_value=0.0, step=0.1)
    p_in = st.number_input("Price", min_value=0.0, step=0.01)
    note_in = st.text_input("Reasoning")
    if st.form_submit_button("Sync to Cloud 💾"):
        if t_in and q_in > 0:
            if act_in == "Buy":
                if t_in in st.session_state.portfolio.index:
                    oq, op = st.session_state.portfolio.loc[t_in]
                    nq = oq + q_in
                    st.session_state.portfolio.loc[t_in] = [nq, ((oq*op)+(q_in*p_in))/nq]
                else: st.session_state.portfolio.loc[t_in] = [q_in, p_in]
            else:
                if t_in in st.session_state.portfolio.index:
                    nq = max(0, st.session_state.portfolio.loc[t_in, 'Quantity'] - q_in)
                    if nq == 0: st.session_state.portfolio = st.session_state.portfolio.drop(t_in)
                    else: st.session_state.portfolio.loc[t_in, 'Quantity'] = nq
            if save_cloud_portfolio(st.session_state.portfolio):
                log_activity(t_in, act_in, q_in, p_in, note_in)
                st.rerun()

# ==========================================
# 4. Main Application Tabs
# ==========================================
t_port, t_scan, t_ai, t_advisor, t_journal = st.tabs(["💼 Portfolio", "🔎 Global Scanner", "🤖 AI Predictor", "🧠 Strategy", "📜 Journal"])

m_data = fetch_deep_data(WATCHLIST)

with t_port:
    # מצב עריכה
    with st.expander("⚙️ Edit Raw Data"):
        edit_df = st.data_editor(st.session_state.portfolio.reset_index(), num_rows="dynamic", use_container_width=True)
        if st.button("Save Changes"):
            st.session_state.portfolio = edit_df.dropna(subset=['Ticker']).set_index('Ticker')
            save_cloud_portfolio(st.session_state.portfolio)
            st.rerun()

    # חישובים פיננסיים
    cash_val = st.session_state.config['cash_usd'] + (st.session_state.config['cash_ils']/usd_ils_rate)
    stock_val_usd = 0; total_beta = 0; total_inv = 0; rows = []
    
    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]; qty = row['Quantity']; bp = row['PurchasePrice']
            v_u = (d['price']*qty / (usd_ils_rate if d['currency']=="ILS" else 1))
            stock_val_usd += v_u
            total_inv += (bp*qty / (usd_ils_rate if d['currency']=="ILS" else 1))
            total_beta += (d['beta']*v_u)
            rows.append({
                "Ticker": t, "Sector": d['sector'], "Qty": qty, "Last": d['price'],
                "P&L %": ((d['price']-bp)/bp)*100, "Beta": d['beta'], "Div %": d['div']*100,
                "RSI": d['rsi'], "Value USD": v_u, "Curr": "₪" if d['currency']=="ILS" else "$"
            })

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Assets ($)", f"${(stock_val_usd + cash_val):,.2f}")
    col_b.metric("P&L ($)", f"${(stock_val_usd - total_inv):,.2f}")
    col_c.metric("Portfolio Beta", f"{(total_beta/stock_val_usd if stock_val_usd>0 else 1.0):.2f}")

    v_mode = st.radio("Display Mode:", ["Desktop", "Mobile"], horizontal=True)
    if v_mode == "Desktop":
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Value USD", ascending=False).drop(columns="Value USD"), use_container_width=True)
    else:
        for r in sorted(rows, key=lambda x: x['Value USD'], reverse=True):
            with st.container(border=True):
                mc1, mc2 = st.columns([3,1])
                mc1.markdown(f"### {r['Ticker']}")
                mc1.caption(f"{r['Sector']} | RSI: {r['RSI']:.1f}")
                color = "green" if r['P&L %'] >= 0 else "red"
                mc2.markdown(f"<h2 style='text-align:right;color:{color};'>{r['P&L %']:.1f}%</h2>", unsafe_allow_html=True)

with t_scan:
    if st.button("🚀 Run Global Scan"):
        with st.spinner("Scanning S&P500 + NASDAQ + TA35..."):
            all_ticks = get_global_tickers()
            s_data = fetch_deep_data(all_ticks)
            df_s = pd.DataFrame(s_data).T
            c1, c2 = st.columns(2)
            with c1:
                st.write("💰 Value Plays (P/E < 15)")
                st.dataframe(df_s[(df_s['pe']>0)&(df_s['pe']<15)].sort_values('pe').head(20), use_container_width=True)
            with c2:
                st.write("🔥 Momentum (Price > 200SMA & RSI < 55)")
                st.dataframe(df_s[(df_s['price']>df_s['sma200'])&(df_s['rsi']<55)].sort_values('rsi').head(20), use_container_width=True)

with t_ai:
    at = st.text_input("Ticker for AI Analysis", value="NVDA").upper()
    if st.button("Forecast"):
        with st.spinner("Training Random Forest..."):
            stock = yf.Ticker(at, session=session)
            df = stock.history(period="5y")
            if len(df) > 200:
                df['SMA10'] = df['Close'].rolling(10).mean(); df['SMA50'] = df['Close'].rolling(50).mean()
                df['Vol'] = df['Close'].pct_change().rolling(20).std(); df['Mom'] = df['Close']/df['Close'].shift(10)-1
                train = df.dropna()
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(train[['SMA10', 'SMA50', 'Vol', 'Mom']], train['Close'].shift(-7).dropna())
                p = model.predict(df.tail(7)[['SMA10', 'SMA50', 'Vol', 'Mom']].bfill())
                fd = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.tail(60).index, y=df.tail(60)['Close'], name='Actual'))
                fig.add_trace(go.Scatter(x=fd, y=p, name='AI Forecast', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)

with t_advisor:
    for r in rows:
        with st.expander(f"Strategy: {r['Ticker']}"):
            if r['rsi'] > 70: st.warning("Overbought: RSI is very high. Consider taking profits.")
            if r['Last'] < r['Beta']: st.error("Price below long term trend.")
            st.write(f"Beta: {r['Beta']} | Dividend: {r['Div %']:.2f}%")

with t_journal:
    st.subheader("📜 Activity Log")
    try:
        logs = conn.read(worksheet="Activity", ttl=0)
        if logs is not None and not logs.empty:
            st.dataframe(logs.sort_values("Date", ascending=False), use_container_width=True)
            st.markdown("---")
            st.subheader("🧠 Performance Feedback")
            st.info("Insights: You are currently balanced. Focus on high RSI stocks for potential exits.")
    except: st.warning("Could not sync Journal. Check GSheets permissions.")
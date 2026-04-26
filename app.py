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
# 1. Configuration & Cloud Sync
# ==========================================
st.set_page_config(page_title="AlonStocks: Strategic Quant", layout="wide", page_icon="📈")

# Browser disguise
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

conn = st.connection("gsheets", type=GSheetsConnection)

def load_cloud_portfolio():
    try:
        df = conn.read(worksheet="Portfolio", ttl=0)
        if df is None or df.empty:
            return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')
        df = df.dropna(subset=['Ticker'])
        df = df[df['Ticker'].astype(str).str.strip() != ""]
        return df.set_index('Ticker')
    except:
        return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')

def save_cloud_portfolio(df):
    try:
        df_save = df.reset_index()
        conn.update(worksheet="Portfolio", data=df_save)
        st.cache_data.clear() 
        return True
    except Exception as e:
        st.error(f"Sync Error: {e}")
        return False

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

CONFIG_FILE = "config.json"
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_cloud_portfolio()
if 'config' not in st.session_state:
    st.session_state.config = load_config()

TA35 = ['ESLT.TA', 'NICE.TA', 'LUMI.TA', 'POLI.TA', 'ICL.TA', 'DSCT.TA']
p_tickers = [t for t in st.session_state.portfolio.index if isinstance(t, str) and t.strip()]
WATCHLIST = list(set(['META', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'MU', 'SPY', 'QQQ'] + p_tickers + TA35))
WATCHLIST.sort()

# ==========================================
# 2. Market Data Engine
# ==========================================
@st.cache_data(ttl=600)
def get_usd_ils():
    try: return yf.Ticker("USDILS=X").fast_info['lastPrice']
    except: return 3.75

@st.cache_data(ttl=1200)
def fetch_deep_data(tickers_to_fetch):
    data = {}
    valid_list = [t for t in tickers_to_fetch if isinstance(t, str) and t.strip()]
    def fetch_single(t):
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="1y")
            if len(hist) < 20: return None
            info = stock.info
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
            h52 = info.get('fiftyTwoWeekHigh', hist['Close'].max())
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]
            return t, {
                'price': curr, 'change': ((curr - prev)/prev)*100, 'sector': info.get('sector', 'Unknown'),
                'pe': info.get('trailingPE', 0), 'beta': info.get('beta', 1.0),
                'div': info.get('dividendYield', 0) or 0, 'h_drop': ((curr-h52)/h52)*100, 
                'sma200': sma200, 'rsi': rsi, 'currency': "ILS" if str(t).endswith(".TA") else "USD"
            }
        except: return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_single, valid_list)
        for r in results:
            if r: data[r[0]] = r[1]
    return data

usd_ils_rate = get_usd_ils()

# ==========================================
# 3. Sidebar UI
# ==========================================
st.sidebar.header("💰 Bank & Liquidity")
c1, c2 = st.sidebar.columns(2)
n_usd = c1.number_input("Cash $", value=float(st.session_state.config['cash_usd']))
n_ils = c2.number_input("Cash ₪", value=float(st.session_state.config['cash_ils']))

if n_usd != st.session_state.config['cash_usd'] or n_ils != st.session_state.config['cash_ils']:
    st.session_state.config = {"cash_usd": n_usd, "cash_ils": n_ils}
    with open(CONFIG_FILE, 'w') as f: json.dump(st.session_state.config, f)

st.sidebar.markdown("---")
st.sidebar.subheader("🛒 Fast Trade Entry")
with st.sidebar.form("trade_form", clear_on_submit=True):
    t_in = st.text_input("Ticker Symbol").upper()
    act_in = st.selectbox("Action", ["Buy", "Sell"])
    q_in = st.number_input("Qty", min_value=0.0, step=0.1)
    p_in = st.number_input("Price", min_value=0.0, step=0.01)
    note_in = st.text_input("Notes")
    if st.form_submit_button("Sync Trade 🚀"):
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
# 4. Main Tabs
# ==========================================
t_port, t_scan, t_ai, t_advisor, t_journal = st.tabs(["💼 My Portfolio", "🔎 Global Scanner", "🤖 AI Predictor", "🧠 Master Advisor", "📜 Journal"])

m_data = fetch_deep_data(WATCHLIST)

with t_port:
    # Logic for customized Profit calculation
    cash_usd = st.session_state.config['cash_usd']
    cash_ils = st.session_state.config['cash_ils']
    
    stock_val_usd = 0; rows = []
    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]; qty = row['Quantity']; bp = row['PurchasePrice']
            v_u = (d['price']*qty / (usd_ils_rate if d['currency']=="ILS" else 1))
            stock_val_usd += v_u
            rows.append({
                "Ticker": t, "Sector": d['sector'], "Qty": qty, "Price": d['price'],
                "P&L %": ((d['price']-bp)/bp)*100, "Value USD": v_u, "Beta": d['beta'], "Div %": d['div']*100,
                "RSI": d['rsi'], "Curr": "₪" if d['currency']=="ILS" else "$", "Dist.High": d['h_drop']
            })

    total_equity_usd = stock_val_usd + cash_usd + (cash_ils / usd_ils_rate)
    total_equity_ils = total_equity_usd * usd_ils_rate

    # Custom Baselines: 7,000 USD / 25,500 ILS
    profit_usd = total_equity_usd - 7000
    profit_ils = total_equity_ils - 25500
    profit_pct_usd = (profit_usd / 7000) * 100
    profit_pct_ils = (profit_ils / 25500) * 100

    st.markdown("### 📊 Performance Summary (Vs. Baselines)")
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Total Assets ($)", f"${total_equity_usd:,.2f}")
    c_m1.metric("Total Assets (₪)", f"₪{total_equity_ils:,.2f}")

    c_m2.metric("Profit ($) vs. 7k", f"${profit_usd:,.2f}", f"{profit_pct_usd:.2f}%")
    c_m3.metric("Profit (₪) vs. 25.5k", f"₪{profit_ils:,.2f}", f"{profit_pct_ils:.2f}%")

    st.divider()
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Value USD", ascending=False).drop(columns="Value USD"), use_container_width=True)

with t_advisor:
    st.subheader("🧠 Master Quant Advisor: Knowledge Base")
    st.info("להלן ניתוח מעמיק של המניות שלך עם הסברים על המדדים.")
    
    for r in rows:
        with st.expander(f"Strategy Review for {r['Ticker']}"):
            # Row 1: RSI & Beta
            col_a1, col_a2 = st.columns(2)
            
            rsi_val = r['RSI']
            col_a1.markdown(f"#### RSI: {rsi_val:.1f}")
            if rsi_val > 70: col_a1.error("🔥 Overbought (קניית יתר)")
            elif rsi_val < 30: col_a1.success("❄️ Oversold (מכירת יתר)")
            else: col_a1.info("⚖️ Neutral")
            col_a1.caption("💡 **החוק:** RSI בודק אם המניה עלתה מהר מדי. נמוך מ-30 זה 'טוב' לקנייה. גבוה מ-70 זה 'מסוכן' ומרמז על תיקון קרוב.")

            beta_val = r['Beta']
            col_a2.markdown(f"#### Beta: {beta_val:.2f}")
            if beta_val > 1.2: col_a2.warning("⚡ High Volatility")
            elif beta_val < 0.8: col_a2.success("🛡️ Defensive")
            else: col_a2.write("📏 Market-like")
            col_a2.caption("💡 **החוק:** Beta מודדת סיכון. גבוה מ-1.0 אומר שהמניה תנודתית יותר מהשוק (רווח גבוה/הפסד גבוה). נמוך מ-1.0 אומר שהיא יציבה יותר.")

            st.divider()
            # Row 2: Dividends & P/E
            col_b1, col_b2 = st.columns(2)
            
            div_val = r['Div %']
            col_b1.markdown(f"#### Div Yield: {div_val:.2f}%")
            if div_val > 4: col_b1.success("💰 High Income")
            elif div_val > 0: col_b1.info("💵 Paying")
            else: col_b1.write("🚫 No Dividend")
            col_b1.caption("💡 **החוק:** אחוז המזומן שאתה מקבל מהחברה בשנה. גבוה זה טוב למי שמחפש 'משכורת' מהמניות.")

            pe_val = m_data[r['Ticker']]['pe']
            col_b2.markdown(f"#### P/E Ratio: {pe_val:.1f}")
            if pe_val < 15 and pe_val > 0: col_b2.success("💎 Value Stock")
            elif pe_val > 35: col_b2.warning("🚀 Growth/Expensive")
            else: col_b2.write("📊 Average")
            col_b2.caption("💡 **החוק:** מכפיל רווח. נמוך אומר שהמניה 'זולה' ביחס לרווחים שלה. גבוה אומר שאתה משלם הרבה על פוטנציאל עתידי.")

            # Row 3: Distance from High
            dh_val = r['Dist.High']
            st.markdown(f"#### Distance from 52-Week High: **{dh_val:.2f}%**")
            if dh_val < -20: st.success(f"📉 'Buy the Dip' Opportunity: המניה ירדה משמעותית מהשיא שלה.")
            st.caption("💡 **החוק:** כמה המניה רחוקה מהשיא השנתי שלה. מספר שלילי גבוה (למשל 20%-) יכול להיות הזדמנות קנייה בזול.")

# (Rest of the tabs: Scanner, AI, Journal remain with existing robust logic)
with t_scan:
    if st.button("🚀 Run Global Scan"):
        with st.spinner("Analyzing..."):
            all_ticks = get_global_tickers(); s_data = fetch_deep_data(all_ticks); df_s = pd.DataFrame(s_data).T
            c_s1, c_s2 = st.columns(2)
            c_s1.write("💰 Value (P/E < 15)"); c_s1.dataframe(df_s[(df_s['pe']>0)&(df_s['pe']<15)].sort_values('pe').head(15), use_container_width=True)
            c_s2.write("🔥 Momentum (RSI < 55)"); c_s2.dataframe(df_s[(df_s['price']>df_s['sma200'])&(df_s['rsi']<55)].sort_values('rsi').head(15), use_container_width=True)

with t_ai:
    a_ticker = st.text_input("Ticker to Predict", value="NVDA").upper()
    if st.button("Forecast Next 7 Days 🔮"):
        stock = yf.Ticker(a_ticker); df = stock.history(period="5y")
        if len(df) > 250:
            df['SMA10'] = df['Close'].rolling(10).mean(); df['SMA50'] = df['Close'].rolling(50).mean()
            df['Vol'] = df['Close'].pct_change().rolling(20).std(); df['Mom'] = df['Close']/df['Close'].shift(10)-1
            train = df.dropna(); y = train['Close'].shift(-7).dropna(); X = train.loc[y.index, ['SMA10', 'SMA50', 'Vol', 'Mom']]
            model = RandomForestRegressor(n_estimators=100, random_state=42); model.fit(X, y)
            preds = model.predict(df.tail(7)[['SMA10', 'SMA50', 'Vol', 'Mom']].bfill())
            f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
            fig = go.Figure(); fig.add_trace(go.Scatter(x=df.tail(60).index, y=df.tail(60)['Close'], name='Actual')); fig.add_trace(go.Scatter(x=f_dates, y=preds, name='AI Forecast', line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

with t_journal:
    st.subheader("📜 Professional Activity Journal")
    try:
        activity = conn.read(worksheet="Activity", ttl=0)
        if activity is not None and not activity.empty:
            st.dataframe(activity.sort_values("Date", ascending=False), use_container_width=True)
    except: st.warning("Trading log is currently empty.")
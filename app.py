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
# Configuration & Cloud Connection
# ==========================================
st.set_page_config(page_title="AlonStocks Omni-Terminal", layout="wide", page_icon="🦅")

# Browser disguise for Yahoo Finance
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

# Database Connection
conn = st.connection("gsheets", type=GSheetsConnection)

def load_cloud_portfolio():
    try:
        df = conn.read(worksheet="Portfolio", ttl=0)
        if df.empty or 'Ticker' not in df.columns:
            # כאן תוקן הבאג: הוספתי את 'Ticker' לרשימת העמודות
            return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')
        return df.set_index('Ticker')
    except Exception as e:
        # אם יש שגיאת חיבור, נחזיר טבלה ריקה תקינה כדי למנוע קריסה
        return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')

def save_cloud_portfolio(df):
    try:
        df_save = df.reset_index()
        conn.update(worksheet="Portfolio", data=df_save)
        return True
    except Exception as e:
        st.sidebar.error("שגיאה בשמירה לענן. ודא שהגיליון מוגדר כ'פתוח לעריכה'.")
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
        except:
            existing = pd.DataFrame()
            
        updated = pd.concat([existing, new_log], ignore_index=True) if not existing.empty else new_log
        conn.update(worksheet="Activity", data=updated)
    except: pass

# Local Config for Cash
CONFIG_FILE = "config.json"
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_cloud_portfolio()
if 'config' not in st.session_state:
    st.session_state.config = load_config()

DEFAULT_TICKERS = ['META', 'ESLT', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'MU', 'SPY', 'NBIS']
TA35_TICKERS = ['ESLT.TA', 'NICE.TA', 'LUMI.TA', 'POLI.TA', 'ICL.TA', 'DSCT.TA', 'ENOG.TA', 'AZRG.TA']
TICKERS = list(set(DEFAULT_TICKERS + list(st.session_state.portfolio.index)))
TICKERS.sort()

# ==========================================
# Core Data Engine
# ==========================================
@st.cache_data(ttl=600)
def get_usd_ils():
    try: return yf.Ticker("USDILS=X", session=session).fast_info['lastPrice']
    except: return 3.75

@st.cache_data(ttl=3600)
def get_global_tickers():
    tickers = set()
    try:
        url_sp = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(StringIO(requests.get(url_sp, headers=session.headers).text))[0]
        tickers.update([t.replace('.', '-') for t in table['Symbol'].tolist()])
        url_ndx = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        table_ndx = pd.read_html(StringIO(requests.get(url_ndx, headers=session.headers).text))[4]
        tickers.update([t.replace('.', '-') for t in table_ndx['Ticker'].tolist()])
    except: pass
    tickers.update(TA35_TICKERS)
    return list(tickers)

@st.cache_data(ttl=1200)
def fetch_deep_data(tickers_to_fetch):
    data = {}
    def fetch_single(t):
        try:
            time.sleep(random.uniform(0.05, 0.15))
            stock = yf.Ticker(t, session=session)
            hist = stock.history(period="6mo")
            if len(hist) < 10: return None
            info = stock.info
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            sma50 = hist['Close'].rolling(50).mean().iloc[-1]
            sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else sma50
            h52 = info.get('fiftyTwoWeekHigh', hist['Close'].max())
            
            # RSI
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]

            return t, {
                'price': curr, 'change': ((curr-prev)/prev)*100, 'sector': info.get('sector', 'Unknown'),
                'pe': info.get('trailingPE', 0), 'growth': info.get('revenueGrowth', 0),
                'beta': info.get('beta', 1.0), 'dividend': info.get('dividendYield', 0) or 0,
                'high_drop': ((curr - h52) / h52) * 100, 'sma50': sma50, 'sma200': sma200, 'rsi': rsi,
                'currency': "ILS" if str(t).endswith(".TA") else "USD"
            }
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_single, tickers_to_fetch)
        for r in results:
            if r: data[r[0]] = r[1]
    return data

@st.cache_data(ttl=3600)
def train_and_predict(ticker):
    try:
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period="5y")
        if len(df) < 200: return None, None, "Insufficient data."
        df['SMA10'] = df['Close'].rolling(10).mean(); df['SMA50'] = df['Close'].rolling(50).mean()
        df['Vol'] = df['Close'].pct_change().rolling(20).std(); df['Mom'] = df['Close']/df['Close'].shift(10)-1
        df['Target'] = df['Close'].shift(-7)
        train = df.dropna()
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(train[['SMA10', 'SMA50', 'Vol', 'Mom']], train['Target'])
        last = df.tail(7)[['SMA10', 'SMA50', 'Vol', 'Mom']].bfill()
        preds = model.predict(last)
        f_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
        return df[['Close']], pd.DataFrame({'Predicted': preds}, index=f_dates), "AI Forecast ready."
    except: return None, None, "Error training AI."

usd_ils = get_usd_ils()

# ==========================================
# Sidebar UI
# ==========================================
st.sidebar.header("💰 Liquidity")
c1, c2 = st.sidebar.columns(2)
n_usd = c1.number_input("Cash $", value=float(st.session_state.config['cash_usd']))
n_ils = c2.number_input("Cash ₪", value=float(st.session_state.config['cash_ils']))
if n_usd != st.session_state.config['cash_usd'] or n_ils != st.session_state.config['cash_ils']:
    st.session_state.config = {"cash_usd": n_usd, "cash_ils": n_ils}
    with open(CONFIG_FILE, 'w') as f: json.dump(st.session_state.config, f)

st.sidebar.markdown("---")
st.sidebar.subheader("🛒 Execute Trade")
with st.sidebar.form("trade_f", clear_on_submit=True):
    t_in = st.text_input("Ticker").upper()
    act_in = st.selectbox("Action", ["Buy", "Sell"])
    q_in = st.number_input("Qty", min_value=0.0)
    p_in = st.number_input("Price", min_value=0.0)
    note_in = st.text_input("Notes")
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
# Main Tabs
# ==========================================
t1, t2, t3, t4, t5 = st.tabs(["💼 Portfolio", "🔎 Global Scanner", "🤖 AI Predictor", "🧠 Strategy", "📜 Journal"])

m_data = fetch_deep_data(TICKERS)

with t1:
    with st.expander("⚙️ Interactive Data Editor"):
        edit_df = st.data_editor(st.session_state.portfolio.reset_index(), num_rows="dynamic", use_container_width=True)
        if st.button("Save Editor Changes"):
            st.session_state.portfolio = edit_df.dropna(subset=['Ticker']).set_index('Ticker')
            save_cloud_portfolio(st.session_state.portfolio)
            st.rerun()

    cash_usd = st.session_state.config['cash_usd'] + (st.session_state.config['cash_ils']/usd_ils)
    rows = []; total_val = cash_usd; total_inv = 0; total_beta = 0
    
    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]; qty = row['Quantity']; bp = row['PurchasePrice']
            v = d['price']*qty; inv = bp*qty; is_ils = d['currency']=="ILS"
            v_u = v/usd_ils if is_ils else v; inv_u = inv/usd_ils if is_ils else inv
            total_val += v_u; total_inv += inv_u; total_beta += (d['beta']*v_u)
            rows.append({
                "Ticker": t, "Qty": qty, "Price": d['price'], "P&L %": ((d['price']-bp)/bp)*100,
                "Value USD": v_u, "Beta": d['beta'], "Div %": d['dividend']*100, "Sector": d['sector'], "Curr": "₪" if is_ils else "$"
            })

    c_a, c_b, c_c = st.columns(3)
    c_a.metric("Total Equity ($)", f"${total_val:,.2f}")
    c_b.metric("Unrealized P&L", f"${(total_val - total_inv - cash_usd):,.2f}")
    stock_v = total_val - cash_usd
    c_c.metric("Portfolio Beta", f"{(total_beta/stock_v if stock_v>0 else 1.0):.2f}")

    v_m = st.radio("View:", ["Desktop", "Mobile"], horizontal=True)
    if v_m == "Desktop":
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Value USD", ascending=False).drop(columns="Value USD"), width='stretch')
    else:
        for r in sorted(rows, key=lambda x: x['Value USD'], reverse=True):
            with st.container(border=True):
                col_m1, col_m2 = st.columns([3,1])
                col_m1.markdown(f"### {r['Ticker']} ({r['Sector']})")
                col_m1.write(f"Value: {r['Curr']}{r['Price']*r['Qty']:,.0f} | Beta: {r['Beta']}")
                col_m2.markdown(f"<h2 style='color:{'green' if r['P&L %']>=0 else 'red'};'>{r['P&L %']:.1f}%</h2>", unsafe_allow_html=True)

with t2:
    if st.button("🚀 Run Global Scan"):
        with st.spinner("Scanning S&P500 + NASDAQ + TA35..."):
            s_tickers = get_global_tickers()
            s_data = fetch_deep_data(s_tickers)
            df_s = pd.DataFrame(s_data).T
            c1, c2 = st.columns(2)
            c1.write("💰 Value Plays (P/E < 15)"); c1.dataframe(df_s[(df_s['pe']>0)&(df_s['pe']<15)].sort_values('pe').head(20), width='stretch')
            c2.write("🔥 Momentum (Price > 50SMA > 200SMA)"); c2.dataframe(df_s[(df_s['price']>df_s['sma50'])&(df_s['sma50']>df_s['sma200'])].sort_values('rsi', ascending=False).head(20), width='stretch')

with t3:
    p_t = st.text_input("Predict Ticker", value="NVDA").upper()
    if st.button("Run AI"):
        h, f, m = train_and_predict(p_t)
        if h is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h.tail(60).index, y=h.tail(60)['Close'], name='Actual'))
            fig.add_trace(go.Scatter(x=f.index, y=f['Predicted'], name='AI Forecast', line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

with t4:
    if st.button("Generate Strategy Blueprint"):
        for r in rows:
            with st.expander(f"Strategy for {r['Ticker']}"):
                if r['P&L %'] > 20: st.success("Take Profits? RSI is high.")
                elif r['P&L %'] < -10: st.error("Stop Loss? Trend is weak.")
                else: st.info("Hold. Position is within normal volatility.")

with t5:
    st.subheader("📜 Activity Journal")
    logs = conn.read(worksheet="Activity", ttl=0)
    if not logs.empty: st.dataframe(logs.sort_values("Date", ascending=False), width='stretch')
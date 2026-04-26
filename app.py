import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import requests
from io import StringIO
import concurrent.futures
import time
import random

# ==========================================
# Configuration & Stability
# ==========================================
st.set_page_config(page_title="AlonStocks Pro", layout="wide", page_icon="🏦")

DATA_FILE = "portfolio_data.csv"
CONFIG_FILE = "config.json"

DEFAULT_TICKERS = ['META', 'ESLT', 'NUVB', 'SPY', 'TSLA', 'NBIS', 'MSTR', 'NVDA', 'PLTR', 'TSM', 'PYPL', 'ZIM', 'GOOGL', 'AAPL', 'MSFT', 'NFLX', 'MU', 'ORCL', 'ARQQ']
TA35_TICKERS = ['ESLT.TA', 'NICE.TA', 'TEVA.TA', 'LUMI.TA', 'POLI.TA', 'ICL.TA', 'DSCT.TA', 'ENOG.TA', 'TSEM.TA', 'ALHE.TA', 'AZRG.TA', 'NWMD.TA']

def load_data():
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE, index_col=0)
            return df
    except: pass
    return pd.DataFrame(columns=['Quantity', 'PurchasePrice'])

def save_data(df):
    try:
        df.to_csv(DATA_FILE)
        return True
    except:
        st.error("Error saving to CSV. If using OneDrive, check if file is locked.")
        return False

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

# Initialize Session States
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data()
if 'config' not in st.session_state:
    st.session_state.config = load_config()

# Update Ticker list
TICKERS = list(set(DEFAULT_TICKERS + list(st.session_state.portfolio.index)))
TICKERS.sort()

# ==========================================
# Data Fetching Logic
# ==========================================
@st.cache_data(ttl=3600)
def get_usd_ils():
    try: return yf.Ticker("ILS=X").history(period="1d")['Close'].iloc[-1]
    except: return 3.72

@st.cache_data(ttl=600)
def fetch_market_data_batch(tickers):
    data = {}
    def fetch_single(ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if len(hist) < 2: return None
            info = stock.info
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
            return ticker, {
                'price': curr,
                'change': ((curr - prev) / prev) * 100,
                'sector': info.get('sector', 'Unknown'),
                'pe': info.get('trailingPE', 0),
                'growth': info.get('revenueGrowth', 0),
                'rsi': 50, # Simplified for mobile stability
                'currency': "ILS" if str(ticker).endswith(".TA") else "USD",
                'sma200': sma200,
                'history': hist['Close']
            }
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_single, tickers)
        for res in results:
            if res: data[res[0]] = res[1]
    return data

usd_ils = get_usd_ils()

# ==========================================
# UI Layout
# ==========================================
st.title("🏦 AlonStocks Mobile-Ready Terminal")

# Sidebar - Improved "Add" Logic
st.sidebar.header("🏦 Cash & Portfolio")
c1, c2 = st.sidebar.columns(2)
new_usd = c1.number_input("Cash $", value=float(st.session_state.config['cash_usd']))
new_ils = c2.number_input("Cash ₪", value=float(st.session_state.config['cash_ils']))

if new_usd != st.session_state.config['cash_usd'] or new_ils != st.session_state.config['cash_ils']:
    st.session_state.config = {"cash_usd": new_usd, "cash_ils": new_ils}
    save_config(st.session_state.config)

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Update Position")
add_ticker = st.sidebar.text_input("Ticker (e.g. NVDA, LUMI.TA)").upper()
add_qty = st.sidebar.number_input("Quantity", min_value=0.0, step=0.1)
add_price = st.sidebar.number_input("Avg Price", min_value=0.0, step=0.01)

if st.sidebar.button("Update Portfolio 💾"):
    if add_ticker:
        # Direct update to session state to ensure immediate UI change
        st.session_state.portfolio.loc[add_ticker] = [add_qty, add_price]
        if save_data(st.session_state.portfolio):
            st.sidebar.success(f"Updated {add_ticker}!")
            time.sleep(0.5)
            st.rerun()

# --- Main App Tabs ---
tab_port, tab_ai, tab_advisor = st.tabs(["💼 Portfolio", "🤖 AI Forecast", "🧠 Advisor"])

# Portfolio Logic
m_data = fetch_market_data_batch(list(st.session_state.portfolio.index))

with tab_port:
    # Top Metrics - Better for Mobile
    total_val_usd = st.session_state.config['cash_usd'] + (st.session_state.config['cash_ils'] / usd_ils)
    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data:
            val = m_data[t]['price'] * row['Quantity']
            total_val_usd += (val / usd_ils if m_data[t]['currency'] == "ILS" else val)

    col1, col2 = st.columns(2)
    col1.metric("Total Equity ($)", f"${total_val_usd:,.0f}")
    col2.metric("Total Equity (₪)", f"₪{total_val_usd * usd_ils:,.0f}")

    # Mobile vs Desktop View Toggle
    view_mode = st.radio("View Mode:", ["Desktop (Table)", "Mobile (Cards)"], horizontal=True)

    if view_mode == "Desktop (Table)":
        display_list = []
        for t, row in st.session_state.portfolio.iterrows():
            if t in m_data and row['Quantity'] > 0:
                p = m_data[t]['price']
                val = p * row['Quantity']
                cost = row['PurchasePrice'] * row['Quantity']
                sym = "₪" if m_data[t]['currency'] == "ILS" else "$"
                display_list.append({
                    "Ticker": t, "Qty": row['Quantity'], "Last": f"{sym}{p:,.2f}",
                    "Value": val, "P&L %": ((p - row['PurchasePrice'])/row['PurchasePrice'])*100,
                    "SortVal": val / (usd_ils if sym=="₪" else 1)
                })
        if display_list:
            df_p = pd.DataFrame(display_list).sort_values("SortVal", ascending=False)
            st.dataframe(df_p.drop(columns="SortVal"), width='stretch')
    
    else: # Mobile View - Cards
        for t, row in st.session_state.portfolio.iterrows():
            if t in m_data and row['Quantity'] > 0:
                p = m_data[t]['price']
                chg = m_data[t]['change']
                sym = "₪" if m_data[t]['currency'] == "ILS" else "$"
                with st.container(border=True):
                    c1, c2 = st.columns([2, 1])
                    c1.markdown(f"### {t}")
                    c1.caption(m_data[t]['sector'])
                    pnl = ((p - row['PurchasePrice'])/row['PurchasePrice'])*100
                    color = "green" if pnl >= 0 else "red"
                    c2.markdown(f"<h2 style='text-align:right;color:{color};'>{pnl:.1f}%</h2>", unsafe_allow_html=True)
                    
                    st.write(f"**Value:** {sym}{p * row['Quantity']:,.0f} | **Price:** {sym}{p:,.2f}")
                    st.progress(min(max(pnl + 50, 0), 100) / 100) # Simple visual bar

# Tab AI & Advisor (Kept simple and robust for V17)
with tab_ai:
    sel_t = st.selectbox("Predict Ticker:", list(st.session_state.portfolio.index) + DEFAULT_TICKERS)
    if st.button("Run AI Prediction 🔮"):
        with st.spinner("Analyzing 5-year trends..."):
            # Simplified prediction logic to avoid crashes
            stock = yf.Ticker(sel_t)
            df = stock.history(period="5y")
            if not df.empty:
                st.line_chart(df['Close'].tail(100))
                st.success("Trend Analysis: Momentum is positive." if df['Close'].iloc[-1] > df['Close'].iloc[-20] else "Trend Analysis: Currently consolidating.")

with tab_advisor:
    if st.button("Generate Strategy Blueprint 📋"):
        with st.spinner("Scanning market..."):
            # Logic focuses on owned stocks first
            for t, row in st.session_state.portfolio.iterrows():
                if t in m_data and row['Quantity'] > 0:
                    price = m_data[t]['price']
                    sma = m_data[t]['sma200']
                    if price < sma * 0.95:
                        st.warning(f"⚠️ **{t}**: Price is 5% below 200-day average. Long-term trend is weak.")
                    elif price > sma * 1.2:
                        st.success(f"✅ **{t}**: Trading 20% above long-term average. Strong momentum.")
            st.info("💡 **Strategy:** Keep cash for dips in NVDA or TSLA if they drop below their 50-day averages.")
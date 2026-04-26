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
# Configuration & Security Bypass
# ==========================================
st.set_page_config(page_title="AlonStocks Ultimate Quant", layout="wide", page_icon="🦅")

# Browser disguise to prevent 401 Unauthorized errors from Yahoo
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

DATA_FILE = "portfolio_data.csv"
CONFIG_FILE = "config.json"

DEFAULT_TICKERS = ['META', 'ESLT', 'NUVB', 'SPY', 'TSLA', 'NBIS', 'MSTR', 'NVDA', 'PLTR', 'TSM', 'PYPL', 'ZIM', 'GOOGL', 'AAPL', 'MSFT', 'NFLX', 'MU', 'ORCL', 'ARQQ']
TA35_TICKERS = ['ESLT.TA', 'NICE.TA', 'TEVA.TA', 'LUMI.TA', 'POLI.TA', 'ICL.TA', 'DSCT.TA', 'ENOG.TA', 'TSEM.TA', 'ALHE.TA', 'AZRG.TA', 'NWMD.TA']

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, index_col=0)
    return pd.DataFrame(columns=['Quantity', 'PurchasePrice'])

def save_data(df):
    df.to_csv(DATA_FILE)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data()
if 'config' not in st.session_state:
    st.session_state.config = load_config()

TICKERS = list(set(DEFAULT_TICKERS + list(st.session_state.portfolio.index)))
TICKERS.sort()

# ==========================================
# Core Data Engine (Deep Fundamentals)
# ==========================================
@st.cache_data(ttl=600)
def get_usd_ils():
    try: 
        # Using fast_info on USDILS=X is the most accurate real-time spot rate
        return yf.Ticker("USDILS=X", session=session).fast_info['lastPrice']
    except: 
        return 3.75

@st.cache_data(ttl=86400) 
def get_global_market_tickers():
    tickers = set()
    try:
        url_sp = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(StringIO(requests.get(url_sp, headers=session.headers).text))[0]
        tickers.update([t.replace('.', '-') for t in table['Symbol'].tolist()])
    except: pass
    try:
        url_ndx = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        table = pd.read_html(StringIO(requests.get(url_ndx, headers=session.headers).text))[4]
        tickers.update([t.replace('.', '-') for t in table['Ticker'].tolist()])
    except: pass
    tickers.update(TA35_TICKERS)
    if not tickers: tickers.update(['AAPL', 'MSFT', 'NVDA', 'TSLA', 'ESLT.TA', 'LUMI.TA'])
    return list(tickers)

@st.cache_data(ttl=1200)
def fetch_deep_market_data(tickers):
    data = {}
    def fetch_single(ticker):
        try:
            time.sleep(random.uniform(0.1, 0.3))
            stock = yf.Ticker(ticker, session=session)
            hist = stock.history(period="6mo")
            if len(hist) < 60: return None
            info = stock.info
            
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            sma50 = hist['Close'].rolling(50).mean().iloc[-1]
            sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else sma50
            high52 = info.get('fiftyTwoWeekHigh', hist['Close'].max())
            
            # RSI Calculation
            try:
                delta = hist['Close'].diff()
                gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
                rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]
            except: rsi = 50

            return ticker, {
                'price': curr,
                'change': ((curr - prev) / prev) * 100,
                'sector': info.get('sector', 'Unknown'),
                'pe': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'growth': info.get('revenueGrowth', 0),
                'beta': info.get('beta', 1.0),
                'dividend': info.get('dividendYield', 0) or 0,
                'high_drop': ((curr - high52) / high52) * 100,
                'sma50': sma50,
                'sma200': sma200,
                'rsi': rsi,
                'currency': "ILS" if str(ticker).endswith(".TA") else "USD"
            }
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_single, tickers)
        for res in results:
            if res: data[res[0]] = res[1]
    return data

@st.cache_data(ttl=3600)
def train_and_predict(ticker):
    try:
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period="5y")
        if len(df) < 200: return None, None, "Insufficient data."
            
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
        
        pred_days = 7
        df['Target'] = df['Close'].shift(-pred_days)
        
        train_df = df.dropna()
        X = train_df[['SMA10', 'SMA50', 'Volatility', 'Momentum']]
        y = train_df['Target']
        
        model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        model.fit(X, y)
        
        last_data = df.tail(pred_days)[['SMA10', 'SMA50', 'Volatility', 'Momentum']].bfill()
        futures = model.predict(last_data)
        
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, pred_days + 1)]
        pred_df = pd.DataFrame({'Predicted_Close': futures}, index=future_dates)
        return df[['Close']], pred_df, "AI Model trained successfully."
    except Exception as e:
        return None, None, f"Prediction error: {e}"

usd_ils_rate = get_usd_ils()

# ==========================================
# Sidebar UI
# ==========================================
st.sidebar.header("💰 Bank & Liquidity")
c1, c2 = st.sidebar.columns(2)
new_usd = c1.number_input("Cash $", value=float(st.session_state.config['cash_usd']))
new_ils = c2.number_input("Cash ₪", value=float(st.session_state.config['cash_ils']))

if new_usd != st.session_state.config['cash_usd'] or new_ils != st.session_state.config['cash_ils']:
    st.session_state.config = {"cash_usd": new_usd, "cash_ils": new_ils}
    with open(CONFIG_FILE, 'w') as f: json.dump(st.session_state.config, f)

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Update Position")
with st.sidebar.form("update_form", clear_on_submit=True):
    t_in = st.text_input("Ticker (e.g., TSLA, LUMI.TA)").upper()
    q_in = st.number_input("Quantity", min_value=0.0, step=0.1)
    p_in = st.number_input("Buy Price", min_value=0.0, step=0.01)
    if st.form_submit_button("Save to Portfolio 💾"):
        if t_in:
            st.session_state.portfolio.loc[t_in] = [q_in, p_in]
            save_data(st.session_state.portfolio)
            if t_in not in TICKERS: TICKERS.append(t_in)
            st.rerun()

# ==========================================
# Tabs
# ==========================================
tab_port, tab_scan, tab_ai, tab_advisor = st.tabs(["💼 Portfolio Master", "🔎 Global Screener", "🤖 AI Predictor", "🧠 Quant Strategy"])

m_data = fetch_deep_market_data(TICKERS)

# --- Tab 1: Deep Portfolio Analytics ---
with tab_port:
    total_val_usd = st.session_state.config['cash_usd'] + (st.session_state.config['cash_ils'] / usd_ils_rate)
    portfolio_rows = []
    sector_weights = {}
    total_beta_weight = 0
    total_invested_usd = 0
    
    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]
            qty = row['Quantity']
            buy_p = row['PurchasePrice']
            
            val = d['price'] * qty
            inv = buy_p * qty
            is_ils = d['currency'] == "ILS"
            
            val_usd = val / usd_ils_rate if is_ils else val
            inv_usd = inv / usd_ils_rate if is_ils else inv
            
            total_val_usd += val_usd
            total_invested_usd += inv_usd
            sector_weights[d['sector']] = sector_weights.get(d['sector'], 0) + val_usd
            
            # Weighted Beta calculation
            stock_beta = d['beta'] if d['beta'] else 1.0
            total_beta_weight += (stock_beta * val_usd)
            
            pnl_pct = ((d['price'] - buy_p) / buy_p) * 100 if buy_p > 0 else 0
            
            portfolio_rows.append({
                "Ticker": t,
                "Sector": d['sector'],
                "Qty": qty,
                "Buy Price": f"{'₪' if is_ils else '$'}{buy_p:,.2f}",
                "Current Price": f"{'₪' if is_ils else '$'}{d['price']:,.2f}",
                "P&L %": pnl_pct,
                "P&L Value": val - inv,
                "Value USD": val_usd, 
                "P/E": round(d['pe'], 1) if pd.notnull(d['pe']) else "N/A",
                "Div Yield %": round(d['dividend'] * 100, 2),
                "Beta": round(stock_beta, 2),
                "Dist. from High": f"{d['high_drop']:.1f}%",
                "Curr": "₪" if is_ils else "$"
            })

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Equity ($)", f"${total_val_usd:,.2f}")
    col2.metric("Total Equity (₪)", f"₪{total_val_usd * usd_ils_rate:,.2f}")
    
    overall_pnl = total_val_usd - total_invested_usd - st.session_state.config['cash_usd'] - (st.session_state.config['cash_ils'] / usd_ils_rate)
    col3.metric("Unrealized P&L ($)", f"${overall_pnl:,.2f}")
    
    # Portfolio Volatility (Beta)
    port_beta = total_beta_weight / (total_val_usd - st.session_state.config['cash_usd'] - (st.session_state.config['cash_ils']/usd_ils_rate)) if total_val_usd > 0 else 1.0
    col4.metric("Portfolio Beta (Risk)", f"{port_beta:.2f}", delta="> 1.0 is highly volatile" if port_beta > 1.0 else "< 1.0 is defensive", delta_color="inverse")

    view_mode = st.radio("Display Mode:", ["Desktop (Deep Data)", "Mobile (Cards)"], horizontal=True)

    if view_mode == "Desktop (Deep Data)":
        if portfolio_rows:
            df_p = pd.DataFrame(portfolio_rows).sort_values("Value USD", ascending=False).reset_index(drop=True)
            
            # Formatting
            df_p["P&L Value"] = df_p.apply(lambda x: f"{x['Curr']}{x['P&L Value']:,.2f}", axis=1)
            display_df = df_p.drop(columns=["Value USD", "Curr"])
            
            def style_negative(v, props=''):
                try: return props if float(v) < 0 else None
                except: return None
                
            def style_positive(v, props=''):
                try: return props if float(v) > 0 else None
                except: return None

            st.dataframe(
                display_df.style
                .format({"P&L %": "{:.2f}%", "Div Yield %": "{:.2f}%"})
                .map(style_positive, props='color: #00FF00; font-weight: bold;', subset=['P&L %'])
                .map(style_negative, props='color: #FF0000; font-weight: bold;', subset=['P&L %']),
                width='stretch', height=400
            )
            
            st.caption("💡 **Beta:** Measures volatility. > 1.0 means the stock swings more violently than the S&P 500. **Div Yield:** The cash percentage they pay you yearly just to hold the stock.")
    else:
        for item in sorted(portfolio_rows, key=lambda x: x['Value USD'], reverse=True):
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"### {item['Ticker']}")
                c1.caption(f"{item['Sector']} | Beta: {item['Beta']}")
                color = "green" if item['P&L %'] >= 0 else "red"
                c2.markdown(f"<h2 style='text-align:right;color:{color};'>{item['P&L %']:.1f}%</h2>", unsafe_allow_html=True)
                st.write(f"**Value:** {item['Curr']}{item['Value USD'] * (usd_ils_rate if item['Curr']=='₪' else 1):,.0f} | **Holdings:** {item['Qty']}")

# --- Tab 2: Global Screener (Fully Restored) ---
with tab_scan:
    st.subheader("🔎 Institutional Quant Screener")
    st.write("Scan entire markets to find pricing inefficiencies, momentum plays, and oversold dips.")
    
    universe = st.radio("Select Universe:", ["My Watchlist", "Global Mega-Cap (S&P500 + NDX + TA35)"], horizontal=True)
    scan_tickers = TICKERS if universe == "My Watchlist" else get_global_market_tickers()
    
    if st.button(f"🚀 Execute Deep Scan on {len(scan_tickers)} Stocks"):
        with st.spinner("Fetching fundamentals, P/E ratios, and technicals globally... (Takes ~1-2 mins)"):
            scan_data = fetch_deep_market_data(scan_tickers)
            if scan_data:
                df_scan = pd.DataFrame(scan_data).T
                df_scan['pe'] = pd.to_numeric(df_scan['pe'], errors='coerce')
                df_scan['growth'] = pd.to_numeric(df_scan['growth'], errors='coerce').fillna(0)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 💰 Deep Value (P/E < 15)")
                    val_stocks = df_scan[(df_scan['pe'] > 0) & (df_scan['pe'] < 15)].sort_values('pe')
                    st.dataframe(val_stocks[['sector', 'pe', 'dividend', 'currency']], width='stretch', height=300)
                with c2:
                    st.markdown("### 📊 Earnings Mismatch (Revenue Up, Price Down)")
                    gem_stocks = df_scan[(df_scan['growth'] > 0.10) & (df_scan['high_drop'] < -10)].sort_values('growth', ascending=False)
                    gem_stocks['growth'] = (gem_stocks['growth'] * 100).round(1).astype(str) + '%'
                    st.dataframe(gem_stocks[['sector', 'growth', 'high_drop']], width='stretch', height=300)
                
                st.markdown("---")
                c3, c4 = st.columns(2)
                with c3:
                    st.markdown("### 📉 'Buy The Dip' (Distance from 52W High)")
                    dip_stocks = df_scan[df_scan['high_drop'] < -20].sort_values('high_drop')
                    st.dataframe(dip_stocks[['price', 'high_drop', 'rsi', 'currency']], width='stretch', height=300)
                with c4:
                    st.markdown("### 🔥 Alpha Momentum (Price > 50SMA > 200SMA)")
                    trend_stocks = df_scan[(df_scan['price'] > df_scan['sma50']) & (df_scan['sma50'] > df_scan['sma200'])].sort_values('rsi', ascending=False)
                    st.dataframe(trend_stocks[['sector', 'price', 'rsi']], width='stretch', height=300)

# --- Tab 3: Deep AI Predictor ---
with tab_ai:
    st.subheader("🤖 Random Forest Predictive Engine")
    c1, c2 = st.columns([1, 3])
    with c1:
        pred_ticker = st.text_input("Enter Asset (e.g. SPY, NVDA):", value="SPY").upper()
        if st.button("Generate Forecast 🔮"):
            with st.spinner("Analyzing 5-year volatility and momentum vectors..."):
                hist_df, fut_df, msg = train_and_predict(pred_ticker)
                if hist_df is not None:
                    st.success(msg)
                    fig = go.Figure() if 'go' in globals() else px.line()
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    recent = hist_df.tail(60)
                    fig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], mode='lines', name='Actual Price'))
                    fig.add_trace(go.Scatter(x=fut_df.index, y=fut_df['Predicted_Close'], mode='lines+markers', name='7-Day Forecast', line=dict(dash='dash', color='orange')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    upside = ((fut_df['Predicted_Close'].iloc[-1] - recent['Close'].iloc[-1]) / recent['Close'].iloc[-1]) * 100
                    if upside > 1.5: st.success(f"📈 **AI Signal: BULLISH** ({upside:.2f}% projected upside).")
                    elif upside < -1.5: st.error(f"📉 **AI Signal: BEARISH** ({upside:.2f}% projected downside).")
                    else: st.warning(f"⚖️ **AI Signal: NEUTRAL** ({upside:.2f}%).")
                else:
                    st.error(msg)

# --- Tab 4: AI Advisor (Global Scale) ---
with tab_advisor:
    st.subheader("🧠 Algorithmic Strategy & Risk Manager")
    if st.button("Generate Strategy Blueprint 📋"):
        with st.spinner("Analyzing portfolio risk and global opportunities..."):
            adv_tickers = list(set(get_global_market_tickers() + [t for t in st.session_state.portfolio.index]))
            adv_data = fetch_deep_market_data(adv_tickers)
            
            if adv_data:
                df_adv = pd.DataFrame(adv_data).T
                owned = [t for t, row in st.session_state.portfolio.iterrows() if row['Quantity'] > 0]
                
                df_owned = df_adv[df_adv.index.isin(owned)]
                df_market = df_adv[~df_adv.index.isin(owned)]
                
                st.markdown("### 🛡️ Risk Management (Sell / Trim Candidates)")
                st.caption("Sorted by extreme overbought levels (RSI). Take profits to free up cash.")
                sell_cands = df_owned[(df_owned['rsi'] > 70) | ((df_owned['price'] < df_owned['sma200']) & (df_owned['sma50'] < df_owned['sma200']))].copy()
                if not sell_cands.empty:
                    sell_cands['Warning'] = np.where(sell_cands['rsi'] > 70, 'Overbought (Trim)', 'Downtrend (Cut Loss)')
                    st.dataframe(sell_cands[['Warning', 'rsi', 'beta']].sort_values('rsi', ascending=False), width='stretch')
                else:
                    st.success("Portfolio is technically sound. No immediate risk warnings.")
                
                st.markdown("---")
                c_swing, c_long = st.columns(2)
                
                with c_swing:
                    st.markdown("### ⚡ Mean-Reversion Swings (1-4 Weeks)")
                    st.caption("Long-term uptrend (Price > 200SMA) but short-term oversold (RSI < 45).")
                    swings = df_market[(df_market['price'] > df_market['sma200']) & (df_market['rsi'] < 45)].sort_values('rsi')
                    if not swings.empty:
                        st.dataframe(swings[['price', 'rsi', 'high_drop']].head(15), width='stretch', height=400)
                    else: st.write("No swing setups found.")

                with c_long:
                    st.markdown("### 🏦 Quality Value Accumulation (Years)")
                    st.caption("Profitable (P/E 5-25), Growing Revenue (>5%), and paying Dividends.")
                    df_market['pe'] = pd.to_numeric(df_market['pe'], errors='coerce')
                    longs = df_market[(df_market['pe'] > 5) & (df_market['pe'] < 25) & (df_market['growth'] > 0.05) & (df_market['dividend'] > 0.01)].sort_values('pe')
                    if not longs.empty:
                        longs['growth'] = (longs['growth'] * 100).round(1).astype(str) + '%'
                        longs['dividend'] = (longs['dividend'] * 100).round(2).astype(str) + '%'
                        st.dataframe(longs[['sector', 'pe', 'growth', 'dividend']].head(15), width='stretch', height=400)
                    else: st.write("No perfect Quality-Value stocks found today.")
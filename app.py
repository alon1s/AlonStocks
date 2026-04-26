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
# Configuration & File Management
# ==========================================
st.set_page_config(page_title="AlonStocks Pro Quant", layout="wide", page_icon="🏦")

DATA_FILE = "portfolio_data.csv"
CONFIG_FILE = "config.json"

DEFAULT_TICKERS = ['META', 'ESLT', 'NUVB', 'SPY', 'TSLA', 'NBIS', 'MSTR', 'NVDA', 'PLTR', 'TSM', 'PYPL', 'ZIM', 'GOOGL', 'AAPL', 'MSFT', 'NFLX', 'MU', 'ORCL', 'ARQQ']
TA35_TICKERS = ['ESLT.TA', 'NICE.TA', 'TEVA.TA', 'LUMI.TA', 'POLI.TA', 'ICL.TA', 'DSCT.TA', 'ENOG.TA', 'TSEM.TA', 'ALHE.TA', 'AZRG.TA', 'NWMD.TA']

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, index_col=0)
    return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')

def save_data(df):
    df.to_csv(DATA_FILE)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data()
if 'config' not in st.session_state:
    st.session_state.config = load_config()

TICKERS = list(set(DEFAULT_TICKERS + list(st.session_state.portfolio.index)))
TICKERS.sort()

# ==========================================
# Advanced Data Fetching
# ==========================================
@st.cache_data(ttl=86400) 
def get_global_market_tickers():
    tickers = set()
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url_sp = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        resp = requests.get(url_sp, headers=headers)
        table = pd.read_html(StringIO(resp.text))[0]
        tickers.update([t.replace('.', '-') for t in table['Symbol'].tolist()])
    except: pass
    try:
        url_ndx = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        resp = requests.get(url_ndx, headers=headers)
        table = pd.read_html(StringIO(resp.text))[4]
        tickers.update([t.replace('.', '-') for t in table['Ticker'].tolist()])
    except: pass
    tickers.update(TA35_TICKERS)
    if not tickers: tickers.update(['AAPL', 'MSFT', 'NVDA', 'TSLA', 'ESLT.TA', 'LUMI.TA'])
    return list(tickers)

@st.cache_data(ttl=3600) 
def get_advanced_market_data(tickers):
    data = {}
    def fetch_single(ticker):
        try:
            time.sleep(random.uniform(0.1, 0.3))
            stock = yf.Ticker(ticker)
            hist = stock.history(period="6mo")
            if len(hist) < 60: return None
            info = stock.info
            
            current_price = hist['Close'].iloc[-1]
            day_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
            high_3m = hist['Close'].tail(60).max()
            sma50 = hist['Close'].rolling(50).mean().iloc[-1]
            sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else sma50
            
            try:
                delta = hist['Close'].diff()
                gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
            except:
                rsi = 50

            currency = "ILS" if str(ticker).endswith(".TA") else "USD"

            return ticker, {
                'current_price': current_price,
                'day_change': day_change,
                'sector': info.get('sector', 'Unknown'),
                'pe_ratio': info.get('trailingPE', 0) if info.get('trailingPE') is not None else 0,
                'revenue_growth': info.get('revenueGrowth', 0) if info.get('revenueGrowth') is not None else 0,
                'high_3m': high_3m,
                'drop_from_high': ((current_price - high_3m) / high_3m) * 100,
                'sma50': sma50,
                'sma200': sma200,
                'rsi': rsi,
                'currency': currency,
                'history': hist['Close']
            }
        except:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_single, tickers)
        for res in results:
            if res is not None:
                data[res[0]] = res[1]
    return data

@st.cache_data(ttl=3600)
def get_usd_ils():
    try: return yf.Ticker("ILS=X").history(period="1d")['Close'].iloc[-1]
    except: return 3.72

@st.cache_data(ttl=3600)
def train_and_predict(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")
        if df.empty: return None, None, "No data available."
        if len(df) < 200: return None, None, "Insufficient data."
            
        df['SMA10'] = df['Close'].rolling(10).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(20).std()
        df['Momentum'] = df['Close'] / df['Close'].shift(10) - 1
        
        prediction_days = 7
        df['Target'] = df['Close'].shift(-prediction_days)
        
        train_df = df.dropna()
        X = train_df[['SMA10', 'SMA50', 'Volatility', 'Momentum']]
        y = train_df['Target']
        
        model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        model.fit(X, y)
        
        last_data = df.tail(prediction_days)[['SMA10', 'SMA50', 'Volatility', 'Momentum']].bfill()
        future_predictions = model.predict(last_data)
        
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
        pred_df = pd.DataFrame({'Predicted_Close': future_predictions}, index=future_dates)
        
        return df[['Close']], pred_df, "Model calculated successfully."
    except Exception as e:
        return None, None, f"Prediction error: {e}"

usd_ils_rate = get_usd_ils()

st.title("🏦 AlonStocks: Institutional Quant Terminal")

# ==========================================
# Sidebar
# ==========================================
st.sidebar.header("💰 Bank & Portfolio")
new_cash_usd = st.sidebar.number_input("Cash Balance (USD)", value=float(st.session_state.config['cash_usd']))
new_cash_ils = st.sidebar.number_input("Cash Balance (ILS)", value=float(st.session_state.config['cash_ils']))

if new_cash_usd != st.session_state.config['cash_usd'] or new_cash_ils != st.session_state.config['cash_ils']:
    st.session_state.config['cash_usd'] = new_cash_usd
    st.session_state.config['cash_ils'] = new_cash_ils
    save_config(st.session_state.config)
    st.rerun()

total_cash_usd = st.session_state.config['cash_usd'] + (st.session_state.config['cash_ils'] / usd_ils_rate)

with st.sidebar.expander("ℹ️ What does this mean? (Cash Management)"):
    st.write("""
    **Free Cash (Buying Power):** This is the money sitting in your account waiting to be used. 
    It's important to always keep some free cash. When the market suddenly crashes (a 'Dip'), 
    you want to have cash ready to buy good stocks at a discount. If you are 100% invested, you miss those opportunities.
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Add Position")
st.sidebar.caption("For Israeli stocks, add .TA (e.g., LUMI.TA)")
with st.sidebar.form("add_stock_form"):
    ticker_to_add = st.text_input("Enter Ticker:").upper()
    qty_to_add = st.number_input("Quantity:", min_value=0.0, step=0.1)
    price_to_add = st.number_input("Average Cost:", min_value=0.0, step=0.01)
    
    if st.form_submit_button("Update Portfolio 💾"):
        if ticker_to_add:
            st.session_state.portfolio.loc[ticker_to_add] = [qty_to_add, price_to_add]
            save_data(st.session_state.portfolio)
            if ticker_to_add not in TICKERS: TICKERS.append(ticker_to_add)
            st.rerun()

# ==========================================
# Tabs
# ==========================================
tab_port, tab_scanner, tab_ai, tab_advisor = st.tabs(["💼 My Portfolio", "🔎 Global Market Scanner", "🤖 5-Year AI Predictor", "🧠 AI Advisor"])

# --- Tab 1: Enhanced Portfolio ---
with tab_port:
    with st.expander("ℹ️ What does this mean? (Portfolio Rules)"):
        st.write("""
        * **Market Value Sorting:** The table puts your biggest financial investments at the top. This reminds you where your biggest risk is. If a stock at the top drops 5%, it hurts much more than a stock at the bottom dropping 20%.
        * **Sector Exposure (The Pie Chart):** This shows 'Diversification'. If your pie chart is 90% 'Technology', a crisis in the tech sector will destroy your portfolio. A healthy portfolio is spread across Tech, Healthcare, Finance, etc.
        * **P&L (Profit & Loss):** How much money you actually made or lost since you bought it.
        """)
        
    portfolio_market_data = get_advanced_market_data(TICKERS)
    portfolio_display = []
    total_market_val_usd = 0
    sector_weights = {}

    for ticker, row in st.session_state.portfolio.iterrows():
        if ticker not in portfolio_market_data or row['Quantity'] <= 0: continue
        
        m_data = portfolio_market_data[ticker]
        curr_p = m_data['current_price']
        day_chg = m_data['day_change']
        qty = row['Quantity']
        cost_p = row['PurchasePrice']
        
        is_ils = m_data['currency'] == "ILS"
        val_in_original_curr = curr_p * qty
        cost_in_original_curr = cost_p * qty
        pnl_val = val_in_original_curr - cost_in_original_curr
        pnl_pct = (pnl_val / cost_in_original_curr * 100) if cost_in_original_curr > 0 else 0
        
        market_val_usd = val_in_original_curr / usd_ils_rate if is_ils else val_in_original_curr
        total_market_val_usd += market_val_usd
        
        sector = m_data['sector']
        sector_weights[sector] = sector_weights.get(sector, 0) + market_val_usd
        
        curr_symbol = "₪" if is_ils else "$"
        
        portfolio_display.append({
            "Ticker": ticker,
            "Sector": sector,
            "Qty": qty,
            "Avg Cost": f"{curr_symbol}{cost_p:,.2f}",
            "Last Price": f"{curr_symbol}{curr_p:,.2f}",
            "Day Chg %": day_chg,
            "Invested": cost_in_original_curr,
            "Market Value": val_in_original_curr,
            "Market Value USD": market_val_usd,
            "P&L": pnl_val,
            "P&L %": pnl_pct,
            "Currency": curr_symbol
        })

    total_assets = total_market_val_usd + total_cash_usd
    
    st.subheader("📊 Executive Portfolio Dashboard")
    col1, col2 = st.columns(2)
    col1.metric("Total Assets (USD)", f"${total_assets:,.2f}")
    col2.metric("Total Assets (ILS)", f"₪{total_assets * usd_ils_rate:,.2f}")
    
    st.markdown("---")
    
    if portfolio_display:
        df_show = pd.DataFrame(portfolio_display)
        df_show = df_show.sort_values(by="Market Value USD", ascending=False).reset_index(drop=True)
        df_show["Invested"] = df_show.apply(lambda x: f"{x['Currency']}{x['Invested']:,.2f}", axis=1)
        df_show["Market Value"] = df_show.apply(lambda x: f"{x['Currency']}{x['Market Value']:,.2f}", axis=1)
        df_show = df_show.drop(columns=["Market Value USD", "Currency"])
        
        def color_pnl(val):
            return 'color: #00FF00; font-weight: bold;' if val > 0 else 'color: #FF0000; font-weight: bold;' if val < 0 else ''
        
        st.dataframe(
            df_show.style
            .format({"Day Chg %": "{:.2f}%", "P&L": "{:,.2f}", "P&L %": "{:.2f}%"})
            .map(color_pnl, subset=['Day Chg %', 'P&L', 'P&L %']),
            width='stretch',
            height=400
        )
    else:
        st.info("Portfolio is empty.")
            
    if sector_weights:
        st.markdown("---")
        st.write("**Sector Exposure (USD)**")
        pie_df = pd.DataFrame(list(sector_weights.items()), columns=['Sector', 'Value'])
        fig = px.pie(pie_df, values='Value', names='Sector', hole=0.4, height=300)
        fig.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Global Market Scanner ---
with tab_scanner:
    with st.expander("ℹ️ What does this mean? (Market Scanner terms)"):
        st.write("""
        * **Value Plays (Undervalued):** Companies that are 'On Sale'. A P/E ratio under 15 means you are paying less for every dollar the company makes compared to the rest of the market.
        * **Earnings Mismatch:** Hidden Gems. These companies reported great sales growth (>10%), but their stock price dropped anyway. The market might have panicked or made a mistake here, creating an opportunity.
        * **Buy The Dip:** Stocks that recently crashed more than 15% from their peak. Often, panic selling creates a discount for brave buyers to buy good companies for cheap.
        * **Trending Momentum:** 'The trend is your friend'. These stocks are riding a strong wave upwards. The market loves them right now, and they keep breaking their own average prices.
        """)
        
    st.subheader("🔎 Deep Actionable Intelligence")
    universe = st.radio("Select Universe:", ["My Watchlist", "Global Mega-Cap (S&P500 + NDX + TA35)"], horizontal=True)
    
    scan_tickers = TICKERS if universe == "My Watchlist" else get_global_market_tickers()
    
    if st.button(f"🚀 Scan {len(scan_tickers)} Global Stocks"):
        with st.spinner(f"Fetching fundamental and technical data... (This may take ~1-2 mins to avoid API bans)"):
            scan_data = get_advanced_market_data(scan_tickers)
            if scan_data:
                df_scan = pd.DataFrame(scan_data).T
                df_scan['revenue_growth'] = pd.to_numeric(df_scan['revenue_growth'], errors='coerce').fillna(0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 💰 Value Plays (P/E < 15)")
                    value_stocks = df_scan[(df_scan['pe_ratio'] > 0) & (df_scan['pe_ratio'] < 15)].sort_values('pe_ratio')
                    st.dataframe(value_stocks[['sector', 'pe_ratio', 'current_price', 'currency']], width='stretch', height=300)
                with col2:
                    st.markdown("### 📊 Earnings Mismatch")
                    gem_stocks = df_scan[(df_scan['revenue_growth'] > 0.10) & (df_scan['drop_from_high'] < -10)]
                    st.dataframe(gem_stocks[['sector', 'revenue_growth', 'drop_from_high']], width='stretch', height=300)
                
                st.markdown("---")
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("### 📉 'Buy The Dip' (Down > 15%)")
                    dip_stocks = df_scan[df_scan['drop_from_high'] < -15].sort_values('drop_from_high')
                    st.dataframe(dip_stocks[['current_price', 'drop_from_high', 'currency']], width='stretch', height=300)
                with col4:
                    st.markdown("### 🔥 Trending Momentum")
                    trend_stocks = df_scan[(df_scan['current_price'] > df_scan['sma50']) & (df_scan['sma50'] > df_scan['sma200'])]
                    st.dataframe(trend_stocks[['sector', 'current_price', 'rsi']], width='stretch', height=300)

# --- Tab 3: Deep AI Price Predictor ---
with tab_ai:
    with st.expander("ℹ️ What does this mean? (AI Predictions)"):
        st.write("""
        * **The Forecast:** The AI looks at 5 years of history to guess where the price will be next week.
        * **BULLISH Signal:** The AI expects the price to GO UP. It identified patterns that historically led to a rise.
        * **BEARISH Signal:** The AI expects the price to GO DOWN. It sees weakness, like a loss of momentum or high volatility.
        * **Important:** No AI can predict the future perfectly (news, wars, or sudden earnings reports can change everything). Use this as a 'second opinion', not a guarantee.
        """)
        
    st.subheader("🤖 Deep AI Forecaster (5-Year Memory)")
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        pred_ticker = st.text_input("Enter Ticker (e.g., TSLA, LUMI.TA):", value="NVDA").upper()
        run_pred = st.button("Generate AI Forecast 🔮")
    with col_p2:
        if run_pred and pred_ticker:
            with st.spinner("Training ML Regressor..."):
                historical_df, future_df, msg = train_and_predict(pred_ticker)
            if historical_df is not None:
                st.success(msg)
                import plotly.graph_objects as go
                fig = go.Figure()
                recent_hist = historical_df.tail(60)
                fig.add_trace(go.Scatter(x=recent_hist.index, y=recent_hist['Close'], mode='lines', name='Actual Price'))
                fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted_Close'], mode='lines+markers', name='AI Forecast (Next 7 Days)', line=dict(dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                
                curr_p = recent_hist['Close'].iloc[-1]
                target_p = future_df['Predicted_Close'].iloc[-1]
                upside = ((target_p - curr_p) / curr_p) * 100
                
                if upside > 2:
                    st.success(f"📈 **BULLISH:** The AI projects the stock will rise by {upside:.2f}% in the next 7 days.")
                elif upside < -2:
                    st.error(f"📉 **BEARISH:** The AI projects the stock will drop by {upside:.2f}%.")
                else:
                    st.warning(f"⚖️ **NEUTRAL:** The AI expects the price to stay mostly the same ({upside:.2f}%).")
            else:
                st.error(msg)

# --- Tab 4: AI Advisor ---
with tab_advisor:
    with st.expander("ℹ️ What does this mean? (Investment Strategy)"):
        st.write("""
        * **Portfolio Defense (Trim):** If a stock you own is 'Overbought' (RSI > 75), it means everyone bought it too fast. It's like a runner sprinting - eventually they need to catch their breath (the price will drop). The advisor tells you to 'Trim' (sell some of your shares) to lock in profits before it falls.
        * **Swing Trades:** Catching a bounce. These are generally strong stocks that had a bad week. The strategy is to buy them low now, and sell them in 2-3 weeks when they recover to their normal price. It's a quick trade, not a marriage.
        * **Long-Term Accumulation:** 'Marriage material'. These companies make actual money, grow steadily, and aren't overpriced. You buy these, ignore the daily news, and hold them for years.
        """)
        
    st.subheader("🧠 Personal Robo-Advisor & Strategy")
    st.write("Live analysis of your holdings vs the broader GLOBAL market. Showing **ALL** valid possibilities, sorted by highest conviction.")
    
    if st.button("Generate Global Strategy Blueprint 📋"):
        with st.spinner("Analyzing portfolio health and scanning the GLOBAL market for opportunities..."):
            
            global_tickers = get_global_market_tickers()
            advisor_tickers = list(set(global_tickers + [t for t in st.session_state.portfolio.index]))
            
            advisor_data = get_advanced_market_data(advisor_tickers)
            
            if advisor_data:
                df_adv = pd.DataFrame(advisor_data).T
                owned_tickers = [t for t, row in st.session_state.portfolio.iterrows() if row['Quantity'] > 0]
                
                df_owned = df_adv[df_adv.index.isin(owned_tickers)]
                df_market = df_adv[~df_adv.index.isin(owned_tickers)]
                
                st.markdown("### 🛡️ Portfolio Defense (Sell / Trim Candidates)")
                st.caption("Sorted from highest risk (Most Overbought) to lowest.")
                sell_candidates = []
                for ticker, row in df_owned.iterrows():
                    reason = ""
                    if row['rsi'] > 75:
                        reason = "Too Hot! (Overbought). Consider selling some to lock in profits before it cools down."
                    elif row['current_price'] < row['sma200'] and row['sma50'] < row['sma200']:
                        reason = "Downtrend. The stock is weak. Consider cutting losses to free up cash."
                    
                    if reason:
                        sell_candidates.append({"Ticker": ticker, "Action": reason, "RSI Score": round(row['rsi'],1)})
                
                if sell_candidates:
                    df_sell = pd.DataFrame(sell_candidates)
                    # SORTING LOGIC: Highest RSI first (Most dangerous)
                    df_sell = df_sell.sort_values(by='RSI Score', ascending=False).reset_index(drop=True)
                    st.dataframe(df_sell, width='stretch')
                else:
                    st.success("Your portfolio looks safe right now. No immediate warnings.")
                
                st.markdown("---")
                col_swing, col_long = st.columns(2)
                
                with col_swing:
                    st.markdown("### ⚡ Swing Trade Ideas (Buy for 1-4 Weeks)")
                    st.caption("Sorted from best setup (Lowest RSI / Most Oversold) downwards.")
                    
                    swing_buys = df_market[(df_market['current_price'] > df_market['sma200']) & (df_market['rsi'] < 55)].copy()
                    if not swing_buys.empty:
                        swing_display = swing_buys[['current_price', 'rsi', 'drop_from_high']]
                        # SORTING LOGIC: Lowest RSI first (Best discount/bounce potential)
                        swing_display = swing_display.sort_values(by='rsi', ascending=True).reset_index()
                        swing_display.rename(columns={'index': 'Ticker'}, inplace=True)
                        st.dataframe(swing_display.set_index('Ticker'), width='stretch', height=400)
                    else:
                        st.write("No swing setups found right now.")

                with col_long:
                    st.markdown("### 🏦 Long-Term Accumulation (Buy for Years)")
                    st.caption("Sorted from cheapest valuation (Lowest P/E) downwards.")
                    
                    long_buys = df_market[(df_market['pe_ratio'] > 0) & (df_market['pe_ratio'] < 45) & (df_market['revenue_growth'] > 0.0) & (df_market['current_price'] > df_market['sma200'])].copy()
                    if not long_buys.empty:
                        long_display = long_buys[['sector', 'pe_ratio', 'revenue_growth']]
                        # SORTING LOGIC: Lowest P/E first (Best value for the money)
                        long_display = long_display.sort_values(by='pe_ratio', ascending=True).reset_index()
                        long_display.rename(columns={'index': 'Ticker'}, inplace=True)
                        long_display['revenue_growth'] = (long_display['revenue_growth'] * 100).round(1).astype(str) + '%'
                        st.dataframe(long_display.set_index('Ticker'), width='stretch', height=400)
                    else:
                        st.write("Market is a bit expensive. No value-growth stocks found.")
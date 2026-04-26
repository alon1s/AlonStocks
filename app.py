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

# ==========================================
# 1. הגדרות וחיבור לענן
# ==========================================
st.set_page_config(page_title="AlonStocks: Strategic Mentor", layout="wide", page_icon="📈")

# מניעת חסימות בויקיפדיה
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
        conn.update(worksheet="Portfolio", data=df.reset_index())
        st.cache_data.clear() 
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

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_cloud_portfolio()

CONFIG_FILE = "config.json"
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}

if 'config' not in st.session_state:
    st.session_state.config = load_config()

# ==========================================
# 2. מנוע נתונים
# ==========================================
@st.cache_data(ttl=600)
def get_usd_ils():
    try: return yf.Ticker("USDILS=X").fast_info['lastPrice']
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
    tickers.update(['ESLT.TA', 'NICE.TA', 'LUMI.TA', 'POLI.TA', 'ICL.TA', 'DSCT.TA'])
    return list(tickers)

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
                'price': curr, 'sector': info.get('sector', 'Unknown'),
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

p_tickers = [t for t in st.session_state.portfolio.index if isinstance(t, str) and t.strip()]
WATCHLIST = list(set(['META', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'MU', 'SPY', 'QQQ'] + p_tickers))
WATCHLIST.sort()

# ==========================================
# 3. תפריט צדדי וניהול הגדרות
# ==========================================
st.sidebar.header("⚙️ הגדרות תצוגה")
# כפתור גלובלי שמחליף את כל האפליקציה למצב טלפון מפושט
app_mode = st.sidebar.radio("מצב תצוגה:", ["💻 מחשב (מלא)", "📱 טלפון (פשוט וברור)"])
is_mobile = app_mode == "📱 טלפון (פשוט וברור)"

st.sidebar.divider()
st.sidebar.header("💰 קופת מזומנים")
n_usd = st.sidebar.number_input("מזומן בדולר $", value=float(st.session_state.config['cash_usd']))
n_ils = st.sidebar.number_input("מזומן בשקל ₪", value=float(st.session_state.config['cash_ils']))

if n_usd != st.session_state.config['cash_usd'] or n_ils != st.session_state.config['cash_ils']:
    st.session_state.config = {"cash_usd": n_usd, "cash_ils": n_ils}
    with open(CONFIG_FILE, 'w') as f: json.dump(st.session_state.config, f)

st.sidebar.markdown("---")
st.sidebar.subheader("🛒 ביצוע טרייד")
with st.sidebar.form("trade_form", clear_on_submit=True):
    t_in = st.text_input("סימול מניה").upper()
    act_in = st.selectbox("פעולה", ["Buy", "Sell"])
    q_in = st.number_input("כמות", min_value=0.0, step=0.1)
    p_in = st.number_input("מחיר", min_value=0.0, step=0.01)
    if st.form_submit_button("עדכן לענן 🚀"):
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
                log_activity(t_in, act_in, q_in, p_in)
                st.rerun()

# ==========================================
# 4. הטאבים המרכזיים
# ==========================================
t_port, t_scan, t_ai, t_advisor, t_journal = st.tabs(["💼 תיק השקעות", "🔎 סורק", "🤖 AI", "🧠 יועץ", "📜 יומן"])

m_data = fetch_deep_data(WATCHLIST)

with t_port:
    if not is_mobile:
        with st.expander("⚙️ עריכה ידנית של התיק"):
            edit_df = st.data_editor(st.session_state.portfolio.reset_index(), num_rows="dynamic", width='stretch')
            if st.button("שמור שינויים"):
                clean_df = edit_df.dropna(subset=['Ticker'])
                clean_df = clean_df[clean_df['Ticker'].astype(str).str.strip() != ""]
                st.session_state.portfolio = clean_df.set_index('Ticker')
                save_cloud_portfolio(st.session_state.portfolio)
                st.rerun()

    stock_val_usd = 0; rows = []
    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]; qty = row['Quantity']; bp = row['PurchasePrice']
            v_u = (d['price']*qty / (usd_ils_rate if d['currency']=="ILS" else 1))
            stock_val_usd += v_u
            rows.append({
                "Ticker": t, "Qty": qty, "Price": d['price'], "P&L %": ((d['price']-bp)/bp)*100, 
                "Value USD": v_u, "Beta": d['beta'], "Div %": d['div']*100, "RSI": d['rsi'], 
                "Curr": "₪" if d['currency']=="ILS" else "$", "Dist.High": d['h_drop']
            })

    total_equity_usd = stock_val_usd + n_usd + (n_ils / usd_ils_rate)
    total_equity_ils = total_equity_usd * usd_ils_rate

    profit_usd = total_equity_usd - 7000
    profit_ils = total_equity_ils - 25500
    p_pct_usd = (profit_usd / 7000) * 100
    p_pct_ils = (profit_ils / 25500) * 100

    if not is_mobile:
        st.markdown("### 📊 סיכום רווחים והפסדים")
        c_p1, c_p2 = st.columns(2)
        c_p1.metric("רווח/הפסד בדולר ($) מול 7,000$", f"${profit_usd:,.2f}", f"{p_pct_usd:.2f}%")
        c_p2.metric("רווח/הפסד בשקל (₪) מול 25,500₪", f"₪{profit_ils:,.2f}", f"{p_pct_ils:.2f}%")
        st.divider()
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Value USD", ascending=False).drop(columns="Value USD"), width='stretch')
    else:
        # תצוגה נקייה לטלפון
        st.markdown("### 📱 סיכום מהיר")
        st.info(f"רווח: ₪{profit_ils:,.0f} | דולר: ${profit_usd:,.0f}")
        for r in sorted(rows, key=lambda x: x['Value USD'], reverse=True):
            with st.container(border=True):
                mc1, mc2 = st.columns([2,1])
                mc1.markdown(f"#### {r['Ticker']}")
                val_display = r['Value USD']*usd_ils_rate if r['Curr']=='₪' else r['Value USD']
                mc1.caption(f"שווי: {r['Curr']}{val_display:,.0f}")
                color = "green" if r['P&L %'] >= 0 else "red"
                mc2.markdown(f"<h3 style='text-align:left;color:{color};'>{r['P&L %']:.1f}%</h3>", unsafe_allow_html=True)

with t_advisor:
    st.subheader("🧠 מורה נבוך להשקעות")
    for r in rows:
        with st.expander(f"{r['Ticker']}"):
            col_ad1, col_ad2 = st.columns(2)
            
            rsi_val = r['RSI']
            col_ad1.markdown(f"**RSI: {rsi_val:.1f}**")
            if rsi_val > 70: col_ad1.error("🔥 חם מדי (Overbought)")
            elif rsi_val < 30: col_ad1.success("✅ זול טכנית (Oversold)")
            else: col_ad1.info("⚖️ נייטרלי")
            if not is_mobile: col_ad1.caption("💡 נמוך מ-30 זה 'טוב' לקנייה. גבוה מ-70 מרמז על תיקון.")

            beta_val = r['Beta']
            col_ad2.markdown(f"**Beta: {beta_val:.2f}**")
            if beta_val > 1.2: col_ad2.warning("⚡ תנודתי (מסוכן)")
            elif beta_val < 0.8: col_ad2.success("🛡️ הגנתי (בטוח)")
            else: col_ad2.write("📏 רגיל")
            if not is_mobile: col_ad2.caption("💡 מעל 1.0 המניה תנודתית יותר מהשוק.")

            st.divider()
            pe = m_data[r['Ticker']]['pe']
            st.write(f"**P/E (מכפיל רווח):** {pe:.1f} {'(זול 💎)' if 0 < pe < 15 else '(צמיחה 🚀)' if pe > 30 else ''}")
            st.write(f"**דיבידנד:** {r['Div %']:.1f}%")

with t_scan:
    if st.button("🚀 הפעל סורק שוק חכם"):
        with st.spinner("סורק נתונים..."):
            all_ticks = get_global_tickers(); s_data = fetch_deep_data(all_ticks); df_s = pd.DataFrame(s_data).T
            
            if not is_mobile:
                c_s1, c_s2 = st.columns(2)
                c_s1.write("💰 מניות זולות (P/E < 15)"); c_s1.dataframe(df_s[(df_s['pe']>0)&(df_s['pe']<15)].sort_values('pe').head(15), width='stretch')
                c_s2.write("🔥 מומנטום חזק"); c_s2.dataframe(df_s[(df_s['price']>df_s['sma200'])&(df_s['rsi']<55)].sort_values('rsi').head(15), width='stretch')
            else:
                st.write("💰 חברות ערך זולות:")
                st.dataframe(df_s[(df_s['pe']>0)&(df_s['pe']<15)].sort_values('pe')[['price','pe']].head(5), width='stretch')
                st.write("🔥 מומנטום:")
                st.dataframe(df_s[(df_s['price']>df_s['sma200'])&(df_s['rsi']<55)].sort_values('rsi')[['price','rsi']].head(5), width='stretch')

with t_ai:
    a_ticker = st.text_input("סימול לחיזוי AI", value="NVDA").upper()
    if st.button("צור תחזית 7 ימים 🔮"):
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
    st.subheader("📜 יומן פעולות")
    try:
        activity = conn.read(worksheet="Activity", ttl=0)
        if not activity.empty: 
            if not is_mobile:
                st.dataframe(activity.sort_values("Date", ascending=False), width='stretch')
            else:
                for _, log in activity.sort_values("Date", ascending=False).head(10).iterrows():
                    color = "green" if log['Action'] == "Buy" else "red"
                    st.info(f"{log['Date'][:10]} | {log['Ticker']} | {log['Action']} {log['Quantity']} @ ${log['Price']:.2f}")
    except: st.info("היומן ריק.")
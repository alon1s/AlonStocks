import streamlit as st
from streamlit_gsheets import GSheetsConnection
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
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
st.set_page_config(page_title="AlonStocks: Expert Advisor", layout="wide", page_icon="🏦")

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

conn = st.connection("gsheets", type=GSheetsConnection)

def load_cloud_portfolio():
    try:
        df = conn.read(worksheet="Portfolio", ttl=0)
        if df is None or df.empty: return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')
        df = df.dropna(subset=['Ticker'])
        df = df[df['Ticker'].astype(str).str.strip() != ""]
        return df.set_index('Ticker')
    except: return pd.DataFrame(columns=['Ticker', 'Quantity', 'PurchasePrice']).set_index('Ticker')

def save_cloud_portfolio(df):
    try:
        conn.update(worksheet="Portfolio", data=df.reset_index())
        st.cache_data.clear() 
        return True
    except: return False

def log_activity(ticker, action, qty, price, notes=""):
    try:
        new_log = pd.DataFrame([{"Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Ticker": ticker, "Action": action, "Quantity": float(qty), "Price": float(price), "Notes": notes}])
        try: existing = conn.read(worksheet="Activity", ttl=0)
        except: existing = pd.DataFrame()
        updated = pd.concat([existing, new_log], ignore_index=True) if not existing.empty else new_log
        conn.update(worksheet="Activity", data=updated)
    except: pass

if 'portfolio' not in st.session_state: st.session_state.portfolio = load_cloud_portfolio()

CONFIG_FILE = "config.json"
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f: return json.load(f)
    return {"cash_usd": 86.67, "cash_ils": 0.0}
if 'config' not in st.session_state: st.session_state.config = load_config()

# ==========================================
# 2. מנוע נתונים ואנליזה מתקדמת
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
    return list(tickers)

@st.cache_data(ttl=1200)
def fetch_expert_data(tickers_to_fetch):
    data = {}
    valid_list = [t for t in tickers_to_fetch if isinstance(t, str) and t.strip()]
    
    def fetch_single(t):
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="1y")
            if len(hist) < 50: return None
            info = stock.info
            
            curr = hist['Close'].iloc[-1]
            high_1y = hist['Close'].max()
            low_1y = hist['Close'].min()
            
            # Fibonacci Levels
            diff = high_1y - low_1y
            fib_382 = high_1y - (diff * 0.382)
            fib_618 = high_1y - (diff * 0.618)
            
            # Trend Analysis (HH/HL vs LH/LL proxy using Moving Averages)
            sma20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma50 = hist['Close'].rolling(50).mean().iloc[-1]
            sma200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else hist['Close'].mean()
            
            if sma20 > sma50 > sma200 and curr > sma20: trend = "Strong Uptrend (HH/HL)"
            elif sma20 < sma50 < sma200 and curr < sma20: trend = "Strong Downtrend (LH/LL)"
            elif curr > sma50: trend = "Moderate Uptrend"
            else: trend = "Consolidation/Weak"
            
            # RSI
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(com=13, adjust=False).mean()
            rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]
            
            return t, {
                'price': curr, 'sector': info.get('sector', 'Unknown'),
                'pe': info.get('trailingPE', 0), 'beta': info.get('beta', 1.0),
                'div': info.get('dividendYield', 0) or 0, 'h_drop': ((curr-high_1y)/high_1y)*100, 
                'sma50': sma50, 'sma200': sma200, 'rsi': rsi,
                'fib_382': fib_382, 'fib_618': fib_618, 'trend': trend,
                'analyst': info.get('recommendationKey', 'none'), 'growth_yoy': info.get('revenueGrowth', 0) * 100,
                'currency': "ILS" if str(t).endswith(".TA") else "USD"
            }
        except: return None
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_single, valid_list)
        for r in results:
            if r: data[r[0]] = r[1]
    return data

usd_ils_rate = get_usd_ils()
p_tickers = [t for t in st.session_state.portfolio.index if isinstance(t, str) and t.strip()]

# ==========================================
# 3. Sidebar UI
# ==========================================
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
# 4. Main Tabs
# ==========================================
t_port, t_tech_port, t_timeframes, t_ideas, t_scan, t_journal = st.tabs([
    "💼 התיק שלי", "📈 ניתוח טכני (התיק)", "⏱️ טווחי זמן", "🎯 רעיונות מסחר", "🌎 סורק S&P 500", "📜 יומן פעולות"
])

# Fetch data only for portfolio to keep it fast, scan tab does its own thing
m_data = fetch_expert_data(p_tickers + ['SPY', 'QQQ'])

# --- TAB 1: PORTFOLIO & SECTOR ROTATION ---
with t_port:
    st.subheader("סקירה כללית והמלצות מבנה (Sector Rotation)")
    stock_val_usd = 0; rows = []; sector_weights = {}
    for t, row in st.session_state.portfolio.iterrows():
        if t in m_data and row['Quantity'] > 0:
            d = m_data[t]; qty = row['Quantity']; bp = row['PurchasePrice']
            v_u = (d['price']*qty / (usd_ils_rate if d['currency']=="ILS" else 1))
            stock_val_usd += v_u
            sec = d['sector']
            sector_weights[sec] = sector_weights.get(sec, 0) + v_u
            rows.append({
                "Ticker": t, "Sector": sec, "Qty": qty, "Price": d['price'],
                "P&L %": ((d['price']-bp)/bp)*100, "Value USD": v_u, "Trend": d['trend']
            })

    total_usd = stock_val_usd + n_usd + (n_ils / usd_ils_rate)
    st.metric("שווי נכסים כולל", f"${total_usd:,.2f}")
    
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Value USD", ascending=False).drop(columns="Value USD"), width='stretch')
        
        st.markdown("### 🔄 יועץ מבנה תיק (Sector Analysis)")
        c_sec1, c_sec2 = st.columns(2)
        with c_sec1:
            st.write("**חשיפה סקטוריאלית נוכחית:**")
            st.json({k: f"{(v/stock_val_usd)*100:.1f}%" for k,v in sector_weights.items()})
        with c_sec2:
            st.write("**המלצת היועץ:**")
            if sector_weights.get('Technology', 0) / stock_val_usd > 0.5:
                st.warning("⚠️ חשיפת יתר לטכנולוגיה. שקול להוסיף סקטורים דפנסיביים (Healthcare, Utilities) לאיזון במקרה של ירידות.")
            else:
                st.success("✅ התיק מפוזר בצורה סבירה בין הסקטורים.")

# --- TAB 2: TECHNICAL ANALYSIS (PORTFOLIO) ---
with t_tech_port:
    st.subheader("ניתוח טכני ופונדמנטלי עמוק - מניות התיק")
    for r in rows:
        t = r['Ticker']
        d = m_data[t]
        with st.expander(f"📊 {t} - {d['trend']}"):
            c1, c2, c3 = st.columns(3)
            # Trends & Fibonacci
            c1.markdown("**📈 ניתוח מגמה ופיבונאצ'י**")
            c1.write(f"מגמה נוכחית: **{d['trend']}**")
            c1.write(f"תמיכה קרובה (Fib 38.2%): **${d['fib_382']:.2f}**")
            c1.write(f"תמיכה עמוקה (Fib 61.8%): **${d['fib_618']:.2f}**")
            if d['price'] < d['fib_382'] and d['price'] > d['fib_618']:
                c1.info("המניה נמצאת באזור קנייה קלאסי של פיבונאצ'י.")
                
            # Fundamentals & Analyst
            c2.markdown("**🏢 פונדמנטלי ואנליסטים**")
            c2.write(f"קונצנזוס אנליסטים (וול סטריט): **{d['analyst'].upper()}**")
            c2.write(f"צמיחה בהכנסות YoY (שנה לשנה): **{d['growth_yoy']:.1f}%**")
            pe = d['pe']
            c2.write(f"מכפיל רווח P/E: **{pe:.1f}**")
            if pe > 40: c2.warning("תמחור יקר ביחס לרווח.")
            
            # Oscillators
            c3.markdown("**⏱️ מתנדים (Oscillators)**")
            c3.write(f"RSI: **{d['rsi']:.1f}**")
            if d['rsi'] > 70: c3.error("Overbought - סיכוי לירידה קרובה.")
            elif d['rsi'] < 30: c3.success("Oversold - הזדמנות אפשרית.")
            c3.write(f"מרחק משיא שנתי: **{d['h_drop']:.1f}%**")

# --- TAB 3: TIMEFRAMES ---
with t_timeframes:
    st.subheader("⏱️ תחזיות וניתוח לטווחי זמן שונים")
    st.info("תחזיות המבוססות על מומנטום היסטורי ותנודתיות (הערכות סטטיסטיות).")
    
    tf_tabs = st.tabs(["ימים (Swing)", "שבועות (Short-Term)", "חודשים (Mid-Term)", "שנים (Long-Term)"])
    
    for t in p_tickers:
        if t not in m_data: continue
        d = m_data[t]
        p = d['price']
        volatility = d['beta'] * 0.02 # Proxy for daily vol
        
        with tf_tabs[0]: # Days
            est = p * (1 + (volatility if d['rsi'] < 50 else -volatility))
            st.write(f"**{t} (ימים):** צפי למחיר ${est:.2f}. מגמה: {d['trend']}.")
        with tf_tabs[1]: # Weeks
            est = p * (1 + (volatility * 3))
            st.write(f"**{t} (שבועות):** יעד התנגדות/תמיכה קרוב: ${est:.2f}.")
        with tf_tabs[2]: # Months
            est = p * (1 + (d['growth_yoy']/100 * 0.25)) # Growth based
            st.write(f"**{t} (חודשים):** השפעת דוחות קרובים. יעד משוער ${est:.2f}.")
        with tf_tabs[3]: # Years
            est = p * (1 + (d['growth_yoy']/100))
            st.write(f"**{t} (שנים):** חזון פונדמנטלי (צמיחה של {d['growth_yoy']:.1f}% לשנה). יעד: ${est:.2f}")

# --- TAB 4: TRADE IDEAS ---
with t_ideas:
    st.subheader("🎯 רעיונות מסחר מהיועץ (Trade Ideas)")
    c_sw, c_lt = st.columns(2)
    
    with c_sw:
        st.markdown("### ⚡ עסקאות סווינג (ימים-שבועות)")
        st.write("מניות שעברו תיקון חד אבל שומרות על מגמת עלייה, או מניות ב-Oversold:")
        for t, d in m_data.items():
            if d['rsi'] < 40 and "Uptrend" in d['trend']:
                st.success(f"**{t}:** RSI נמוך ({d['rsi']:.1f}) במגמת עלייה. תמיכה ב-${d['fib_382']:.2f}")
                
    with c_lt:
        st.markdown("### 🏦 השקעות לטווח ארוך (חודשים-שנים)")
        st.write("חברות עם צמיחה חזקה, הכנסות יציבות, ומכפיל סביר:")
        for t, d in m_data.items():
            if d['growth_yoy'] > 15 and 0 < d['pe'] < 35 and "buy" in d['analyst']:
                st.info(f"**{t}:** צמיחה של {d['growth_yoy']:.1f}%, המלצת {d['analyst'].upper()}, ומכפיל רווח סביר ({d['pe']:.1f}).")

# --- TAB 5: S&P SCANNER ---
with t_scan:
    st.subheader("🌎 סורק שוק עולמי (S&P 500)")
    if st.button("סרוק הזדמנויות בשוק"):
        with st.spinner("שואב נתונים (עשוי לקחת מעט זמן)..."):
            s_ticks = get_global_tickers()
            # To prevent crashing, we scan a random sample of 50 stocks from S&P for speed
            sample_ticks = random.sample(s_ticks, min(50, len(s_ticks))) 
            s_data = fetch_expert_data(sample_ticks)
            df_s = pd.DataFrame(s_data).T
            
            c_s1, c_s2 = st.columns(2)
            c_s1.write("🔥 **מגמות עלייה חזקות (HH/HL)**")
            up_df = df_s[df_s['trend'].astype(str).str.contains("Strong Up", na=False)]
            c_s1.dataframe(up_df[['price', 'rsi', 'pe']], width='stretch')
            
            c_s2.write("📉 **נשחטו לאחרונה (RSI < 30)**")
            os_df = df_s[df_s['rsi'] < 30]
            c_s2.dataframe(os_df[['price', 'rsi', 'trend']], width='stretch')

# --- TAB 6: JOURNAL ---
with t_journal:
    st.subheader("📜 יומן פעולות")
    try:
        activity = conn.read(worksheet="Activity", ttl=0)
        if not activity.empty: st.dataframe(activity.sort_values("Date", ascending=False), width='stretch')
    except: st.info("היומן ריק.")
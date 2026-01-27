import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Pro Scanner", layout="wide")

# --- AI Analyse Functie ---
def get_full_analysis(ticker):
    try:
        data = yf.download(ticker, period="100d", interval="1d", progress=False)
        if data.empty or len(data) < 30: return None
        
        # 1. RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        current_rsi = rsi.iloc[-1]

        # 2. AI Trend & Target
        current_price = float(data['Close'].iloc[-1])
        y = data['Close'].values.reshape(-1, 1)
        X = np.array(range(len(y))).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        ai_pred = float(model.predict(np.array([[len(y)]]))[0][0])
        
        # 3. ATR (Volatility) voor Stoploss
        high_low = data['High'] - data['Low']
        atr = high_low.rolling(14).mean().iloc[-1]
        
        # Bereken realistische swing-doelen
        real_target = current_price + (atr * 3) 
        real_stop = current_price - (atr * 2)

        # 4. Score Logica
        score = 0
        reasons = []
        if ai_pred > current_price: 
            score += 1
            reasons.append("Trend: Up")
        if current_rsi < 45: 
            score += 2
            reasons.append("RSI: Buy Zone")
        elif current_rsi > 70: 
            score -= 2
            reasons.append("RSI: Overbought")

        status = "HOLD"
        if score >= 2: status = "STRONG BUY"
        elif score == 1: status = "BUY"
        elif score <= -1: status = "SELL"

        return {
            "ticker": ticker, "status": status, "price": current_price,
            "target": real_target, "stop": real_stop, "rsi": current_rsi,
            "score": score, "reasons": reasons, "data": data
        }
    except:
        return None

# --- SIDEBAR CONTROL ---
st.sidebar.title("🎮 Dashboard Control")
user_list = st.sidebar.text_area("Watchlist", "AAPL, TSLA, NVDA, AMD, MSFT, BTC-USD")
tickers = [t.strip().upper() for t in user_list.split(",")]
start_scan = st.sidebar.button("🚀 RUN FULL MARKET SCAN")

# --- MAIN PAGE ---
st.title("📊 AI Swing Trader & Scanner")

# TABS voor overzicht
tab_scan, tab_deep = st.tabs(["📡 Market Scanner", "🔍 Single Ticker Deep Dive"])

with tab_scan:
    if start_scan:
        st.subheader("Live Scanner Results")
        scan_results = []
        progress = st.progress(0)
        
        for i, t in enumerate(tickers):
            analysis = get_full_analysis(t)
            if analysis:
                scan_results.append({
                    "Ticker": t, "Signal": analysis['status'], "Price": f"${analysis['price']:.2f}",
                    "Target": f"${analysis['target']:.2f}", "Stoploss": f"${analysis['stop']:.2f}",
                    "RSI": round(analysis['rsi'], 1), "Score": analysis['score']
                })
            progress.progress((i + 1) / len(tickers))
        
        if scan_results:
            df = pd.DataFrame(scan_results)
            def color_sig(v):
                if v == 'STRONG BUY': return 'background-color: #0ecb81; color: white'
                if v == 'SELL': return 'color: #f6465d'
                return ''
            st.table(df.style.applymap(color_sig, subset=['Signal']))
    else:
        st.info("Gebruik de sidebar om de scanner te starten.")

with tab_deep:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        ticker_select = st.text_input("Enter Ticker", "TSLA").upper()
        btn_analyze = st.button("Analyze Now")
    
    if btn_analyze:
        res = get_full_analysis(ticker_select)
        if res:
            # Stats Display
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${res['price']:.2f}")
            m2.metric("AI Swing Target", f"${res['target']:.2f}", delta=f"{(res['target']-res['price']):.2f}")
            m3.metric("Smart Stoploss", f"${res['stop']:.2f}", delta=f"{(res['stop']-res['price']):.2f}", delta_color="inverse")
            
            # Score en Signal
            st.subheader(f"Signal: {res['status']} (Score: {res['score']})")
            st.write(f"**Reasons:** {', '.join(res['reasons'])}")
            
            # Chart
            st.line_chart(res['data']['Close'])



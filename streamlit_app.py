import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Setup
st.set_page_config(page_title="AI Stock Advisor", layout="wide")
st.title("📈 AI Market Advisor & Scanner")

# --- AI Logica ---
def get_analysis(ticker):
    try:
        data = yf.download(ticker, period="60d", interval="1d", progress=False)
        if data.empty: return None
        
        current_price = float(data['Close'].iloc[-1])
        
        # Eenvoudige AI Trend
        y = data['Close'].values.reshape(-1, 1)
        X = np.array(range(len(y))).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        prediction = float(model.predict(np.array([[len(y)]]))[0][0])
        
        # Volatiliteit (ATR-stijl) voor Target & Stop
        daily_range = (data['High'] - data['Low']).mean()
        target = current_price + (daily_range * 2.5)
        stop = current_price - (daily_range * 1.5)
        
        # Score
        score = 0
        if prediction > current_price: score += 1
        if current_price > data['Close'].mean(): score += 1
        
        status = "BUY" if score >= 1 else "HOLD"
        if prediction < current_price * 0.98: status = "SELL"
        
        return {
            "ticker": ticker, "price": current_price, "target": target, 
            "stop": stop, "score": score, "status": status, "data": data
        }
    except:
        return None

# --- Sidebar ---
st.sidebar.header("Scanner")
watchlist_input = st.sidebar.text_area("Watchlist (tickers gescheiden door komma)", "AAPL, TSLA, NVDA, AMD, BTC-USD")
run_scan = st.sidebar.button("Start Scanner")

# --- Main Layout ---
col_main, col_side = st.columns([2, 1])

with col_main:
    ticker_input = st.text_input("Voer Ticker in voor Analyse", "AAPL").upper()
    res = get_analysis(ticker_input)
    
    if res:
        st.subheader(f"Analyse voor {ticker_input}")
        
        # De vertrouwde statistieken
        c1, c2, c3 = st.columns(3)
        c1.metric("Huidige Prijs", f"${res['price']:.2f}")
        c2.metric("AI Target", f"${res['target']:.2f}")
        c3.metric("Stop Loss", f"${res['stop']:.2f}")
        
        st.write(f"**AI Score:** {res['score']} | **Advies:** {res['status']}")
        
        # De werkende grafiek
        st.line_chart(res['data']['Close'])

with col_side:
    if run_scan:
        st.subheader("Scanner Resultaten")
        tickers = [t.strip().upper() for t in watchlist_input.split(",")]
        scan_data = []
        for t in tickers:
            s = get_analysis(t)
            if s:
                scan_data.append({"Ticker": t, "Prijs": f"${s['price']:.2f}", "Advies": s['status']})
        
        st.table(pd.DataFrame(scan_data))
    else:
        st.write("Klik op 'Start Scanner' in de zijbalk om je watchlist te controleren.")




import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Market Hunter", layout="wide")

# --- AI Logica Functie ---
def analyze_swing_trade(ticker):
    try:
        data = yf.download(ticker, period="100d", interval="1d", progress=False)
        if data.empty or len(data) < 30: return None
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rsi = 100 - (100 / (1 + (gain / loss)))
        current_rsi = rsi.iloc[-1]

        # Trend & ATR
        current_price = float(data['Close'].iloc[-1])
        y = data['Close'].values.reshape(-1, 1)
        X = np.array(range(len(y))).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        ai_target = float(model.predict(np.array([[len(y)]]))[0][0])
        
        high_low = data['High'] - data['Low']
        atr = high_low.rolling(14).mean().iloc[-1]
        
        # Score
        score = 0
        if ai_target > current_price: score += 1
        if current_rsi < 40: score += 2
        elif current_rsi > 70: score -= 2

        status = "HOLD"
        if score >= 2: status = "STRONG BUY"
        elif score == 1: status = "BUY"
        elif score <= -1: status = "SELL"

        return {
            "Ticker": ticker, "Signal": status, "Price": f"${current_price:.2f}",
            "Target": f"${ai_target:.2f}", "RSI": round(current_rsi, 1),
            "Stop": f"${(current_price - (atr*2)):.2f}"
        }
    except:
        return None

# --- SIDEBAR ---
st.sidebar.header("⚙️ Control Panel")
# Standaard lijst als de gebruiker niks invult
default_list = "AAPL, TSLA, NVDA, AMD, MSFT, META, AMZN, GOOGL, NFLX, BTC-USD"
user_list = st.sidebar.text_area("Watchlist (comma separated)", default_list)
tickers = [t.strip().upper() for t in user_list.split(",")]

# DE START KNOP (Nu prominent in de sidebar)
run_scan = st.sidebar.button("🚀 START AI SCANNER", use_container_width=True)

# --- MAIN PAGE ---
st.title("🏹 AI Multi-Factor Swing Scanner")

if run_scan:
    st.subheader("📊 Market Scan Results")
    results = []
    
    # Voortgangsbalk
    progress_bar = st.progress(0)
    for i, t in enumerate(tickers):
        res = analyze_swing_trade(t)
        if res:
            results.append(res)
        progress_bar.progress((i + 1) / len(tickers))
    
    if results:
        df = pd.DataFrame(results)
        
        # Styling voor signalen
        def color_signal(val):
            if val == 'STRONG BUY': return 'background-color: #0ecb81; color: white'
            if val == 'BUY': return 'color: #0ecb81'
            if val == 'SELL': return 'color: #f6465d'
            return ''

        st.dataframe(df.style.applymap(color_signal, subset=['Signal']), use_container_width=True, height=500)
    else:
        st.error("No data could be retrieved. Check your tickers.")
else:
    st.info("👈 Enter your tickers in the sidebar and click 'START AI SCANNER' to begin.")

# Individuele check onderaan
st.divider()
st.subheader("🔍 Single Ticker Deep Dive")
single_t = st.text_input("Enter one ticker for chart", "TSLA").upper()
if st.button("Show Chart"):
    st.line_chart(yf.download(single_t, period="100d")['Close'])



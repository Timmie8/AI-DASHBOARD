import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Swing Trader", layout="wide")

def analyze_swing_trade(ticker):
    try:
        # Haal data op (100 dagen voor indicatoren)
        data = yf.download(ticker, period="100d", interval="1d", progress=False)
        if data.empty or len(data) < 30: return None
        
        # --- 1. RSI (Momentum) ---
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # --- 2. AI Trend (Linear Regression) ---
        current_price = float(data['Close'].iloc[-1])
        y = data['Close'].values.reshape(-1, 1)
        X = np.array(range(len(y))).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        ai_target = float(model.predict(np.array([[len(y)]]))[0][0])
        trend_move = ((ai_target - current_price) / current_price) * 100

        # --- 3. Volatility & Risk (ATR) ---
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        atr = np.max(ranges, axis=1).rolling(14).mean().iloc[-1]
        
        stop_loss = current_price - (atr * 2)
        target_price = current_price + (atr * 3) # Swing target: 3x de ATR
        risk_reward = (target_price - current_price) / (current_price - stop_loss)

        # --- AI SWING LOGIC ---
        score = 0
        reasons = []

        if trend_move > 0: 
            score += 1
            reasons.append("Positive AI Trend")
        if current_rsi < 40: 
            score += 2  # Sterke koop-factor (oversold)
            reasons.append("Oversold (RSI < 40)")
        elif current_rsi > 70:
            score -= 2  # Risicovol (overbought)
            reasons.append("Overbought (RSI > 70)")
        if data['Volume'].iloc[-1] > data['Volume'].rolling(20).mean().iloc[-1]:
            score += 1
            reasons.append("High Volume Confirmation")

        status = "HOLD"
        if score >= 3: status = "STRONG BUY"
        elif score >= 1: status = "BUY"
        elif score <= -2: status = "STRONG SELL"

        return {
            "price": current_price, "target": target_price, "stop": stop_loss,
            "rsi": current_rsi, "status": status, "rr": risk_reward, "reasons": reasons
        }
    except Exception as e:
        return None

# --- UI Layout ---
st.title("🏹 AI Multi-Factor Swing Scanner")

tab1, tab2 = st.tabs(["🎯 Deep Swing Analysis", "📋 Multi-Stock Scanner"])

with tab1:
    t_input = st.text_input("Analyze Ticker", "TSLA").upper().strip()
    res = analyze_swing_trade(t_input)
    
    if res:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${res['price']:.2f}")
        col2.metric("Swing Target", f"${res['target']:.2f}")
        col3.metric("RSI (Momentum)", f"{res['rsi']:.1f}")
        col4.metric("Risk/Reward Ratio", f"{res['rr']:.2f}")

        st.info(f"**AI Verdict:** {res['status']}")
        st.write("**Key Factors:** " + ", ".join(res['reasons']))
        
        st.line_chart(yf.download(t_input, period="100d")['Close'])

with tab2:
    watchlist = st.session_state.get('watchlist', ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "META"])
    if st.button("Run Market Swing Scan"):
        results = []
        for t in watchlist:
            m = analyze_swing_trade(t)
            if m:
                results.append({
                    "Ticker": t, "Signal": m['status'], "Price": f"${m['price']:.2f}",
                    "Target": f"${m['target']:.2f}", "RSI": round(m['rsi'], 1), "R/R": round(m['rr'], 2)
                })
        st.dataframe(pd.DataFrame(results), use_container_width=True)

st.caption("Swing Strategy: Entries based on RSI exhaustion and Trend confirmation. Exit at 3x ATR.")



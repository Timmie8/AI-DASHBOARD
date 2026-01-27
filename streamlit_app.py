import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Layout ---
st.set_page_config(page_title="AI Pro Stock Dashboard", layout="centered")
st.title("🏹 AI Visual Strategy Dashboard")

ticker_input = st.text_input("Enter Ticker Symbol", "AAPL").upper()

if ticker_input:
    try:
        # 1. Fetch Data
        data = yf.download(ticker_input, period="100d", interval="1d", progress=False)

        if data is None or data.empty or len(data) < 50:
            st.error("Not enough data found for this ticker.")
        else:
            data = data.copy().dropna()
            current_price = float(data['Close'].iloc[-1])
            
            # --- 2. STRATEGY CALCULATIONS ---
            # A. Basis Trend
            y = data['Close'].values.reshape(-1, 1)
            X = np.array(range(len(y))).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            pred = float(model.predict(np.array([[len(y)]]))[0][0])
            
            # B. Swingtrade (RSI)
            delta = data['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # C. Levels
            recent_high = float(data['High'].iloc[-21:-1].max())
            sma50 = float(data['Close'].iloc[-50:].mean())

            # --- 3. SIGNAL LOGIC ---
            signals = []
            if pred > current_price:
                signals.append(f"🟢 **Trend Analysis:** Upside predicted to ${pred:.2f}")
            if rsi < 45:
                signals.append(f"🟢 **Swingtrade:** RSI is Oversold ({rsi:.1f}). Rebound likely.")
            if current_price >= recent_high:
                signals.append(f"🟢 **Breakout:** Price cleared 20-day resistance of ${recent_high:.2f}")
            if current_price < (sma50 * 0.92):
                signals.append(f"🟢 **Reversal:** Price is 8%+ below SMA50. Mean reversion expected.")

            # --- 4. DISPLAY KEY METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("AI Basis Target", f"${pred:.2f}")
            col3.metric("Stop Loss (5%)", f"${current_price * 0.95:.2f}")

            # --- 5. DYNAMIC ALERT BARS ---
            if signals:
                for s in signals:
                    st.success(s)
            else:
                st.info("No active BUY signals for the current selection.")
            
            st.line_chart(data['Close'])

            # --- 6. STRATEGY SCOREBOARD ---
            st.divider()
            st.subheader("🚀 Strategy Scoreboard")
            
            methods_data = [
                {"Method": "Basis Trend", "Action": "BUY" if pred > current_price else "HOLD", "Target": f"${pred:.2f}"},
                {"Method": "Swingtrade", "Action": "BUY" if rsi < 45 else "HOLD", "Target": f"${(current_price * 1.08):.2f}"},
                {"Method": "Breakout", "Action": "BUY" if current_price >= recent_high else "HOLD", "Target": f"${(recent_high * 1.15):.2f}"},
                {"Method": "Reversal", "Action": "BUY" if current_price < (sma50 * 0.92) else "HOLD", "Target": f"${sma50:.2f}"}
            ]
            
            buy_count = sum(1 for m in methods_data if m["Action"] == "BUY")
            score_color = "green" if buy_count >= 2 else "orange"
            st.markdown(f"### Overall Strength: <span style='color:{score_color}'>{buy_count} / 4 BUY Signals</span>", unsafe_allow_html=True)

            df_results = pd.DataFrame(methods_data)
            def highlight_buy(s):
                return ['background-color: #d4edda; color: #155724; font-weight: bold' if v == 'BUY' else '' for v in s]

            st.table(df_results.style.apply(highlight_buy, subset=['Action']))

    except Exception as e:
        st.error(f"Analysis error: {e}")

st.caption("AI Disclaimer: Analysis based on technical indicators. Not financial advice.")










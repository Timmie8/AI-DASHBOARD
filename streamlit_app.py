import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Layout ---
st.set_page_config(page_title="AI Pro Dashboard", layout="centered")
st.title("🏹 AI Visual Strategy Dashboard")

ticker_input = st.text_input("Voer Ticker Symbool in", "AAPL").upper()

if ticker_input:
    try:
        # 1. Haal data op
        data = yf.download(ticker_input, period="100d", interval="1d", progress=False)

        if data is None or data.empty or len(data) < 50:
            st.error("Niet genoeg data gevonden.")
        else:
            data = data.copy().dropna()
            current_price = float(data['Close'].iloc[-1])
            
            # --- 2. ALLE BEREKENINGEN ---
            # Basis Trend
            y = data['Close'].values.reshape(-1, 1)
            X = np.array(range(len(y))).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            pred = float(model.predict(np.array([[len(y)]]))[0][0])
            
            # Swingtrade (RSI)
            delta = data['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # Omliggende levels
            recent_high = float(data['High'].iloc[-21:-1].max())
            sma50 = float(data['Close'].iloc[-50:].mean())

            # --- 3. DISPLAY HOOFDSTATS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Prijs", f"${current_price:.2f}")
            col2.metric("AI Target", f"${pred:.2f}")
            col3.metric("Stop Loss (5%)", f"${current_price * 0.95:.2f}")

            # Signaal boven de grafiek
            if pred > current_price:
                st.success(f"🟢 KOOP-SIGNAAL: De AI voorspelt een stijging naar ${pred:.2f}")
            
            st.line_chart(data['Close'])

            # --- 4. HET SCOREBOARD (Tabel met kleuren) ---
            st.divider()
            st.subheader("🚀 Strategie Scoreboard")
            
            methods_data = [
                {"Methode": "Basis Trend", "Advies": "BUY" if pred > current_price else "HOLD", "Target": f"${pred:.2f}"},
                {"Methode": "Swingtrade", "Advies": "BUY" if rsi < 45 else "HOLD", "Target": f"${(current_price * 1.08):.2f}"},
                {"Methode": "Breakout", "Advies": "BUY" if current_price >= recent_high else "HOLD", "Target": f"${(recent_high * 1.15):.2f}"},
                {"Methode": "Reversal", "Advies": "BUY" if current_price < (sma50 * 0.92) else "HOLD", "Target": f"${sma50:.2f}"}
            ]
            
            df_results = pd.DataFrame(methods_data)

            # Styling functie voor de tabel
            def highlight_buy(s):
                return ['background-color: #d4edda; color: #155724; font-weight: bold' if v == 'BUY' else '' for v in s]

            st.table(df_results.style.apply(highlight_buy, subset=['Advies']))

            # --- 5. VISUELE "BUY" BOXEN (Groen omlijnd) ---
            st.write("### Details per Methode")
            for m in methods_data:
                if m["Advies"] == "BUY":
                    st.info(f"✅ **{m['Methode']}** staat op **BUY**. Het koersdoel is **{m['Target']}**.")
                else:
                    st.write(f"⚪ {m['Methode']} staat momenteel op HOLD.")

    except Exception as e:
        st.error(f"Fout bij analyse: {e}")








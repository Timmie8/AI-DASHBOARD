import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Layout ---
st.set_page_config(page_title="AI Multi-Strategy Dashboard", layout="centered")
st.title("🛡️ AI Multi-Strategy Dashboard")

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
            
            # --- 2. ALLE STRATEGIE BEREKENINGEN (Achtergrond) ---
            results = []

            # A. BASIS TREND (Linear Regression)
            y = data['Close'].values.reshape(-1, 1)
            X = np.array(range(len(y))).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            pred = float(model.predict(np.array([[len(y)]]))[0][0])
            status_basis = "BUY" if pred > current_price else "HOLD"
            results.append({"Methode": "Basis Trend", "Advies": status_basis, "Target": f"${pred:.2f}"})

            # B. SWINGTRADE (RSI)
            delta = data['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rsi = float(100 - (100 / (1 + (ema_up / ema_down).iloc[-1])))
            status_swing = "BUY" if rsi < 45 else "HOLD"
            results.append({"Methode": "Swingtrade", "Advies": status_swing, "Target": f"${(current_price * 1.08):.2f}"})

            # C. BREAKOUT (20-day high)
            recent_high = float(data['High'].iloc[-21:-1].max())
            status_break = "BUY" if current_price >= recent_high else "HOLD"
            results.append({"Methode": "Breakout", "Advies": status_break, "Target": f"${(recent_high * 1.15):.2f}"})

            # D. REVERSAL (SMA50)
            sma50 = float(data['Close'].iloc[-50:].mean())
            status_rev = "BUY" if current_price < (sma50 * 0.92) else "HOLD"
            results.append({"Methode": "Reversal", "Advies": status_rev, "Target": f"${sma50:.2f}"})

            # --- 3. DISPLAY HOOFDSTATS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Huidige Prijs", f"${current_price:.2f}")
            col2.metric("Basis AI Target", f"${pred:.2f}")
            col3.metric("Stop Loss (5%)", f"${current_price * 0.95:.2f}")

            st.line_chart(data['Close'])

            # --- 4. MULTI-STRATEGY SCOREBOARD ---
            st.divider()
            st.subheader("🚀 Strategie Scoreboard")
            st.write("Bekijk hieronder het advies van alle AI-modellen voor dit aandeel:")
            
            # Maak een mooie tabel
            df_results = pd.DataFrame(results)
            
            # Kleur de cellen voor extra duidelijkheid
            def color_advice(val):
                color = '#0ecb81' if val == 'BUY' else '#808080'
                return f'color: {color}; font-weight: bold'

            st.table(df_results.style.applymap(color_advice, subset=['Advies']))

    except Exception as e:
        st.error(f"Fout bij analyse: {e}")

st.caption("Alle berekeningen worden in real-time uitgevoerd op basis van de geselecteerde ticker.")






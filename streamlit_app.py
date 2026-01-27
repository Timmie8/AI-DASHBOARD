import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Visual Trader", layout="centered")
st.title("🏹 AI Visual Strategy Dashboard")

ticker_input = st.text_input("Voer Ticker Symbool in", "AAPL").upper()

if ticker_input:
    try:
        data = yf.download(ticker_input, period="100d", interval="1d", progress=False)

        if data is None or data.empty or len(data) < 50:
            st.error("Niet genoeg data gevonden.")
        else:
            data = data.copy().dropna()
            current_price = float(data['Close'].iloc[-1])
            
            # --- 1. BEREKENINGEN ---
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
            rsi = float(100 - (100 / (1 + (ema_up / ema_down).iloc[-1])))
            
            # Breakout & Reversal
            recent_high = float(data['High'].iloc[-21:-1].max())
            sma50 = float(data['Close'].iloc[-50:].mean())

            # --- 2. SIGNALEER OP DE KAART ---
            # We maken een kolom 'Signal' voor de grafiek
            data['Buy_Signal'] = np.nan
            if pred > current_price: data.iloc[-1, data.columns.get_loc('Buy_Signal')] = current_price * 0.98
            
            # --- 3. DISPLAY STATS ---
            st.subheader(f"Analyse: {ticker_input}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Prijs", f"${current_price:.2f}")
            col2.metric("AI Target", f"${pred:.2f}")
            col3.metric("Stop Loss", f"${current_price * 0.95:.2f}")

            # --- 4. DE GRAFIEK MET PIJLTJE ---
            # We tonen de prijslijn en een stip (pijltje) bij een BUY
            st.line_chart(data['Close'])
            if pred > current_price:
                st.success(f"▲ AI Trend Koopsignaal gedetecteerd op ${current_price:.2f}")

            # --- 5. HET GROENE SCOREBOARD ---
            st.divider()
            st.subheader("🚀 Strategie Scoreboard")
            
            methods = [
                {"Methode": "Basis Trend", "Advies": "BUY" if pred > current_price else "HOLD", "Target": f"${pred:.2f}"},
                {"Methode": "Swingtrade", "Advies": "BUY" if rsi < 45 else "HOLD", "Target": f"${(current_price * 1.08):.2f}"},
                {"Methode": "Breakout", "Advies": "BUY" if current_price >= recent_high else "HOLD", "Target": f"${(recent_high * 1.15):.2f}"},
                {"Methode": "Reversal", "Advies": "BUY" if current_price < (sma50 * 0.92) else "HOLD", "Target": f"${sma50:.2f}"}
            ]

            # Weergave met groene omlijning/styling
            for m in methods:
                if m["Advies"] == "BUY":
                    st.markdown(f"""
                        <div style="border: 2px solid #0ecb81; border-radius: 10px; padding: 15px; margin-bottom: 10px; background-color: rgba(14, 203, 129, 0.1);">
                            <h4 style="margin:0; color: #0ecb81;">✅ {m['Methode']}: BUY</h4>
                            <p style="margin:0;">Target: <b>{m['Target']}</b></p>
                        </div>
                    """, unsafe_input=True, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="border: 1px solid #808080; border-radius: 10px; padding: 10px; margin-bottom: 10px; opacity: 0.6;">
                            <h4 style="margin:0; color: #808080;">⚪ {m['Methode']}: HOLD</h4>
                            <p style="margin:0;">Geen optimaal instappunt</p>
                        </div>
                    """, unsafe_input=True, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Fout bij analyse: {e}")







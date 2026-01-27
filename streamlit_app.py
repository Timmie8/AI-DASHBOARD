import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Dashboard Layout ---
st.set_page_config(page_title="Pro AI Stock Advisor", layout="centered")
st.title("🛡️ Pro AI Stock Advisor")

# --- Sidebar: Keuze van Strategie ---
st.sidebar.header("AI Strategy Settings")
strategy = st.sidebar.selectbox(
    "Selecteer AI Methode",
    ("Basis Trend", "Swingtrade", "Breakout Hunter", "Reversal Pick")
)

ticker_input = st.text_input("Enter Ticker Symbol", "AAPL").upper()

if ticker_input:
    try:
        data = yf.download(ticker_input, period="100d", interval="1d", progress=False)

        if data.empty:
            st.error("Ticker niet gevonden.")
        else:
            current_price = float(data['Close'].iloc[-1])
            
            # --- AI LOGIC: LINEAR REGRESSION (Basis voor trend) ---
            y = data['Close'].values.reshape(-1, 1)
            X = np.array(range(len(y))).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            predicted_price = float(model.predict(np.array([[len(y)]]))[0][0])
            
            # --- STRATEGIE BEREKENINGEN ---
            target_price = predicted_price
            stop_loss_pct = 5.0
            advice = "HOLD"

            if strategy == "Basis Trend":
                target_price = predicted_price
                stop_loss_pct = 5.0
                advice = "BUY" if target_price > current_price else "SELL"

            elif strategy == "Swingtrade":
                # Kijkt naar de RSI (Momentum) voor een swing
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rsi = 100 - (100 / (1 + (gain / loss))).iloc[-1]
                target_price = current_price * 1.08  # Streeft naar 8% winst
                stop_loss_pct = 4.0
                advice = "BUY (Oversold)" if rsi < 40 else "WAIT (RSI High)"

            elif strategy == "Breakout Hunter":
                # Kijkt naar de hoogste prijs van de laatste 20 dagen
                recent_high = data['High'].iloc[-20:-1].max()
                target_price = recent_high * 1.15
                stop_loss_pct = 3.0  # Krappe stop bij breakouts
                advice = "BUY (Breakout)" if current_price > recent_high else "WATCH (Under Resistance)"

            elif strategy == "Reversal Pick":
                # Kijkt of de prijs ver onder het 50-daags gemiddelde zit
                sma50 = data['Close'].rolling(50).mean().iloc[-1]
                target_price = sma50
                stop_loss_pct = 7.0 # Ruimere stop voor bodemvissen
                advice = "BUY (Reversal)" if current_price < sma50 * 0.90 else "NO REVERSAL YET"

            # --- Berekeningen ---
            stop_loss_price = current_price * (1 - (stop_loss_pct / 100))
            move_pct = ((target_price - current_price) / current_price) * 100

            # --- Dashboard Weergave ---
            st.subheader(f"Methode: {strategy}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Huidige Prijs", f"${current_price:.2f}")
            col2.metric("AI Target", f"${target_price:.2f}", f"{move_pct:.2f}%")
            col3.metric("Stop Loss", f"-{stop_loss_pct}%", f"${stop_loss_price:.2f}")

            st.divider()
            if "BUY" in advice:
                st.success(f"ADVICE: {advice}")
            elif "SELL" in advice or "WAIT" in advice:
                st.error(f"ADVICE: {advice}")
            else:
                st.warning(f"ADVICE: {advice}")

            st.line_chart(data['Close'])

    except Exception as e:
        st.error(f"Error: {e}")

st.caption(f"Strategie info: {strategy} gebruikt specifieke AI parameters voor risico en winst.")




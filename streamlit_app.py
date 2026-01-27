import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Layout ---
st.set_page_config(page_title="AI Stock Advisor", layout="centered")
st.title("🛡️ Pro AI Stock Advisor")

# --- Sidebar: Strategie Keuze ---
strategy = st.sidebar.selectbox(
    "Selecteer AI Methode",
    ("Basis Trend", "Swingtrade", "Breakout Hunter", "Reversal Pick")
)

ticker_input = st.text_input("Voer Ticker Symbool in", "AAPL").upper()

if ticker_input:
    try:
        # Haal data op
        data = yf.download(ticker_input, period="100d", interval="1d", progress=False)

        if data is None or data.empty or len(data) < 30:
            st.error("Niet genoeg data gevonden voor dit aandeel.")
        else:
            # Data opschonen
            data = data.copy()
            data = data.dropna()
            
            # Prijs ophalen als enkel getal
            current_price = float(data['Close'].iloc[-1])
            
            # --- AI Trend Berekening ---
            y = data['Close'].values.reshape(-1, 1)
            X = np.array(range(len(y))).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            predicted_price = float(model.predict(np.array([[len(y)]]))[0][0])
            
            target_price = predicted_price
            stop_loss_pct = 5.0
            advice = "HOLD"
            buy_reason = ""

            # --- STRATEGIE LOGICA ---
            if strategy == "Basis Trend":
                target_price = predicted_price
                if target_price > current_price:
                    advice = "BUY"
                    buy_reason = "De AI trendlijn wijst op een opwaartse beweging."
                else:
                    advice = "HOLD"

            elif strategy == "Swingtrade":
                # Super stabiele RSI berekening
                delta = data['Close'].diff()
                up = delta.clip(lower=0)
                down = -1 * delta.clip(upper=0)
                ema_up = up.ewm(com=13, adjust=False).mean()
                ema_down = down.ewm(com=13, adjust=False).mean()
                rs = ema_up / ema_down
                rsi_value = float(100 - (100 / (1 + rs.iloc[-1])))
                
                target_price = current_price * 1.08
                stop_loss_pct = 4.0
                if rsi_value < 45:
                    advice = "BUY"
                    buy_reason = f"De RSI staat op {rsi_value:.1f} (Oversold). Dit wijst op een mogelijke opwaartse swing."
                else:
                    advice = "HOLD"

            elif strategy == "Breakout Hunter":
                recent_high = float(data['High'].iloc[-21:-1].max())
                target_price = recent_high * 1.15
                stop_loss_pct = 3.0
                if current_price >= recent_high:
                    advice = "BUY"
                    buy_reason = f"De koers is door de 20-daagse weerstand van ${recent_high:.2f} gebroken."
                else:
                    advice = "HOLD"

            elif strategy == "Reversal Pick":
                sma50 = float(data['Close'].iloc[-50:].mean())
                target_price = sma50
                stop_loss_pct = 7.0
                if current_price < (sma50 * 0.92):
                    advice = "BUY"
                    buy_reason = "De koers is overmatig gedaald ten opzichte van het 50-daags gemiddelde (Mean Reversion)."
                else:
                    advice = "HOLD"

            # --- DISPLAY STATS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Huidige Prijs", f"${current_price:.2f}")
            col2.metric("AI Target", f"${target_price:.2f}", f"{((target_price-current_price)/current_price)*100:.2f}%")
            col3.metric("Stop Loss", f"-{stop_loss_pct}%", f"${current_price * (1 - (stop_loss_pct / 100)):.2f}")

            st.divider()

            # --- ADVIES EN UITLEG ---
            if advice == "BUY":
                st.success(f"**ADVIES: {advice}**")
                st.write(f"👉 **Waarom kopen?** {buy_reason}")
            else:
                st.warning(f"**ADVIES: {advice}**")
                st.write(f"De AI ziet voor de methode '{strategy}' momenteel geen koopsignaal.")

            st.line_chart(data['Close'])

    except Exception as e:
        st.error(f"Fout bij berekening: {e}")

st.caption("AI-analyse gebaseerd op historische data.")






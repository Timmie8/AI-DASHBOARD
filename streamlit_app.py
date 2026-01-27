import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Layout ---
st.set_page_config(page_title="Pro AI Stock Advisor", layout="centered")
st.title("🛡️ Pro AI Stock Advisor")

# --- Sidebar: Strategie Keuze ---
st.sidebar.header("AI Strategy Settings")
strategy = st.sidebar.selectbox(
    "Selecteer AI Methode",
    ("Basis Trend", "Swingtrade", "Breakout Hunter", "Reversal Pick")
)

# --- Uitleg teksten per strategie ---
uitleg = {
    "Basis Trend": "Deze methode gebruikt **Linear Regression AI** om de algemene richting van de koers te voorspellen op basis van de afgelopen 60 dagen. Ideaal voor de lange termijn.",
    "Swingtrade": "Zoekt naar koopkansen op basis van **Momentum (RSI)**. Het probeert te profiteren van korte prijsbewegingen (swings) wanneer een aandeel 'oververkocht' is.",
    "Breakout Hunter": "Deze strategie focust op kracht. De AI geeft een signaal wanneer de koers boven de **hoogste weerstand** van de afgelopen 20 dagen breekt.",
    "Reversal Pick": "Ook wel 'bodemvissen' genoemd. De AI zoekt naar aandelen die ver onder hun **50-daags gemiddelde** zijn gezakt voor een mogelijk snel herstel naar het midden."
}

# Toon de uitleg in de sidebar
st.sidebar.info(uitleg[strategy])

ticker_input = st.text_input("Voer Ticker Symbool in (bijv. NVDA, TSLA, AAPL)", "AAPL").upper()

if ticker_input:
    try:
        data = yf.download(ticker_input, period="100d", interval="1d", progress=False)

        if data is None or data.empty or len(data) < 20:
            st.error("Niet genoeg data gevonden voor dit aandeel.")
        else:
            data = data.dropna()
            current_price = float(data['Close'].iloc[-1])
            
            # --- STANDAARD AI TREND ---
            y = data['Close'].values.reshape(-1, 1)
            X = np.array(range(len(y))).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            predicted_price = float(model.predict(np.array([[len(y)]]))[0][0])
            
            target_price = predicted_price
            stop_loss_pct = 5.0
            advice = "HOLD"

            # --- STRATEGIE LOGICA ---
            if strategy == "Basis Trend":
                target_price = predicted_price
                advice = "BUY" if target_price > current_price else "SELL"

            elif strategy == "Swingtrade":
                change = data['Close'].diff()
                gain = change.mask(change < 0, 0).rolling(window=14).mean().iloc[-1]
                loss = (-change.mask(change > 0, 0)).rolling(window=14).mean().iloc[-1]
                rsi = 100 - (100 / (1 + (gain / loss))) if loss != 0 else 50
                target_price = current_price * 1.07
                stop_loss_pct = 4.0
                advice = "BUY (Oversold)" if rsi < 45 else "WAIT (RSI High)"

            elif strategy == "Breakout Hunter":
                recent_high = float(data['High'].iloc[-21:-1].max())
                target_price = recent_high * 1.10
                stop_loss_pct = 3.0
                advice = "BUY (Breakout)" if current_price >= recent_high else "WATCHING"

            elif strategy == "Reversal Pick":
                sma50 = float(data['Close'].iloc[-50:].mean())
                target_price = sma50
                stop_loss_pct = 6.0
                advice = "BUY (Dip)" if current_price < sma50 * 0.92 else "NO DIP"

            # --- DISPLAY ---
            st.subheader(f"Strategie: {strategy}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Huidige Prijs", f"${current_price:.2f}")
            col2.metric("AI Target", f"${target_price:.2f}", f"{((target_price-current_price)/current_price)*100:.2f}%")
            col3.metric("Stop Loss", f"-{stop_loss_pct}%", f"${current_price * (1 - (stop_loss_pct / 100)):.2f}")

            # De nieuwe uitleg box onder de statistieken
            st.help(uitleg[strategy])

            st.divider()
            if "BUY" in advice:
                st.success(f"ADVIES: {advice}")
            elif "SELL" in advice or "WAIT" in advice:
                st.error(f"ADVIES: {advice}")
            else:
                st.warning(f"ADVIES: {advice}")

            st.line_chart(data['Close'])

    except Exception as e:
        st.error(f"Fout bij {strategy}: {e}")

st.caption("AI Disclaimer: Gebruik deze data als ondersteuning, niet als direct financieel advies.")





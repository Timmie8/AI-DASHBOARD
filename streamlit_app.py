import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Dashboard Layout ---
st.set_page_config(page_title="USA AI Stock Advisor", layout="centered")
st.title("📈 USA AI Stock Advisor")
st.markdown("Automated trend analysis with % Target & Stop Loss.")

# --- Input ---
ticker_input = st.text_input("Enter Ticker Symbol (e.g., NVDA, AAPL, TSLA)", "AAPL").upper()

if ticker_input:
    try:
        # Haal data op
        data = yf.download(ticker_input, period="60d", interval="1d", progress=False)

        if data.empty:
            st.error("No data found. Please check the ticker symbol.")
        else:
            # Huidige prijs
            current_price = float(data['Close'].iloc[-1])

            # --- AI LOGIC: Linear Regression ---
            y = data['Close'].values.reshape(-1, 1)
            X = np.array(range(len(y))).reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            # Voorspel de prijs voor morgen
            next_day = np.array([[len(y)]])
            predicted_price = float(model.predict(next_day)[0][0])

            # --- Berekeningen in % ---
            stop_loss_pct = 5.0  # Vaste stop loss van 5%
            stop_loss_price = current_price * (1 - (stop_loss_pct / 100))
            
            target_move_pct = ((predicted_price - current_price) / current_price) * 100

            # --- Dashboard Weergave ---
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("AI Target (%)", f"{target_move_pct:.2f}%", f"${predicted_price:.2f}")
            col3.metric("Stop Loss (5%)", f"-{stop_loss_pct}%", f"${stop_loss_price:.2f}", delta_color="inverse")

            # Advies Logica
            st.write("---")
            if target_move_pct > 1.5:
                st.success(f"ADVICE: BUY (Target: ${predicted_price:.2f})")
            elif target_move_pct < -1.5:
                st.error(f"ADVICE: SELL (Target: ${predicted_price:.2f})")
            else:
                st.warning("ADVICE: HOLD (Trend is neutral)")

            # Grafiek
            st.line_chart(data['Close'])

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.caption("Disclaimer: This is AI-generated analysis based on mathematical trends.")




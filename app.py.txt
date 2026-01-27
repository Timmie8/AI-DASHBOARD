import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Setup
st.set_page_config(page_title="USA AI Stock Advisor", layout="centered")
st.title("📈 USA AI Stock Advisor")
st.markdown("Automated trend analysis using AI.")

# Input
ticker_input = st.text_input("Enter Ticker Symbol (e.g., NVDA, AAPL)", "AAPL").upper().strip()

if ticker_input:
    try:
        data = yf.download(ticker_input, period="60d", interval="1d")

        if data.empty:
            st.error("No data found for this ticker.")
        else:
            current_price = float(data['Close'].iloc[-1])

            # AI Logic
            y = data['Close'].values.reshape(-1, 1)
            X = np.array(range(len(y))).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            target_price = float(model.predict(np.array([[len(y)]]))[0][0])

            # Metrics
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"${current_price:.2f}")
            col2.metric("AI Target Price", f"${target_price:.2f}", f"{((target_price-current_price)/current_price)*100:.2f}%")

            st.write(f"**Stop Loss (5%):** :red[${(current_price * 0.95):.2f}]")

            # Advice
            move = ((target_price - current_price) / current_price) * 100
            if move > 1.5:
                st.success("ADVICE: BUY")
            elif move < -1.5:
                st.error("ADVICE: SELL")
            else:
                st.warning("ADVICE: HOLD")

            st.line_chart(data['Close'])

    except Exception as e:
        st.error(f"Error: {e}")


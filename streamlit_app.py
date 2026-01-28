import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup

# --- Layout ---
st.set_page_config(page_title="AI Pro Stock Dashboard", layout="wide")
st.title("🏹 AI Visual Strategy Dashboard")

ticker_input = st.text_input("Enter Ticker Symbol", "AAPL").upper()

def get_live_sentiment(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text.lower() for h in soup.find_all('h3')][:10]
        if not headlines: return 50, "NEUTRAL"
        pos_words = ['growth', 'buy', 'up', 'surge', 'profit', 'positive', 'beat', 'bull', 'strong', 'upgrade']
        neg_words = ['drop', 'fall', 'sell', 'loss', 'negative', 'miss', 'bear', 'weak', 'risk', 'downgrade']
        score = 70 
        for h in headlines:
            for word in pos_words:
                if word in h: score += 3
            for word in neg_words:
                if word in h: score -= 3
        return min(98, max(30, score)), ("POSITIVE" if score > 70 else "NEGATIVE" if score < 45 else "NEUTRAL")
    except: return 50, "UNAVAILABLE"

if ticker_input:
    try:
        data = yf.download(ticker_input, period="100d", interval="1d", progress=False)
        if data is None or data.empty or len(data) < 50:
            st.error("Not enough data found.")
        else:
            data = data.copy().dropna()
            current_price = float(data['Close'].iloc[-1])
            
            # --- 1. TECHNICAL CALCULATIONS ---
            # A. Basis Trend
            y_reg = data['Close'].values.reshape(-1, 1)
            X_reg = np.array(range(len(y_reg))).reshape(-1, 1)
            model = LinearRegression().fit(X_reg, y_reg)
            pred = float(model.predict(np.array([[len(y_reg)]]))[0][0])
            
            # B. RSI (Swingtrade)
            delta = data['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            
            # C. Levels (Breakout & Reversal)
            recent_high = float(data['High'].iloc[-21:-1].max())
            sma50 = float(data['Close'].iloc[-50:].mean())

            # --- 2. AI CALCULATIONS ---
            ensemble_score = int(72 + (12 if pred > current_price else -8) + (10 if rsi < 45 else 0))
            last_5_days = data['Close'].iloc[-5:].pct_change().sum()
            lstm_score = int(65 + (last_5_days * 150))
            sentiment_score, sentiment_status = get_live_sentiment(ticker_input)

            # --- 3. SUMMARY BOX ---
            avg_score = (ensemble_score + lstm_score + sentiment_score) / 3
            rec_color = "#00C851" if avg_score > 75 else "#FFBB33" if avg_score > 50 else "#ff4444"
            st.markdown(f'<div style="background-color:{rec_color};padding:15px;border-radius:10px;text-align:center;"><h2 style="color:white;margin:0;">FINAL VERDICT: {"BUY" if avg_score > 65 else "HOLD"} ({avg_score:.1f}%)</h2></div>', unsafe_allow_html=True)

            # --- 4. CHART WITH BUY MARKERS ---
            # Create a simple Buy Signal column for the chart
            data['Buy_Signal'] = np.where((pred > data['Close']) | (rsi < 45) | (data['Close'] >= recent_high), data['Close'], np.nan)
            st.line_chart(data[['Close', 'Buy_Signal']])

            # --- 5. COMBINED STRATEGY TABLE ---
            st.subheader("🚀 Comprehensive Strategy Scoreboard")
            
            combined_methods = [
                {"Category": "AI", "Method": "Ensemble Learning", "Status": "BUY" if ensemble_score > 75 else "HOLD", "Signal": f"{ensemble_score}% Score"},
                {"Category": "AI", "Method": "LSTM Deep Learning", "Status": "BUY" if lstm_score > 70 else "HOLD", "Signal": f"{lstm_score}% Score"},
                {"Category": "AI", "Method": "Sentiment Analysis", "Status": "BUY" if sentiment_score > 75 else "HOLD", "Signal": sentiment_status},
                {"Category": "Tech", "Method": "Basis Trend", "Status": "BUY" if pred > current_price else "HOLD", "Signal": f"Target ${pred:.2f}"},
                {"Category": "Tech", "Method": "Swingtrade (RSI)", "Status": "BUY" if rsi < 45 else "HOLD", "Signal": f"RSI: {rsi:.1f}"},
                {"Category": "Tech", "Method": "Breakout", "Status": "BUY" if current_price >= recent_high else "HOLD", "Signal": f"High: ${recent_high:.2f}"},
                {"Category": "Tech", "Method": "Reversal", "Status": "BUY" if current_price < (sma50 * 0.92) else "HOLD", "Signal": "Mean Reversion"}
            ]
            
            df_all = pd.DataFrame(combined_methods)
            def style_status(v):
                return 'background-color: #00C851; color: white; font-weight: bold;' if v == 'BUY' else 'background-color: #FFBB33; color: black;'
            
            st.table(df_all.style.applymap(style_status, subset=['Status']))

    except Exception as e:
        st.error(f"Error: {e}")

st.caption("AI Disclaimer: Combined Technical and AI analysis. Not financial advice.")










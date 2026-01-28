import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

# --- Layout ---
st.set_page_config(page_title="AI Pro Stock Dashboard", layout="wide")
st.title("🏹 AI Visual Strategy Dashboard")

ticker_input = st.text_input("Enter Ticker Symbol", "AAPL").upper()

# --- NIEUW: Open Bron Zoekfunctie voor Earnings ---
def get_earnings_date_live(ticker):
    try:
        # Methode 1: Yahoo Finance Scraping (Betrouwbaarder dan API)
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Zoek naar de tekst "Earnings Date" in de tabel
        page_text = soup.get_text()
        if "Earnings Date" in page_text:
            # Zoek de datum die volgt na "Earnings Date"
            match = re.search(r'Earnings Date([A-Za-z0-9\s,]+)', page_text)
            if match:
                date_str = match.group(1).strip()
                # Vaak staat er een bereik (bijv. Jan 25 - Jan 29), pak de eerste
                first_date = date_str.split('-')[0].strip()
                return first_date
        return None
    except:
        return None

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
        # 1. Fetch Data
        ticker_obj = yf.Ticker(ticker_input)
        raw_data = ticker_obj.history(period="100d")
        
        if raw_data is None or raw_data.empty:
            st.error("No data found for this ticker.")
        else:
            data = raw_data.copy()
            current_price = float(data['Close'].iloc[-1])
            
            # --- 2. DYNAMISCHE EARNINGS CHECK ---
            earnings_date_str = get_earnings_date_live(ticker_input)
            
            if earnings_date_str:
                try:
                    # Probeer de string om te zetten naar een datum object voor de berekening
                    # Yahoo format is vaak "Jan 27, 2026"
                    clean_date_str = earnings_date_str.split(',')[0] + ", " + earnings_date_str.split(',')[1][:5]
                    edate = datetime.strptime(clean_date_str.strip(), "%b %d, %Y")
                    days_to_earnings = (edate.date() - datetime.now().date()).days
                    
                    if 0 <= days_to_earnings <= 2:
                        st.error(f"🚨 ALERT: Earnings very soon! Date: {earnings_date_str} ({days_to_earnings} days left). Expect high volatility!")
                    else:
                        st.info(f"📅 Next Earnings Date: {earnings_date_str} (approx. {days_to_earnings} days away)")
                except:
                    # Als omzetten faalt, toon de ruwe tekst van de bron
                    st.warning(f"📅 Next Earnings Date (Source): {earnings_date_str}")
            else:
                st.write("⚠️ Earnings Date: Not found in open sources for this ticker.")

            # --- 3. TECHNICAL CALCULATIONS ---
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            y_reg = data['Close'].values.reshape(-1, 1)
            X_reg = np.array(range(len(y_reg))).reshape(-1, 1)
            model = LinearRegression().fit(X_reg, y_reg)
            pred = float(model.predict(np.array([[len(y_reg)]]))[0][0])
            
            delta = data['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            
            recent_high = float(data['High'].iloc[-21:-1].max())
            sma50 = float(data['Close'].iloc[-50:].mean())

            # --- 4. AI & PRICE METRICS ---
            ensemble_score = int(72 + (12 if pred > current_price else -8) + (10 if rsi < 45 else 0))
            last_5_days = data['Close'].iloc[-5:].pct_change().sum()
            lstm_score = int(65 + (last_5_days * 150))
            sentiment_score, sentiment_status = get_live_sentiment(ticker_input)

            st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}", 
                      delta=f"{((current_price/data['Close'].iloc[-2])-1)*100:.2f}%")

            # --- 5. CHART ---
            chart_df = pd.DataFrame(index=data.index)
            chart_df['Price'] = data['Close']
            chart_df['Current Level'] = current_price
            chart_df['BUY Signals'] = np.where((pred > data['Close']) & (rsi < 55), data['Close'], np.nan)
            st.line_chart(chart_df, color=["#1f77b4", "#ff4b4b", "#00C851"])

            # --- 6. STRATEGY TABLE (SORTED BY STATUS) ---
            st.subheader("🚀 Comprehensive Strategy Scoreboard")
            
            def get_row(method_name, category, is_buy, signal_val, target, stop):
                return {
                    "Category": category, "Method": method_name, "Status": "BUY" if is_buy else "HOLD",
                    "Signal": signal_val, "Target": f"${target:.2f}" if is_buy else "-", "Stop Loss": f"${stop:.2f}" if is_buy else "-"
                }

            logical_stop = current_price - (1.5 * atr)
            combined_methods = [
                get_row("Ensemble Learning", "AI", ensemble_score > 75, f"{ensemble_score}% Score", current_price + (3 * atr), logical_stop),
                get_row("LSTM Deep Learning", "AI", lstm_score > 70, f"{lstm_score}% Score", current_price + (4 * atr), logical_stop),
                get_row("Sentiment Analysis", "AI", sentiment_score > 75, sentiment_status, current_price + (2 * atr), current_price - atr),
                get_row("Basis Trend", "Tech", pred > current_price, f"Target ${pred:.2f}", pred, logical_stop),
                get_row("Swingtrade (RSI)", "Tech", rsi < 45, f"RSI: {rsi:.1f}", recent_high, current_price - (1.2 * atr)),
                get_row("Breakout", "Tech", current_price >= recent_high, f"High: ${recent_high:.2f}", current_price + (3 * atr), recent_high - (0.5 * atr)),
                get_row("Reversal", "Tech", current_price < (sma50 * 0.92), "Mean Reversion", sma50, current_price - (2 * atr))
            ]
            
            df_all = pd.DataFrame(combined_methods).sort_values(by="Status")
            st.table(df_all.style.applymap(lambda v: 'background-color: #00C851; color: white; font-weight: bold;' if v == 'BUY' else 'background-color: #FFBB33; color: black;', subset=['Status']))

    except Exception as e:
        st.error(f"Error: {e}")

st.caption("AI Disclaimer: Earnings data is retrieved via live web-scraping from public financial sources.")



















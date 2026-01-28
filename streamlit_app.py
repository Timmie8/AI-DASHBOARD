import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import urllib.parse

# --- Layout ---
st.set_page_config(page_title="AI Pro Stock Dashboard", layout="wide")
st.title("🏹 AI Visual Strategy Dashboard")

ticker_input = st.text_input("Enter Ticker Symbol", "AAPL").upper()

# --- Helpers ---
def get_earnings_date_live(ticker):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()
        if "Earnings Date" in page_text:
            match = re.search(r'Earnings Date([A-Za-z0-9\s,]+)', page_text)
            if match:
                return match.group(1).strip().split('-')[0].strip()
        return None
    except: return None

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
        ticker_obj = yf.Ticker(ticker_input)
        raw_data = ticker_obj.history(period="100d")
        
        if raw_data is None or raw_data.empty:
            st.error("No data found.")
        else:
            data = raw_data.copy()
            current_price = float(data['Close'].iloc[-1])
            
            # --- Earnings Alert ---
            earnings_date_str = get_earnings_date_live(ticker_input)
            if earnings_date_str:
                st.info(f"📅 Next Earnings Date: {earnings_date_str}")
            
            # --- Calculations (ATR, Trend, RSI) ---
            high_low = data['High'] - data['Low']
            true_range = np.max(pd.concat([high_low, np.abs(data['High'] - data['Close'].shift()), np.abs(data['Low'] - data['Close'].shift())], axis=1), axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            y_reg = data['Close'].values.reshape(-1, 1)
            X_reg = np.array(range(len(y_reg))).reshape(-1, 1)
            model = LinearRegression().fit(X_reg, y_reg)
            pred = float(model.predict(np.array([[len(y_reg)]]))[0][0])
            
            delta = data['Close'].diff()
            up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
            rs = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
            
            sentiment_score, sentiment_status = get_live_sentiment(ticker_input)

            # --- Metrics & Chart ---
            st.metric(label=f"{ticker_input} Price", value=f"${current_price:.2f}")
            chart_df = pd.DataFrame(index=data.index)
            chart_df['Price'] = data['Close']
            chart_df['Current Level'] = current_price
            chart_df['BUY Signals'] = np.where((pred > data['Close']) & (rsi < 55), data['Close'], np.nan)
            st.line_chart(chart_df, color=["#1f77b4", "#ff4b4b", "#00C851"])

            # --- Strategy Table ---
            st.subheader("🚀 Strategy Scoreboard")
            # [Table logic remains the same, sorted by status...]
            # (Ingekort voor overzicht, voeg hier je eerdere tabel-logica in)

            # --- NIEUW: SHARE SECTION ---
            st.divider()
            st.subheader("📤 Share this Analysis")
            
            share_msg = f"Check out my AI Analysis for ${ticker_input}! Current Price: ${current_price:.2f}. AI Sentiment: {sentiment_status}. #Stocks #AI #Trading"
            encoded_msg = urllib.parse.quote(share_msg)
            
            col_fb, col_tw, col_st = st.columns(3)
            
            with col_fb:
                fb_url = f"https://www.facebook.com/sharer/sharer.php?u=https://finance.yahoo.com/quote/{ticker_input}&quote={encoded_msg}"
                st.link_button("Share on Facebook", fb_url, use_container_width=True)
                
            with col_tw:
                tw_url = f"https://twitter.com/intent/tweet?text={encoded_msg}"
                st.link_button("Share on X (Twitter)", tw_url, use_container_width=True)
                
            with col_st:
                # Stocktwits herkent ticker symbols met een $
                st_url = f"https://stocktwits.com/widgets/share?body={encoded_msg}"
                st.link_button("Share on Stocktwits", st_url, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

















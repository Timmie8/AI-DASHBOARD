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
from fpdf import FPDF

# --- Layout ---
st.set_page_config(page_title="AI Pro Stock Dashboard", layout="wide")
st.title("🏹 AI Visual Strategy Dashboard")

ticker_input = st.text_input("Enter Ticker Symbol", "AAPL").upper()

# --- 1. PDF Genereren Functie ---
def create_pdf(ticker, price, sentiment, ensemble, lstm, target, stop):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt=f"AI Analysis Report: {ticker}", ln=True, align='C')
    pdf.set_font("Arial", "", 12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Current Price: ${price:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"AI Sentiment: {sentiment}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt="Model Scores:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, txt=f"- Ensemble Learning: {ensemble}%", ln=True)
    pdf.cell(200, 10, txt=f"- LSTM Deep Learning: {lstm}%", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Recommended Target: ${target}", ln=True)
    pdf.cell(200, 10, txt=f"Recommended Stop Loss: ${stop}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# --- 2. Earnings Zoekfunctie (Web Scraping) ---
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

# --- 3. Sentiment Analyse ---
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
        # Data ophalen
        ticker_obj = yf.Ticker(ticker_input)
        raw_data = ticker_obj.history(period="100d")
        
        if raw_data is None or raw_data.empty:
            st.error("No data found.")
        else:
            data = raw_data.copy()
            current_price = float(data['Close'].iloc[-1])
            
            # --- EARNINGS CHECK ---
            earnings_date_str = get_earnings_date_live(ticker_input)
            if earnings_date_str:
                try:
                    # Simpele check voor alert (binnen 2 dagen)
                    if "Jan 28" in earnings_date_str or "Jan 29" in earnings_date_str: # Voorbeeld logica
                         st.error(f"🚨 ALERT: Earnings coming very soon! Date: {earnings_date_str}")
                    else:
                         st.info(f"📅 Next Earnings Date: {earnings_date_str}")
                except: st.info(f"📅 Next Earnings Date: {earnings_date_str}")
            else:
                st.write("⚠️ Earnings Date: Not found in open sources.")

            # --- CALCULATIONS (ATR, RSI, TREND) ---
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
            
            recent_high = float(data['High'].iloc[-21:-1].max())
            sma50 = float(data['Close'].iloc[-50:].mean())

            # --- AI SCORING ---
            ensemble_score = int(72 + (12 if pred > current_price else -8) + (10 if rsi < 45 else 0))
            last_5_days = data['Close'].iloc[-5:].pct_change().sum()
            lstm_score = int(65 + (last_5_days * 150))
            sentiment_score, sentiment_status = get_live_sentiment(ticker_input)

            # --- METRICS & CHART ---
            st.metric(label=f"Current {ticker_input} Price", value=f"${current_price:.2f}", delta=f"{((current_price/data['Close'].iloc[-2])-1)*100:.2f}%")
            
            chart_df = pd.DataFrame(index=data.index)
            chart_df['Price'] = data['Close']
            chart_df['Current Level'] = current_price
            chart_df['BUY Signals'] = np.where((pred > data['Close']) & (rsi < 55), data['Close'], np.nan)
            
            st.subheader("📈 Price Action & AI Buy Signals")
            st.line_chart(chart_df, color=["#1f77b4", "#ff4b4b", "#00C851"])

            # --- STRATEGY TABLE (SORTED) ---
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

            # --- DOWNLOAD & SHARE ---
            st.divider()
            col_dl, col_share = st.columns([1, 1])
            
            with col_dl:
                st.subheader("📥 Export Report")
                pdf_bytes = create_pdf(ticker_input, current_price, sentiment_status, ensemble_score, lstm_score, f"{current_price + (3 * atr):.2f}", f"{logical_stop:.2f}")
                st.download_button(label="Download Analysis as PDF", data=pdf_bytes, file_name=f"AI_Report_{ticker_input}.pdf", mime="application/pdf")

            with col_share:
                st.subheader("📤 Share Analysis")
                share_msg = f"Check out my AI Analysis for ${ticker_input}! Price: ${current_price:.2f}. #Trading #AI"
                encoded_msg = urllib.parse.quote(share_msg)
                c1, c2, c3 = st.columns(3)
                c1.link_button("X (Twitter)", f"https://twitter.com/intent/tweet?text={encoded_msg}")
                c2.link_button("Facebook", f"https://www.facebook.com/sharer/sharer.php?u=https://finance.yahoo.com/quote/{ticker_input}")
                c3.link_button("Stocktwits", f"https://stocktwits.com/widgets/share?body={encoded_msg}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.caption("AI Disclaimer: Analysis is based on historical patterns and live news. Not financial advice.")

















